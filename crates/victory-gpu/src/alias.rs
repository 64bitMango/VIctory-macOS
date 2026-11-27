//! Zero-copy title buffers on Apple silicon through D3DMetal
//! (`mach_vm_remap` aliasing; from upstream #204, #322, #390 and #471).
//!
//! D3DMetal refuses `OpenExistingHeapFromAddress*`, so title buffers have
//! host copies (`crate::shadow`) and pages go back and forth. On Apple
//! silicon (unified memory), D3DMetal's own mapped write-back L0 buffers
//! are backed by the same physical RAM. Under Wine and Rosetta 2, raw
//! `mach_msg2_trap` MIG calls (`mach_vm_remap` with `copy = FALSE`,
//! `VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) point the title's `XMemVirtualAlloc`
//! range at such a buffer's physical pages: the copy is title memory itself,
//! and nothing is refreshed, copied back or tracked for it.
//!
//! Off unless `VICTORY_ALIAS_BUFFERS=1` (upstream has it on by default with
//! at least 20 GiB of RAM), never with WARP or without Rosetta. The probe at
//! start-up ([`supported`]) falls back to the copies where the remap does
//! not work; a region whose buffer or remap fails keeps plain memory, and
//! after [`REFUSED_MAX`] such failures no more regions are aliased (Metal
//! out of memory).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

use crate::host::host;
use crate::watch::PAGE;

#[cfg(target_arch = "x86_64")]
mod mach {
    use core::arch::asm;
    use std::sync::{Mutex, OnceLock};

    const MACH: u64 = 0x0100_0000;
    const TRAP_REPLY_PORT: u64 = 26;
    const TRAP_TASK_SELF: u64 = 28;
    /// `mach_msg2_trap` (8 arguments). `mach_msg_trap` (31) is fatal on
    /// macOS 27+ for kernel-object calls (`EXC_GUARD`).
    const TRAP_MSG2: u64 = 47;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;

    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    unsafe fn trap(n: u64, a: [u64; 6]) -> u64 {
        // SAFETY: forwarded to `trap8`.
        unsafe { trap8(n, a, 0, 0) }
    }

    unsafe fn trap8(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
        let ret: u64;
        // SAFETY: x86-64 XNU Mach trap ABI under Rosetta 2; 7th and 8th args
        // sit at `[rsp + 8]` and `[rsp + 16]`.
        unsafe {
            asm!(
                "sub rsp, 32",
                "mov qword ptr [rsp], 0",
                "mov qword ptr [rsp + 8], {a7}",
                "mov qword ptr [rsp + 16], {a8}",
                "syscall",
                "add rsp, 32",
                inlateout("rax") MACH | n => ret,
                in("rdi") a[0],
                in("rsi") a[1],
                in("rdx") a[2],
                in("r10") a[3],
                in("r8") a[4],
                in("r9") a[5],
                a7 = in(reg) a7,
                a8 = in(reg) a8,
                lateout("rcx") _,
                lateout("r11") _,
            );
        }
        ret
    }

    pub fn task_self() -> u32 {
        // SAFETY: argument-less Mach trap.
        unsafe { trap(TRAP_TASK_SELF, [0; 6]) as u32 }
    }

    fn reply_port() -> u32 {
        static PORT: OnceLock<u32> = OnceLock::new();
        *PORT.get_or_init(|| {
            // SAFETY: argument-less Mach trap.
            unsafe { trap(TRAP_REPLY_PORT, [0; 6]) as u32 }
        })
    }

    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct Header {
        bits: u32,
        size: u32,
        remote: u32,
        local: u32,
        voucher: u32,
        id: i32,
    }

    /// `mach_vm_remap` MIG request (`msgh_id = 4813`, `#pragma pack(4)`).
    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct RemapRequest {
        head: Header,
        descriptors: u32,
        src_task: u32,
        pad1: u32,
        disp_type: u32,
        ndr: [u8; 8],
        target_address: u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_address: u64,
        copy: u32,
        inheritance: u32,
    }

    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    struct RemapReply {
        head: Header,
        ndr: [u8; 8],
        ret: i32,
        target_address: u64,
        cur_protection: i32,
        max_protection: i32,
        trailer: [u8; 128],
    }

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    static REMAP_LOCK: Mutex<()> = Mutex::new(());

    /// Maps `[src, src + size)` of this task at `target`, sharing physical
    /// pages (`copy = FALSE`).
    ///
    /// # Safety
    /// Replaces whatever is mapped at `target`; both ranges must be page
    /// aligned and owned by the caller.
    pub unsafe fn remap_shared(target: u64, src: u64, size: u64, flags: i32) -> Remapped {
        let _guard = REMAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let task = task_self();
        let reply = reply_port();
        let mut msg = [0u8; size_of::<RemapReply>()];
        let req = RemapRequest {
            head: Header {
                // MACH_MSGH_BITS(COPY_SEND=19, MAKE_SEND_ONCE=21) | MACH_MSGH_BITS_COMPLEX
                bits: 19 | (21 << 8) | 0x8000_0000,
                size: size_of::<RemapRequest>() as u32,
                remote: task,
                local: reply,
                voucher: 0,
                id: 4813,
            },
            descriptors: 1,
            src_task: task,
            pad1: 0,
            disp_type: 19 << 16,
            ndr: [0, 0, 0, 0, 1, 0, 0, 0],
            target_address: target,
            size,
            mask: 0,
            flags,
            src_address: src,
            copy: 0,
            inheritance: 1,
        };
        // SAFETY: `req` and `msg` are plain-old-data structs of known size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const req).cast::<u8>(),
                msg.as_mut_ptr(),
                size_of::<RemapRequest>(),
            );
        }
        let pack = |lo: u32, hi: u32| (u64::from(hi) << 32) | u64::from(lo);
        let args = [
            msg.as_mut_ptr() as u64,
            3 | MACH64_SEND_KOBJECT_CALL,
            pack(req.head.bits, size_of::<RemapRequest>() as u32),
            pack(task, reply),
            pack(0, 4813),
            pack(1, reply),
        ];
        // SAFETY: `msg` holds the request and space for the reply and trailer.
        let r = unsafe { trap8(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
        // SAFETY: the kernel wrote at most `msg.len()` bytes into `msg`.
        let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
        Remapped {
            msg: r,
            kr: rep.ret,
            address: rep.target_address,
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod mach {
    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    pub fn task_self() -> u32 {
        0
    }

    pub unsafe fn remap_shared(_target: u64, _src: u64, _size: u64, _flags: i32) -> Remapped {
        Remapped {
            msg: 1,
            kr: -1,
            address: 0,
        }
    }
}

/// Physical memory (20 GiB) below which upstream leaves aliasing off by
/// default: pinned aliased regions there can run Metal out of memory. Here
/// it is off by default anyway; asked for below this, it warns.
const LOW_RAM_BYTES: u64 = 20 << 30;

/// Failed regions (no buffer, or the remap refused) after which no more
/// are aliased.
const REFUSED_MAX: u64 = 8;

fn total_ram_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is plain-old-data with `dwLength` initialized.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is valid and `dwLength` is set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.ullTotalPhys == 0 {
        None
    } else {
        Some(status.ullTotalPhys)
    }
}

/// `asked`: `Some(on)` when `VICTORY_ALIAS_BUFFERS` is set (`0` is off, any
/// other value on), `None` when it is not: off.
fn decide_enabled(asked: Option<bool>, rosetta: bool, warp: bool) -> bool {
    asked.unwrap_or(false) && rosetta && !warp
}

/// Whether title buffers are aliased on this host: asked for with
/// `VICTORY_ALIAS_BUFFERS=1`, under Rosetta with D3DMetal. Logs once what
/// was decided, and with it on, the start-up probe's result.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let rosetta = crate::watch::rosetta();
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        let asked = std::env::var_os("VICTORY_ALIAS_BUFFERS").map(|_| crate::switch::on("VICTORY_ALIAS_BUFFERS"));
        let ram = total_ram_bytes();
        let on = decide_enabled(asked, rosetta, warp);
        let ram_gib = ram.map(|b| b / (1 << 30));
        if !on {
            tracing::info!(
                asked = ?asked,
                rosetta,
                warp,
                "zero-copy title buffers off: title buffers have host copies (VICTORY_ALIAS_BUFFERS=1 turns aliasing on under Rosetta)"
            );
            return false;
        }
        if ram.is_some_and(|r| r < LOW_RAM_BYTES) {
            tracing::warn!(ram_gib, "zero-copy title buffers asked for with less than 20 GiB of RAM: Metal may run out of memory");
        }
        // The probe now, so that the run shows at start-up whether it works.
        match supported() {
            Some(zeroed) => tracing::info!(
                ram_gib,
                zeroed,
                "zero-copy title buffers on (VICTORY_ALIAS_BUFFERS=1): title memory the title commits read-write is aliased onto host buffers (mach_vm_remap); aliased so far: 0 regions, 0 MB"
            ),
            None => tracing::warn!(
                ram_gib,
                "zero-copy title buffers asked for (VICTORY_ALIAS_BUFFERS=1) but the remap probe failed: title buffers keep host copies"
            ),
        }
        true
    })
}

/// Copies by the host GPU itself, to prove that what the title writes is
/// what the GPU reads and the other way round: a remap the kernel accepts
/// could still leave the GPU on other pages.
mod gpu {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::{Error, Interface, Result};

    fn committed(
        device: &ID3D12Device,
        kind: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource> {
        let props = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut res = None;
        // SAFETY: a plain committed buffer, its slot on this stack.
        unsafe {
            device.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut res,
            )?;
        }
        res.ok_or_else(Error::empty)
    }

    /// Records one list with `f` on a queue of its own and waits for it
    /// (five seconds at most: false then).
    fn run(device: &ID3D12Device, f: impl FnOnce(&ID3D12GraphicsCommandList)) -> Result<bool> {
        // SAFETY: a standalone queue, list and fence, waited for here.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let alloc: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
            f(&list);
            list.Close()?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            queue.Signal(&fence, 1)?;
            if fence.GetCompletedValue() < 1 {
                let event = CreateEventW(None, false, false, None)?;
                fence.SetEventOnCompletion(1, event)?;
                WaitForSingleObject(event, 5000);
                let _ = CloseHandle(event);
            }
            Ok(fence.GetCompletedValue() >= 1)
        }
    }

    /// `len` bytes of `src` at `offset`, as the GPU reads them.
    pub(super) fn read(
        device: &ID3D12Device,
        src: &ID3D12Resource,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>> {
        let back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            len as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: both buffers hold the range.
        if !run(device, |l| unsafe {
            l.CopyBufferRegion(&back, 0, src, offset, len as u64)
        })? {
            return Err(Error::empty());
        }
        let mut out = vec![0u8; len];
        // SAFETY: a readback buffer of `len` bytes, the copy done.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            back.Map(0, None, Some(&mut ptr))?;
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), len);
            back.Unmap(0, None);
        }
        Ok(out)
    }

    /// Writes `bytes` into `dst` at `offset` by a GPU copy.
    pub(super) fn write(
        device: &ID3D12Device,
        dst: &ID3D12Resource,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        let up = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        // SAFETY: an upload buffer of `bytes.len()` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut ptr))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: both buffers hold the range.
        if run(device, |l| unsafe {
            l.CopyBufferRegion(dst, offset, &up, 0, bytes.len() as u64)
        })? {
            Ok(())
        } else {
            Err(Error::empty())
        }
    }

    /// Whether title memory at `title` (`len` bytes) and `buffer` are the
    /// same bytes to the GPU, both ways, compared byte for byte: a pattern
    /// the title writes is what a GPU copy of the buffer reads, and a
    /// pattern a GPU copy writes into the buffer is what the title reads.
    /// Leaves the memory zeroed.
    pub(super) fn same_bytes(
        device: &ID3D12Device,
        buffer: &ID3D12Resource,
        title: *mut u8,
        len: usize,
    ) -> bool {
        let pattern = |seed: u8| -> Vec<u8> {
            (0..len)
                .map(|i| (i as u8).wrapping_mul(37) ^ (i >> 8) as u8 ^ seed)
                .collect()
        };
        let written = pattern(0x5a);
        // SAFETY: (all three) `title` holds `len` bytes.
        unsafe { std::ptr::copy_nonoverlapping(written.as_ptr(), title, len) };
        let gpu_reads_title = read(device, buffer, 0, len).is_ok_and(|seen| seen == written);
        let from_gpu = pattern(0xa5);
        let gpu_wrote = write(device, buffer, 0, &from_gpu).is_ok();
        // SAFETY: as above.
        let title_reads_gpu =
            gpu_wrote && unsafe { std::slice::from_raw_parts(title, len) } == &from_gpu[..];
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(title, 0, len) };
        gpu_reads_title && title_reads_gpu
    }
}

/// Probes once whether `mach_vm_remap` onto a D3DMetal write-back L0
/// buffer works, proven by GPU copies both ways compared byte for byte
/// ([`gpu::same_bytes`]), and whether fresh buffers arrive zero-initialized:
/// `None` where it does not work, else whether they do.
fn supported() -> Option<bool> {
    static PROBE: OnceLock<Option<bool>> = OnceLock::new();
    *PROBE.get_or_init(|| {
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        if !crate::watch::rosetta() || warp || mach::task_self() == 0 {
            return None;
        }
        let h = host()?;
        let (buffer, gpu_mem) = h.shared_buffer(PAGE as u64)?;
        // SAFETY: `gpu_mem` is our `PAGE`-byte mapped buffer.
        let zeroed = unsafe {
            gpu_mem.cast::<u64>().read_volatile() == 0
                && gpu_mem.add(PAGE - 8).cast::<u64>().read_volatile() == 0
        };
        // SAFETY: standalone `PAGE`-byte allocation freed below.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        if title.is_null() {
            return None;
        }
        // SAFETY: `title` and `gpu_mem` are both `PAGE`-aligned and `PAGE` bytes.
        let r = unsafe {
            mach::remap_shared(
                title as u64,
                gpu_mem as u64,
                PAGE as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        let ok = if r.msg == 0 && r.kr == 0 && r.address == title as u64 {
            // SAFETY: both pointers point to the same `PAGE` of RAM.
            let cpu = unsafe {
                title.add(128).write_volatile(0x5a);
                let seen_gpu = gpu_mem.add(128).read_volatile() == 0x5a;
                gpu_mem.add(256).write_volatile(0xa5);
                let seen_title = title.add(256).read_volatile() == 0xa5;
                seen_gpu && seen_title
            };
            // Two CPU views agreeing is not enough: the GPU itself must read
            // what the title wrote, and the title what the GPU wrote.
            let gpu = cpu && gpu::same_bytes(&h.device, &buffer, title, PAGE);
            if cpu && !gpu {
                tracing::warn!(
                    "mach_vm_remap probe: the CPU views agree but a GPU copy saw other bytes"
                );
            }
            gpu
        } else {
            false
        };
        // SAFETY: `title` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
        drop(buffer);
        if ok {
            tracing::info!(
                zeroed,
                "mach_vm_remap probe: title memory can alias host buffers"
            );
            Some(zeroed)
        } else {
            tracing::warn!(
                ?r,
                "mach_vm_remap probe failed: title buffers keep host copies"
            );
            None
        }
    })
}

/// Regions aliased now and their bytes, regions that failed, and the bytes
/// at which the next total is logged.
static REGIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static NEXT_LOG: AtomicU64 = AtomicU64::new(0);

/// Regions aliased now and their bytes.
pub fn aliased_now() -> (u64, u64) {
    (REGIONS.load(Relaxed), BYTES.load(Relaxed))
}

/// The first region, and each time the bytes aliased pass 256 MiB, then
/// each doubling: logged.
fn note_aliased(len: u64) {
    let regions = REGIONS.fetch_add(1, Relaxed) + 1;
    let bytes = BYTES.fetch_add(len, Relaxed) + len;
    let next = NEXT_LOG.load(Relaxed);
    if bytes >= next
        && NEXT_LOG
            .compare_exchange(
                next,
                (bytes + 1).next_power_of_two().max(256 << 20),
                Relaxed,
                Relaxed,
            )
            .is_ok()
    {
        tracing::info!(
            regions,
            mb = bytes >> 20,
            "zero-copy title buffers: aliased so far (the title's own memory, not extra)"
        );
    }
}

fn note_gone(shadow: &crate::shadow::Shadow) {
    REGIONS.fetch_sub(1, Relaxed);
    BYTES.fetch_sub(shadow.range().1, Relaxed);
}

/// A region failed: past [`REFUSED_MAX`], no more are aliased.
fn refused() {
    if REFUSED.fetch_add(1, Relaxed) + 1 == REFUSED_MAX {
        let (regions, bytes) = aliased_now();
        tracing::warn!(
            regions,
            mb = bytes >> 20,
            "zero-copy title buffers: too many regions failed (Metal out of memory?): new title memory keeps host copies"
        );
    }
}

/// Aliases `[base, base + len)` onto a new write-back L0 D3D12 buffer so CPU
/// and GPU share physical pages with zero copy (`preserve_contents`: the
/// bytes there go along; else it reads zero). False, with the memory left as
/// it was, where it cannot be.
pub fn alias_region(base: usize, len: usize, preserve_contents: bool) -> bool {
    static LOCK: Mutex<()> = Mutex::new(());
    let Some(zeroed) = supported() else {
        return false;
    };
    if REFUSED.load(Relaxed) >= REFUSED_MAX
        || base == 0
        || len < PAGE
        || !base.is_multiple_of(PAGE)
        || !len.is_multiple_of(PAGE)
    {
        return false;
    }
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if crate::shadow::has_aliased_overlap(base as u64, end as u64) {
        return false;
    }
    let Some(h) = host() else {
        return false;
    };
    let Some((buffer, gpu_mem)) = h.shared_buffer(len as u64) else {
        refused();
        return false;
    };
    // SAFETY: `base` and `gpu_mem` are `len`-byte owned mappings.
    unsafe {
        if preserve_contents {
            std::ptr::copy_nonoverlapping(base as *const u8, gpu_mem, len);
        } else if !zeroed {
            std::ptr::write_bytes(gpu_mem, 0, len);
        }
        let r = mach::remap_shared(
            base as u64,
            gpu_mem as u64,
            len as u64,
            mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
        );
        if r.msg != 0 || r.kr != 0 || r.address != base as u64 {
            tracing::warn!(
                base = format_args!("{base:#x}"),
                len,
                ?r,
                "mach_vm_remap failed; keeping plain memory"
            );
            refused();
            return false;
        }
    }
    crate::host::name(&buffer, || {
        format!("aliased title memory {base:#x}+{len:#x}")
    });
    // SAFETY: `buffer` is alive.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = crate::shadow::Shadow::new_aliased(buffer, gpu_mem, gpu, base as u64, len as u64);
    crate::shadow::register_aliased(shadow);
    note_aliased(len as u64);
    if len >= 16 << 20 {
        tracing::info!(
            base = format_args!("{base:#x}"),
            len,
            "large title memory region aliased zero-copy"
        );
    } else {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            "title memory region aliased zero-copy"
        );
    }
    true
}

/// Removes the aliased region registered at `base` (the title frees it).
pub fn unalias_region(base: usize) -> bool {
    crate::shadow::unregister_aliased(base as u64)
        .inspect(|s| note_gone(s))
        .is_some()
}

/// Removes every aliased region overlapping `[base, base + len)`: the
/// memory there is new (a fresh reservation), so any left are stale ones
/// the title freed without telling (`VirtualFree`). Returns how many.
pub fn unalias_within(base: usize, len: usize) -> usize {
    let gone =
        crate::shadow::unregister_aliased_within(base as u64, base.saturating_add(len) as u64);
    for s in &gone {
        note_gone(s);
    }
    if !gone.is_empty() {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            stale = gone.len(),
            "stale zero-copy regions dropped for a new reservation"
        );
    }
    gone.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
    };
    use windows_sys::Win32::System::Memory::{
        MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };

    struct AliasedAlloc {
        ptr: *mut u8,
        len: usize,
        aliased: bool,
    }

    impl AliasedAlloc {
        fn new(len: usize, init: Option<&dyn Fn(*mut u8)>, preserve: bool) -> Self {
            // SAFETY: allocating `len` bytes of RW memory for the test.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            if let Some(f) = init {
                f(ptr);
            }
            let aliased = alias_region(ptr as usize, len, preserve);
            assert!(aliased, "alias_region failed");
            Self { ptr, len, aliased }
        }
    }

    impl Drop for AliasedAlloc {
        fn drop(&mut self) {
            if self.aliased {
                unalias_region(self.ptr as usize);
            }
            // As the runtime does when the title unmaps memory: the guards
            // let go of its pages. A tracker some test or copy still holds
            // left them in the guard set otherwise, and a test running at
            // the same time that was given the address had them skipped as
            // guarded already, so they stayed writable (#31:
            // `takes_guard_one_view_per_call` after
            // `aliased_gpu_writes_notify_overlapping_texture_tracker`).
            crate::watch::title_memory_remapped(self.ptr as usize, self.len);
            // SAFETY: `self.ptr` was allocated by `VirtualAlloc`.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn gpu_read(device: &ID3D12Device, src: &ID3D12Resource, offset: u64, len: usize) -> Vec<u8> {
        gpu::read(device, src, offset, len).expect("a GPU read-back")
    }

    fn gpu_write(device: &ID3D12Device, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) {
        gpu::write(device, dst, offset, bytes).expect("a GPU write")
    }

    #[test]
    fn the_probe_says_whether_the_remap_works_here() {
        let rosetta = crate::watch::rosetta();
        let gpu = host().is_some();
        let probe = supported();
        eprintln!(
            "zero-copy probe: rosetta {rosetta}, host GPU {gpu}, remap works {}, buffers zeroed {:?}",
            probe.is_some(),
            probe
        );
        if !rosetta {
            assert!(probe.is_none(), "never without Rosetta");
        }
        // Where it works, a region the way the runtime aliases one is the
        // same bytes to the GPU as to the title, every byte, both ways.
        if probe.is_some() {
            let len = 4 * PAGE;
            let alloc = AliasedAlloc::new(len, None, false);
            let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
            let h = host().expect("a host GPU");
            assert!(gpu::same_bytes(&h.device, &shadow.buffer, alloc.ptr, len));
        }
    }

    #[test]
    fn switch_respects_rosetta_and_warp() {
        assert!(!decide_enabled(Some(false), true, false));
        assert!(!decide_enabled(Some(true), false, false));
        assert!(!decide_enabled(Some(true), true, true));
        assert!(decide_enabled(Some(true), true, false));
    }

    /// Unlike upstream, unset is off whatever the RAM: the copies stay
    /// until aliasing is compared in the game.
    #[test]
    fn unset_is_off() {
        assert!(!decide_enabled(None, true, false));
        assert!(!decide_enabled(None, false, false));
        assert!(!decide_enabled(None, true, true));
        assert!(total_ram_bytes().is_some());
    }

    #[test]
    fn aliased_buffer_is_coherent_with_gpu_copies() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 4 * PAGE;
        let alloc = AliasedAlloc::new(
            len,
            Some(&|p| {
                for i in (0..len).step_by(4096) {
                    // SAFETY: `p` has `len` bytes.
                    unsafe { p.add(i).write((i >> 12) as u8) };
                }
            }),
            true,
        );
        // Initial contents survived remap.
        for i in (0..len).step_by(4096) {
            // SAFETY: `alloc.ptr` has `len` bytes.
            assert_eq!(unsafe { alloc.ptr.add(i).read_volatile() }, (i >> 12) as u8);
        }

        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(shadow.clean());
        assert_eq!(shadow.update(alloc.ptr as usize, len), 0);
        assert_eq!(shadow.copy_back(alloc.ptr as usize, len), 0);

        for round in 0..8u8 {
            let off = len - 2 * PAGE;
            for i in 0..PAGE {
                let b = (i as u8).wrapping_mul(31) ^ round;
                // SAFETY: inside `alloc`.
                unsafe { alloc.ptr.add(off + i).write_volatile(b) };
            }
            let got = gpu_read(&h.device, &shadow.buffer, off as u64, PAGE);
            for (i, &b) in got.iter().enumerate() {
                assert_eq!(b, (i as u8).wrapping_mul(31) ^ round);
            }

            let new: Vec<u8> = (0..PAGE).map(|i| (i as u8) ^ round ^ 0xc3).collect();
            gpu_write(&h.device, &shadow.buffer, (len - PAGE) as u64, &new);
            // SAFETY: inside `alloc`.
            let cpu = unsafe { std::slice::from_raw_parts(alloc.ptr.add(len - PAGE), PAGE) };
            assert_eq!(cpu, &new[..]);
        }
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn aliased_gpu_writes_notify_overlapping_texture_tracker() {
        if !crate::watch::rosetta() || !crate::watch::fault_tracking() {
            return;
        }
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 2 * PAGE;
        // SAFETY: allocating `len` bytes with `MEM_WRITE_WATCH` for tracker test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let alloc = AliasedAlloc {
            ptr,
            len,
            aliased: true,
        };
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let base = alloc.ptr as usize;

        let tex = crate::watch::track(base, len, false).expect("texture tracker");
        let _ = tex.take(base, base + len);
        assert!(tex.clean());

        assert_eq!(shadow.copy_back(base, PAGE), 0);
        assert!(tex.changed(base / PAGE));
        assert!(!tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert!(tex.clean());
        assert_eq!(shadow.copy_back_marked(&[base + PAGE]), 0);
        assert!(!tex.changed(base / PAGE));
        assert!(tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert_eq!(shadow.copy_back_page(base), 0);
        assert!(tex.changed(base / PAGE));
    }

    #[test]
    fn aliased_memory_supports_virtual_query_and_write_guard() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        static GUARD_AT: AtomicUsize = AtomicUsize::new(0);
        static GUARD_HIT: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: `info` is provided by Windows/Wine exception dispatch.
            unsafe {
                let rec = &*(*info).ExceptionRecord;
                let page = GUARD_AT.load(Ordering::SeqCst);
                if rec.ExceptionCode as u32 == 0xC000_0005 && page != 0 {
                    let addr = rec.ExceptionInformation[1];
                    if addr >= page && addr < page + PAGE {
                        let mut old = 0;
                        VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old);
                        GUARD_HIT.store(true, Ordering::SeqCst);
                        return -1;
                    }
                }
                0
            }
        }

        let len = 2 * PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        // SAFETY: querying our own allocation.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let q = VirtualQuery(alloc.ptr.add(PAGE).cast(), &mut info, size_of_val(&info));
            assert_ne!(q, 0);
            assert_eq!(info.State, MEM_COMMIT);
            assert_eq!(info.Protect, PAGE_READWRITE);

            let handler = AddVectoredExceptionHandler(1, Some(on_fault));
            let page = alloc.ptr.add(PAGE);
            GUARD_AT.store(page as usize, Ordering::SeqCst);
            GUARD_HIT.store(false, Ordering::SeqCst);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), PAGE, PAGE_READONLY, &mut old),
                0
            );
            page.add(100).write_volatile(0x77);
            let hit = GUARD_HIT.load(Ordering::SeqCst);
            GUARD_AT.store(0, Ordering::SeqCst);
            RemoveVectoredExceptionHandler(handler);
            assert!(hit, "VirtualProtect write guard did not fault");
        }
        assert_eq!(
            gpu_read(&h.device, &shadow.buffer, (PAGE + 100) as u64, 1),
            vec![0x77]
        );
    }

    #[test]
    fn unalias_region_removes_shadow_and_allows_virtual_free() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: allocating `len` bytes for explicit unalias + free test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let shadow = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(unalias_region(ptr as usize));
        assert!(crate::shadow::find(ptr as u64, len as u64).is_none_or(|s| !s.is_aliased()));
        // SAFETY: `ptr` was allocated by `VirtualAlloc`.
        assert_ne!(unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) }, 0);
        drop(shadow);
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }
}
