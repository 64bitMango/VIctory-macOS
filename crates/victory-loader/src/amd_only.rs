//! Fewer exceptions from the AMD-only instructions a title uses: how often
//! each kind is emulated ([`crate::crash`]'s handler), and the ones replaced
//! in the title's code so that they raise none.
//!
//! An exception that reaches a vectored handler under Wine and Rosetta costs
//! tens of microseconds, and titles run these instructions in loops:
//!
//! - `clzero` loops (the two forms the handler does whole, one exception
//!   each) are found when the image is mapped and made to jump to the same
//!   loop with plain stores ([`patch_clzero_loops`], `VICTORY_PATCH_CLZERO=0`:
//!   left to the handler).
//! - `monitorx` and `mwaitx`, wait hints the handler steps over, are replaced
//!   at their first exception by a jump over themselves ([`replace_wait`],
//!   `VICTORY_PATCH_WAITS=0`: emulated each time). Found by the exception and
//!   not by their three bytes, which other instructions' operands hold too.
//!
//! Diagnostics builds log the counts per frame ([`log_emulated`]).

/// `cmp rm, reg` between 64-bit registers (`REX.W 39 /r`): (`rm`, `reg`).
pub(crate) fn compare_registers(code: &[u8]) -> Option<(u8, u8)> {
    let &[rex @ 0x48..=0x4f, 0x39, modrm] = code else {
        return None;
    };
    (modrm >> 6 == 3).then_some((modrm & 7 | (rex & 1) << 3, (modrm >> 3) & 7 | (rex & 4) << 1))
}

/// A `clzero` loop as the handler knows it (`crash::clzero`): one of
///
/// ```text
/// top: clzero                 top: mov rax, a
///      add rax, 0x40               add a, 0x40
///      cmp rax, b                  clzero
///      jb  top                     cmp a, b
///                                  jb  top
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClzeroLoop {
    /// Where the loop starts and where the instruction after its `jb` is,
    /// as offsets in the code searched.
    pub top: usize,
    pub end: usize,
    /// The instructions ahead of the stores in a pass (`mov rax, a` and
    /// `add a, 0x40`; none in the first form) and after them (`add rax,
    /// 0x40` in the first form; then `cmp`), as the title has them.
    before: [u8; 7],
    before_len: usize,
    after: [u8; 7],
    after_len: usize,
    /// Where the instruction is whose first bytes the jump takes besides the
    /// loop's first instruction: nothing may jump there.
    pub cut: usize,
}

/// The `clzero` loop whose `clzero` is at `code[at]`, if it is one the
/// handler does whole and neither of its registers is `rsp` (the stores
/// that take the loop's place use the stack for a moment).
pub(crate) fn clzero_loop(code: &[u8], at: usize) -> Option<ClzeroLoop> {
    if code.get(at..at + 3)? != [0x0f, 0x01, 0xfc] {
        return None;
    }
    const RSP: u8 = 4;
    if let Some(&[0x48, 0x83, 0xc0, 0x40, c0, c1, c2, 0x72, 0xf4]) = code.get(at + 3..at + 12)
        && let Some((0, limit)) = compare_registers(&[c0, c1, c2])
        && limit != 0
        && limit != RSP
    {
        return Some(ClzeroLoop {
            top: at,
            end: at + 12,
            before: [0; 7],
            before_len: 0,
            after: [0x48, 0x83, 0xc0, 0x40, c0, c1, c2],
            after_len: 7,
            cut: at + 3,
        });
    }
    let before: [u8; 7] = code.get(at.checked_sub(7)?..at)?.try_into().ok()?;
    let [rex @ (0x48 | 0x4c), 0x89, load, add_rex @ (0x48 | 0x49), 0x83, add, 0x40] = before else {
        return None;
    };
    let &[c0, c1, c2, 0x72, 0xf1] = code.get(at + 3..at + 8)? else {
        return None;
    };
    let (a, b) = compare_registers(&[c0, c1, c2])?;
    let found = load & 0xc7 == 0xc0
        && a == (load >> 3) & 7 | (rex & 4) << 1
        && add & 0xf8 == 0xc0
        && a == add & 7 | (add_rex & 1) << 3
        && a != 0
        && b != 0
        && a != RSP
        && b != RSP;
    found.then_some(ClzeroLoop {
        top: at - 7,
        end: at + 8,
        before,
        before_len: 7,
        after: [c0, c1, c2, 0, 0, 0, 0],
        after_len: 3,
        cut: at - 4,
    })
}

/// Every [`clzero_loop`] of `code`, in order.
pub(crate) fn clzero_loops(code: &[u8]) -> Vec<ClzeroLoop> {
    let mut found = Vec::new();
    let mut at = 0;
    while at + 3 <= code.len() {
        if code[at] == 0x0f
            && let Some(l) = clzero_loop(code, at)
        {
            at = l.end;
            found.push(l);
        } else {
            at += 1;
        }
    }
    found
}

/// Whether bytes near `target` read as a short jump to it (`jcc`, `jmp`,
/// `loop`, `jrcxz` with an 8-bit offset). Bytes, not decoded instructions:
/// it says yes too often, never no wrongly for such a jump.
pub(crate) fn short_jump_to(code: &[u8], target: usize) -> bool {
    let from = target.saturating_sub(129);
    let to = (target + 127).min(code.len().saturating_sub(1));
    (from..to).any(|p| {
        matches!(code[p], 0x70..=0x7f | 0xeb | 0xe0..=0xe3)
            && (p as i64 + 2 + i64::from(code[p + 1] as i8)) == target as i64
    })
}

/// `jmp rel32` at `at` to `to`, if it reaches.
fn jump(at: u64, to: u64) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The loop `l` (of code mapped at `code_at`) as code at `stub`, `clzero`
/// done by eight plain stores, ending with a jump to the instruction after
/// the loop. It leaves every register, the flags and memory as the loop
/// does, and so as the handler's emulation of it does:
///
/// ```text
/// pass: [mov rax, a; add a, 0x40]      the second form's, as the title has them
///       push rcx
///       mov  rcx, rax
///       and  rcx, -64                  clzero's line: the 64 bytes holding rax
///       mov  qword [rcx + 8k], 0       k = 0..8
///       pop  rcx                       before the compare, whose register it may be
///       [add rax, 0x40]                the first form's
///       cmp  .., b                     the title's own; the flags are its flags
///       jb   pass
///       jmp  the instruction after the loop
/// ```
///
/// (The eight bytes under the stack pointer are written, which a `clzero`
/// leaves alone; an exception writes far more there, and Windows code keeps
/// nothing below the stack pointer.)
pub(crate) fn clzero_stub(l: &ClzeroLoop, code_at: u64, stub: u64) -> Option<Vec<u8>> {
    let mut code = l.before[..l.before_len].to_vec();
    code.extend([0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
    for k in 0..8u8 {
        code.extend([0x48, 0xc7, 0x41, k * 8, 0, 0, 0, 0]);
    }
    code.push(0x59);
    code.extend(&l.after[..l.after_len]);
    let back = i8::try_from(-(code.len() as i64 + 2)).ok()?;
    code.extend([0x72, back as u8]);
    let here = stub + code.len() as u64;
    code.extend(jump(here, code_at + l.end as u64)?);
    Some(code)
}

/// Whether `clzero` loops are replaced when the image is mapped: 0 not
/// (`VICTORY_PATCH_CLZERO=0`), 1 unless bytes nearby read as a jump to the
/// one instruction the replacement cuts in two ([`short_jump_to`]), 2 those
/// too (`VICTORY_PATCH_CLZERO=all`).
fn patches_clzero(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("0") => 0,
        Some("all") => 2,
        _ => 1,
    }
}

/// Whether wait instructions are replaced at their first exception; not
/// with `VICTORY_PATCH_WAITS=0`.
fn patches_waits_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

#[cfg(windows)]
pub use on_windows::*;

#[cfg(windows)]
mod on_windows {
    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `IMAGE_SCN_MEM_EXECUTE`.
    const EXECUTE: u32 = 0x2000_0000;

    /// What the handler emulated, by kind (an exception each).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        /// `clzero`, one line.
        Clzero,
        /// A whole `clzero` loop in one exception.
        ClzeroLoop,
        Monitorx,
        Mwaitx,
        Rdpru,
        /// `clwb`, one instruction.
        Clwb,
        /// A whole `clwb` write-back loop in one exception.
        ClwbLoop,
        /// `extrq`, `insertq`.
        Sse4a,
        /// A wait instruction that was replaced before its exception was
        /// handled (or that the translator still runs as it was).
        Replaced,
    }
    const KINDS: usize = Kind::Replaced as usize + 1;

    static EMULATED: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    /// Nanoseconds in the handler for the exceptions counted.
    #[cfg(feature = "diagnostics")]
    static HANDLER_NS: AtomicU64 = AtomicU64::new(0);
    /// `clzero` loops replaced at load, and wait instructions replaced at
    /// their first exception, so far.
    static LOOPS_REPLACED: AtomicU64 = AtomicU64::new(0);
    static WAITS_REPLACED: AtomicU64 = AtomicU64::new(0);

    /// The handler emulated one of `kind`.
    pub(crate) fn count(kind: Kind) {
        EMULATED[kind as usize].fetch_add(1, Relaxed);
    }

    /// How many of `kind` the handler emulated since the last
    /// [`log_emulated`].
    #[cfg(test)]
    pub(crate) fn counted(kind: Kind) -> u64 {
        EMULATED[kind as usize].load(Relaxed)
    }

    /// Adds the time since it was made to the handler's time when dropped.
    #[cfg(feature = "diagnostics")]
    pub(crate) struct Timed(pub(crate) std::time::Instant);

    #[cfg(feature = "diagnostics")]
    impl Drop for Timed {
        fn drop(&mut self) {
            HANDLER_NS.fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
        }
    }

    /// Logs what the handler emulated since the last call, per frame over
    /// `frames` frames: every one of them was an exception. With the frame
    /// statistics (`victory_gpu::also_log_per_frame`).
    #[cfg(feature = "diagnostics")]
    pub fn log_emulated(frames: f64) {
        let n = |kind: Kind| EMULATED[kind as usize].swap(0, Relaxed) as f64 / frames.max(1.0);
        tracing::info!(
            "  AMD-only instructions emulated per frame (an exception each): clzero {:.1} single lines and {:.1} whole loops, \
             monitorx {:.1}, mwaitx {:.1}, rdpru {:.1}, clwb {:.1} single and {:.1} whole loops, extrq/insertq {:.1}, \
             waits already replaced {:.1}; {:.3} ms in their handler (the exception's own cost not counted); \
             replaced so far: {} clzero loops at load, {} wait instructions at their first exception",
            n(Kind::Clzero),
            n(Kind::ClzeroLoop),
            n(Kind::Monitorx),
            n(Kind::Mwaitx),
            n(Kind::Rdpru),
            n(Kind::Clwb),
            n(Kind::ClwbLoop),
            n(Kind::Sse4a),
            n(Kind::Replaced),
            HANDLER_NS.swap(0, Relaxed) as f64 / 1e6 / frames.max(1.0),
            LOOPS_REPLACED.load(Relaxed),
            WAITS_REPLACED.load(Relaxed),
        );
    }

    /// Whether `monitorx` and `mwaitx` are replaced at their first exception
    /// (`VICTORY_PATCH_WAITS=0`: emulated each time, as before).
    pub fn patches_waits() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| patches_waits_from(std::env::var("VICTORY_PATCH_WAITS").ok().as_deref()))
    }

    /// One thread at a time changes code: a second one would take the
    /// first one's loosened protection for the page's own.
    static PATCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Code tests let [`replace_wait`] change: (start, end) each.
    #[cfg(test)]
    pub(crate) static TEST_CODE: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

    /// The first byte of what replaces a wait instruction: `jmp` over one
    /// byte (`eb 01`), which with the instruction's last byte left behind it
    /// is three bytes that do nothing.
    pub(crate) const JUMP_OVER: [u8; 2] = [0xeb, 0x01];

    /// Replaces the `monitorx` or `mwaitx` (`0f 01 fa`, `0f 01 fb`) at `at`,
    /// where it has just raised an exception in a title's code, by a jump
    /// over itself (`eb 01 fa`, `eb 01 fb`): whether this call did.
    ///
    /// What the handler does for these two is nothing to registers, flags
    /// or memory, and neither does the jump. The difference is in time: no
    /// exception, and for `mwaitx` no yield to another ready thread
    /// (`SwitchToThread`), so a loop waiting with it turns faster.
    ///
    /// Safe with other threads running the same bytes: one byte changes,
    /// in one store, so a thread fetches either the old instruction (it
    /// faults and is emulated, as ever) or the jump, never a mix, and no
    /// thread can be inside the instruction. A thread whose exception was
    /// raised by the old bytes and handled after the change is known by the
    /// new bytes at its instruction pointer (`crash::emulate_amd_only`).
    ///
    /// # Safety
    /// `at` is where an illegal-instruction exception was just raised, and
    /// its three bytes were read as one of the two instructions.
    pub(crate) unsafe fn replace_wait(at: usize) -> bool {
        if !patches_waits() {
            return false;
        }
        #[cfg(test)]
        let known = TEST_CODE.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, b)| (a..b).contains(&at));
        #[cfg(not(test))]
        let known = false;
        // Only a title's own code: never the host's, or memory that merely
        // holds the bytes.
        if !known && !crate::threads::in_title_image(at) {
            return false;
        }
        let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
        let mut old = 0;
        // SAFETY: mapped code (it just ran), writable for the one store and
        // put back; the store is of one byte, whole to every other thread.
        unsafe {
            if VirtualProtect(at as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return false;
            }
            let done = AtomicU8::from_ptr(at as *mut u8).compare_exchange(0x0f, JUMP_OVER[0], SeqCst, SeqCst).is_ok();
            VirtualProtect(at as *const _, 1, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, 3);
            if done {
                WAITS_REPLACED.fetch_add(1, Relaxed);
            }
            done
        }
    }

    /// Makes the `clzero` loops in the executable sections of the image
    /// mapped at `base` jump to plain stores ([`clzero_stub`]), and returns
    /// how many, and how many it left to the handler because bytes nearby
    /// read as a jump into them ([`short_jump_to`]; none are left with
    /// `VICTORY_PATCH_CLZERO=all`). Nothing with `VICTORY_PATCH_CLZERO=0`.
    /// Before any title thread runs.
    ///
    /// The jump takes the loop's first five bytes: its first instruction
    /// and the start of its second. The rest stays, so code that enters the
    /// loop at its compare still finds it, and jumps back to the top; only
    /// a jump straight to that second instruction would land in the middle
    /// of ours, which is what is looked for.
    pub fn patch_clzero_loops(base: u64, info: &victory_pe::ImageInfo) -> (usize, usize) {
        let mode = patches_clzero(std::env::var("VICTORY_PATCH_CLZERO").ok().as_deref());
        if mode == 0 {
            return (0, 0);
        }
        let (mut patched, mut left) = (0, 0);
        let mut pool = None;
        for s in info.sections.iter().filter(|s| s.characteristics & EXECUTE != 0) {
            let start = base + u64::from(s.rva);
            let len = s.virtual_size.min(s.file_size) as usize;
            // SAFETY: the mapped section, before any title thread runs.
            let (p, l) = unsafe { patch_clzero_loops_in(start, len, base, mode == 2, &mut pool) };
            patched += p;
            left += l;
        }
        seal(pool);
        LOOPS_REPLACED.fetch_add(patched as u64, Relaxed);
        (patched, left)
    }

    /// [`patch_clzero_loops`] of the code `[start, start + len)`, its plain
    /// stores in `pool` (where it is and the bytes used; made near `base`
    /// when first needed); `every`: also the loops bytes nearby read as a
    /// jump into. Returns the loops replaced and left.
    ///
    /// # Safety
    /// The range is mapped code that no thread runs meanwhile.
    pub(crate) unsafe fn patch_clzero_loops_in(
        start: u64,
        len: usize,
        base: u64,
        every: bool,
        pool: &mut Option<(u64, usize)>,
    ) -> (usize, usize) {
        let (mut patched, mut left) = (0, 0);
        // SAFETY: mapped, readable code (the caller's).
        let code = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
        for l in clzero_loops(code) {
            if !every && short_jump_to(code, l.cut) {
                left += 1;
                continue;
            }
            if pool.is_none() {
                *pool = near(base).map(|at| (at, 0));
            }
            let Some((at, used)) = pool.as_mut() else {
                tracing::warn!("no memory near the image for clzero loops: left to the handler");
                left += 1;
                continue;
            };
            let stub = *at + *used as u64;
            let site = start + l.top as u64;
            let (Some(body), Some(jump)) = (clzero_stub(&l, start, stub), jump(site, stub)) else {
                left += 1;
                continue;
            };
            if *used + body.len() > POOL {
                left += 1;
                continue;
            }
            let mut old = 0;
            // SAFETY: our pool has room for the body; the site is the
            // title's code, writable for the moment of the write, which no
            // thread runs.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
                if VirtualProtect(site as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    left += 1;
                    continue;
                }
                std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
                VirtualProtect(site as *const _, jump.len(), old, &mut old);
            }
            *used += body.len().next_multiple_of(16);
            patched += 1;
        }
        (patched, left)
    }

    /// The plain stores are written: their memory becomes code.
    pub(crate) fn seal(pool: Option<(u64, usize)>) {
        if let Some((at, _)) = pool {
            let mut old = 0;
            // SAFETY: our pool, written; then our process's whole cache.
            unsafe {
                VirtualProtect(at as *const _, POOL, PAGE_EXECUTE_READ, &mut old);
                FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
            }
        }
    }

    /// Bytes for the loops' plain stores: 64 KiB, some 600 loops.
    const POOL: usize = 64 << 10;

    /// Memory for code within reach of a `jmp rel32` from the image at
    /// `base`: below it, as the masked stores' (`crate::masked::patch`), or
    /// above it where nothing is free below (an image mapped low).
    fn near(base: u64) -> Option<u64> {
        let below = (1..2048u64).filter_map(|step| base.checked_sub(step << 20));
        let above = (1..2048u64).filter_map(|step| base.checked_add(step << 20));
        below.chain(above).find_map(|want| {
            // SAFETY: a plain allocation at a hint; failure returns null.
            let got = unsafe { VirtualAlloc(want as *const _, POOL, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            (!got.is_null()).then_some(got as u64)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `clzero; add rax, 0x40; cmp rax, r10; jb`.
    const FIRST: [u8; 12] = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];
    /// `mov rax, rdx; add rdx, 0x40; clzero; cmp rdx, r8; jb`.
    const SECOND: [u8; 15] =
        [0x48, 0x89, 0xd0, 0x48, 0x83, 0xc2, 0x40, 0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2, 0x72, 0xf1];

    #[test]
    fn both_loops_are_found_with_their_ends() {
        let mut code = vec![0x90; 4];
        code.extend(FIRST);
        code.extend([0x90; 3]);
        code.extend(SECOND);
        code.push(0xc3);
        let found = clzero_loops(&code);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].top, found[0].end, found[0].cut), (4, 16, 7));
        assert_eq!((found[1].top, found[1].end, found[1].cut), (19, 34, 22));
        // A lone clzero, and the loops the handler does one line at a time:
        // against rax itself, with another step, with rax as the limit.
        assert!(clzero_loops(&[0x0f, 0x01, 0xfc, 0xc3]).is_empty());
        let mut own = FIRST;
        own[7..10].copy_from_slice(&[0x48, 0x39, 0xc0]);
        assert!(clzero_loops(&own).is_empty());
        let mut step = FIRST;
        step[6] = 0x20;
        assert!(clzero_loops(&step).is_empty());
        let mut limit = SECOND;
        limit[10..13].copy_from_slice(&[0x48, 0x39, 0xc2]);
        assert!(clzero_loops(&limit).is_empty());
        // Against the stack pointer: left to the handler.
        let mut stack = FIRST;
        stack[7..10].copy_from_slice(&[0x48, 0x39, 0xe0]);
        assert!(clzero_loops(&stack).is_empty());
        // Cut short.
        assert!(clzero_loops(&FIRST[..11]).is_empty());
        assert!(clzero_loops(&SECOND[7..]).is_empty());
    }

    #[test]
    fn the_stores_keep_the_titles_own_instructions() {
        let l = clzero_loop(&FIRST, 0).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        // push rcx; mov rcx, rax; and rcx, -64; eight stores; pop rcx.
        assert_eq!(stub[..8], [0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
        assert_eq!(stub[8..16], [0x48, 0xc7, 0x41, 0, 0, 0, 0, 0]);
        assert_eq!(stub[64..72], [0x48, 0xc7, 0x41, 56, 0, 0, 0, 0]);
        assert_eq!(stub[72], 0x59);
        // add rax, 0x40; cmp rax, r10; jb to the start.
        assert_eq!(stub[73..80], FIRST[3..10]);
        assert_eq!(stub[80..82], [0x72, (-82i8) as u8]);
        // jmp to the instruction after the loop.
        assert_eq!(stub[82], 0xe9);
        let rel = i32::from_le_bytes(stub[83..87].try_into().unwrap());
        assert_eq!(0x9000 + 87 + i64::from(rel), 0x1_0000 + 12);
        assert_eq!(stub.len(), 87);
        // The second form: its mov and add first, its compare after.
        let l = clzero_loop(&SECOND, 7).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        assert_eq!(stub[..7], SECOND[..7]);
        assert_eq!(stub[7], 0x51);
        assert_eq!(stub[80..83], SECOND[10..13]);
        assert_eq!(stub[83..85], [0x72, (-85i8) as u8]);
        let rel = i32::from_le_bytes(stub[86..90].try_into().unwrap());
        assert_eq!(0x9000 + 90 + i64::from(rel), 0x1_0000 + 15);
        // Out of a jump's reach.
        assert!(clzero_stub(&l, 0x1_0000, 0x9000_0000_0000).is_none());
    }

    #[test]
    fn short_jumps_to_an_address_are_seen() {
        let mut code = vec![0x90; 64];
        assert!(!short_jump_to(&code, 30));
        // jne +8 at 20: to 30.
        code[20..22].copy_from_slice(&[0x75, 8]);
        assert!(short_jump_to(&code, 30));
        assert!(!short_jump_to(&code, 31));
        // jmp -12 at 40: to 30.
        code[20] = 0x90;
        code[40..42].copy_from_slice(&[0xeb, (-12i8) as u8]);
        assert!(short_jump_to(&code, 30));
        // The loops' own `jb` goes to their top, not to what the jump cuts.
        let first = clzero_loop(&FIRST, 0).unwrap();
        assert!(!short_jump_to(&FIRST, first.cut));
        let second = clzero_loop(&SECOND, 7).unwrap();
        assert!(!short_jump_to(&SECOND, second.cut));
    }

    #[test]
    fn the_switches_are_on_unless_zero() {
        assert_eq!(patches_clzero(None), 1);
        assert_eq!(patches_clzero(Some("1")), 1);
        assert_eq!(patches_clzero(Some("0")), 0);
        assert_eq!(patches_clzero(Some("all")), 2);
        assert!(patches_waits_from(None));
        assert!(patches_waits_from(Some("1")));
        assert!(!patches_waits_from(Some(" 0 ")));
    }
}
