//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;
#[cfg(feature = "diagnostics")]
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicU64, Ordering::{Acquire, Release, SeqCst}};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::paged::Pages;
use super::{VIEW_TAG, View, view_at, views_current};
use crate::host::host;

struct DynamicSlots {
    cursor: usize,
    taken: Vec<bool>,
    /// Once a search fails, this size and larger cannot fit until release.
    first_failed_size: Option<usize>,
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each title slot was last
    /// made from, 0 for none; read without a lock, written with the slot's
    /// lock held ([`HostHeap::lock_slot`]).
    ///
    /// One mutex held this table, and every draw took it for every table
    /// entry it read, as every view written and every slot a heap sync made
    /// did: the title's recording threads waited over a second a frame on it
    /// between them (runs ab-locks-on, ab-locks-off). What it gave, and what
    /// gives it now:
    /// - One thread at a time makes a slot, and the one that makes it last
    ///   read its record last: the slot's lock, one of [`STRIPES`] by the
    ///   slot's run of [`RUN`] slots, held from reading the record to
    ///   entering what the slot views ([`HostHeap::make_locked`]). Slots of
    ///   other runs are made beside it.
    /// - A slot found made from a record holds that record's view, whole:
    ///   its entry here is written last, after the host view, its stride,
    ///   mark and address, what it views (`crate::bindless`) and what it
    ///   holds ([`HostHeap::keep`], which sets its bit in `held` first). So
    ///   a draw that finds its record's key here, and the slot holding a
    ///   texture where it needs one, takes no lock
    ///   ([`HostHeap::made_already`]): it makes nothing, so it cannot undo
    ///   another thread's making, and what it finds was complete when it
    ///   looked. While a slot is looked at under its lock its entry is
    ///   [`BUSY`], set before the record is read: a thread that wrote a
    ///   record and then finds the entry that record's key (after a fence,
    ///   so that its write is seen by whoever sets `BUSY` next) knows that
    ///   no thread is making the slot from a record read before its write.
    ///   Without that a slot written back to what it was made from, while
    ///   another thread made it from the record in between, was taken for
    ///   made and ended as the record in between (seen once in six runs of
    ///   `the_last_of_several_threads_writing_a_slot_at_once_makes_it`).
    /// - Passes that need every slot still ([`HostHeap::hold_slots`]: the
    ///   release of idle copies, giving textures up, the first fill of the
    ///   view addresses) hold every stripe.
    ///
    /// `VICTORY_SLOT_LOCKS=0`: one lock for all slots, taken for every look.
    made: Box<[AtomicU64]>,
    /// A bit per title slot: whether `kept` holds a host resource for it.
    /// Written by [`HostHeap::keep`] under `kept`'s lock, read without one.
    held: Box<[AtomicU64]>,
    /// The slots' locks ([`HostHeap::lock_slot`]).
    locks: Box<[Stripe]>,
    /// Whether slots have locks by run and are looked at without one
    /// ([`slot_locks`]), and whether a sync makes a run's slots under one
    /// hold ([`sync_runs`]), as they were when the heap was made: every
    /// thread takes the same lock for a slot.
    by_run: [bool; 2],
    /// Diagnostics, per title slot: the frame (plus 1) a draw last read it
    /// through a table, and a heap sync last made it null for a record that
    /// names no view.
    #[cfg(feature = "diagnostics")]
    noted: [Box<[std::sync::atomic::AtomicU32]>; 2],
    /// With `VICTORY_TABLES_TO_DRAWS=1` ([`tables_to_draws`], as it was when
    /// the heap was made), per title slot: the frame (plus 1) a draw last
    /// read it through a table, made by the draw (`mirror` for a draw).
    tabled: Option<Box<[std::sync::atomic::AtomicU32]>>,
    /// With it, per title slot: the key of the record a sync last left to
    /// the draws or to wait ([`HostHeap::left_as`]), so that the write
    /// marks' fallback does not take it for a record no mark told of.
    left: Option<Box<[AtomicU64]>>,
    /// New constant and placed views a heap sync found, waiting for a draw
    /// to read them through a table ([`HostHeap::due_runs`]), by the frame
    /// they were found in.
    waiting: Mutex<std::collections::VecDeque<(u64, Vec<u32>)>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one. Paged:
    /// the title views through a fraction of the slots, and a dense table
    /// of them all was 37.7 MB, written when the heap was made.
    kept: Pages<Held>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<DynamicSlots>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per title slot, the console descriptor's dwords 0-3 as far as the
    /// host knows them (shaders reading a descriptor as data, `crate::memory`;
    /// [`write_address`]): the buffer, its mapping and host address, made on
    /// first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
}

/// What a slot's entry in [`HostHeap::made`] holds while a thread looks at
/// the slot under its lock: no record's key (no view has this index).
const BUSY: u64 = u64::MAX;

/// Slot locks to a heap ([`HostHeap::made`]).
const STRIPES: usize = 256;

/// Slots in a row under one lock: a table's slots are made together (a heap
/// sync takes the lock once for those of a run that changed), and other
/// tables' beside them.
const RUN: u32 = 64;

/// A slot lock, on a cache line of its own.
#[derive(Default)]
#[repr(align(64))]
struct Stripe {
    lock: Mutex<()>,
    /// Who holds it, or held it last ([`Who`]; diagnostics).
    #[cfg(feature = "diagnostics")]
    holder: std::sync::atomic::AtomicU8,
}

/// `n` zeros, their pages untouched until written (a view heap has over a
/// million slots).
fn zeroed(n: usize) -> Box<[AtomicU64]> {
    let zeros = Box::into_raw(vec![0u64; n].into_boxed_slice());
    // SAFETY: an `AtomicU64` is laid out as a `u64`.
    unsafe { Box::from_raw(zeros as *mut [AtomicU64]) }
}

fn zeroed32(n: usize) -> Box<[std::sync::atomic::AtomicU32]> {
    let zeros = Box::into_raw(vec![0u32; n].into_boxed_slice());
    // SAFETY: an `AtomicU32` is laid out as a `u32`.
    unsafe { Box::from_raw(zeros as *mut [std::sync::atomic::AtomicU32]) }
}

/// Whether slots have locks by run and are looked at without one
/// ([`HostHeap::made`]): unless `VICTORY_SLOT_LOCKS=0`.
fn slot_locks() -> bool {
    #[cfg(test)]
    if tests::ONE_LOCK.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_SLOT_LOCKS"))
}

/// Whether a heap sync makes the slots of a run that changed under one
/// hold of their lock ([`HostHeap::sync_slots`]): unless
/// `VICTORY_SYNC_RUNS=0`, which takes it for each.
fn sync_runs() -> bool {
    #[cfg(test)]
    if tests::ONE_LOCK.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_SYNC_RUNS"))
}

/// Slots of one run a heap sync is to make, with the records read of them
/// ([`HostHeap::due_runs`]).
pub(crate) type Run = Vec<(u32, [u32; 3])>;

/// Slots to make below which a heap sync makes them on its own thread:
/// waking the helpers costs more than making a few slots.
const SPREAD: usize = 192;

/// How many threads make a heap sync's runs of slots, the submitting
/// thread among them: `VICTORY_SYNC_THREADS`, 1 without it (all on the
/// submitting thread). On four the helpers waited for each other and for
/// the title's threads, at what the slots view and hold: 457 ms of threads'
/// time for what one thread did in 67, and 119 ms of the submitting
/// thread's against 72 (run ab-all-on3).
fn sync_threads() -> usize {
    // Tests: as the thread running the sync says, for any number of slots;
    // on one thread unless it says.
    #[cfg(test)]
    return tests::SYNC_THREADS.with(|threads| threads.get()).unwrap_or(1);
    #[cfg(not(test))]
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    #[cfg(not(test))]
    *THREADS.get_or_init(|| {
        let asked = std::env::var("VICTORY_SYNC_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
        asked.map_or(1, |n| n.clamp(1, 16))
    })
}

/// Whether what a heap's slots hold is under a lock a page of slots
/// (`super::paged::Pages`): unless `VICTORY_KEPT_PAGES=0`, which puts one
/// lock before them all.
fn kept_pages() -> bool {
    #[cfg(test)]
    if tests::ONE_LOCK.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_KEPT_PAGES"))
}

/// Whether a slot that shows nothing keeps the null view it has when its
/// record comes to name no view (`VICTORY_NULL_ONCE=1`). Off: the null
/// view it has is one of the kind its last view was
/// ([`null_for`]), not the plain one a slot without a view gets, so what
/// the slot holds is not what it would hold; whether that is ever seen is
/// not known, and how many such slots there are is counted first.
fn null_once() -> bool {
    #[cfg(test)]
    if tests::NULL_ONCE.with(|once| once.get()) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_NULL_ONCE"))
}

/// Whether a heap sync leaves a slot as it is when its record comes to name
/// no view because the title wrote something other than a record of ours
/// there, zeros or other bytes (on by default; `VICTORY_STALE_LEFT=0` turns it
/// off). Measured: heap syncs 35.4 -> 23.4 ms a frame (median), no visible
/// change in the picture (#14).
/// Most slots a sync makes are such (32,000 of 47,000 a world frame, most
/// of them bytes the title keeps in heap memory between its descriptors),
/// and no draw reads one through a table. Left, the slot keeps the view
/// and what it holds that it had: a draw that reads it through a table
/// makes it null then ([`HostHeap::made_already`] finds it not made from
/// its record), a title thread writing it makes it, and the title writing
/// the record back that it was made from costs nothing. A record of ours
/// that names no view (a view of a released resource) is still made null.
/// shortcut: a shader that takes such a slot from the heap at an index
/// from its data reads the view the slot had, not nothing; and what the
/// slot views stays entered (`crate::bindless`), so its memory is kept
/// current and its copy held until the slot is written again.
fn stale_left() -> bool {
    // Tests choose for themselves (off unless a test turns it on), whatever
    // the default, so the tests of making such slots null still run.
    #[cfg(test)]
    {
        tests::STALE_LEFT.with(|left| left.get())
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| !crate::switch::off("VICTORY_STALE_LEFT"))
    }
}

/// Whether heap syncs leave the new records of draws' tables to the draws
/// (on by default since the A/B: heap syncs 24.1 -> 19.2 ms; `VICTORY_TABLES_TO_DRAWS=0` turns it off). A slot a
/// draw read through a table in the last [`TABLED_FRAMES`] frames is a
/// table's: when its record changes, a sync passes it by, and the draw that
/// reads it makes it (`made_already` finds it not made from its record).
/// A new constant or placed view elsewhere waits [`WAIT_FRAMES`] frames for
/// a draw to read it so, and is made by a sync after that if none did.
/// After avatarmanz's "draws' own records left to the draws" (commit
/// 1f1a4336 of that fork), in our heap sync.
/// shortcut: a shader that takes such a slot from the heap at an index from
/// its data meanwhile reads the view the slot had (a table's slot of an
/// earlier frame, or a slot whose view waits).
fn tables_to_draws() -> bool {
    #[cfg(test)]
    {
        tests::TABLES_TO_DRAWS.with(|on| on.get())
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| !crate::switch::off("VICTORY_TABLES_TO_DRAWS"))
    }
}

/// Frames for which a slot a draw read through a table is left to draws.
const TABLED_FRAMES: u32 = 8;

/// Frames a new constant or placed view waits for a draw.
const WAIT_FRAMES: u64 = 4;

/// How a heap sync looks at slots ([`HostHeap::due_runs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Look {
    /// Slots written since the last sync (changed pages, or write marks).
    Fresh,
    /// Slots whose wait for a draw is over ([`HostHeap::waited_out`]).
    Waited,
    /// Pages the write marks' fallback reads whole. As [`Look::Fresh`] for
    /// draws' table records: a slot whose mark a sync took and left to the
    /// draws (or to wait) is due still when the fallback reads its page,
    /// and was counted as unmarked and made then, ~1,800 a frame in the
    /// first A/B run with both switches (#14).
    Check,
}

/// What a record that is neither ours nor zeros looks like (diagnostics):
/// a render target or depth view's record; a current view's entry and
/// generation under another tag; a title address in its first eight bytes;
/// anything else.
#[cfg(feature = "diagnostics")]
fn other_bytes(record: [u32; 3]) -> crate::copystat::C {
    use crate::copystat::C;
    let [tag, a, b] = record;
    let first = u64::from(tag) | u64::from(a) << 32;
    if tag == super::RTV_TAG || tag == super::DSV_TAG {
        C::SyncOtherTarget
    } else if views_current(&[[VIEW_TAG, a, b]])[0] {
        C::SyncOtherTagged
    } else if a != 0 && (0x1_0000..0x8000_0000_0000).contains(&first) {
        C::SyncOtherAddress
    } else {
        C::SyncOtherRest
    }
}

/// The `step` bytes of the title slot at `handle`, as dwords in hex.
#[cfg(feature = "diagnostics")]
fn slot_bytes(handle: u64, step: u64) -> String {
    let words: Vec<String> = (0..step / 4)
        // SAFETY: a descriptor slot in title memory, `step` bytes.
        .map(|k| format!("{:08x}", unsafe { ((handle + 4 * k) as *const u32).read_unaligned() }))
        .collect();
    words.join(" ")
}

/// Who makes or reads a slot (diagnostics: whose waits for the slot locks,
/// and behind whom).
#[derive(Clone, Copy)]
enum Who {
    /// A title thread writing a view into its heap.
    Title,
    /// A list recorder reading a table for a draw or dispatch.
    Draw,
    /// The submitting thread: heap syncs, and slots shaders read.
    Sync,
}

impl Who {
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn now(for_draw: bool) -> Self {
        if crate::contend::in_sync() {
            Who::Sync
        } else if for_draw {
            Who::Draw
        } else {
            Who::Title
        }
    }
}

/// Times the steps of every eighth slot a heap sync makes (diagnostics;
/// nothing without them).
struct Laps(#[cfg(feature = "diagnostics")] Option<std::time::Instant>);

impl Laps {
    fn start(_index: u32) -> Self {
        Laps(
            #[cfg(feature = "diagnostics")]
            (_index.is_multiple_of(8) && crate::contend::in_sync()).then(std::time::Instant::now),
        )
    }

    /// The time since the last lap (or the start) was `_step`'s.
    fn lap(&mut self, #[cfg(feature = "diagnostics")] _step: crate::copystat::C) {
        #[cfg(feature = "diagnostics")]
        if let Some(at) = &mut self.0 {
            let now = std::time::Instant::now();
            crate::copystat::add(_step, (now - *at).as_nanos() as u64);
            *at = now;
        }
    }
}

/// [`Laps::lap`] of a step named by its counter; nothing without
/// diagnostics.
macro_rules! lap {
    ($laps:expr, $step:ident) => {
        $laps.lap(
            #[cfg(feature = "diagnostics")]
            crate::copystat::C::$step,
        )
    };
}

/// What [`HostHeap::make_locked`] found or did.
enum Made {
    /// The slot is made from its record: the view recorded there.
    Is(Option<View>),
    /// What the record's view needs is to be made first ([`prepare`]),
    /// without the lock: the view and the record.
    Needs(View, [u32; 3]),
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59). On the sister fork a world
/// frame of some 7,000 draws held over 400,000 slots, and tables of 150
/// slots no longer fit: the draws that needed them were dropped (the
/// subsurface resolve, tree imposters). 2^19 with the title's 1,245,184
/// slots stays below `victory_shader::emit::HEAP_VIEWS` (2^21), so the
/// heap keeps its size; the host copies' slot pairs get what is left.
const DYNAMIC: u32 = 1 << 19;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // Other threads' submissions held off from here until the heap is
        // filled (`Host::quiet`; a title heap's `host` holds them already).
        let _quiet = (!sampler).then(|| host.quiet()).flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below `HEAP_VIEWS` or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let (views, samplers) = (victory_shader::emit::HEAP_VIEWS, victory_shader::emit::HEAP_SAMPLERS);
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            // shortcut: a host that caps its heaps below `HEAP_VIEWS` (NVIDIA)
            // faults on an index past them, as before the clamp.
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: zeroed(count as usize),
            held: zeroed((count as usize).div_ceil(64)),
            locks: (0..STRIPES).map(|_| Stripe::default()).collect(),
            by_run: [slot_locks(), sync_runs()],
            #[cfg(feature = "diagnostics")]
            noted: [zeroed32(count as usize), zeroed32(count as usize)],
            tabled: (!sampler && tables_to_draws()).then(|| zeroed32(count as usize)),
            left: (!sampler && tables_to_draws()).then(|| zeroed(count as usize)),
            waiting: Mutex::default(),
            kept: Pages::new((count + extra) as usize, !kept_pages()),
            next: Mutex::new(DynamicSlots {
                cursor: 0,
                taken: vec![false; extra as usize],
                first_failed_size: None,
            }),
            pairs: Mutex::new((count + extra, Vec::new())),
            pairs_end: match victory_shader::emit::NULL_BUFFER_VIEW {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        let null = victory_shader::emit::NULL_BUFFER_VIEW;
        if !sampler && null > count + extra && null < total {
            h.raw_view_at(null, None, 0, 0, false);
            h.raw_view_at(null - 1, None, 0, 0, true);
        } else if !sampler {
            tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let n = n as usize;
        if guard.first_failed_size.is_some_and(|size| n >= size) {
            return None;
        }
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = guard.cursor;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if guard.taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            guard.first_failed_size = Some(n);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        guard.taken[first..first + n].fill(true);
        guard.cursor = (first + n) % end;
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.taken.get_mut(at as usize..end) {
            slots.fill(false);
            guard.first_failed_size = None;
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        self.make_null_view(index, true, &mut Laps::start(1));
    }

    /// [`HostHeap::make_null`]; the host slot itself is written only if
    /// `write` (not: it holds a null view already, [`null_once`]).
    fn make_null_view(&self, index: u32, write: bool, laps: &mut Laps) {
        if !write {
        } else if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        lap!(laps, SlotHostNs);
        self.keep(index, None);
        lap!(laps, SlotKeepNs);
        self.set_stride(index, None);
        self.set_mark(index, None);
        lap!(laps, SlotTablesNs);
        if index < self.count {
            self.viewed.set(index, None, false);
        }
        lap!(laps, SlotViewedNs);
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the integer-format
    /// bits of a typed buffer view ([`view_kind`]).
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, |v| view_stride(v) | view_kind(v)));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        #[cfg(feature = "diagnostics")]
        let now = resource.as_ref().map(|held| windows::core::Interface::as_raw(&held.0));
        // A slot that held nothing and gets nothing is left alone (no page).
        let old = self.kept.replace(index as usize, resource, |holds| {
            if let Some(bits) = self.held.get(index as usize / 64).filter(|_| index < self.count) {
                // Under the page's lock: the bit is what the slot holds.
                match holds {
                    true => bits.fetch_or(1 << (index % 64), Release),
                    false => bits.fetch_and(!(1 << (index % 64)), Release),
                };
            }
        });
        // With no lock held: the page's lock is for the slots.
        #[cfg(feature = "diagnostics")]
        if let Some((old, _)) = &old
            && !self.sampler
            && now != Some(windows::core::Interface::as_raw(old))
            // SAFETY: a live resource.
            && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        {
            use windows::core::Interface;
            let left = crate::resource::life::holders(old);
            crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
        }
        // Title threads rewrite slots while the GPU runs a submission;
        // one of its shaders may hold the old descriptor (or read the
        // slot at an index from wrong data): what it views is freed
        // only once that work has run.
        if let Some((old, _)) = old {
            crate::host::retire(old);
        }
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            let (buffer, data) = host()?.upload_buffer(u64::from(self.count) * 16)?;
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            // No slot is made meanwhile: one made before the table exists
            // does not enter its address (`note_address`).
            let _still = self.hold_slots();
            for (i, key) in self.made.iter().enumerate() {
                let key = key.load(Acquire);
                let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
                write_address(data, i, view.as_ref());
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if let Some(Some((_, data, _))) = self.addresses.get() {
            write_address(*data, index as usize, view);
        }
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// A heap sync of slots `first..last`, whose records are at `start` +
    /// index × `step`: each slot whose record differs from what the slot
    /// was made from is made as [`HostHeap::mirror`] makes it, the rest
    /// passed by as `mirror` would. The records are compared with what the
    /// slots were made from without a lock (a world frame looks at
    /// ~500,000 slots), and the slots to make of one run are made under one
    /// hold of the run's lock, what their views need made before it is
    /// taken: one take per slot was 3,000 waits and 100 ms a frame beside
    /// the title's threads (run ab-locks-on).
    #[cfg(test)]
    pub(crate) fn sync_slots(&self, first: u32, last: u32, start: u64, step: u64) {
        let mut runs = Vec::new();
        self.due_runs(first, last, start, step, crate::frame::current(), Look::Fresh, &mut runs);
        self.make_runs(&runs, start, step);
    }

    /// Of slots `first..last`, whose records are at `start` + index ×
    /// `step`, those to make ([`due`]), by run, with the records read:
    /// added to `runs`. A sync in `frame` looking as `look` says; with
    /// [`tables_to_draws`] the new records of draws' tables are left to the
    /// draws, and new constant and placed views wait for one.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn due_runs(&self, first: u32, last: u32, start: u64, step: u64, frame: u64, look: Look, runs: &mut Vec<Run>) {
        let at = |i: u32| start + u64::from(i) * step;
        #[cfg(feature = "diagnostics")]
        let looking = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncLookUs);
        // SAFETY: records in title heap memory the caller found mapped.
        let records: Vec<[u32; 3]> =
            (first..last).map(|i| unsafe { (at(i) as *const [u32; 3]).read_unaligned() }).collect();
        let current = views_current(&records);
        // Why each is made (diagnostics).
        let mut why = [0u64; 3];
        let mut made = 0u64;
        // Left to draws, and made after a wait; slots that wait.
        let mut left = [0u64; 2];
        let mut wait = Vec::new();
        for (i, (&record, current)) in (first..last).zip(records.iter().zip(current)) {
            let [_tag, view, generation] = record;
            let key = u64::from(generation) << 32 | u64::from(view);
            let from = self.made.get(i as usize).map(|made| made.load(Acquire));
            let Some(reason) = due(from, current.then_some(key)) else {
                continue;
            };
            if reason == Some(Due::Stale) && record[0] != VIEW_TAG && stale_left() {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::SyncStaleLeft, 1);
                continue;
            }
            // (The fallback's look makes a record no sync left so: one no
            // mark told of, counted by the caller.)
            if self.tabled.is_some()
                && matches!(reason, Some(Due::First | Due::Rewrote))
                && (look != Look::Check || self.left_as(i, key))
            {
                if self.tabled_lately(i, frame) {
                    self.note_left(i, key);
                    left[0] += 1;
                    continue;
                }
                if look != Look::Waited && matches!(generation >> super::KIND_SHIFT, super::CONSTANT | super::PLACED) {
                    self.note_left(i, key);
                    wait.push(i);
                    continue;
                }
            }
            left[1] += u64::from(look == Look::Waited);
            if let Some(reason) = reason {
                why[reason as usize] += 1;
                #[cfg(feature = "diagnostics")]
                self.note_due(i, reason, record, at(i), step);
            }
            made += 1;
            match runs.last_mut() {
                Some(run) if run.last().is_some_and(|&(last, _)| last / RUN == i / RUN) => run.push((i, record)),
                _ => runs.push(vec![(i, record)]),
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            drop(looking);
            add(C::SyncMade, made);
            add(C::SyncFirst, why[Due::First as usize]);
            add(C::SyncRewrote, why[Due::Rewrote as usize]);
            add(C::SyncStale, why[Due::Stale as usize]);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = why;
        {
            use crate::write_marks::{S_LOOKED, S_MADE, S_TO_DRAWS, S_WAITED_MADE, S_WAITING, STATS};
            use std::sync::atomic::Ordering::Relaxed;
            STATS[S_LOOKED].fetch_add(u64::from(last.saturating_sub(first)), Relaxed);
            STATS[S_MADE].fetch_add(made, Relaxed);
            STATS[S_TO_DRAWS].fetch_add(left[0], Relaxed);
            STATS[S_WAITED_MADE].fetch_add(left[1], Relaxed);
            STATS[S_WAITING].fetch_add(wait.len() as u64, Relaxed);
        }
        if !wait.is_empty() {
            self.waiting.lock().unwrap_or_else(|e| e.into_inner()).push_back((frame, wait));
        }
    }

    /// Whether a draw read title slot `index` through a table within the
    /// last [`TABLED_FRAMES`] frames before `frame` (never without
    /// [`tables_to_draws`]).
    pub(crate) fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let Some(read) = self.tabled.as_ref().and_then(|t| t.get(index as usize)).map(|t| t.load(Acquire)) else {
            return false;
        };
        read != 0 && (frame as u32).wrapping_add(1).wrapping_sub(read) <= TABLED_FRAMES
    }

    /// Whether a sync left title slot `index` to the draws or to wait when
    /// its record's key was `key` (the record it holds now).
    pub(crate) fn left_as(&self, index: u32, key: u64) -> bool {
        self.left.as_ref().and_then(|l| l.get(index as usize)).is_some_and(|l| l.load(std::sync::atomic::Ordering::Relaxed) == key)
    }

    fn note_left(&self, index: u32, key: u64) {
        if let Some(l) = self.left.as_ref().and_then(|l| l.get(index as usize)) {
            l.store(key, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// A draw read title slot `index` through a table in `frame`.
    pub(crate) fn note_tabled(&self, index: u32, frame: u64) {
        if let Some(t) = self.tabled.as_ref().and_then(|t| t.get(index as usize)) {
            t.store((frame as u32).wrapping_add(1).max(1), Release);
        }
    }

    /// The slots whose wait for a draw is over in `frame` ([`HostHeap::due_runs`]),
    /// each once, in order.
    pub(crate) fn waited_out(&self, frame: u64) -> Vec<u32> {
        let mut waiting = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        while waiting.front().is_some_and(|&(found, _)| found + WAIT_FRAMES <= frame) {
            out.extend(waiting.pop_front().into_iter().flat_map(|(_, slots)| slots));
        }
        drop(waiting);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Makes the slots of `runs` ([`HostHeap::due_runs`]): on this thread
    /// and helper threads at once where there are enough of them
    /// ([`sync_threads`]). A run is made by one thread, under its lock, as
    /// a draw's thread or a title thread makes a slot beside others; when
    /// this returns every run is made.
    pub(crate) fn make_runs(&self, runs: &[Run], start: u64, step: u64) {
        #[cfg(feature = "diagnostics")]
        let _making = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncRunsUs);
        if runs.is_empty() {
            return;
        }
        let slots: usize = runs.iter().map(Vec::len).sum();
        let threads = if slots >= SPREAD || cfg!(test) { sync_threads() } else { 1 };
        // The helpers work for this thread, which waits for them: where it
        // holds the submission lock, so do they (`Host::quiet`).
        let holding = crate::host::holds_submissions();
        crate::back::spread(runs.len(), threads, |k| {
            let _holding = holding.then(crate::host::holding_too);
            // Counted as the submitting thread's work (diagnostics).
            let _syncing = crate::contend::syncing();
            self.make_run(&runs[k], start, step);
        });
    }

    /// Makes the slots of `run`, slots of one run of [`RUN`] with the
    /// records read of them.
    fn make_run(&self, run: &Run, start: u64, step: u64) {
        let at = |i: u32| start + u64::from(i) * step;
        let made = |_view: &Option<View>| {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                match _view {
                    None => add(C::SyncNulled, 1),
                    Some(View::Constant(..)) => add(C::SyncConstant, 1),
                    Some(view) if is_placed_texture(view) => add(C::SyncLeft, 1),
                    Some(_) => {}
                }
            }
        };
        let Some(&(head, _)) = run.first() else { return };
        if !self.by_run[1] {
            for &(i, _) in run {
                made(&self.mirror(i, at(i), false));
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        let preparing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncPrepareUs);
        // What the views need, before the lock (`mirror` makes it
        // without the lock too): for the records as read.
        for &(_, [tag, view, generation]) in run {
            if let Some(view) = (tag == VIEW_TAG).then(|| view_at(view, generation)).flatten()
                && !is_placed_texture(&view)
            {
                prepare(&view);
            }
        }
        #[cfg(feature = "diagnostics")]
        drop(preparing);
        #[cfg(feature = "diagnostics")]
        let _making = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncMakeUs);
        // One by one where the title wrote another record meanwhile
        // (what that one needs is not made), and past the host heap.
        let mut again = Vec::new();
        {
            let _slots = self.lock_slot(head, false);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SyncRuns, 1);
            for &(i, record) in run {
                match (i < self.count).then(|| self.make_locked(i, at(i), false, Some(record))) {
                    Some(Made::Is(view)) => made(&view),
                    Some(Made::Needs(..)) | None => again.push(i),
                }
            }
        }
        for i in again {
            made(&self.mirror(i, at(i), false));
        }
    }

    /// Diagnostics of a slot a heap sync is to make for `reason`, its
    /// record `record`: what a record that names no view is (a view whose
    /// resource the title released, zeros, other bytes), whether the slot
    /// was made null for one in the two frames before too, and whether a
    /// draw read the slot through a table in the last eight frames (such
    /// slots the draws would make themselves).
    #[cfg(feature = "diagnostics")]
    fn note_due(&self, index: u32, reason: Due, record: [u32; 3], handle: u64, step: u64) {
        use crate::copystat::{C, add};
        let frame = crate::frame::current() as u32 + 1;
        let since = |noted: &std::sync::atomic::AtomicU32| match noted.load(Relaxed) {
            0 => u32::MAX,
            was => frame.wrapping_sub(was),
        };
        let Some((tabled, nulled)) = self.noted[0].get(index as usize).zip(self.noted[1].get(index as usize)) else {
            return;
        };
        match reason {
            Due::Stale => {
                add(
                    match record {
                        [VIEW_TAG, ..] => C::SyncStaleReleased,
                        [0, 0, 0] => C::SyncStaleZeros,
                        _ => C::SyncStaleOther,
                    },
                    1,
                );
                if !matches!(record, [VIEW_TAG, ..] | [0, 0, 0]) {
                    add(other_bytes(record), 1);
                    crate::copystat::sample(|| {
                        let was = self.made[index as usize].load(Acquire);
                        let was = view_at(was as u32, (was >> 32) as u32);
                        format!(
                            "{} slot {index} made from {} now holds {}",
                            if self.sampler { "sampler" } else { "view" },
                            was.map_or("a view since released".into(), |v| format!("{v:?}").chars().take(60).collect::<String>()),
                            slot_bytes(handle, step)
                        )
                    });
                }
                if since(nulled) <= 2 {
                    add(C::SyncStaleAgain, 1);
                }
                if !self.sampler && !self.holds(index) {
                    add(C::SyncStaleNothing, 1);
                }
                nulled.store(frame, Relaxed);
            }
            Due::First | Due::Rewrote => {
                if since(tabled) <= 8 {
                    add(C::SyncTabled, 1);
                }
            }
        }
    }

    /// What title slot `index` is made from (0: nothing, or past the heap),
    /// waiting out a thread that looks at it under its lock.
    fn made_from(&self, index: u32) -> u64 {
        let Some(entry) = self.made.get(index as usize) else {
            return 0;
        };
        match entry.load(SeqCst) {
            BUSY => {
                let _slot = self.lock_slot(index, false);
                entry.load(SeqCst)
            }
            made => made,
        }
    }

    /// Whether `kept` holds a host resource for title slot `index`.
    fn holds(&self, index: u32) -> bool {
        self.held.get(index as usize / 64).is_some_and(|bits| bits.load(Acquire) >> (index % 64) & 1 != 0)
    }

    /// The lock of slot `index` ([`HostHeap::made`]), for a draw or not
    /// (diagnostics: the wait is counted by who waited and who held it).
    fn lock_slot(&self, index: u32, _for_draw: bool) -> std::sync::MutexGuard<'_, ()> {
        let stripe = &self.locks[if self.by_run[0] { (index / RUN) as usize % STRIPES } else { 0 }];
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            let who = Who::now(_for_draw) as usize;
            let guard = match stripe.lock.try_lock() {
                Ok(guard) => guard,
                Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    let holder = usize::from(stripe.holder.load(Relaxed)).min(2);
                    let since = std::time::Instant::now();
                    let guard = stripe.lock.lock().unwrap_or_else(|e| e.into_inner());
                    let us = since.elapsed().as_micros() as u64;
                    add([C::SlotWaitTitleUs, C::SlotWaitDrawUs, C::SlotWaitSyncUs][who], us);
                    add([C::SlotContendedTitle, C::SlotContendedDraw, C::SlotContendedSync][who], 1);
                    add([C::SlotBehindTitleUs, C::SlotBehindDrawUs, C::SlotBehindSyncUs][holder], us);
                    guard
                }
            };
            stripe.holder.store(who as u8, Relaxed);
            add([C::SlotTakesTitle, C::SlotTakesDraw, C::SlotTakesSync][who], 1);
            guard
        }
        #[cfg(not(feature = "diagnostics"))]
        stripe.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Without a lock: the view recorded at `handle` if slot `index` is
    /// made from that record already (for a draw: and holds its texture, if
    /// it views one), `Some(None)` if the record names no view and the slot
    /// shows none; `None` if the slot is to be made, or looked at under its
    /// lock ([`HostHeap::made`]).
    fn made_already(&self, index: u32, handle: u64, for_draw: bool) -> Option<Option<View>> {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // The record this thread (or the title, for it) wrote is seen by
        // any thread that sets the entry `BUSY` after the load below, which
        // then reads the record: one of the two sees the other
        // ([`HostHeap::made`]).
        std::sync::atomic::fence(SeqCst);
        let made = self.made[index as usize].load(SeqCst);
        let Some(view) = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten() else {
            return (made == 0).then_some(None);
        };
        if made != (u64::from(generation) << 32 | u64::from(i)) {
            return None;
        }
        // A texture's view left null is made for the draw (`make_locked`).
        if for_draw && is_texture(&view) && !self.holds(index) {
            return None;
        }
        Some(Some(view))
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        let view = self.mirror_slot(index, handle, for_draw);
        // A draw's table entry that names no view: what its slot holds
        // (diagnostics; bytes that are no record of ours there would be a
        // descriptor the title made that we do not know).
        #[cfg(feature = "diagnostics")]
        if for_draw && view.is_none() && index < self.count && !crate::contend::in_sync() {
            use crate::copystat::{C, add};
            // SAFETY: the handle is a descriptor slot in title memory.
            let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            match record {
                [VIEW_TAG, ..] => add(C::DrawNoneReleased, 1),
                [0, 0, 0] => add(C::DrawNoneZeros, 1),
                _ => {
                    add(C::DrawNoneOther, 1);
                    crate::copystat::sample(|| {
                        let kind = if self.sampler { "sampler" } else { "view" };
                        format!("a draw's table entry at {kind} slot {index} holds {}", slot_bytes(handle, u64::from(self.title_step())))
                    });
                }
            }
        }
        view
    }

    /// Bytes per title slot of this heap.
    #[cfg(feature = "diagnostics")]
    fn title_step(&self) -> u32 {
        if self.sampler { 16 } else { 32 }
    }

    fn mirror_slot(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        #[cfg(feature = "diagnostics")]
        if for_draw && !crate::contend::in_sync() {
            self.noted[0][index as usize].store(crate::frame::current() as u32 + 1, Relaxed);
        }
        if for_draw && self.tabled.is_some() && !crate::contend::in_sync() {
            self.note_tabled(index, crate::frame::current());
        }
        if self.by_run[0]
            && let Some(found) = self.made_already(index, handle, for_draw)
        {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::C;
                let quick = [C::SlotQuickTitle, C::SlotQuickDraw, C::SlotQuickSync];
                crate::copystat::add(quick[Who::now(for_draw) as usize], 1);
            }
            return found;
        }
        // What the view needs (the texture it views, the copy of the
        // memory) is made without the slot lock, which title threads
        // binding and writing views wait on (run evict5): for the record
        // then read again under it.
        let mut prepared = None;
        loop {
            let slot = self.lock_slot(index, for_draw);
            match self.make_locked(index, handle, for_draw, prepared) {
                Made::Is(view) => return view,
                Made::Needs(view, record) => {
                    drop(slot);
                    prepare(&view);
                    prepared = Some(record);
                }
            }
        }
    }

    /// [`HostHeap::mirror`] with the slot's lock held. `prepared`: the
    /// record whose view has what it needs made ([`prepare`]); for another
    /// record that needs something, nothing is made and the caller is told
    /// ([`Made::Needs`]).
    fn make_locked(&self, index: u32, handle: u64, for_draw: bool, prepared: Option<[u32; 3]>) -> Made {
        let entry = &self.made[index as usize];
        let mut laps = Laps::start(index);
        // Before the record is read: a thread that finds the entry its
        // record's key meanwhile does not take the slot for made
        // ([`HostHeap::made`]). `was`: what the slot is made from.
        let was = entry.swap(BUSY, SeqCst);
        // Read with the slot lock held: whichever thread makes the slot
        // last read its record last. A title thread's write is followed by
        // its own call here, so a heap sync or draw that read the record
        // from before cannot leave the slot made from that one (heap syncs
        // run beside title threads, `crate::descriptor::SHADER_HEAPS`).
        // SAFETY: the handle is a descriptor slot in title memory.
        let record = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let [tag, i, generation] = record;
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        lap!(laps, SlotLookNs);
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if was != 0 {
                // A slot that shows nothing has a null view already.
                let shows = self.sampler || self.holds(index);
                #[cfg(feature = "diagnostics")]
                if !shows && null_once() {
                    crate::copystat::add(crate::copystat::C::NullKept, 1);
                }
                self.make_null_view(index, shows || !null_once(), &mut laps);
                self.note_address(index, None);
                lap!(laps, SlotTablesNs);
            }
            // Last: the slot is whole for whoever finds this.
            entry.store(0, SeqCst);
            return Made::Is(None);
        };
        // A placed texture's view left null below is made when a draw reads
        // the slot through a table; until then the slot is not made again
        // (heap syncs revisit it at every submission).
        let lazy = for_draw && is_texture(&view) && !self.holds(index);
        if was == key && !lazy {
            entry.store(was, SeqCst);
            return Made::Is(Some(view));
        }
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        // shortcut: views of textures the title placed itself are made
        // when a draw reads them through a table: a host texture (and a
        // staging copy) for each of the ~20,000 the title has views of
        // ran the host out of memory (run b2).
        let later = !for_draw && is_placed_texture(&view);
        if !later && prepared != Some(record) {
            entry.store(was, SeqCst);
            return Made::Needs(view, record);
        }
        self.set_stride(index, Some(&view));
        self.note_address(index, Some(&view));
        if later {
            // Made once a shader reads it (`crate::shadow::take_reads`).
            crate::shadow::forget_read(index);
        }
        lap!(laps, SlotTablesNs);
        let (kept, mark, found) = if later {
            (None, None, None)
        } else {
            make_found(&view, self.cpu(index))
        };
        if kept.is_none() && !self.sampler {
            null_for(&view, self.cpu(index));
        }
        lap!(laps, SlotHostNs);
        self.set_mark(index, mark);
        lap!(laps, SlotTablesNs);
        if !self.sampler {
            self.viewed.set_found(index, &view, kept.is_some(), found_copy(&view, found));
        }
        lap!(laps, SlotViewedNs);
        self.keep(index, kept.flatten());
        lap!(laps, SlotKeepNs);
        // Last: the slot is whole for whoever finds this (`made`).
        entry.store(key, SeqCst);
        Made::Is(Some(view))
    }

    /// Holds off the making of this heap's title slots until dropped
    /// (`crate::descriptor::with_slots_still`).
    pub(crate) fn hold_slots(&self) -> Vec<std::sync::MutexGuard<'_, ()>> {
        // Every slot's lock, in order (a thread making a slot holds one).
        self.locks.iter().map(|stripe| stripe.lock.lock().unwrap_or_else(|e| e.into_inner())).collect()
    }

    /// Of host resources `resources` (raw pointers), those a title slot
    /// views.
    pub(crate) fn viewing(&self, resources: &std::collections::HashSet<usize>) -> Vec<usize> {
        let mut viewing = Vec::new();
        self.kept.each(self.count as usize, |_, k| {
            let raw = windows::core::Interface::as_raw(&k.0) as usize;
            if resources.contains(&raw) {
                viewing.push(raw);
            }
        });
        viewing
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made_from(index);
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.with(index as usize, |held| held.is_some());
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Slots viewing one of the host textures `textures` (raw pointers),
    /// which are given up: each gets a null view of its kind and holds the
    /// texture no more (retired until the running submission has run), its
    /// record stays noted so heap syncs leave it, and the next draw reading
    /// it through a table or shader reading it through the heap (its read
    /// mark, cleared) makes it again.
    /// The slots are found by a walk of the heap, once per pass that gives
    /// textures up: a map of each texture's slots, kept at every slot the
    /// title rewrote, cost ~25 us a slot (a texture viewed in thousands of
    /// slots had its list searched each time; 62,000 slots a frame, run
    /// evict3).
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        let mut slots: Vec<u32> = Vec::new();
        self.kept.each(usize::MAX, |i, k| {
            if k.1.is_none() && textures.contains(&(windows::core::Interface::as_raw(&k.0) as usize)) {
                slots.push(i as u32);
            }
        });
        // No slot is made meanwhile, as under the one lock there was.
        let _still = self.hold_slots();
        for index in slots {
            let key = self.made.get(index as usize).map_or(0, |made| made.load(Acquire));
            let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
            match &view {
                Some(v) => null_for(v, self.cpu(index)),
                None => null_view(self.cpu(index)),
            }
            self.keep(index, None);
            if index < self.count {
                self.viewed.set(index, view.as_ref(), false);
                crate::shadow::forget_read(index);
            }
        }
    }

    /// The host textures (raw pointers) slots `slots` view.
    pub(crate) fn textures_at(&self, slots: &[u32]) -> Vec<usize> {
        let texture = |held: Option<&Held>| {
            held.filter(|k| k.1.is_none()).map(|k| windows::core::Interface::as_raw(&k.0) as usize)
        };
        slots.iter().filter_map(|&i| self.kept.with(i as usize, texture)).collect()
    }

    /// The view of slot `index` if it is a placed texture's left without a
    /// host view (made when read).
    pub(crate) fn unmade_placed(&self, index: u32) -> Option<View> {
        let key = self.made_from(index);
        let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten()?;
        let kept = self.kept.with(index as usize, |held| held.is_some());
        (is_placed_texture(&view) && !kept).then_some(view)
    }

    /// A shader read slot `index` through the heap: the texture it views
    /// was used now.
    pub(crate) fn stamp_read(&self, index: u32) {
        let key = self.made_from(index);
        let Some(view) = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten() else {
            return;
        };
        let view = placed_texture(&view).unwrap_or(view);
        if let View::Shader(r, _) = view
            && let Some(t) = crate::resource::made_copy(r as crate::object::This)
        {
            t.stamp();
        }
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made_from(index) != 0
    }

    /// The record slot `index` was last made from, 0 for none.
    pub(crate) fn made_key(&self, index: u32) -> u64 {
        self.made_from(index)
    }

    /// Whether slot `index` holds a host view of a resource (tests).
    #[cfg(test)]
    pub(crate) fn kept(&self, index: u32) -> bool {
        self.kept.with(index as usize, |held| held.is_some())
    }

    /// Whether the bit of title slot `index` says what the slot holds, as
    /// its page in the table does (tests; with no slot being made).
    #[cfg(test)]
    pub(crate) fn held_agrees(&self, index: u32) -> bool {
        self.kept(index) == self.holds(index)
    }
}

/// Why a heap sync makes a slot again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Due {
    /// Nothing was made there, and the title's record names a view.
    First,
    /// The title's record names another view than the slot was made from.
    Rewrote,
    /// The slot was made from a view, and the title's record names none
    /// now: its resource was released, or the title wrote something else.
    Stale,
}

/// Whether a heap sync makes a slot again, and why: `made` is the record
/// the slot was last made from (0: none; `None`: a title slot past the host
/// heap, passed on to [`HostHeap::mirror`], which makes nothing there),
/// `record` the title's record if it names a current view.
fn due(made: Option<u64>, record: Option<u64>) -> Option<Option<Due>> {
    match (made, record) {
        (None, _) => Some(None),
        (Some(0), Some(_)) => Some(Some(Due::First)),
        (Some(made), Some(record)) => (made != record).then_some(Some(Due::Rewrote)),
        (Some(made), None) => (made != 0).then_some(Some(Due::Stale)),
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// The view-stride table's format-kind bits of a buffer view: a typed view
/// (no structure stride, not raw) of an integer format gets
/// `VIEW_INTEGER`, a signed one also `VIEW_SIGNED`; anything else none.
/// Translated shaders pass 16-bit typed data of such views as bits, not as
/// halves (`victory_shader::emit`, "d16").
fn view_kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let (unordered, words): (bool, &[u32; 10]) = match view {
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => (*unordered, words),
            Err(_) => return 0,
        },
        _ => return 0,
    };
    // Buffer views only (dimension 1); a structured or raw one is no typed view.
    let (format, typed) = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (desc.Format.0 as u32, b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 == 0)
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (desc.Format.0 as u32, b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0)
    };
    match (words[1] == 1 && typed).then(|| integer_format(format)).flatten() {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    }
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// Writes slot `index`'s words into the view addresses at `data`, 16 bytes
/// a slot: a buffer view's title address (dword 0 low, dword 1 bits 0-15
/// high), a texture view's size fields ([`texture_words`]), zeros for
/// anything else.
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let address = view.and_then(buffer_range).map_or(0, |(a, _)| a);
    let words = match view.and_then(texture_words) {
        Some(words) => words,
        None => [address as u32, (address >> 32) as u32 & 0xFFFF, 0, 0],
    };
    // SAFETY: the mapped view addresses hold 16 bytes per title slot.
    unsafe { ((data + 16 * index) as *mut [u32; 4]).write_unaligned(words) };
}

/// A texture view's console image descriptor dwords 0-3 as far as its size
/// goes: width - 1 and height - 1 of the resource's level 0 (dword 1 bits
/// 30-31, dword 2 bits 0-13 and 14-29), dword 3 an identity swizzle, the
/// view's first and last level (bits 12-15, 16-19) and the image type (bits
/// 28-31). `None` for buffers and samplers.
// shortcut: the address, format and tiling fields read 0 (no translated
// shader reads them: the shader refuses such reads where it can tell),
// and the swizzle is the identity whatever the view's component mapping.
fn texture_words(view: &View) -> Option<[u32; 4]> {
    let (resource, desc, unordered) = match view {
        View::Shader(r, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, false),
        View::Unordered(r, _, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, true),
        View::Placed(unordered, resource, desc) if resource[0] != 1 => {
            (*resource, desc[..10].try_into().ok(), *unordered)
        }
        _ => return None,
    };
    Some(image_words(&resource, desc.as_ref(), unordered))
}

/// [`texture_words`] of a texture of resource description `resource` (as
/// words) through a view of description `desc` (none: the resource's own).
fn image_words(resource: &[u32; 14], desc: Option<&[u32; 10]>, unordered: bool) -> [u32; 4] {
    let width = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let height = resource[6];
    let (depth, mips) = (resource[7] & 0xFFFF, resource[7] >> 16);
    let (samples, dimension) = (resource[9], resource[0]);
    // Mip count 0: the whole chain.
    let deepest = if dimension == 4 { depth } else { 1 };
    let chain = 64 - (width.max(u64::from(height)).max(u64::from(deepest)) | 1).leading_zeros();
    let mips = if mips == 0 { chain } else { mips };
    // Image types: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array,
    // 14 2D MSAA, 15 2D MSAA array.
    let (ty, first, count) = match desc {
        // Shader and unordered view dimensions number alike up to 3D (8).
        Some(d) => {
            let ty = [0, 0, 8, 12, 9, 13, 14, 15, 10, 11, 11].get(d[1] as usize).copied().unwrap_or(0);
            match (unordered, d[1]) {
                (_, 6 | 7) => (ty, 0, 1),
                // An unordered view's one level is its third word (its
                // union starts at byte 8: `MipSlice` of every texture
                // member, as `uav_desc` hands it to the host); a shader
                // view's first level and count are words 4 and 5 (its union
                // starts at byte 16). Word 4 of an unordered view is an
                // array's size or a 3D view's depth.
                (true, _) => (ty, d[2], 1),
                (false, _) => (ty, d[4], if d[5] == u32::MAX { mips.saturating_sub(d[4]) } else { d[5] }),
            }
        }
        None => {
            let array = depth > 1;
            let ty = match dimension {
                2 if array => 12,
                2 => 8,
                3 if samples > 1 => 14 + u32::from(array),
                3 => 9 + 4 * u32::from(array),
                _ => 10,
            };
            (ty, 0, if unordered { 1 } else { mips })
        }
    };
    let (w, h) = (width.saturating_sub(1) as u32, height.saturating_sub(1));
    let last = first + count.max(1) - 1;
    [
        0,
        (w & 3) << 30,
        (w >> 2 & 0x3FFF) | (h & 0xFFFF) << 14,
        0xFAC | (first & 0xF) << 12 | (last & 0xF) << 16 | ty << 28,
    ]
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, u64::from(view_stride(view)))?;
    Some((shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, and the view's own copy
/// of title memory ([`crate::shadow::for_view`]): the slot holds it, so that
/// draws reading the view find and refresh that copy.
type Held = (
    ID3D12Resource,
    Option<std::sync::Arc<crate::shadow::Shadow>>,
);

/// What a slot made by [`make`] holds, if anything.
type Kept = Option<Held>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let start = address + buffer.FirstElement * element;
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    buffer.FirstElement = shadow.offset(start) / element;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    FOUND.with(|f| *f.borrow_mut() = Some((shadow.clone(), start, u64::from(count) * element, element)));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    FOUND.with(|f| *f.borrow_mut() = Some((shadow.clone(), start, u64::from(count) * element, element)));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
    /// The copy the buffer view [`make`] last made on this thread reads,
    /// with the view's title range and element size ([`make_marked`]).
    static FOUND: std::cell::RefCell<Option<Found>> = const { std::cell::RefCell::new(None) };
}

/// The copy a buffer view made on the host reads ([`crate::shadow::for_view`]
/// as `make` asked it), the view's title start and length, and its element
/// size: what `crate::bindless` would look up again for the slot.
pub(crate) type Found = (std::sync::Arc<crate::shadow::Shadow>, u64, u64, u64);

/// Whether what a heap's slots view is entered from the copy the slot's
/// buffer view was just made on ([`found_copy`]) rather than looked up
/// again: unless `VICTORY_SEEN_FROM_MADE=0`. Each buffer view made asked the
/// registry of copies twice for the same range (and the table of copies of
/// views off their copy's elements, under one lock, twice: 21,000 such
/// views a world frame); the copy `make` found is the one the host view
/// reads, which the second asking is after (a copy registered in between
/// is the only way the two differ, and then the first is the right one).
fn seen_from_made() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_SEEN_FROM_MADE"))
}

/// Of the copy `found` that [`make`] made `view` on: the copy, the view's
/// title start and length, as `crate::bindless` finds them for the slot
/// (`crate::descriptor::host::view_copy`), when they are the same asking:
/// a buffer view (not a constant buffer's, which it finds otherwise) whose
/// title range and element size `make` used. None to look them up.
pub(crate) fn found_copy(view: &View, found: Option<Found>) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (shadow, start, len, element) = found?;
    if !seen_from_made() || matches!(view, View::Constant(..)) {
        return None;
    }
    (buffer_range(view) == Some((start, len)) && u64::from(view_stride(view)) == element).then_some((shadow, start, len))
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    let (kept, mark, _) = make_found(view, at);
    (kept, mark)
}

/// [`make_marked`], and the copy a buffer view was made on ([`Found`]).
fn make_found(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>, Option<Found>) {
    MARK.with(|m| m.set(None));
    FOUND.with(|f| f.borrow_mut().take());
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take), FOUND.with(|f| f.borrow_mut().take()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A packed 16-bit texture made R8G8B8A8_UNORM on this GPU
    // (`crate::packed`): views name the format it has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    if cast && family(view) != family(own) && unordered && !typed_store(view) {
        once!(view, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
            // Shaders indexing the heap write through this view without
            // moving the texture to a write state, and a slot left null is
            // made again only when read: kept from now on.
            copy.note_state(0x8);
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// Makes what a host view of `view` needs, if it is not made yet: the
/// texture it views (made on first use) or the host copy of the buffer
/// memory it views. The slot's view is then quick to make.
fn prepare(view: &View) {
    #[cfg(test)]
    tests::while_preparing();
    match view {
        View::Placed(..) if is_placed_texture(view) => {
            let _ = placed_texture(view);
        }
        View::Shader(r, _) | View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some() => {
            let _ = crate::resource::host_resource(*r as crate::object::This);
        }
        _ => {
            let _ = view_copy(view);
        }
    }
}

/// Whether `view` is a shader view of a texture (one placed in memory, or a
/// resource object's): such a slot left null is made when read.
fn is_texture(view: &View) -> bool {
    is_placed_texture(view)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, a title shader, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A heap sync makes a slot whose record differs from what it was made
    /// from, and passes by one that holds what its record says: a view
    /// where none was made, another view, no view where one was made; the
    /// same view, and nothing where nothing was made, are left.
    #[test]
    fn a_sync_makes_the_slots_whose_record_differs() {
        let (a, b) = (1 << 32 | 7, 2 << 32 | 7);
        assert_eq!(due(Some(0), Some(a)), Some(Some(Due::First)));
        assert_eq!(due(Some(a), Some(b)), Some(Some(Due::Rewrote)));
        assert_eq!(due(Some(a), None), Some(Some(Due::Stale)));
        assert_eq!(due(Some(a), Some(a)), None);
        assert_eq!(due(Some(0), None), None);
        assert_eq!(due(None, Some(a)), Some(None));
        assert_eq!(due(None, None), Some(None));
    }

    /// Typed buffer views of integer formats carry their kind in the
    /// view-stride table; float, normalized, raw and structured ones none.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(57, 0, 0)), VIEW_INTEGER, "R16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(64, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R8_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_kind(&srv), VIEW_INTEGER, "R8_UINT");
        assert_eq!(view_stride(&uav(12, 0, 0)) | view_kind(&uav(12, 0, 0)), 8 | VIEW_INTEGER);
    }

    /// The copy a buffer view is made on is the one what the slot views is
    /// entered with (`VICTORY_SEEN_FROM_MADE`): for structured views on and
    /// off their copy's elements (the latter on a copy of their own), raw
    /// unordered views and typed ones, the same copy and range as asking
    /// again; constant buffer views, found otherwise, are asked again.
    #[test]
    fn a_buffer_slot_views_the_copy_its_view_was_made_on() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let host = crate::descriptor::heap(heap).host().expect("a host view heap");
        let p = crate::watch::tests::pool(2);
        let len = 2 * crate::watch::PAGE as u64;
        crate::shadow::register(p, p as u64, len);
        // A placed buffer of the whole pool, and views of it at `at`.
        let mut buffer = [0u32; 14];
        (buffer[0], buffer[4]) = (1, len as u32);
        let placed = |unordered: bool, at: u64, words: [u32; 10]| {
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&words);
            (desc[10], desc[11]) = ((p as u64 + at) as u32, ((p as u64 + at) >> 32) as u32);
            View::Placed(unordered, buffer, desc)
        };
        // Format, dimension, mapping, (pad), first element (2), count, stride, flags.
        let structured = |first: u32, count: u32, stride: u32| [0, 1, 5768, 0, first, 0, count, stride, 0, 0];
        // Format, dimension, first element (2), count, stride, counter (2), flags.
        let raw_uav = [39, 1, 4, 0, 32, 0, 0, 0, 1, 0];
        let typed_uav = [2, 1, 3, 0, 16, 0, 0, 0, 0, 0];
        let views = [
            ("structured", placed(false, 0, structured(2, 64, 16))),
            ("off its copy's elements", placed(false, 8, structured(1, 32, 24))),
            ("raw unordered", placed(true, 256, raw_uav)),
            ("typed unordered", placed(true, 4096, typed_uav)),
        ];
        for (k, (what, view)) in views.iter().enumerate() {
            let (kept, _, found) = make_found(view, host.cpu(k as u32));
            assert!(matches!(kept, Some(Some(_))), "{what}: made");
            let (copy, start, bytes) = found_copy(view, found.clone()).expect("from the copy it was made on");
            let (again, start_again, bytes_again) = view_copy(view).expect("found asking again");
            assert!(std::sync::Arc::ptr_eq(&copy, &again) && (start, bytes) == (start_again, bytes_again), "{what}: the same copy and range");
            assert!(crate::bindless::found_as_seen(view, Some((copy, start, bytes))), "{what}: as what the slot views is found");
        }
        let constant = View::Constant(p as u64 + 512, 256);
        let (kept, _, found) = make_found(&constant, host.cpu(7));
        assert!(matches!(kept, Some(Some(_))) && found.is_some(), "a constant buffer view, made");
        assert!(found_copy(&constant, found).is_none(), "asked again");
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A placed texture's view the title writes into a heap is left null
    /// and noted once: heap syncs (every submission) pass it by, a draw
    /// reading it through a table makes it.
    #[test]
    fn a_placed_texture_slot_left_null_is_made_once_a_draw_reads_it() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        for _ in 0..2 {
            assert!(host.mirror(0, title_heap.start, false).is_some());
            assert!(host.made(0), "the record is noted");
            assert!(!host.kept(0), "no host texture for it yet");
        }
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
    }

    /// A slot viewing a texture given up (`crate::resource::evict`) holds it
    /// no more: heap syncs pass it by, a draw reading it through a table
    /// makes it again.
    #[test]
    fn a_slot_viewing_a_texture_given_up_is_made_again_when_read() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!host.kept(0), "the slot holds it no more");
        assert!(host.made(0), "its record stays noted");
        assert!(host.mirror(0, title_heap.start, false).is_some());
        assert!(!host.kept(0), "heap syncs pass it by");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made again for the draw");
        assert!(crate::resource::made_copy(texture).is_some(), "with a host texture again");
    }

    thread_local! {
        /// Heaps made on this thread have one lock for all their slots, taken
        /// for every look and each slot a sync makes (as
        /// `VICTORY_SLOT_LOCKS=0` and `VICTORY_SYNC_RUNS=0`).
        pub(crate) static ONE_LOCK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// How many threads make the runs of a heap sync this thread runs
        /// (as `VICTORY_SYNC_THREADS`).
        pub(crate) static SYNC_THREADS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
        /// This thread leaves the null view of a slot that shows nothing (as
        /// `VICTORY_NULL_ONCE=1`).
        pub(crate) static NULL_ONCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// Heap syncs this thread runs leave slots whose record came to be
        /// other bytes (as `VICTORY_STALE_LEFT=1`).
        pub(crate) static STALE_LEFT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// Heaps made on this thread leave draws' table records to the draws
        /// (as `VICTORY_TABLES_TO_DRAWS=1`).
        pub(crate) static TABLES_TO_DRAWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    const HEAP_IID: crate::object::Iid =
        crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);

    /// A draw reading a table entry whose slot is made from its record
    /// waits for no lock: with every slot lock of the heap held it comes
    /// back with the view. One whose record changed waits for the lock, and
    /// then makes the slot from the new record.
    #[test]
    fn a_draw_reads_a_made_slot_without_its_lock_and_waits_to_make_one() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        // A shader-visible sampler heap of 4 slots, in memory of its own.
        let heap_desc: [u32; 8] = [1, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &HEAP_IID, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (start, at) = (title_heap.start, heap as usize);
        let sampler = |address: u32| [21, address, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        crate::descriptor::create_sampler(null, &sampler(1), start);
        let slots = title_heap.host().expect("a host sampler heap");
        let first = slots.made_key(0);
        assert!(first != 0, "made when written");
        // As a draw on another thread would read the entry.
        let read = move || {
            let (done, result) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let slots = crate::descriptor::heap(at as crate::object::This).host().expect("made");
                let _ = done.send(slots.mirror(0, start, true));
            });
            result
        };
        let held = slots.hold_slots();
        let found = read().recv_timeout(std::time::Duration::from_secs(20));
        assert_eq!(found, Ok(Some(View::Sampler(sampler(1)))), "read while every slot lock is held");
        // Another record, copied in as bytes: the slot is to be made.
        let elsewhere = [0u64; 2];
        crate::descriptor::create_sampler(null, &sampler(2), elsewhere.as_ptr() as u64);
        // SAFETY: a record of ours into the heap's first slot.
        unsafe { std::ptr::copy_nonoverlapping(elsewhere.as_ptr().cast::<u8>(), start as *mut u8, 16) };
        let waiting = read();
        assert!(waiting.recv_timeout(std::time::Duration::from_millis(300)).is_err(), "waits for the slot's lock");
        assert_eq!(slots.made_key(0), first, "not made meanwhile");
        drop(held);
        let found = waiting.recv_timeout(std::time::Duration::from_secs(20));
        assert_eq!(found, Ok(Some(View::Sampler(sampler(2)))));
        assert!(slots.made_key(0) != first && slots.made_key(0) != 0, "made from the new record");
    }

    /// A slot whose record comes to name no view is made null whether or
    /// not it showed anything: it is noted as made from nothing, and views
    /// nothing. With `VICTORY_NULL_ONCE=1` too, which only leaves the null
    /// view of a slot that showed nothing where it is.
    #[test]
    fn a_slot_whose_record_names_no_view_is_made_null() {
        if host().is_none() {
            return; // no host GPU here
        }
        for once in [false, true] {
            NULL_ONCE.with(|n| n.set(once));
            let null = std::ptr::null();
            let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
            // A slot no other test's pass over read slots names.
            const SLOT: u32 = 777;
            let at = |title: &crate::descriptor::Heap| title.start + u64::from(SLOT) * 32;
            let mut heap = null;
            assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &HEAP_IID, &mut heap), 0);
            let title_heap = crate::descriptor::heap(heap);
            // A placed texture's view: left null until read, so the slot
            // shows nothing.
            let memory = vec![0u8; 1 << 16];
            let address = memory.as_ptr() as u64;
            let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, at(title_heap));
            let slots = title_heap.host().expect("a host view heap");
            assert!(slots.made(SLOT) && !slots.kept(SLOT) && !slots.holds(SLOT), "noted, showing nothing");
            assert_eq!(slots.viewed.slots(), 1);
            // SAFETY: a slot of the heap.
            unsafe { std::ptr::write_bytes(at(title_heap) as *mut u8, 0, 32) };
            assert!(slots.mirror(SLOT, at(title_heap), false).is_none());
            assert!(!slots.made(SLOT), "made from nothing (leaving its null view: {once})");
            assert_eq!(slots.viewed.slots(), 0);
            NULL_ONCE.with(|n| n.set(false));
        }
    }

    /// With `VICTORY_STALE_LEFT=1` a heap sync leaves a slot whose record
    /// came to be other bytes or zeros as it is, made from the view it had:
    /// the record written back costs nothing, and a draw reading the slot
    /// through a table makes it null. A record of ours that names no view
    /// is made null all the same. Without the switch the sync makes the
    /// slot null.
    #[test]
    fn a_sync_leaves_slots_whose_record_came_to_be_other_bytes_with_the_switch() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        // A shader-visible sampler heap of 4 slots, in memory of its own.
        let heap_desc: [u32; 8] = [1, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &HEAP_IID, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        let slots = title_heap.host().expect("a host sampler heap");
        let sync = || slots.sync_slots(0, 1, start, 16);
        let write = |bytes: [u32; 4]| {
            // SAFETY: the heap's first slot, 16 bytes.
            unsafe { (start as *mut [u32; 4]).write_unaligned(bytes) }
        };
        let sampler = [21, 7, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        crate::descriptor::create_sampler(null, &sampler, start);
        // SAFETY: the heap's first slot, 16 bytes.
        let record = unsafe { (start as *const [u32; 4]).read_unaligned() };
        let made = slots.made_key(0);
        assert!(made != 0, "made when written");
        let other = [0x3f80_0000, 0x4120_0000, 0xffff_ffff, 0x0006_c098];
        STALE_LEFT.with(|left| left.set(true));
        for bytes in [other, [0; 4]] {
            write(bytes);
            sync();
            assert_eq!(slots.made_key(0), made, "left as made from the view ({bytes:x?})");
            write(record);
            sync();
            assert_eq!(slots.made_key(0), made, "the record written back is made already");
        }
        write(other);
        sync();
        assert_eq!(slots.mirror(0, start, true), None, "a draw reads no view there");
        assert_eq!(slots.made_key(0), 0, "and makes the slot null");
        // A record of ours naming no view (an entry never given): null.
        write(record);
        sync();
        assert_eq!(slots.made_key(0), made, "made again from the record");
        write([record[0], u32::MAX >> 2, 1, 0]);
        sync();
        assert_eq!(slots.made_key(0), 0, "a record of ours naming no view is made null");
        // Without the switch other bytes make the slot null.
        STALE_LEFT.with(|left| left.set(false));
        write(record);
        sync();
        assert_eq!(slots.made_key(0), made);
        write(other);
        sync();
        assert_eq!(slots.made_key(0), 0, "made null without the switch");
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(heap, 0x10)(heap);
    }

    /// A view heap of 1024 slots in memory of its own, with a placed
    /// texture's view written by our call at slot `at`: the heap, its host
    /// heap, and the record's 32 bytes. Placed texture views are made
    /// cheaply by syncs (left null until read).
    fn heap_with_a_placed_view(at: u32, memory: &[u8]) -> (crate::object::This, &'static crate::descriptor::Heap, [u8; 32]) {
        heap_of_with_a_placed_view(1024, at, memory)
    }

    /// [`heap_with_a_placed_view`] of `count` slots.
    fn heap_of_with_a_placed_view(count: u32, at: u32, memory: &[u8]) -> (crate::object::This, &'static crate::descriptor::Heap, [u8; 32]) {
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, count, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &HEAP_IID, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let handle = title_heap.start + u64::from(at) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, handle);
        assert!(title_heap.host().expect("a host view heap").made(at), "made when written");
        // SAFETY: a slot of the heap.
        let record = unsafe { (handle as *const [u8; 32]).read_unaligned() };
        (heap, title_heap, record)
    }

    /// Writes `record` into slot `index` of `heap` as bytes, as the title
    /// copies records (no call, no mark).
    fn copy_into(heap: &crate::descriptor::Heap, index: u32, record: &[u8; 32]) {
        // SAFETY: a slot of the heap.
        unsafe { ((heap.start + u64::from(index) * 32) as *mut [u8; 32]).write_unaligned(*record) };
    }

    fn slots_of(runs: &[Run]) -> Vec<u32> {
        runs.iter().flatten().map(|&(i, _)| i).collect()
    }

    /// With write marks, a sync takes the marked slots alone; the fallback's
    /// page read whole finds the view record nobody marked, makes it and
    /// counts it, and leaves out the slots taken by their marks in the same
    /// sync (made once, not counted as unmarked).
    #[test]
    fn write_marks_sync_marked_slots_and_the_fallback_counts_an_unmarked_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let _one = crate::write_marks::tests::ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let memory = vec![0u8; 1 << 16];
        let (heap, title_heap, record) = heap_with_a_placed_view(10, &memory);
        let slots = title_heap.host().expect("made");
        let start = title_heap.start;
        let maps = crate::write_marks::tests::adopt_for_test(start, 1024);
        *crate::write_marks::FALLBACK.lock().unwrap() = crate::write_marks::Fallback::new();
        let sync = |frame: u64| {
            let mut runs = Vec::new();
            let checked = crate::descriptor::due_in_heap(slots, start, 32, 1024, None, frame, &mut runs);
            slots.make_runs(&runs, start, 32);
            (slots_of(&runs), checked)
        };
        // Copied without a mark: not looked at.
        copy_into(title_heap, 20, &record);
        // Copied and marked, as a patched store does.
        copy_into(title_heap, 30, &record);
        maps.mark_at(start + 30 * 32);
        let (made, checked) = sync(100);
        assert_eq!(made, [30], "the marked slot alone");
        assert_eq!((checked.pages, checked.unmarked, checked.tail_unmarked), (0, 0, 0));
        assert!(slots.made(30) && !slots.made(20));
        // The fallback reads the heap's page whole; another marked slot in it
        // is taken by its mark, once.
        copy_into(title_heap, 40, &record);
        maps.mark_at(start + 40 * 32);
        crate::write_marks::FALLBACK.lock().unwrap().add(&[(start as usize, start as usize + 1024 * 32)], crate::watch::PAGE);
        let (made, checked) = sync(101);
        assert_eq!(made, [40, 20], "the marked slot, then what the page read whole found");
        // (The heap's 32 KiB may lie across two 64 KiB pages.)
        assert!((1..=2).contains(&checked.pages), "{checked:?}");
        assert_eq!((checked.unmarked, checked.pending, checked.other), (1, 0, 0));
        assert!(slots.made(20) && slots.made(40));
        // Nothing left: the page is not read again until the watch tells it.
        let (made, checked) = sync(102);
        assert!(made.is_empty());
        assert_eq!(checked.pages, 0);
        crate::write_marks::let_go(start);
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(heap, 0x10)(heap);
    }

    /// With write marks, the two records before the end of each 64 KiB
    /// block are read once a frame: one no mark told of there is made and
    /// counted apart, one elsewhere waits for the fallback.
    #[test]
    fn write_marks_read_the_tail_of_every_block_once_a_frame() {
        if host().is_none() {
            return; // no host GPU here
        }
        let _one = crate::write_marks::tests::ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let memory = vec![0u8; 1 << 16];
        // 4,096 slots: 128 KiB, over at least one block's end.
        let (heap, title_heap, record) = heap_of_with_a_placed_view(4096, 1, &memory);
        let slots = title_heap.host().expect("made");
        let start = title_heap.start;
        let _maps = crate::write_marks::tests::adopt_for_test(start, 4096);
        *crate::write_marks::FALLBACK.lock().unwrap() = crate::write_marks::Fallback::new();
        let block_end = (start as usize & !0xffff) + 0x1_0000;
        let tail = ((block_end - 0x60) as u64 - start) as u32 / 32;
        let sync = |frame: u64| {
            let mut runs = Vec::new();
            let checked = crate::descriptor::due_in_heap(slots, start, 32, 4096, None, frame, &mut runs);
            slots.make_runs(&runs, start, 32);
            (slots_of(&runs), checked)
        };
        let frame = 1_000_000 + crate::frame::current();
        copy_into(title_heap, tail, &record);
        copy_into(title_heap, tail + 1, &record);
        copy_into(title_heap, tail - 10, &record);
        let (made, checked) = sync(frame);
        assert_eq!(made, [tail, tail + 1], "the block's tail alone");
        assert!(checked.tail_blocks >= 1, "{checked:?}");
        assert_eq!((checked.tail_unmarked, checked.unmarked), (2, 0));
        assert!(!slots.made(tail - 10));
        // Once a frame.
        copy_into(title_heap, tail, &[0x5a; 32]);
        copy_into(title_heap, tail, &record);
        assert!(slots.made(tail));
        let other = {
            let mut r = record;
            r[0] ^= 1; // no record of ours: nothing to make, but a look
            r
        };
        copy_into(title_heap, tail + 1, &other);
        let (_, checked) = sync(frame);
        assert_eq!(checked.tail_blocks, 0, "not again in the same frame");
        let (_, checked) = sync(frame + 1);
        assert!(checked.tail_blocks >= 1);
        crate::write_marks::let_go(start);
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(heap, 0x10)(heap);
    }

    /// With write marks and `VICTORY_TABLES_TO_DRAWS=1` both, a marked slot
    /// the sync left to the draws, or to wait, is not counted as unmarked
    /// when the fallback reads its page in the next sync, nor made then.
    #[test]
    fn slots_left_to_draws_are_not_unmarked_records() {
        if host().is_none() {
            return; // no host GPU here
        }
        let _one = crate::write_marks::tests::ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        TABLES_TO_DRAWS.with(|on| on.set(true));
        let memory = vec![0u8; 1 << 16];
        let (heap, title_heap, record) = heap_with_a_placed_view(10, &memory);
        TABLES_TO_DRAWS.with(|on| on.set(false));
        let slots = title_heap.host().expect("made");
        let start = title_heap.start;
        let maps = crate::write_marks::tests::adopt_for_test(start, 1024);
        *crate::write_marks::FALLBACK.lock().unwrap() = crate::write_marks::Fallback::new();
        let sync = |frame: u64| {
            let mut runs = Vec::new();
            let checked = crate::descriptor::due_in_heap(slots, start, 32, 1024, None, frame, &mut runs);
            slots.make_runs(&runs, start, 32);
            (slots_of(&runs), checked)
        };
        // A table's slot and another, written by marked stores.
        slots.note_tabled(20, 100);
        for i in [20, 30] {
            copy_into(title_heap, i, &record);
            maps.mark_at(start + u64::from(i) * 32);
        }
        let (made, _) = sync(101);
        assert!(made.is_empty(), "one left to the draws, one waiting");
        crate::write_marks::FALLBACK.lock().unwrap().add(&[(start as usize, start as usize + 1024 * 32)], crate::watch::PAGE);
        let (made, checked) = sync(102);
        assert!(made.is_empty(), "the fallback leaves them as they were left");
        assert_eq!((checked.unmarked, checked.pending), (0, 0), "{checked:?}");
        assert!(!slots.made(20) && !slots.made(30));
        // An unmarked one is still found, counted and made.
        copy_into(title_heap, 40, &record);
        crate::write_marks::FALLBACK.lock().unwrap().add(&[(start as usize, start as usize + 1024 * 32)], crate::watch::PAGE);
        let (made, checked) = sync(103);
        assert_eq!(made, [40]);
        assert_eq!((checked.unmarked, checked.unmarked_rewrote), (1, 0));
        assert!(slots.made(40));
        assert_eq!(slots.waited_out(200), [30]);
        crate::write_marks::let_go(start);
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(heap, 0x10)(heap);
    }

    /// With `VICTORY_TABLES_TO_DRAWS=1` a new record at a slot a draw read
    /// through a table lately is left to the draws; a new placed view
    /// elsewhere waits a few frames for a draw and is made after; the
    /// fallback's look leaves nothing to draws. A draw's read notes the slot.
    #[test]
    fn draws_table_records_are_left_to_the_draws_with_the_switch() {
        if host().is_none() {
            return; // no host GPU here
        }
        TABLES_TO_DRAWS.with(|on| on.set(true));
        let memory = vec![0u8; 1 << 16];
        let (heap, title_heap, record) = heap_with_a_placed_view(500, &memory);
        TABLES_TO_DRAWS.with(|on| on.set(false));
        let slots = title_heap.host().expect("made");
        let start = title_heap.start;
        let due = |first: u32, frame: u64, look: Look| {
            let mut runs = Vec::new();
            slots.due_runs(first, first + 1, start, 32, frame, look, &mut runs);
            slots.make_runs(&runs, start, 32);
            slots_of(&runs)
        };
        // A table's slot: left to the draws within eight frames.
        slots.note_tabled(510, 10);
        copy_into(title_heap, 510, &record);
        assert!(due(510, 12, Look::Fresh).is_empty(), "left to the draw");
        assert!(!slots.made(510));
        assert!(slots.waited_out(100).is_empty(), "and not waiting");
        // A draw reading it makes it.
        assert!(slots.mirror(510, start + 510 * 32, true).is_some());
        assert!(slots.made(510));
        assert!(slots.tabled_lately(510, crate::frame::current()), "noted by the draw's read");
        // Elsewhere: a placed view waits, and is made once its wait is over.
        copy_into(title_heap, 520, &record);
        assert!(due(520, 20, Look::Fresh).is_empty(), "waits for a draw");
        assert!(slots.waited_out(23).is_empty());
        let mut runs = Vec::new();
        crate::descriptor::due_in_heap(slots, start, 32, 1024, None, 24, &mut runs);
        slots.make_runs(&runs, start, 32);
        assert_eq!(slots_of(&runs), [520], "made after its wait");
        assert!(slots.made(520));
        // Long after a draw read it, a table's slot is no table's any more.
        slots.note_tabled(530, 10);
        copy_into(title_heap, 530, &record);
        assert!(due(530, 30, Look::Fresh).is_empty());
        assert_eq!(slots.waited_out(34), [530], "waits as any other");
        // The fallback's look leaves what a sync left as it was left, and
        // makes a record no sync left so (no mark told of it).
        slots.note_tabled(540, 40);
        copy_into(title_heap, 540, &record);
        assert!(due(540, 41, Look::Fresh).is_empty());
        assert!(due(540, 42, Look::Check).is_empty(), "left as it was");
        copy_into(title_heap, 550, &record);
        assert!(due(550, 41, Look::Fresh).is_empty(), "waits");
        assert!(due(550, 42, Look::Check).is_empty(), "waits still");
        assert_eq!(slots.waited_out(45), [550]);
        slots.note_tabled(560, 40);
        copy_into(title_heap, 560, &record);
        assert_eq!(due(560, 41, Look::Check), [560], "never left: made");
        // Without the switch nothing is left.
        let (other, other_heap, record) = heap_with_a_placed_view(500, &memory);
        let plain = other_heap.host().expect("made");
        plain.note_tabled(510, 10);
        copy_into(other_heap, 510, &record);
        let mut runs = Vec::new();
        plain.due_runs(510, 511, other_heap.start, 32, 12, Look::Fresh, &mut runs);
        assert_eq!(slots_of(&runs), [510]);
        assert!(!plain.tabled_lately(510, 12));
        for h in [heap, other] {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(h, 0x10)(h);
        }
    }

    thread_local! {
        /// Work to run while this thread prepares a view's texture, once
        /// (tests). The thread's own: tests run at once, and another's
        /// thread preparing a view took the work and ran it beside the test
        /// that set it (four of five runs of the suite, once a test
        /// prepared thousands of views).
        pub(crate) static WHILE_PREPARING: std::cell::RefCell<Option<crate::descriptor::tests::Work>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(crate) fn while_preparing() {
        let work = WHILE_PREPARING.with(|work| work.borrow_mut().take());
        if let Some(work) = work {
            work();
        }
    }

    /// A record the title writes into a slot while another thread prepares
    /// the slot's old record is the one the slot ends up viewing.
    #[test]
    fn a_slot_rewritten_while_its_view_is_prepared_views_the_new_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let s = 960u32;
        let at = title_heap.start + u64::from(s) * 32;
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let view_of = |address: u64| {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            view
        };
        let (first, second) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + (1 << 16));
        crate::descriptor::create_placed_srv(null, &words, &view_of(first), at);
        let host = title_heap.host().expect("a host view heap");
        let (done, waited) = std::sync::mpsc::channel();
        let second_view = view_of(second);
        let work: crate::descriptor::tests::Work = Box::new(move || {
            // The title writes the other record into the slot meanwhile.
            std::thread::spawn(move || {
                crate::descriptor::create_placed_srv(std::ptr::null(), &words, &second_view, at);
                done.send(()).unwrap();
            });
            assert!(waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(), "the write waits");
        });
        WHILE_PREPARING.with(|hook| *hook.borrow_mut() = Some(work));
        // A draw reads the slot: its first record is prepared.
        assert!(host.mirror(s, at, true).is_some());
        let second_texture = crate::resource::texture_at(second, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(second_texture).expect("made").texture) as usize;
        assert_eq!(host.textures_at(&[s]), [raw], "the slot views the record written last");
    }

    /// What a view needs is made without the slot lock: another thread
    /// binding a view of the same heap meanwhile is not held up.
    #[test]
    fn views_bound_while_another_slot_gets_its_texture_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (first, second) = (950u32, 951u32);
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        for (i, at) in [(first, 0u64), (second, 1 << 16)] {
            let address = memory.as_ptr() as u64 + at;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, slot(i));
        }
        let host = title_heap.host().expect("a host view heap");
        let heap_at = title_heap.start;
        let heap = heap as usize;
        let (done, waited) = std::sync::mpsc::channel();
        let work: crate::descriptor::tests::Work = Box::new(move || {
            std::thread::spawn(move || {
                let h = crate::descriptor::heap(heap as crate::object::This);
                h.host().unwrap().mirror(second, heap_at + u64::from(second) * 32, true);
                done.send(()).unwrap();
            });
            assert!(
                waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                "a view bound while another slot's texture is made waits"
            );
        });
        WHILE_PREPARING.with(|hook| *hook.borrow_mut() = Some(work));
        assert!(host.mirror(first, slot(first), true).is_some());
        assert!(WHILE_PREPARING.with(|work| work.borrow().is_none()), "the prepare ran");
        assert!(host.kept(first) && host.kept(second), "both made");
    }

    /// The textures of slots shaders read are made without the heaps' lock:
    /// a title thread writing a view meanwhile is not held up by the make.
    #[test]
    fn views_written_while_read_slots_get_their_textures_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (read, written) = (900u32, 901u32);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, slot(read));
        let host = title_heap.host().expect("a host view heap");
        assert!(!host.kept(read), "left null until read");
        let (done, waited) = std::sync::mpsc::channel();
        let at = slot(written);
        *crate::descriptor::tests::WHILE_MAKING.lock().unwrap() = Some((
            read,
            Box::new(move || {
                std::thread::spawn(move || {
                    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, at);
                    done.send(()).unwrap();
                });
                assert!(
                    waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                    "a view written while a read slot's texture is made waits"
                );
            }),
        ));
        crate::descriptor::make_read(&[read]);
        assert!(crate::descriptor::tests::WHILE_MAKING.lock().unwrap().is_none(), "the make ran");
        assert!(host.kept(read), "made for the read");
    }

    /// A heap sync makes the slots whose records the title changed as bytes
    /// (a copy of another slot's record, a record cleared) and passes the
    /// others by.
    #[test]
    fn heap_syncs_make_the_slots_whose_records_changed() {
        if host().is_none() {
            return; // no host GPU here
        }
        // Whichever way records are found current.
        for from_entries in [false, true] {
            crate::descriptor::tests::FROM_ENTRIES.with(|f| f.set(from_entries));
            heap_sync_of_changed_records();
        }
        crate::descriptor::tests::FROM_ENTRIES.with(|f| f.set(false));
    }

    fn heap_sync_of_changed_records() {
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // A view of no memory: a view of the test's own heap memory would get
        // a host copy whose 4 MiB window the guards may cover (the process
        // heap guarded read-only hung the suite). A table entry, not named
        // whole in its record: only an entry's record can name a generation
        // its view does not have (below).
        crate::descriptor::tests::ENTRIES_ONLY.with(|only| only.set(true));
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start);
        crate::descriptor::tests::ENTRIES_ONLY.with(|only| only.set(false));
        let host = title_heap.host().expect("a host view heap");
        // SAFETY: slots 0 and 1 of the heap's memory, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32) };
        assert!(!host.made(1), "copied as bytes: not seen yet");
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && host.made(1) && !host.made(2));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes((start + 32) as *mut u8, 0, 32) };
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && !host.made(1), "a cleared record leaves its slot");
        // A record naming a generation its view does not have (as one of a
        // view whose resource the title released): not made, and a slot made
        // from the record before leaves it.
        // SAFETY: slots 0, 1 and 2 of the heap's memory.
        unsafe {
            std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32);
            let [tag, view, generation] = (start as *const [u32; 3]).read_unaligned();
            ((start + 64) as *mut [u32; 3]).write_unaligned([tag, view, generation.wrapping_add(7)]);
            host.sync_slots(0, 4, start, 32);
            assert!(host.made(0) && host.made(1) && !host.made(2), "a stale record makes nothing");
            ((start + 32) as *mut [u32; 3]).write_unaligned([tag, view, generation.wrapping_add(7)]);
        }
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && !host.made(1), "a record gone stale leaves its slot");
        // A constant buffer view named whole in its record (when not
        // `VICTORY_VIEW_INLINE=0`), copied as bytes: made at the sync too.
        crate::descriptor::create_cbv(null, &[0x100, 0, 256, 0], start + 64);
        // SAFETY: slots 2 and 3 of the heap's memory.
        unsafe { std::ptr::copy_nonoverlapping((start + 64) as *const u8, (start + 96) as *mut u8, 32) };
        assert!(!host.made(3), "copied as bytes: not seen yet");
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(2) && host.made(3));
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(heap.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_VIEWS);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            // Released on a thread of its own.
            let since = std::time::Instant::now();
            while references(&buffer) > 1 && since.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        assert_eq!(heap.dynamic(51), None);
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// A texture view's size words are the console image descriptor's
    /// fields: level 0's size of the resource, the view's levels, its type.
    #[test]
    fn a_texture_view_gives_its_image_descriptor_size_fields() {
        // A 1000x600 2D texture of 10 levels, then an array of 6 with the
        // whole chain (mip count 0, 1024 wide: 11 levels).
        let mut texture = [0u32; 14];
        (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 1000, 600, 1 | 10 << 16, 1);
        let mut array = texture;
        (array[4], array[7]) = (1024, 6);
        let mut view = [0u32; 10];
        // Levels 2-4 of the texture: width - 1 = 999 (0b11_1110_0111).
        (view[1], view[4], view[5]) = (4, 2, 3);
        let words = image_words(&texture, Some(&view), false);
        assert_eq!(words, [0, 3 << 30, 249 | 599 << 14, 0xFAC | 2 << 12 | 4 << 16 | 9 << 28]);
        // Its whole chain from level 1.
        (view[4], view[5]) = (1, u32::MAX);
        assert_eq!(image_words(&texture, Some(&view), false)[3], 0xFAC | 1 << 12 | 9 << 16 | 9 << 28);
        // An unordered view of level 3: that level alone.
        let uav = [0, 4, 3, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&texture, Some(&uav), true)[3], 0xFAC | 3 << 12 | 3 << 16 | 9 << 28);
        // An unordered view of level 2 of six array slices: the level is
        // not the slice count (word 4).
        let uav = [0, 5, 2, 0, 6, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&array, Some(&uav), true)[3], 0xFAC | 2 << 12 | 2 << 16 | 13 << 28);
        // The array's own view: a 2D array, levels 0-10.
        let words = image_words(&array, None, false);
        assert_eq!(words[1..], [3 << 30, 255 | 599 << 14, 0xFAC | 10 << 16 | 13 << 28]);
    }

    /// The words `image_words` reads are where the host's view descriptions
    /// have their levels: an unordered view's `MipSlice` in word 2, a shader
    /// view's `MostDetailedMip` and `MipLevels` in words 4 and 5.
    #[test]
    fn view_description_words_hold_the_levels_where_image_words_reads_them() {
        let words = |uav: &D3D12_UNORDERED_ACCESS_VIEW_DESC| -> [u32; 10] {
            // SAFETY: 40 plain bytes.
            unsafe { std::ptr::from_ref(uav).cast::<[u32; 10]>().read_unaligned() }
        };
        assert_eq!(size_of::<D3D12_UNORDERED_ACCESS_VIEW_DESC>(), 40);
        assert_eq!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>(), 40);
        let mut uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        uav.Anonymous.Texture2D = D3D12_TEX2D_UAV { MipSlice: 3, PlaneSlice: 0 };
        assert_eq!(words(&uav)[1..5], [4, 3, 0, 0]);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Anonymous.Texture2DArray =
            D3D12_TEX2D_ARRAY_UAV { MipSlice: 2, FirstArraySlice: 1, ArraySize: 6, PlaneSlice: 0 };
        assert_eq!(words(&uav)[1..5], [5, 2, 1, 6]);
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        uav.Anonymous.Texture3D = D3D12_TEX3D_UAV { MipSlice: 1, FirstWSlice: 0, WSize: 32 };
        assert_eq!(words(&uav)[1..5], [8, 1, 0, 32]);
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        srv.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 2, MipLevels: 3, ..Default::default() };
        // SAFETY: 40 plain bytes.
        let w = unsafe { std::ptr::from_ref(&srv).cast::<[u32; 10]>().read_unaligned() };
        assert_eq!((w[1], w[4], w[5]), (4, 2, 3));
    }
}
