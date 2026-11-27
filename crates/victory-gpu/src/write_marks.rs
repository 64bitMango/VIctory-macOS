//! Write marks at the source (`VICTORY_WRITE_MARKS=1`, off by default until
//! measured; #14, stage 2 of the heap-sync plan in `docs/heap-slots.md`).
//!
//! Before: each heap sync took the write watch's changed 64 KiB pages of
//! the title's shader-visible view heap and read and compared every record
//! on them (~434,000 a frame, ~4.9 ms, plus ~0.6 ms asking the watch). The
//! title fills its draws' and dispatches' descriptor tables in that heap
//! with six record stores of its own, and copies records with its
//! `memcpy`/`memmove` imports; our own `Create*View` calls write records
//! there too.
//!
//! Now: the launcher patches the six stores (`victory_loader::mark_stores`,
//! bytes checked first) to set a byte per slot and a byte per 128-slot chunk
//! in maps of ours ([`Maps`], read by the stubs through [`CELLS`]); the
//! title's `memcpy`/`memmove` mark each slot whose copied record is a view
//! record ([`import`]); our calls mark theirs (`descriptor::mirror`).
//! `descriptor::sync_heaps` then takes the marked slots ([`Maps::take`])
//! in place of the changed pages and compares only those.
//!
//! The fallback, for writers nobody patched: every [`check_every`] frames
//! the write watch is asked for the heap's changed pages, and each frame a
//! few of them ([`check_budget`]) are read whole as before, leaving out the
//! slots this sync took by their marks (so the two parts never hold the
//! same slot; the fix of PR #354 of the fork this idea came from). A new
//! view record found there that no mark told of is made and counted
//! ([`Checked::unmarked`]), a writer the patches miss shows in the log.
//!
//! When any store could not be patched (another build), no heap is
//! adopted and every sync reads changed pages as before.
//!
//! What it leaves: a slot whose view record the title overwrote with other
//! bytes through an unmarked writer keeps its view until the fallback reads
//! its page (with `VICTORY_STALE_LEFT`, on by default, a sync leaves such
//! slots anyway).
//!
//! The idea and the store sites are from avatarmanz's "renderer port 2"
//! (`VICTORY_PORT_DESCRIPTORS`), with the fallback fix of Belobug PR #354;
//! this is our own code.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{
    AtomicPtr, AtomicU8, AtomicU64, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};

/// Whether write marks are asked for (`VICTORY_WRITE_MARKS=1`).
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_WRITE_MARKS"))
}

/// Slots a byte of the map of chunks stands for.
pub(crate) const CHUNK: usize = 128;

/// Bytes of a view record (a slot).
const SLOT: u64 = 32;

/// What the patched stores read (`victory_loader::mark_stores::CELLS_LEN`
/// bytes, in this order): the heap's first byte, its length in bytes (0:
/// none), the map of slots, the map of chunks.
#[repr(C)]
pub struct Cells {
    start: AtomicU64,
    len: AtomicU64,
    slots: AtomicPtr<AtomicU8>,
    chunks: AtomicPtr<AtomicU8>,
}

const _: () = assert!(size_of::<Cells>() == 32);

pub static CELLS: Cells = Cells {
    start: AtomicU64::new(0),
    len: AtomicU64::new(0),
    slots: AtomicPtr::new(std::ptr::null_mut()),
    chunks: AtomicPtr::new(std::ptr::null_mut()),
};

/// Where [`CELLS`] is, for the stubs.
pub fn cells_address() -> usize {
    std::ptr::from_ref(&CELLS) as usize
}

/// The title's image (base, size), its stores patched, and of how many.
static IMAGE: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

/// Told by the launcher: `patched` of the title's `of` record stores in the
/// image at `base` (`size` bytes) mark their slots.
pub fn hooked(base: usize, size: usize, patched: usize, of: usize) {
    IMAGE[0].store(base, Relaxed);
    IMAGE[1].store(size, Relaxed);
    IMAGE[2].store(patched, Relaxed);
    IMAGE[3].store(of, Relaxed);
    if patched != of {
        tracing::warn!(
            patched,
            of,
            "write marks (VICTORY_WRITE_MARKS): not every store of the title was patched; heap syncs read changed pages as before"
        );
    }
}

/// Whether marks stand for the heap's changed pages: asked for, and every
/// store patched.
fn working() -> bool {
    let of = IMAGE[3].load(Relaxed);
    on() && of > 0 && IMAGE[2].load(Relaxed) == of
}

/// The file name of the module holding code address `at`, and `at`'s
/// offset in it (a DLL the title's copies go through, our own code).
#[cfg(windows)]
fn module_of(at: usize) -> Option<(String, usize)> {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    // SAFETY: looks up the module holding the address, taking no reference.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            at as *const u16,
            &mut module,
        )
    };
    if found == 0 {
        return None;
    }
    let mut path = [0u16; 260];
    // SAFETY: a live module handle and room for the path.
    let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&path[..n.min(path.len())]);
    let name = path.rsplit(['\\', '/']).next().unwrap_or("?").to_string();
    Some((name, at - module as usize))
}

#[cfg(not(windows))]
fn module_of(_: usize) -> Option<(String, usize)> {
    None
}

/// A heap's maps: a byte per slot and per [`CHUNK`] slots, set by whoever
/// writes a slot, taken (cleared) by the heap sync.
pub(crate) struct Maps {
    pub(crate) start: u64,
    count: usize,
    slots: &'static [AtomicU8],
    chunks: &'static [AtomicU8],
    /// Per 64 KiB block: the code that last opened it with a write fault.
    writers: &'static [AtomicUsize],
}

impl Maps {
    /// Maps for a heap of `count` slots at `start`. Leaked: a stub may be
    /// marking while the heap goes (a few MB at most, once).
    pub(crate) fn new(start: u64, count: u32) -> Self {
        let count = count as usize;
        let leak = |n: usize| -> &'static [AtomicU8] { Box::leak((0..n).map(|_| AtomicU8::new(0)).collect()) };
        let writers = Box::leak((0..(count * SLOT as usize).div_ceil(1 << 16)).map(|_| AtomicUsize::new(0)).collect());
        Maps { start, count, slots: leak(count), chunks: leak(count.div_ceil(CHUNK)), writers }
    }

    /// Marks slot `slot` (in the heap: checked by the caller), as a stub does.
    fn mark_slot(&self, slot: usize) {
        self.slots[slot].store(1, Release);
        self.chunks[slot / CHUNK].store(1, Release);
    }

    /// Marks the slot at `handle`, if it lies in the heap (tests).
    #[cfg(test)]
    pub(crate) fn mark_at(&self, handle: u64) {
        let slot = handle.wrapping_sub(self.start) / SLOT;
        if (slot as usize) < self.count && handle >= self.start {
            self.mark_slot(slot as usize);
        }
    }

    /// Who last opened the block of `handle` with a write fault, as text:
    /// an offset in the title's image, or other code.
    fn writer(&self, handle: u64) -> String {
        let rip = self.writers.get(handle.wrapping_sub(self.start) as usize >> 16).map_or(0, |w| w.load(Relaxed));
        let (base, size) = (IMAGE[0].load(Relaxed), IMAGE[1].load(Relaxed));
        match rip {
            0 => "none seen".into(),
            rip if (base..base + size).contains(&rip) => format!("title+{:#x}", rip - base),
            rip => match module_of(rip) {
                Some((name, offset)) => format!("{name}+{offset:#x}"),
                None => format!("{rip:#x} (not the title's code)"),
            },
        }
    }

    /// Whether slot `slot` is marked (told, not yet taken).
    pub(crate) fn marked(&self, slot: u32) -> bool {
        self.slots.get(slot as usize).is_some_and(|s| s.load(Acquire) != 0)
    }

    /// The marked slots, each mark cleared, as sorted runs `(first, count)`.
    /// A chunk's mark is cleared before its slots are read: a store that
    /// marks meanwhile sets it again, for the next take.
    pub(crate) fn take(&self) -> Vec<(u32, u32)> {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for (c, chunk) in self.chunks.iter().enumerate() {
            if chunk.load(Relaxed) == 0 || chunk.swap(0, AcqRel) == 0 {
                continue;
            }
            let first = c * CHUNK;
            for (k, slot) in self.slots[first..(first + CHUNK).min(self.count)].iter().enumerate() {
                if slot.load(Relaxed) == 0 || slot.swap(0, AcqRel) == 0 {
                    continue;
                }
                let index = (first + k) as u32;
                match runs.last_mut() {
                    Some((at, n)) if *at + *n == index => *n += 1,
                    _ => runs.push((index, 1)),
                }
            }
        }
        runs
    }
}

/// The heap whose slots are marked (one: the title's placed view heap).
static ADOPTED: std::sync::RwLock<Option<&'static Maps>> = std::sync::RwLock::new(None);

/// A placed shader-visible view heap of `count` slots at `start` was made:
/// the first such is marked, when the stores were patched.
pub(crate) fn adopt(start: u64, count: u32) {
    if working() {
        adopt_now(start, count);
    }
}

fn adopt_now(start: u64, count: u32) -> bool {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.is_some() {
        tracing::info!(start = format_args!("{start:#x}"), count, "write marks: another placed view heap, read by its changed pages as before");
        return false;
    }
    let maps: &'static Maps = Box::leak(Box::new(Maps::new(start, count)));
    CELLS.slots.store(maps.slots.as_ptr().cast_mut(), Release);
    CELLS.chunks.store(maps.chunks.as_ptr().cast_mut(), Release);
    CELLS.start.store(start, Release);
    // Last: a store that sees a length sees the rest.
    CELLS.len.store(u64::from(count) * SLOT, Release);
    *adopted = Some(maps);
    tracing::info!(
        start = format_args!("{start:#x}"),
        count,
        stores = IMAGE[2].load(Relaxed),
        "write marks: the title's view heap is synced by the slots its stores, memory functions and our calls mark (VICTORY_WRITE_MARKS)"
    );
    true
}

/// The heap at `start` goes: nothing more is marked for it.
pub(crate) fn let_go(start: u64) {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.is_some_and(|m| m.start == start) {
        CELLS.len.store(0, Release);
        *adopted = None;
        *FALLBACK.lock().unwrap_or_else(|e| e.into_inner()) = Fallback::new();
        tracing::info!(start = format_args!("{start:#x}"), "write marks: the marked view heap went");
    }
}

/// The maps of the heap at `start`, if it is the marked one.
pub(crate) fn maps_for(start: u64) -> Option<&'static Maps> {
    ADOPTED.read().unwrap_or_else(|e| e.into_inner()).filter(|m| m.start == start)
}

/// Marks the slot at `handle` (a record our own call wrote), if it lies in
/// the marked heap.
#[inline]
pub(crate) fn mark(handle: u64) {
    let len = CELLS.len.load(Acquire);
    if len == 0 {
        return;
    }
    let at = handle.wrapping_sub(CELLS.start.load(Relaxed));
    if at >= len {
        return;
    }
    // SAFETY: the maps are leaked, as long as the length says.
    unsafe { mark_cell((at / SLOT) as usize) };
    STATS[S_OURS].fetch_add(1, Relaxed);
}

/// Marks slot `slot` of the published maps.
///
/// # Safety
/// `slot` is below the published length's slots.
#[inline]
unsafe fn mark_cell(slot: usize) {
    // SAFETY: by the caller; the maps are leaked.
    unsafe {
        (*CELLS.slots.load(Relaxed).add(slot)).store(1, Release);
        (*CELLS.chunks.load(Relaxed).add(slot / CHUNK)).store(1, Release);
    }
}

/// The runtime memory functions wrapped by [`import`].
static REAL: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// What the title's import `name` (bound to `address`) is bound to: with the
/// switch, memory copies and fills mark any descriptor records they touch.
pub fn import(name: &str, address: usize) -> usize {
    let (which, ours) = match name {
        "memcpy" => (0, copy as *const () as usize),
        "memmove" => (1, shift as *const () as usize),
        "memset" => (2, fill as *const () as usize),
        _ => return address,
    };
    if !on() || address == 0 {
        return address;
    }
    REAL[which].store(address, Relaxed);
    ours
}

type Copy = unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8;

unsafe extern "system-unwind" fn copy(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: the title's call, passed on to the function it asked for.
    unsafe { copied(to, from, n, REAL[0].load(Relaxed)) }
}

unsafe extern "system-unwind" fn shift(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: as above.
    unsafe { copied(to, from, n, REAL[1].load(Relaxed)) }
}

/// # Safety
/// `real` copies as `memcpy` does; the rest are the title's.
#[inline]
unsafe fn copied(to: *mut u8, from: *const u8, n: usize, real: usize) -> *mut u8 {
    // SAFETY: the host's function, noted by `import` before the title ran.
    let real: Copy = unsafe { std::mem::transmute::<usize, Copy>(real) };
    // SAFETY: by the caller.
    let out = unsafe { real(to, from, n) };
    mark_written(to as u64, n);
    out
}

unsafe extern "system-unwind" fn fill(to: *mut u8, value: i32, n: usize) -> *mut u8 {
    type Fill = unsafe extern "system-unwind" fn(*mut u8, i32, usize) -> *mut u8;
    // SAFETY: import saved the runtime function with this signature.
    let real: Fill = unsafe { std::mem::transmute(REAL[2].load(Relaxed)) };
    // SAFETY: forward the original call's arguments.
    let out = unsafe { real(to, value, n) };
    mark_written(to as u64, n);
    out
}

/// Mark writes to any part of a record's tag, index, or generation.
#[inline]
fn mark_written(to: u64, n: usize) {
    let len = CELLS.len.load(Acquire);
    if len == 0 || n == 0 {
        return;
    }
    let start = CELLS.start.load(Relaxed);
    let end = to.saturating_add(n as u64);
    if to >= start + len || end <= start {
        return;
    }
    let Some(maps) = maps_for(start) else {
        return;
    };
    let lo = to.max(maps.start);
    let hi = end.min(maps.start + maps.count as u64 * SLOT);
    if lo >= hi {
        return;
    }
    let offset = lo - start;
    let mut first = offset / SLOT;
    // The remaining bytes in a 32-byte slot do not affect its view record.
    first += u64::from(offset % SLOT >= 12);
    let last = (hi - start).div_ceil(SLOT);
    for slot in first..last {
        maps.mark_slot(slot as usize);
    }
    if first < last {
        STATS[S_COPIES].fetch_add(1, Relaxed);
        STATS[S_COPY_MARKS].fetch_add(last - first, Relaxed);
    }
}

/// Frames between askings of the write watch for the fallback
/// (`VICTORY_WRITE_MARKS_EVERY=<frames>`, 8 if not given).
pub(crate) fn check_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_WRITE_MARKS_EVERY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8u64).max(1))
}

/// Pages the fallback reads whole a frame (`VICTORY_WRITE_MARKS_CHECK=<pages>`,
/// 16 if not given; 0: none, and the watch is never asked).
pub(crate) fn check_budget() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_WRITE_MARKS_CHECK").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(16usize))
}

/// The fallback's state: pages the watch said were written and not read
/// yet, the frame it last asked in and last read in, and the page it goes on
/// from (round the heap).
pub(crate) struct Fallback {
    pages: std::collections::BTreeSet<usize>,
    asked: Option<u64>,
    looked: Option<u64>,
    next: usize,
}

impl Fallback {
    pub(crate) const fn new() -> Self {
        Fallback { pages: std::collections::BTreeSet::new(), asked: None, looked: None, next: 0 }
    }

    /// Whether to ask the write watch in `frame`, every `every` frames.
    pub(crate) fn ask_due(&mut self, frame: u64, every: u64, budget: usize) -> bool {
        if budget == 0 || self.asked.is_some_and(|asked| frame < asked + every) {
            return false;
        }
        self.asked = Some(frame);
        true
    }

    /// What the watch said was written: byte ranges, kept by page.
    pub(crate) fn add(&mut self, ranges: &[(usize, usize)], page: usize) {
        for &(from, to) in ranges {
            self.pages.extend((from & !(page - 1)..to).step_by(page));
        }
    }

    /// Up to `budget` pages to read whole in `frame`, once a frame.
    pub(crate) fn pages(&mut self, frame: u64, budget: usize, page: usize) -> Vec<usize> {
        if self.looked == Some(frame) {
            return Vec::new();
        }
        self.looked = Some(frame);
        let mut out = Vec::new();
        while out.len() < budget {
            let Some(&p) = self.pages.range(self.next..).next().or_else(|| self.pages.first()) else {
                break;
            };
            self.pages.remove(&p);
            self.next = p + page;
            out.push(p);
        }
        out
    }
}

pub(crate) static FALLBACK: std::sync::Mutex<Fallback> = std::sync::Mutex::new(Fallback::new());

/// Slots `first..last` without those of `taken` (sorted, disjoint runs
/// `(first, count)`): what a page the fallback reads leaves to it, so that a
/// slot is never held by both parts of a sync.
pub(crate) fn without(first: u32, last: u32, taken: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut at = first;
    let from = taken.partition_point(|&(f, n)| f + n <= first);
    for &(f, n) in &taken[from..] {
        if f >= last {
            break;
        }
        if f > at {
            out.push((at, f));
        }
        at = at.max(f + n);
    }
    if at < last {
        out.push((at, last));
    }
    out
}

/// What the fallback found in a sync.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Checked {
    /// Pages read whole.
    pub(crate) pages: u64,
    /// New view records no mark told of (made now).
    pub(crate) unmarked: u64,
    /// Of them, at slots made from another record before (rewritten; the
    /// rest were made from nothing).
    pub(crate) unmarked_rewrote: u64,
    /// New view records whose mark is set and was not taken yet.
    pub(crate) pending: u64,
    /// Slots to make whose record names no current view.
    pub(crate) other: u64,
    /// Blocks whose tail was read ([`tail_due`]), and new view records no
    /// mark told of found there (made now; not in `unmarked`).
    pub(crate) tail_blocks: u64,
    pub(crate) tail_unmarked: u64,
}

impl Checked {
    /// A slot `index` (record at `handle`) the fallback found to make:
    /// counted by what it is.
    pub(crate) fn found(&mut self, maps: &Maps, index: u32, handle: u64, view: bool, made_before: bool) {
        if !view {
            self.other += 1;
            return;
        }
        if maps.marked(index) {
            self.pending += 1;
            return;
        }
        self.unmarked += 1;
        self.unmarked_rewrote += u64::from(made_before);
        let n = UNMARKED_ALL.fetch_add(1, Relaxed);
        let writer = maps.writer(handle);
        *WRITERS.lock().unwrap_or_else(|e| e.into_inner()).entry(writer.clone()).or_default() += 1;
        if n < 20 || n.is_power_of_two() {
            tracing::warn!(
                slot = index,
                handle = format_args!("{handle:#x}"),
                in_block = format_args!("{:#x}", handle & 0xffff),
                made_before,
                %writer,
                since_start = n + 1,
                "write marks: a view record in the title's heap that no mark told of (found by the fallback; made now)"
            );
        }
    }

}

/// Bytes at the end of each 64 KiB block of the heap read once a frame
/// ([`tail_due`]): two records, before the title allocator's footer.
pub(crate) const TAIL_BYTES: usize = 0x60;
pub(crate) const TAIL_SLOTS: u32 = 2;

/// Whether the tails of the heap's blocks are read in `frame`: once a frame,
/// unless `VICTORY_WRITE_MARKS_TAIL=0`. The fallback's first A/B runs found
/// most view records no mark told of at offset 0xffa0 of a block (14 of 23
/// warnings; ~1 a frame), written by code not found; reading those two
/// records of each block (~1,200 a frame) makes them within a frame.
pub(crate) fn tail_due(frame: u64) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| crate::switch::off("VICTORY_WRITE_MARKS_TAIL")) {
        return false;
    }
    static LAST: AtomicU64 = AtomicU64::new(0);
    LAST.swap(frame + 1, Relaxed) != frame + 1
}

/// Since the start: unmarked view records the fallback found.
static UNMARKED_ALL: AtomicU64 = AtomicU64::new(0);

/// Since the start: unmarked view records by the code that last opened
/// their 64 KiB block with a write fault ([`faulted`]).
static WRITERS: std::sync::Mutex<std::collections::BTreeMap<String, u64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// A title write fault at `at` from code at `rip` opened a guarded block
/// (`crate::watch`): noted per 64 KiB block of the marked heap, for the
/// unmarked records' writers. (The first writer since the block was
/// guarded again: a hint, not proof.)
pub(crate) fn faulted(at: usize, rip: usize) {
    if CELLS.len.load(Relaxed) == 0 {
        return;
    }
    let Ok(adopted) = ADOPTED.try_read() else { return };
    if let Some(maps) = *adopted
        && let Some(w) = maps.writers.get((at as u64).wrapping_sub(maps.start) as usize >> 16)
    {
        w.store(rip, Relaxed);
    }
}

const S_OURS: usize = 0;
const S_COPIES: usize = 1;
const S_COPY_MARKS: usize = 2;
pub(crate) const S_TAKEN: usize = 3;
pub(crate) const S_TAKE_NS: usize = 4;
pub(crate) const S_SYNCS: usize = 5;
pub(crate) const S_SYNC_NS: usize = 6;
pub(crate) const S_MADE: usize = 7;
pub(crate) const S_ASK_NS: usize = 8;
pub(crate) const S_PAGES: usize = 9;
pub(crate) const S_UNMARKED: usize = 10;
pub(crate) const S_UNMARKED_REWROTE: usize = 11;
pub(crate) const S_PENDING: usize = 12;
pub(crate) const S_OTHER: usize = 13;
/// Draws' table records left to the draws (`VICTORY_TABLES_TO_DRAWS`).
pub(crate) const S_TO_DRAWS: usize = 14;
/// Constant and placed views that wait for a draw first.
pub(crate) const S_WAITING: usize = 15;
/// Of those, made by a sync after their wait.
pub(crate) const S_WAITED_MADE: usize = 16;
/// Slots a sync looked at (read and compared).
pub(crate) const S_LOOKED: usize = 17;
/// Block tails read ([`tail_due`]), and unmarked view records found there.
pub(crate) const S_TAIL_BLOCKS: usize = 18;
pub(crate) const S_TAIL_UNMARKED: usize = 19;

/// Counters since the last line.
pub(crate) static STATS: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];

/// With the stats lines: what heap syncs did per frame over `frames`, with
/// the write marks and the draws' table records (`VICTORY_TABLES_TO_DRAWS`).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn line(frames: f64) {
    let s: Vec<u64> = STATS.iter().map(|c| c.swap(0, Relaxed)).collect();
    if s[S_SYNCS] == 0 {
        return;
    }
    let f = |i: usize| s[i] as f64 / frames;
    let marks = if working() && maps_for(CELLS.start.load(Relaxed)).is_some() {
        format!(
            "write marks (VICTORY_WRITE_MARKS, {} stores): {:.0} slots taken by their marks in {:.2} ms ({:.0} marked by our calls; \
             {:.0} title memory writes into the heap marked {:.0}); fallback (asked every {} frames, {} pages a frame): {:.1} pages read whole, \
             {:.2} unmarked view records ({:.2} of them at slots made from another record before), {:.1} marked and not taken yet, {:.0} other records, \
             asking {:.2} ms; block tails: {:.0} read, {:.2} unmarked view records there; unmarked since the start {} (by the code that last opened their block with a write fault: {})",
            IMAGE[2].load(Relaxed),
            f(S_TAKEN),
            f(S_TAKE_NS) / 1e6,
            f(S_OURS),
            f(S_COPIES),
            f(S_COPY_MARKS),
            check_every(),
            check_budget(),
            f(S_PAGES),
            f(S_UNMARKED),
            f(S_UNMARKED_REWROTE),
            f(S_PENDING),
            f(S_OTHER),
            f(S_ASK_NS) / 1e6,
            f(S_TAIL_BLOCKS),
            f(S_TAIL_UNMARKED),
            UNMARKED_ALL.load(Relaxed),
            {
                let by = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
                let mut v: Vec<(&String, &u64)> = by.iter().collect();
                v.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
                let top: Vec<String> = v.iter().take(6).map(|(w, n)| format!("{w} {n}")).collect();
                if top.is_empty() { "none".into() } else { top.join(", ") }
            },
        )
    } else if on() {
        "write marks asked for (VICTORY_WRITE_MARKS=1) but not working: changed pages read as before".into()
    } else {
        "write marks off: changed pages read".into()
    };
    tracing::info!(
        "heap syncs per frame: {:.1} ms in {:.1} syncs, {:.0} slots looked at, {:.0} made; {marks}; \
         draws' table records left to the draws (VICTORY_TABLES_TO_DRAWS): {:.0}, constant and placed views that waited {:.0}, {:.0} of them made after the wait",
        f(S_SYNC_NS) / 1e6,
        f(S_SYNCS),
        f(S_LOOKED),
        f(S_MADE),
        f(S_TO_DRAWS),
        f(S_WAITING),
        f(S_WAITED_MADE),
    );
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Tests that adopt a heap (one at a time: the cells are one).
    pub(crate) static ADOPTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Adopts the heap at `start` for a test, whatever the switch.
    pub(crate) fn adopt_for_test(start: u64, count: u32) -> &'static Maps {
        // (A test that failed holding a heap left it adopted.)
        let stale = ADOPTED.read().unwrap_or_else(|e| e.into_inner()).map(|m| m.start);
        if let Some(stale) = stale {
            let_go(stale);
        }
        assert!(adopt_now(start, count), "no other heap adopted");
        maps_for(start).expect("adopted")
    }

    #[test]
    fn the_cells_are_laid_out_as_the_stubs_read_them() {
        // `victory_loader::mark_stores::CELLS_LEN` and its offsets (no
        // dependency on the loader here).
        assert_eq!(size_of::<Cells>(), 32);
        assert_eq!(std::mem::offset_of!(Cells, start), 0);
        assert_eq!(std::mem::offset_of!(Cells, len), 8);
        assert_eq!(std::mem::offset_of!(Cells, slots), 16);
        assert_eq!(std::mem::offset_of!(Cells, chunks), 24);
    }

    /// Slots marked as the stubs mark them (the cells), by our calls and on
    /// the maps come out once each, in order, as runs; nothing outside the
    /// heap is marked; after the heap goes nothing is.
    #[test]
    fn marked_slots_are_taken_once_as_runs() {
        let _one = ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x7_1000_0000u64;
        let maps = adopt_for_test(start, 1000);
        assert!(maps_for(start + 32).is_none());
        // As a stub: through the cells.
        let stub = |handle: u64| {
            let at = handle - CELLS.start.load(Acquire);
            assert!(at < CELLS.len.load(Acquire));
            // SAFETY: in the published length.
            unsafe { mark_cell((at / 32) as usize) };
        };
        stub(start + 5 * 32);
        mark(start + 6 * 32 + 7);
        mark(start + 300 * 32);
        maps.mark_at(start + 999 * 32);
        mark(start - 32);
        mark(start + 1000 * 32);
        maps.mark_at(start - 32);
        maps.mark_at(start + 1000 * 32);
        assert!(maps.marked(5) && maps.marked(999) && !maps.marked(7));
        assert_eq!(maps.take(), [(5, 2), (300, 1), (999, 1)]);
        assert_eq!(maps.take(), []);
        assert!(!maps.marked(5));
        // Chunks: 127 and 128 are in two, and run on.
        maps.mark_at(start + 127 * 32);
        maps.mark_at(start + 128 * 32);
        assert_eq!(maps.take(), [(127, 2)]);
        let_go(start);
        mark(start + 5 * 32);
        assert!(maps_for(start).is_none());
        assert_eq!(CELLS.len.load(Acquire), 0);
        assert!(!maps.marked(5), "nothing marked once the heap went");
    }

    #[test]
    fn bulk_writes_mark_partial_records_and_skip_padding() {
        let _one = ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x7_2000_0000;
        let maps = adopt_for_test(start, 4);
        mark_written(start - 2, 3);
        mark_written(start + SLOT + 8, 1);
        mark_written(start + SLOT * 2 + 12, 20);
        mark_written(start + SLOT * 4 - 32, 64);
        mark_written(start + SLOT * 2, 0);
        assert_eq!(maps.take(), [(0, 2), (3, 1)]);
        let_go(start);
    }

    #[test]
    fn fill_wrapper_forwards_arguments_marks_writes_and_returns_destination() {
        let _one = ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let mut buffer = [0u8; 128];
        let start = buffer.as_mut_ptr() as u64;
        let maps = adopt_for_test(start, 4);

        unsafe extern "system-unwind" fn mock_memset(to: *mut u8, value: i32, n: usize) -> *mut u8 {
            // SAFETY: caller passes a valid writable buffer of `n` bytes.
            unsafe { std::ptr::write_bytes(to, value as u8, n) };
            to
        }

        REAL[2].store(mock_memset as *const () as usize, Relaxed);
        // SAFETY: offset 8 is within the 128-byte local buffer.
        let dst = unsafe { buffer.as_mut_ptr().add(8) };
        // SAFETY: writing 12 bytes at offset 8 stays within `buffer`.
        let returned = unsafe { fill(dst, 0x5a, 12) };
        assert_eq!(returned, dst);
        assert_eq!(&buffer[..8], &[0u8; 8]);
        assert_eq!(&buffer[8..20], &[0x5a; 12]);
        assert_eq!(&buffer[20..], &[0u8; 108]);
        assert_eq!(maps.take(), [(0, 1)]);

        // SAFETY: offset 36 is within the 128-byte local buffer.
        let dst2 = unsafe { buffer.as_mut_ptr().add(32 + 4) };
        // SAFETY: writing 8 bytes at offset 36 stays within `buffer`.
        let ret2 = unsafe { fill(dst2, 0xa5, 8) };
        assert_eq!(ret2, dst2);
        assert_eq!(&buffer[36..44], &[0xa5; 8]);
        assert_eq!(maps.take(), [(1, 1)]);

        let_go(start);
    }

    #[test]
    fn a_page_read_whole_leaves_out_the_slots_taken_by_marks() {
        let taken = [(100, 10), (2040, 15), (2060, 10), (4090, 12)];
        assert_eq!(without(2048, 4096, &taken), [(2055, 2060), (2070, 4090)]);
        assert_eq!(without(0, 2048, &taken), [(0, 100), (110, 2040)]);
        assert_eq!(without(4096, 6144, &taken), [(4102, 6144)]);
        assert_eq!(without(5000, 6000, &taken), [(5000, 6000)]);
        assert_eq!(without(100, 110, &taken), []);
        assert_eq!(without(0, 10, &[]), [(0, 10)]);
        // Every slot of the page once, in one part or the other.
        let mut all: Vec<u32> = without(2048, 4096, &taken).iter().flat_map(|&(a, b)| a..b).collect();
        all.extend(taken.iter().flat_map(|&(f, n)| f..f + n).filter(|i| (2048..4096).contains(i)));
        all.sort_unstable();
        assert_eq!(all, (2048..4096).collect::<Vec<_>>());
    }

    #[test]
    fn the_fallback_asks_every_few_frames_and_reads_a_few_pages_a_frame() {
        let page = 0x10000;
        let mut f = Fallback::new();
        assert!(f.ask_due(1, 8, 16));
        assert!(!f.ask_due(5, 8, 16));
        assert!(f.ask_due(9, 8, 16));
        assert!(!Fallback::new().ask_due(1, 8, 0), "a budget of 0 never asks");
        f.add(&[(0x30_0000 + 5, 0x30_0000 + 3 * page), (0x10_0000, 0x10_0001)], page);
        assert_eq!(f.pages(9, 2, page), [0x10_0000, 0x30_0000]);
        assert_eq!(f.pages(9, 2, page), Vec::<usize>::new(), "once a frame");
        f.add(&[(0x20_0000, 0x20_0000 + page)], page);
        // On from where it left, then round.
        assert_eq!(f.pages(10, 2, page), [0x31_0000, 0x32_0000]);
        assert_eq!(f.pages(11, 2, page), [0x20_0000]);
        assert_eq!(f.pages(12, 2, page), Vec::<usize>::new());
    }

    /// The fallback counts a view record nobody marked once each, apart from
    /// one whose mark is waiting and records that name no view.
    #[test]
    fn the_fallback_counts_a_view_record_nobody_marked() {
        let _one = ADOPTING.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x7_2000_0000u64;
        let maps = adopt_for_test(start, 256);
        let mut checked = Checked::default();
        maps.mark_at(start + 5 * 32);
        checked.found(maps, 5, start + 5 * 32, true, true);
        assert_eq!(checked, Checked { pending: 1, ..Checked::default() });
        checked.found(maps, 6, start + 6 * 32, true, true);
        checked.found(maps, 7, start + 7 * 32, true, false);
        checked.found(maps, 8, start + 8 * 32, false, false);
        assert_eq!(checked, Checked { pages: 0, unmarked: 2, unmarked_rewrote: 1, pending: 1, other: 1, ..Checked::default() });
        // Who opened the block last is told with them.
        assert_eq!(maps.writer(start + 6 * 32), "none seen");
        faulted((start + 0x40) as usize, 0x1234_5678);
        assert_eq!(maps.writer(start + 6 * 32), "0x12345678 (not the title's code)");
        assert_eq!(maps.writer(start + 0x1_0000), "none seen", "another block");
        // Code of a module is named by the module.
        #[cfg(windows)]
        {
            faulted((start + 0x40) as usize, module_of as *const () as usize + 3);
            let writer = maps.writer(start + 6 * 32);
            assert!(writer.contains(".exe+0x"), "{writer}");
        }
        let_go(start);
    }
}
