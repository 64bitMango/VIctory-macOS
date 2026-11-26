//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::{PAGE, Take};

/// The copy side of the checker of `crate::unwaited`.
mod check;
pub(crate) use check::exempt as check_exempt;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<crate::quick::Set<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
    /// When it was last asked for ([`Shadow::ask`]): milliseconds and
    /// presented frames.
    asked: [AtomicU64; 2],
    /// Lists recorded and not yet run that use it ([`Flight`]).
    flights: AtomicU64,
    /// Parts of its memory (title addresses, in order, apart) a copy found
    /// with nothing committed, and whether too many were found to remember:
    /// see [`Shadow::copy`].
    gaps: Mutex<(Vec<(usize, usize)>, bool)>,
}

/// Whether a snapshot takes only the bytes no segment of its copy holds
/// yet (`crate::segments`), instead of rebuilding one segment over every
/// range that overlaps or touches: on unless `VICTORY_SNAPSHOTS_APART=0`.
fn snapshots_apart() -> bool {
    #[cfg(test)]
    if tests::JOINED.with(|joined| joined.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_SNAPSHOTS_APART"))
}

/// Gaps remembered per copy at most: past it the copy remembers none.
const GAPS: usize = 256;

/// Whether copies remember where their memory has nothing committed
/// ([`Shadow::copy`]): on unless `VICTORY_GAP_MEMO=0`.
fn gap_memo() -> bool {
    #[cfg(test)]
    if tests::NO_GAP_MEMO.with(|off| off.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_GAP_MEMO"))
}

/// A list's use of a copy from its recording until it has run (or is
/// reset): the copy is not released for idleness meanwhile
/// ([`release_idle`]), so every list of a submission uses the one copy of
/// its memory.
pub(crate) struct Flight(Arc<Shadow>);

impl Drop for Flight {
    fn drop(&mut self) {
        self.0.flights.fetch_sub(1, Relaxed);
    }
}

/// Snapshot segments (offset, bytes; `crate::segments`) and the pieces
/// compared.
type Snapshot = (Vec<crate::segments::Segment>, crate::quick::Set<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
    // The largest: small ones cost the allocator no page faults, and each
    // kept is looked at whenever a buffer is taken or given back.
    pool.truncate(POOL_BUFFERS);
}

/// Snapshot memory kept for reuse, and how many buffers at most.
const POOL_BYTES: usize = 512 << 20;
const POOL_BUFFERS: usize = 256;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests and
    /// diagnostics).
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// Something asks for it now (a lookup, a list using it submitted, a
    /// shader writing it at an address it computed): it is not idle
    /// ([`release_idle`]).
    pub(crate) fn ask(&self) {
        for (asked, now) in self.asked.iter().zip(&NOW) {
            asked.store(now.load(Relaxed), Relaxed);
        }
    }

    /// A list being recorded uses it: see [`Flight`].
    pub(crate) fn fly(self: &Arc<Self>) -> Flight {
        self.flights.fetch_add(1, Relaxed);
        Flight(self.clone())
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            marks,
            base: slice.map_or(0, |s| s.offset),
            slice,
            asked: [0, 1].map(|i| AtomicU64::new(NOW[i].load(Relaxed))),
            flights: AtomicU64::new(0),
            gaps: Mutex::default(),
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        // From here to the copy-back, "changed" must mean "the title wrote it
        // since the copy took it": pages a refresh left open while they
        // rested are taken for good first.
        crate::unwaited::not_beside("C1", "a range of a host copy is declared written", || {
            format!("{address:#x}+{len:#x} of copy {:#x}+{:#x}", self.start, self.len)
        });
        self.settle(address, len);
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
        if crate::unwaited::checking() {
            self.check_declared(address, len, marked);
        }
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // The bytes no segment holds yet, or (`VICTORY_SNAPSHOTS_APART=0`)
        // one segment with those the range overlaps or touches, rebuilt.
        let (taken, again) = if snapshots_apart() {
            (crate::segments::cover_apart(segments, from, to, host, pooled), 0)
        } else {
            crate::segments::cover_joined(segments, from, to, host, pooled, give_back)
        };
        #[cfg(feature = "diagnostics")]
        if taken > 0 {
            use crate::copystat::{C, add};
            add(C::SnapshotUs, started.elapsed().as_micros() as u64);
            add(C::SnapshotBytes, taken as u64);
            add(C::SnapshotAgain, again as u64);
            crate::copystat::snapshot(self.start, self.len, taken as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (taken, again);
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        // Joined segments never touch: the range is held if one holds it.
        if back
            && from < to
            && crate::segments::covered(segments, from, to)
        {
            let mut gathered = Vec::new();
            let mut laps = Laps::new(false);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            // The pages the range holds whole.
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                // The piece as it was taken, from the segments holding it.
                let old = crate::segments::held(segments, at, at + n, &mut gathered);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                laps.count(Stage::Pieces, 1);
                laps.count(Stage::Bytes, n);
                let differs = host != old;
                if differs {
                    // The comparing since the last changed piece.
                    laps.lap(Stage::CompareUs);
                }
                if differs && crate::host::read_into((start + at) as *const u8, title) {
                    laps.count(Stage::Changed, 1);
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    laps.lap(Stage::ReadUs);
                    if changed {
                        let writing = crate::watch::writing(start + at, n);
                        laps.lap(Stage::OpenUs);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        laps.lap(Stage::WriteUs);
                        drop(writing);
                        laps.lap(Stage::CloseUs);
                        written += n;
                        laps.count(Stage::Written, n);
                        laps.count(Stage::Writes, 1);
                        if self.tracked.is_some() && !(first..last).contains(&((start + at) / PAGE)) {
                            laps.count(Stage::Left, 1);
                        }
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            laps.lap(Stage::CompareUs);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
            laps.lap(Stage::CloseUs);
        }
        self.settled(&mut snapshot, &[(address, len)]);
        written
    }

    /// The submission's copy-back is done with `ranges` (address, bytes) as
    /// declared: every submitted step that declared them is done with them
    /// (one submission runs at a time), and the snapshot with the last
    /// range.
    fn settled(&self, snapshot: &mut Snapshot, ranges: &[(usize, usize)]) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        for &(address, len) in ranges {
            pending.remove(&(address, address + len));
        }
        if pending.is_empty() {
            snapshot.1.clear();
            for (_, bytes) in snapshot.0.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, pages: &[usize]) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        // What goes back (address, bytes), and the pages equal to title
        // memory once it has: written, then guarded, all at once (each
        // opening and guarding is a Wine call and a guard-lock round trip,
        // ~1,000 pages a frame in #50's world).
        let mut writes: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut taken = Vec::new();
        let mut laps = Laps::new(true);
        for &page in pages {
            let (from, to) = (page.max(start), (page + PAGE).min(end));
            if from >= to {
                continue;
            }
            // Our own write below must not save the page for this copy again.
            t.unarm(page / PAGE);
            let saved = t.saved(page / PAGE);
            if saved.is_none() && t.changed(page / PAGE) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::MarkedKept, 1);
                continue;
            }
            let n = to - from;
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
            let mut title = vec![0u8; n];
            if !crate::host::read_into(from as *const u8, &mut title) {
                continue;
            }
            laps.count(Stage::Pieces, 1);
            laps.count(Stage::Bytes, n);
            let differs = match &saved {
                // The title's bytes are the copy's from before: what differs
                // is the shaders'.
                None => {
                    let differs = *host != title[..];
                    if differs {
                        title.copy_from_slice(host);
                    }
                    differs
                }
                Some(before) => {
                    let before = &before[from - page..to - page];
                    let mut changed = false;
                    for ((t, h), b) in title.iter_mut().zip(host.iter_mut()).zip(before) {
                        if *h != *b && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    changed
                }
            };
            if differs {
                writes.push((from, title));
                laps.count(Stage::Changed, 1);
                laps.count(Stage::Writes, 1);
                laps.count(Stage::Written, n);
            }
            // Equal to title memory now: taken, unless it holds the page in part.
            if from == page && to == page + PAGE {
                taken.push(page / PAGE);
            } else if differs {
                laps.count(Stage::Left, 1);
            }
        }
        laps.lap(Stage::CompareUs);
        let mut written = 0;
        {
            // Consecutive writes share one opening.
            let mut i = 0;
            while i < writes.len() {
                let mut j = i + 1;
                while j < writes.len() && writes[j].0 == writes[j - 1].0 + writes[j - 1].1.len() {
                    j += 1;
                }
                let (from, to) = (writes[i].0, writes[j - 1].0 + writes[j - 1].1.len());
                let writing = crate::watch::writing(from, to - from);
                laps.lap(Stage::OpenUs);
                for (at, bytes) in &writes[i..j] {
                    crate::host::write_mapped(*at as *mut u8, bytes.as_ptr(), bytes.len());
                    written += bytes.len();
                }
                laps.lap(Stage::WriteUs);
                drop(writing);
                laps.lap(Stage::CloseUs);
                i = j;
            }
        }
        if !taken.is_empty() {
            t.taken(&taken);
        }
        laps.lap(Stage::CloseUs);
        written
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    #[cfg(test)]
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.update_as(address, len, Take::All)
    }

    /// The pages of `[address, address + len)` (whole pages) that a refresh
    /// left open and marked while they rested (`crate::watch::rest_ms`) are
    /// guarded and copied once more, as [`Shadow::update`] would have done:
    /// before shaders write the range, whose bytes are told from the title's
    /// by which pages the title wrote since the copy took them
    /// ([`Shadow::copy_back_marked`]).
    fn settle(&self, address: usize, len: usize) {
        let Some(t) = &self.tracked else {
            return;
        };
        let (from, to) = (address / PAGE * PAGE, address.saturating_add(len).div_ceil(PAGE) * PAGE);
        if t.rested(from, to) {
            self.update_as(from, to - from, Take::Rested);
        }
    }

    /// [`Shadow::update`], taking the changed pages as `how` says: all of
    /// them guarded (`Take::All`); those the title has just written left
    /// open, to be copied again at the next refresh (`Take::Resting`, for a
    /// submission none of whose shaders write title memory at addresses
    /// they compute: see [`Shadow::copy_back_page`]); or only the ones left
    /// so before, guarded now (`Take::Rested`).
    pub(crate) fn update_as(&self, address: usize, len: usize, how: Take) -> usize {
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them, what the title
            // wrote since it was taken (`update_busy`).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_free(at, a, how);
                }
                if e > a.max(at) {
                    bytes += self.update_busy(a.max(at), e, how);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, how);
            }
            return bytes;
        }
        self.update_free(from, to, how)
    }

    /// [`Shadow::update`] of `[from, to)` where a list submitted earlier
    /// declared that shaders write. A submission waits for the one before it,
    /// so none of this submission's shaders has run yet: the pages the title
    /// wrote since they were taken are copied as anywhere else. Skipped, the
    /// title's constants there stayed stale for the whole submission (a
    /// dispatch read its sizes as zeros and looped until the device was
    /// removed). The same bytes go into the snapshot `writing` took, so
    /// copy-back neither takes them for the GPU's nor puts them back over a
    /// later title write. Untracked memory (not the runtime's, or any under
    /// Rosetta) is copied whole, as [`Shadow::update_free`] copies it.
    ///
    /// Never left resting: shaders are about to write here.
    fn update_busy(&self, from: usize, to: usize, how: Take) -> usize {
        let runs = match &self.tracked {
            Some(t) if how == Take::Rested => t.take_as(from, to, Take::Rested),
            Some(t) => t.take(from, to),
            None if how == Take::Rested => Vec::new(),
            None => vec![(from, to)],
        };
        if runs.is_empty() {
            return 0;
        }
        let start = self.start as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
            if crate::unwaited::checking() {
                self.check_patched(a + start, b + start);
            }
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize, how: Take) -> usize {
        if how == Take::Rested {
            // Only what was left resting: no refresh of the rest.
            let Some(t) = &self.tracked else {
                return 0;
            };
            let (runs, unmapped) = t.settle(from, to);
            let copied = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
            for (a, b) in unmapped {
                // SAFETY: inside the mapping (`settle` cuts to the buffer).
                unsafe { self.data.add(a - self.start as usize).write_bytes(0, b - a) };
            }
            return copied;
        }
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let (runs, unmapped) = t.take_split_as(from, to, how);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let check = how == Take::All && crate::unwaited::checking_refreshes();
            let mut copied: usize = runs
                .into_iter()
                .map(|(a, b)| {
                    let copied = self.copy(a, b);
                    if check {
                        let _cost = crate::unwaited::checked();
                        self.check_equal(a, b, "after a refresh took the page");
                    }
                    copied
                })
                .sum();
            for (a, b) in unmapped {
                // SAFETY: inside the mapping (`take_split` cuts to the buffer).
                unsafe { self.data.add(a - self.start as usize).write_bytes(0, b - a) };
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
            }
            // Parts on memory that is not the runtime's: whole, as below.
            for (a, b) in t.untracked(from, to) {
                #[cfg(feature = "diagnostics")]
                let started = std::time::Instant::now();
                copied += self.copy(a, b);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::WholeBytes, (b - a) as u64);
                    add(C::WholeUs, started.elapsed().as_micros() as u64);
                }
            }
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    ///
    /// Where a copy found nothing committed (free or reserved address
    /// space) it is remembered, and the next copy asks about each such gap
    /// with one query instead of reading into it: a copy that meets an
    /// unreadable page is done again piece by piece, each failed read a
    /// fault (the window at address 0, which shaders reach with null
    /// addresses, is copied whole three times a frame, memory that is not
    /// the runtime's between unmapped stretches: 10-100 ms a frame, run 22).
    /// A gap still uncommitted throughout reads as zeros, as the read
    /// would have found it; one where anything was committed or mapped
    /// since is forgotten and copied as before. Asked at every copy:
    /// the runtime reports only its page mappings, not what the title or
    /// the host heap commits. Zeros are written each time (shaders may have
    /// written the copy there).
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        crate::unwaited::not_beside("C1", "a host copy is refreshed from title memory", || {
            format!("{from:#x}..{to:#x} of copy {start:#x}+{:#x}", self.len)
        });
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let host = |at: usize| unsafe { self.data.add(at - start) };
        #[cfg(feature = "diagnostics")]
        let plain = crate::copystat::mode() != 0;
        #[cfg(not(feature = "diagnostics"))]
        let plain = false;
        if plain || !gap_memo() {
            crate::host::copy_mapped(host(from), from as *const u8, to - from, false);
            return to - from;
        }
        let mut memo = self.gaps.lock().unwrap_or_else(|e| e.into_inner());
        let (gaps, full) = &mut *memo;
        let known: Vec<(usize, usize)> = gaps.iter().copied().filter(|&(a, b)| a < to && from < b).collect();
        let mut found = Vec::new();
        let mut note = |at: usize, until: usize| {
            let list = (!*full).then_some(&mut found);
            crate::host::copy_mapped_noting(host(at), at as *const u8, until - at, false, list);
        };
        let mut at = from;
        for (a, b) in known {
            let (cut_a, cut_b) = (a.max(from), b.min(to));
            if !crate::host::uncommitted(cut_a, cut_b) {
                // Copied with what follows it, which finds what is left of it.
                gaps.retain(|&g| g != (a, b));
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GapsDropped, 1);
                continue;
            }
            if at < cut_a {
                note(at, cut_a);
            }
            // SAFETY: inside the mapping.
            unsafe { host(cut_a).write_bytes(0, cut_b - cut_a) };
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::GapsKept, 1);
                add(C::GapBytes, (cut_b - cut_a) as u64);
                add(C::Unmapped, (cut_b - cut_a) as u64);
            }
            at = cut_b;
        }
        if at < to {
            note(at, to);
        }
        if !found.is_empty() {
            gaps.extend(found);
            gaps.sort_unstable();
            // Joined where they touch or overlap (a gap kept in part).
            let mut joined: Vec<(usize, usize)> = Vec::with_capacity(gaps.len());
            for &(a, b) in gaps.iter() {
                match joined.last_mut() {
                    Some((_, end)) if *end >= a => *end = (*end).max(b),
                    _ => joined.push((a, b)),
                }
            }
            *gaps = joined;
            if gaps.len() > GAPS {
                gaps.clear();
                *full = true;
            }
        }
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        }
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index, then the lock
            // words of emulated 64-bit atomics.
            let bytes = u64::from(victory_shader::emit::VIEW_WRITES_BYTES);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten since first.
pub(crate) fn take_reads() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let mut maps: Vec<&mut [u32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    if marks_quick() {
        // The marks stay set until forgotten, so most dwords hold bits and
        // none new: those are passed sixteen at a time.
        crate::back::new_bits(&*maps[0], maps.get(1).map(|m| &**m), seen, &mut new);
        return new;
    }
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// Whether the shaders' marks are read back the quick way: the bitmaps of
/// written pages and of slots read passed sixteen dwords at a time where
/// nothing is marked (or nothing new), no dword written that holds no mark
/// ([`copy_back_marks`], [`take_reads`]), and the pages marked grouped by
/// copy without a search per page. On unless `VICTORY_MARKS_QUICK=0`.
fn marks_quick() -> bool {
    #[cfg(test)]
    if tests::MARKS_SLOW.with(|slow| slow.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_MARKS_QUICK"))
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        for (w, word) in bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), but for pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; the bitmaps are cleared. Before the copy-backs of
/// declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    let quick = marks_quick();
    #[cfg(feature = "diagnostics")]
    let scanning = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarksScanUs);
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        // Marks of offsets past the bitmap land in its last bit, no copy's.
        bits[words - 1] = 0;
        let used = (marks().end as usize).div_ceil(32).min(words);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::MarksWords, used as u64);
        let into = if memory && memory_withheld { &mut held } else { &mut marked };
        if quick {
            crate::back::take_bits(&mut bits[..used], into);
            continue;
        }
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        drop(scanning);
        crate::copystat::add(crate::copystat::C::MarksBits, marked.len() as u64);
        crate::copystat::add(crate::copystat::C::MarksHeld, held.len() as u64);
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    #[cfg(feature = "diagnostics")]
    let finding = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarksFindUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    // Per copy, in page order, so its pages go back together.
    let mut by_copy: Vec<(Arc<Shadow>, Vec<usize>)> = Vec::new();
    for (copy, page) in pages {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, &copy) && *a < page + PAGE && page < a + n)
        {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarksHeld, 1);
            continue;
        }
        // A copy's bits are one run, so its pages come one after the other:
        // the copy found last, before a search through all (thousands of
        // pages of hundreds of copies cost a search each).
        match by_copy.last_mut().filter(|(c, _)| quick && Arc::ptr_eq(c, &copy)) {
            Some((_, list)) => list.push(page),
            None => match by_copy.iter_mut().find(|(c, _)| Arc::ptr_eq(c, &copy)) {
                Some((_, list)) => list.push(page),
                None => by_copy.push((copy, vec![page])),
            },
        }
    }
    for (_, list) in by_copy.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    #[cfg(feature = "diagnostics")]
    {
        drop(finding);
        crate::copystat::add(crate::copystat::C::MarksCopies, by_copy.len() as u64);
        for (copy, list) in &by_copy {
            crate::copystat::marked(copy.start, copy.len, list.len() as u64);
        }
    }
    if crate::unwaited::checking() {
        check::marked(&by_copy);
    }
    let at_once = if batched() { marked_at_once(&by_copy) } else { None };
    #[cfg(feature = "diagnostics")]
    if let Some(written) = at_once {
        crate::copystat::add(crate::copystat::C::MarkedPages, by_copy.iter().map(|(_, list)| list.len() as u64).sum());
        crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
    }
    if at_once.is_some() {
        return;
    }
    for (copy, list) in by_copy {
        let written = copy.copy_back_marked(&list);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, list.len() as u64);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    }
}

/// Whether a submission's copy-back is done as a batch of pieces on
/// several threads ([`pieces_at_once`]): on unless
/// `VICTORY_COPY_BACK_BATCH=0`, which does it range by range and page by
/// page on the submitting thread, each changed piece written whole, as
/// before.
fn batched() -> bool {
    #[cfg(test)]
    if tests::ONE_BY_ONE.with(|one| one.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_COPY_BACK_BATCH"))
}

/// Whether a batch reads the whole pages its copies took into the copies
/// once more after they are guarded ([`pieces_at_once`]): on unless
/// `VICTORY_COPY_BACK_AGAIN=0`.
fn read_again() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_COPY_BACK_AGAIN"))
}

/// The threads for a pass over `items` pieces: the helpers too
/// (`crate::back::threads`) from `least` pieces on. Waking them costs what
/// a few pieces do.
fn threads_for(items: usize, least: usize) -> usize {
    #[cfg(test)]
    if let threads @ 1.. = tests::THREADS.with(|t| t.get()) {
        return threads;
    }
    if items >= least { crate::back::threads() } else { 1 }
}

/// Title memory, read and written through the kernel (a page that is not
/// there fails the copy, or is left alone, instead of faulting).
struct TitleMemory;

impl crate::back::Title for TitleMemory {
    fn read(&self, at: usize, to: *mut u8, n: usize) -> bool {
        // SAFETY: `n` bytes the caller may write (its working memory, or a
        // piece's bytes of a copy's mapping).
        crate::host::read_into(at as *const u8, unsafe { std::slice::from_raw_parts_mut(to, n) })
    }

    fn write(&self, at: usize, from: *const u8, n: usize) {
        crate::host::write_mapped(at as *mut u8, from, n);
    }
}

/// What copy-back counts and times (diagnostics), for the pieces of
/// declared ranges and of marked pages apart, in the order of
/// `crate::copystat::C::BackPieces` and what follows it: pieces compared
/// and their bytes; pieces the shaders changed; time comparing, opening
/// title memory for our writes (`crate::watch::writing`), on changed pieces
/// in all (title memory read, merged and written), reading and merging,
/// writing, giving the pages back to the trackers (and taking those made
/// equal to title memory), reading taken pages again; bytes and writes put
/// into title memory; pages written that their copy could not take (it
/// holds them in part), so that its next refresh copies them again; pieces
/// opened that needed no write.
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
enum Stage {
    Pieces,
    Bytes,
    Changed,
    CompareUs,
    OpenUs,
    PutUs,
    ReadUs,
    WriteUs,
    CloseUs,
    AgainUs,
    Written,
    Writes,
    Left,
    Idle,
}

/// Diagnostics: counts, and time from one [`Laps::lap`] to the next, added
/// to the frame statistics when dropped (nothing without diagnostics).
struct Laps {
    #[cfg(feature = "diagnostics")]
    last: std::time::Instant,
    /// Counts, and nanoseconds for the timed stages.
    #[cfg(feature = "diagnostics")]
    sums: [u64; Stage::Idle as usize + 1],
    #[cfg(feature = "diagnostics")]
    marked: bool,
}

impl Laps {
    /// For pieces of marked pages (`marked`) or of declared ranges.
    fn new(marked: bool) -> Laps {
        #[cfg(not(feature = "diagnostics"))]
        let _ = marked;
        Laps {
            #[cfg(feature = "diagnostics")]
            last: std::time::Instant::now(),
            #[cfg(feature = "diagnostics")]
            sums: Default::default(),
            #[cfg(feature = "diagnostics")]
            marked,
        }
    }

    /// The time since the last lap (or since it was made) was spent on
    /// `stage`.
    #[inline]
    fn lap(&mut self, stage: Stage) {
        #[cfg(feature = "diagnostics")]
        {
            let now = std::time::Instant::now();
            self.sums[stage as usize] += (now - self.last).as_nanos() as u64;
            self.last = now;
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = stage;
    }

    #[inline]
    fn count(&mut self, stage: Stage, n: usize) {
        #[cfg(feature = "diagnostics")]
        {
            self.sums[stage as usize] += n as u64;
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = (stage, n);
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for Laps {
    fn drop(&mut self) {
        use Stage::*;
        // One by one, the changed pieces' time is their reading and writing.
        if self.sums[PutUs as usize] == 0 {
            self.sums[PutUs as usize] = self.sums[ReadUs as usize] + self.sums[WriteUs as usize];
        }
        for stage in [Pieces, Bytes, Changed, CompareUs, OpenUs, PutUs, ReadUs, WriteUs, CloseUs, AgainUs, Written, Writes, Left, Idle] {
            let timed = matches!(stage, CompareUs | OpenUs | PutUs | ReadUs | WriteUs | CloseUs | AgainUs);
            let n = self.sums[stage as usize] / if timed { 1000 } else { 1 };
            crate::copystat::back(self.marked, stage as usize, n);
        }
    }
}

/// After a submission ran: what its shaders wrote in the ranges its lists
/// declared goes back into title memory (`back`: copy, address, bytes,
/// whether it goes back; each range once), and what is withheld leaves the
/// copies (`crate::memory::drop_withheld`). Returns the bytes of the pieces
/// written.
///
/// The ranges that go back are done as one batch ([`declared_at_once`])
/// and the withheld ones after them, or (`VICTORY_COPY_BACK_BATCH=0`, and
/// where copies overlap) all in the order declared, as before. Withheld
/// ranges write neither title memory nor a copy, and they overlap no range
/// of their copy that goes back, so doing them after changes no byte: at
/// most a page counts as changed in a copy that would have taken it.
pub(crate) fn copy_back_declared(back: &[(Arc<Shadow>, usize, usize, bool)]) -> usize {
    if crate::unwaited::checking() {
        check::declared(back);
    }
    let at_once = if batched() { declared_at_once(back) } else { None };
    let mut written = at_once.unwrap_or(0);
    for (shadow, address, len, goes_back) in back {
        let (address, len) = (*address, *len);
        if *goes_back {
            if at_once.is_none() {
                written += shadow.copy_back(address, len);
            }
        } else {
            shadow.keep_writes(address, len);
            crate::memory::note_withheld(len as u64);
            crate::memory::drop_withheld(address, len);
        }
    }
    written
}

/// Whether pieces of different copies could share memory: `runs` (title
/// memory from, to, copy; each copy's own apart) overlap, or show the same
/// memory at two addresses. Such pieces are done one after the other, in
/// the order declared, as before: the second reads what the first wrote.
fn sharing(runs: &mut [(usize, usize, usize)]) -> bool {
    runs.sort_unstable();
    let mut end = 0;
    for &(from, to, _) in runs.iter() {
        if from < end {
            return true;
        }
        end = to;
    }
    let mut pages: Vec<(usize, usize)> = runs.iter().map(|&(from, to, _)| (from / PAGE, to.div_ceil(PAGE))).collect();
    crate::back::union(&mut pages);
    crate::watch::aliased_among(&pages)
}

/// [`Shadow::copy_back`] of every range of `back` that goes back, as one
/// batch of pieces ([`pieces_at_once`]); the ranges leave their copies'
/// pending sets as there. Returns the bytes of the pieces written; none,
/// with nothing done, if there is nothing to go back or copies share
/// memory ([`sharing`]).
///
/// The pieces are those of the union of each copy's ranges whose bytes its
/// snapshot holds, a whole piece compared once per submission. Range by
/// range, a byte two ranges held was compared again with title memory and
/// found equal, the first having put it there: the union leaves the same
/// bytes. (A page two ranges hold only between them is whole here, and
/// taken by its copy's tracker once all of it is title memory's.)
fn declared_at_once(back: &[(Arc<Shadow>, usize, usize, bool)]) -> Option<usize> {
    use crate::back::{Before, PIECE, Piece};
    // Per copy, in the order first declared: its ranges that go back.
    let mut copies: Vec<(&Shadow, Vec<(usize, usize)>)> = Vec::new();
    let mut index: crate::quick::Map<usize, usize> = Default::default();
    for (shadow, address, len, _) in back.iter().filter(|b| b.3) {
        let i = *index.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| {
            copies.push((shadow, Vec::new()));
            copies.len() - 1
        });
        copies[i].1.push((*address, *len));
    }
    if copies.is_empty() {
        return None;
    }
    // The submitting thread alone takes these locks (one submission at a
    // time), so several at once cannot wait for each other.
    let mut snapshots: Vec<_> = copies.iter().map(|(copy, _)| copy.before.lock().unwrap_or_else(|e| e.into_inner())).collect();
    // What each copy's snapshot holds of its ranges, joined (offsets in the
    // copy), and the same as title memory.
    let mut kept: Vec<Vec<(usize, usize)>> = Vec::with_capacity(copies.len());
    let mut runs = Vec::new();
    for (k, ((copy, ranges), snapshot)) in copies.iter().zip(&snapshots).enumerate() {
        let mut held: Vec<(usize, usize)> = ranges
            .iter()
            .map(|&(address, len)| copy.inside(address, len))
            .filter(|&(from, to)| from < to && crate::segments::covered(&snapshot.0, from, to))
            .collect();
        crate::back::union(&mut held);
        let start = copy.start as usize;
        runs.extend(held.iter().map(|&(from, to)| (start + from, start + to, k)));
        kept.push(held);
    }
    if sharing(&mut runs) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackOneByOne, 1);
        return None;
    }
    let mut written = 0;
    {
        let mut pieces: Vec<Piece> = Vec::new();
        for (k, ((copy, _), snapshot)) in copies.iter().zip(snapshots.iter_mut()).enumerate() {
            let (segments, compared) = &mut **snapshot;
            let segments: &[crate::segments::Segment] = segments;
            let start = copy.start as usize;
            for &(from, to) in &kept[k] {
                crate::back::pieces(start, from, to, |at, end| {
                    let whole = end - at == PIECE;
                    if whole && !compared.insert(start + at) {
                        return;
                    }
                    // SAFETY: inside the mapping.
                    let host = unsafe { copy.data.add(at) };
                    // Only a tracked copy takes pages.
                    let takes = whole && copy.tracked.is_some();
                    pieces.push(Piece::new(start + at, host, end - at, Before::Kept(segments, at), takes, k as u32));
                });
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::CompareBytes, pieces.iter().map(|p| p.n as u64).sum());
        let copies: Vec<&Shadow> = copies.iter().map(|(copy, _)| *copy).collect();
        written += pieces_at_once(&pieces, &copies, false);
    }
    for ((copy, ranges), mut snapshot) in copies.iter().zip(snapshots) {
        copy.settled(&mut snapshot, ranges);
    }
    Some(written)
}

/// [`Shadow::copy_back_marked`] of the marked pages of every copy (`marked`:
/// each copy once, its pages in order), as one batch of pieces
/// ([`pieces_at_once`]). Returns the bytes of the pieces written; none,
/// with nothing done, if copies share memory ([`sharing`]).
fn marked_at_once(marked: &[(Arc<Shadow>, Vec<usize>)]) -> Option<usize> {
    use crate::back::{Before, Piece};
    let mut runs = Vec::new();
    for (k, (copy, pages)) in marked.iter().enumerate() {
        let (start, end) = (copy.start as usize, (copy.start + copy.len) as usize);
        runs.extend(pages.iter().map(|&page| (page.max(start), (page + PAGE).min(end), k)).filter(|r| r.0 < r.1));
    }
    if sharing(&mut runs) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackOneByOne, 1);
        return None;
    }
    #[cfg(feature = "diagnostics")]
    let making = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarksPrepUs);
    let mut pieces: Vec<Piece> = Vec::new();
    for (k, (copy, pages)) in marked.iter().enumerate() {
        let Some(t) = &copy.tracked else {
            continue;
        };
        let (start, end) = (copy.start as usize, (copy.start + copy.len) as usize);
        for &page in pages {
            let (from, to) = (page.max(start), (page + PAGE).min(end));
            if from >= to {
                continue;
            }
            // Our own write below must not save the page for this copy again.
            t.unarm(page / PAGE);
            let before = match t.saved(page / PAGE) {
                Some(saved) => Before::Saved(saved, from - page),
                None if t.changed(page / PAGE) => {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::MarkedKept, 1);
                    continue;
                }
                None => Before::Title,
            };
            // SAFETY: inside the mapping.
            let host = unsafe { copy.data.add(from - start) };
            pieces.push(Piece::new(from, host, to - from, before, to - from == PAGE, k as u32));
        }
    }
    let copies: Vec<&Shadow> = marked.iter().map(|(copy, _)| &**copy).collect();
    #[cfg(feature = "diagnostics")]
    drop(making);
    Some(pieces_at_once(&pieces, &copies, true))
}

/// What shaders wrote in `pieces` goes back into title memory, all pieces
/// at once and on several threads (`crate::back`); `copies` are the copies
/// the pieces name. Returns the bytes of the pieces written into.
///
/// What it relies on, and what stays as it was done piece by piece:
/// - The pieces share no byte of a copy or of title memory, and no memory
///   shown twice (the callers: [`sharing`], and a copy's own pieces apart),
///   so their order does not matter and one thread does each.
/// - One submission runs at a time, and this is its copy-back: no shader
///   writes the copies, nothing refreshes them, and the bytes kept from
///   before stay as they are (the callers hold the snapshots).
/// - Title memory is opened for our writes once for all pieces
///   (`crate::watch::writing`, by runs of pages) before any is written, and
///   only where the shaders changed a piece: the kernel's write would fail
///   on a guarded page, and a tracker taking a page meanwhile leaves it
///   open. Opening marks the pages changed in every tracker holding them,
///   as before; a page whose piece turns out to hold title memory's bytes
///   already is marked without need (counted). A marked page the title has
///   not written is opened without a look (`VICTORY_MARKS_QUICK=0`: read
///   and compared first): telling whether its shaders changed it takes the
///   read its merge makes anyway.
/// - A changed piece is read from title memory, merged and written by one
///   thread without a pause, as before, so a title write is lost only
///   where it lands between that read and that write in bytes we write:
///   runs of bytes the shaders changed (and short gaps between them), not
///   the whole piece any more.
/// - Then the pages are given back to the trackers, and each copy takes
///   the whole pages now equal to title memory, as before (our own write
///   is not taken for the title's at its next refresh). Those are read
///   into the copy once more after they are guarded, which the piece by
///   piece way does not do: there a title write between a piece's read
///   and its page's guard stays out of the copy until the page is next
///   written, and with the batch that time would be longer for small
///   ranges (`VICTORY_COPY_BACK_AGAIN=0` leaves it out).
fn pieces_at_once(pieces: &[crate::back::Piece], copies: &[&Shadow], marked: bool) -> usize {
    let mut laps = Laps::new(marked);
    let title = TitleMemory;
    let mut spread = 0;
    let mut threads = |items: usize, least: usize| {
        let threads = threads_for(items, least);
        spread += usize::from(threads > 1);
        threads
    };
    // Marked pieces whose title wrote nothing are compared with title
    // memory itself, read for it (fewer make a pass worth the helpers), or
    // opened without that read.
    let quick = marks_quick();
    crate::back::compare(pieces, &title, threads(pieces.len(), if marked { 8 } else { 32 }), quick);
    laps.count(Stage::Pieces, pieces.len());
    laps.count(Stage::Bytes, pieces.iter().map(|p| p.n).sum());
    laps.lap(Stage::CompareUs);
    // Title memory opened for the changed pieces, by runs of pages.
    let mut pages: Vec<(usize, usize)> = pieces.iter().filter(|p| p.opens()).map(|p| (p.at / PAGE, p.at / PAGE + 1)).collect();
    laps.count(Stage::Changed, pages.len());
    crate::back::union(&mut pages);
    let open: Vec<crate::watch::Writing> =
        pages.iter().map(|&(first, last)| crate::watch::writing(first * PAGE, (last - first) * PAGE)).collect();
    laps.lap(Stage::OpenUs);
    #[cfg(feature = "diagnostics")]
    let spent = crate::back::Spent::default();
    #[cfg(feature = "diagnostics")]
    let timed = Some(&spent);
    #[cfg(not(feature = "diagnostics"))]
    let timed = None;
    crate::back::put(pieces, &title, threads(pages.iter().map(|&(first, last)| last - first).sum(), 8), timed);
    laps.lap(Stage::PutUs);
    // The pages we opened, to be guarded again without asking what memory
    // they are (`crate::watch::Tracked::taken_known`).
    let known: Vec<(usize, usize)> =
        if quick { open.iter().flat_map(|writing| writing.opened().iter().copied()).collect() } else { Vec::new() };
    drop(open);
    // Whole pages made equal to title memory are current in their copies.
    let mut taken: Vec<Vec<usize>> = vec![Vec::new(); copies.len()];
    let mut written = 0;
    for piece in pieces {
        if piece.whole && piece.synced() {
            taken[piece.copy as usize].push(piece.at / PAGE);
        }
        if piece.wrote() {
            written += piece.n;
            let (bytes, writes) = piece.written();
            laps.count(Stage::Written, bytes as usize);
            laps.count(Stage::Writes, writes as usize);
            laps.count(Stage::Left, usize::from(!piece.whole && copies[piece.copy as usize].tracked.is_some()));
        } else if piece.opens() && piece.synced() {
            laps.count(Stage::Idle, 1);
        }
    }
    let mut again = 0;
    for (copy, pages) in copies.iter().zip(&taken) {
        if let Some(t) = &copy.tracked
            && !pages.is_empty()
        {
            t.taken_known(pages, &known);
            again += pages.len();
        }
    }
    laps.lap(Stage::CloseUs);
    if read_again() {
        crate::back::again(pieces, &title, threads(again, 16));
    }
    laps.lap(Stage::AgainUs);
    if crate::unwaited::checking() && read_again() {
        check::taken(copies, &taken);
    }
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        laps.count(Stage::ReadUs, spent.reading_ns.load(Relaxed) as usize);
        laps.count(Stage::WriteUs, spent.writing_ns.load(Relaxed) as usize);
        crate::copystat::add(crate::copystat::C::BackBatches, 1);
        crate::copystat::add(crate::copystat::C::BackSpread, spread as u64);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = spread;
    written
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
/// A read-write lock: a lookup that finds the copy made ([`find`], for every
/// buffer view made and every draw's buffers) shares it with others;
/// registering, releasing and making a copy hold it alone ([`buffers`]).
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

/// The title buffers, held alone.
fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// Whether lookups of copies already made share their registries
/// ([`BUFFERS`], [`LOOSE`]): unless `VICTORY_COPIES_SHARED=0`, which takes
/// them alone for every lookup, as when they were mutexes.
fn lookups_shared() -> bool {
    #[cfg(test)]
    if tests::LOOKUPS_ALONE.with(|alone| alone.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_COPIES_SHARED"))
}

/// The buffer [`find`] takes for `[address, end)`: the innermost of the
/// nearest 64 starting at or below it.
fn holding(map: &BTreeMap<(u64, usize), Entry>, address: u64, end: u64) -> Option<(u64, usize)> {
    map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end).map(|(k, _)| *k)
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
#[track_caller]
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // The copy already made, or no buffer there: found with the buffers
    // shared. A copy to make is made below with them held alone, the
    // buffer looked up again.
    if lookups_shared() {
        let map = crate::contend::read(&BUFFERS, crate::contend::Lock::Copies);
        match holding(&map, address, end) {
            None => {
                drop(map);
                return loose(address, end);
            }
            Some(key) => {
                if let Some(shadow) = map.get(&key).and_then(|e| e.shadow.clone()) {
                    return Some(shadow);
                }
            }
        }
    }
    let mut map = crate::contend::write(&BUFFERS, crate::contend::Lock::Copies);
    let key = holding(&map, address, end);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches. The title unmaps little
/// of it, so a window nothing asked for in a while is released
/// ([`release_idle`]) and made again when asked for.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length). A
/// read-write lock: a lookup that finds its window ([`loose`]) shares it
/// and stamps the window while it holds it; a release ([`release_idle`])
/// holds it alone while it checks those stamps and takes windows out, so
/// no window goes that a lookup holding the lock has found.
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = std::sync::RwLock::new(BTreeMap::new());

#[track_caller]
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    if lookups_shared()
        && let Some(shadow) = crate::contend::read(&LOOSE, crate::contend::Lock::Copies).get(&(start, len))
    {
        // Under the lock (shared), which a release holds alone.
        shadow.ask();
        return Some(shadow.clone());
    }
    // Looked up again: another thread may have made it meanwhile.
    let mut map = crate::contend::write(&LOOSE, crate::contend::Lock::Copies);
    if let Some(shadow) = map.get(&(start, len)) {
        // Under the lock, where a release checks it.
        shadow.ask();
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    // A null or near-null address something took for title memory: which
    // lookup (the window is then refreshed with every other, run 22).
    if start == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        let by = std::panic::Location::caller();
        ONCE.call_once(|| {
            tracing::info!(
                address = format_args!("{address:#x}"),
                bytes = end - address,
                by = format_args!("{}:{}", by.file(), by.line()),
                "a copy of title memory at address 0 was asked for (logged once)"
            )
        });
    }
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    map.insert((start, len), shadow.clone());
    changed();
    if released().made((start, len)) {
        IDLE_COUNTS[2].fetch_add(1, Relaxed);
        IDLE_COUNTS[3].fetch_add(len, Relaxed);
    }
    Some(shadow)
}

/// The time copies are stamped with when asked for ([`Shadow::ask`]):
/// milliseconds and presented frames, as of the last submission's start
/// ([`idle_sweep`]).
static NOW: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Seconds (and presented frames) without being asked for after which a
/// window is released: `VICTORY_IDLE_RELEASE_S` (default
/// [`crate::idle::IDLE_S`]); none with `VICTORY_IDLE_RELEASE=0`, and in
/// tests, which release what they made themselves.
fn idle_seconds() -> Option<u64> {
    static SECONDS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *SECONDS.get_or_init(|| {
        if cfg!(test) {
            return None;
        }
        let var = |name: &str| std::env::var(name).ok();
        let seconds = crate::idle::setting(var("VICTORY_IDLE_RELEASE").as_deref(), var("VICTORY_IDLE_RELEASE_S").as_deref());
        match seconds {
            Some(seconds) => tracing::info!(
                seconds,
                "copies of title memory outside buffers nothing asked for in this many seconds and frames are released (VICTORY_IDLE_RELEASE_S; VICTORY_IDLE_RELEASE=0 keeps them)"
            ),
            None => tracing::info!("copies of title memory outside buffers are never released (VICTORY_IDLE_RELEASE=0)"),
        }
        seconds
    })
}

/// What is remembered of the windows released. Locked after `LOOSE` where
/// both are held.
static RELEASED: Mutex<crate::idle::Released> = Mutex::new(crate::idle::Released::new());

fn released() -> std::sync::MutexGuard<'static, crate::idle::Released> {
    RELEASED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Since the last [`log_idle`]: windows released and their bytes, windows
/// made again where one was released and their bytes, those of them made
/// again because shaders found no copy there. And as of the last pass: idle
/// windows kept because a heap slot views them, and because a list not yet
/// run uses them or shaders' writes are not copied back.
static IDLE_COUNTS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// After a submission: its shaders read or wrote title memory `[start,
/// end)` at addresses they computed and found no copy (`crate::memory`). A
/// window released there is made again before the next submission
/// ([`idle_sweep`]).
pub(crate) fn note_missed(start: u64, end: u64) {
    released().missed(start, end);
}

/// At each submission, before its lists are prepared and while none runs
/// (`Host::execute`): the stamps' time moves on, windows released where
/// shaders found no copy since are made again, and once a second the idle
/// windows are released ([`release_idle`]).
pub(crate) fn idle_sweep() {
    crate::unwaited::not_beside("C6", "idle copies of title memory are released", String::new);
    let Some(seconds) = idle_seconds() else {
        return;
    };
    let now = (crate::resource::now_ms(), crate::queue::presented());
    NOW[0].store(now.0, Relaxed);
    NOW[1].store(now.1, Relaxed);
    make_missed_again();
    static LAST: AtomicU64 = AtomicU64::new(0);
    if now.0.saturating_sub(LAST.load(Relaxed)) < 1000 {
        return;
    }
    LAST.store(now.0, Relaxed);
    let started = std::time::Instant::now();
    let (windows, bytes) = release_idle(now, seconds, None);
    let took = started.elapsed();
    if took > std::time::Duration::from_millis(20) {
        tracing::info!(windows, bytes, ms = took.as_millis() as u64, "idle copies of title memory released (slow pass)");
    }
}

/// Windows released where shaders found no copy since ([`note_missed`])
/// are made again from title memory, and never released again: such reads
/// ask for nothing, so a window only they use looks idle. Returns how many.
fn make_missed_again() -> u64 {
    let wanted = released().wanted();
    let mut made = 0;
    for (start, len) in wanted {
        // Counted as made again where it is made (`loose`); one a lookup
        // made meanwhile is not counted here.
        let there = LOOSE.write().unwrap_or_else(|e| e.into_inner()).contains_key(&(start, len));
        if !there && loose(start, start + len).is_some() {
            made += 1;
        }
    }
    IDLE_COUNTS[4].fetch_add(made, Relaxed);
    made
}

/// Releases the windows idle at `now` for `seconds`
/// ([`crate::idle::idle`]), only those inside `within` (start, end; tests):
/// they leave the copies the runtime finds, and the next lookup of their
/// memory makes a new window from title memory, as on first use. Returns how
/// many and their bytes. Between submissions (`Host::execute` holds the
/// submission lock; the last one has run).
///
/// Kept, however idle:
/// - a window a title heap slot views (`crate::bindless::shown`, and the
///   slots' own record of the buffers they view): shaders indexing the
///   heap may read any slot, and draws reading the slot refresh the copy
///   they find, which must be the one the slot shows;
/// - a window a list recorded and not yet run uses ([`Flight`]): a new
///   window made meanwhile would not show what that list's shaders write
///   to the other lists of its submission;
/// - a window whose shaders' writes are not copied back yet;
/// - a window made again after shaders missed it ([`make_missed_again`]).
///
/// The slots are looked at and the windows taken out while no title heap
/// slot is being made (`crate::descriptor::with_slots_still`) and under
/// the windows' lock, where a lookup stamps what it finds: a window found
/// since the pass read the time is not idle, so none goes that a slot
/// being made, or a draw being recorded, has just looked up.
///
/// A released window's host buffer lives while anything still holds it:
/// lists that ran and are not reset yet, and `crate::memory`'s page
/// tables, which name it until the next submission using them rewrites
/// them (the copies' generation changed). Its page guards go with it.
fn release_idle(now: crate::idle::Stamp, seconds: u64, within: Option<(u64, u64)>) -> (u64, u64) {
    let inside = |s: &Shadow| within.is_none_or(|(a, b)| a <= s.start && s.start + s.len <= b);
    let asked = |s: &Shadow| (s.asked[0].load(Relaxed), s.asked[1].load(Relaxed));
    let idle = |s: &Shadow| crate::idle::idle(asked(s), now, seconds);
    let at = |s: &Arc<Shadow>| Arc::as_ptr(s) as usize;
    let candidates: Vec<Arc<Shadow>> = LOOSE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter(|s| inside(s) && idle(s))
        .cloned()
        .collect();
    if candidates.is_empty() {
        if within.is_none() {
            IDLE_COUNTS[5].store(0, Relaxed);
            IDLE_COUNTS[6].store(0, Relaxed);
        }
        return (0, 0);
    }
    let (gone, shown, busy) = crate::descriptor::with_slots_still(|slots| {
        use windows::core::Interface;
        let shown = crate::bindless::shown(&candidates.iter().map(at).collect());
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let mut released = released();
        let (mut free, mut busy) = (Vec::new(), 0u64);
        for c in candidates.iter().filter(|c| !shown.contains(&at(c))) {
            let key = (c.start, c.len);
            // Still the window found there, and not asked for since.
            if !map.get(&key).is_some_and(|s| Arc::ptr_eq(s, c)) || !idle(c) || released.pinned(key) {
                continue;
            }
            if c.flights.load(Relaxed) != 0 || !c.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
                busy += 1;
                continue;
            }
            free.push(c);
        }
        // The slots themselves, whatever `crate::bindless` has of them.
        let raw = |c: &Arc<Shadow>| c.buffer.as_raw() as usize;
        let viewed = if free.is_empty() { Default::default() } else { slots(&free.iter().map(|c| raw(c)).collect()) };
        let mut gone = Vec::new();
        for c in free.into_iter().filter(|c| !viewed.contains(&raw(c))) {
            let key = (c.start, c.len);
            gone.extend(map.remove(&key));
            released.release(key);
        }
        (gone, (shown.len() + viewed.len()) as u64, busy)
    });
    drop(candidates);
    if within.is_none() {
        IDLE_COUNTS[5].store(shown, Relaxed);
        IDLE_COUNTS[6].store(busy, Relaxed);
    }
    if gone.is_empty() {
        return (0, 0);
    }
    changed();
    // Page tables name the windows gone until rewritten: now, so that a
    // heap no list uses for a while does not keep their buffers.
    if within.is_none() {
        crate::memory::rewrite_stale();
    }
    let bytes: u64 = gone.iter().map(|c| c.len).sum();
    IDLE_COUNTS[0].fetch_add(gone.len() as u64, Relaxed);
    IDLE_COUNTS[1].fetch_add(bytes, Relaxed);
    tracing::debug!(windows = gone.len(), bytes, "idle copies of title memory released");
    (gone.len() as u64, bytes)
}

/// Logs the windows alive and what idle release did since the last call
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_idle() {
    let (alive, bytes) = {
        let map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        (map.len(), map.values().map(|s| s.len).sum::<u64>())
    };
    let [gone, gone_bytes, again, again_bytes, missed] = [0, 1, 2, 3, 4].map(|i| IDLE_COUNTS[i].swap(0, Relaxed));
    let [shown, busy] = [5, 6].map(|i| IDLE_COUNTS[i].load(Relaxed));
    let pinned = released().pins();
    let rule = match idle_seconds() {
        Some(s) => format!("released when idle {s} s and {s} frames"),
        None => "never released (VICTORY_IDLE_RELEASE=0)".into(),
    };
    tracing::info!(
        "loose windows of title memory: {alive} alive ({:.0} MB), {rule}; since the last line: released {gone} ({:.0} MB),          made again after a release {again} ({:.0} MB), {missed} of them after title-memory shaders missed them;          idle but kept: {shown} viewed by heap slots, {busy} used by lists not yet run, {pinned} made again after a miss",
        bytes as f64 / 1e6,
        gone_bytes as f64 / 1e6,
        again_bytes as f64 / 1e6,
    );
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
#[track_caller]
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice));
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.write().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    thread_local! {
        /// This thread's copies remember no gaps (as `VICTORY_GAP_MEMO=0`).
        pub(super) static NO_GAP_MEMO: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread's snapshots are joined (as `VICTORY_SNAPSHOTS_APART=0`).
        pub(super) static JOINED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread takes the copies' registries alone for every lookup
        /// (as `VICTORY_COPIES_SHARED=0`).
        pub(super) static LOOKUPS_ALONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread reads marks back dword by dword (as
        /// `VICTORY_MARKS_QUICK=0`).
        pub(crate) static MARKS_SLOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// This thread's submissions copy back range by range (as
        /// `VICTORY_COPY_BACK_BATCH=0`).
        pub(crate) static ONE_BY_ONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// The threads this thread's batches use for every pass, however
        /// few the pieces (0: as outside tests).
        pub(crate) static THREADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Threads look up copies of the same buffers and windows at once, none
    /// of them made before: each range gets one copy, whichever thread made
    /// it, and every thread finds that one from then on, with the
    /// registries shared for lookups and taken alone.
    #[test]
    fn lookups_from_several_threads_find_one_copy_each() {
        let Some((p, w)) = windows(2) else { return };
        let b = crate::watch::tests::pool(4);
        let buffer = |k: usize| b + k * PAGE;
        (0..4).for_each(|k| register(buffer(k), buffer(k) as u64, PAGE as u64));
        let found: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..6)
                .map(|t| {
                    scope.spawn(move || {
                        LOOKUPS_ALONE.with(|alone| alone.set(t % 3 == 2));
                        let mut seen = [0usize; 6];
                        for round in 0..2000u64 {
                            for k in 0..6 {
                                let address = match k {
                                    0..4 => buffer(k) as u64 + 16 * (round % 8),
                                    _ => w + (k as u64 - 4) * WINDOW + 4096 * (round % 8),
                                };
                                let copy = find(address, 64).expect("a copy");
                                let at = Arc::as_ptr(&copy) as usize;
                                assert!(seen[k] == 0 || seen[k] == at, "another copy of range {k}");
                                seen[k] = at;
                            }
                        }
                        seen.to_vec()
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().expect("a lookup thread")).collect()
        });
        assert!(found.iter().all(|seen| seen == &found[0]), "{found:x?}");
        let distinct: std::collections::BTreeSet<usize> = found[0].iter().copied().collect();
        assert_eq!(distinct.len(), 6, "a copy for each range");
        (0..4).for_each(|k| unregister(buffer(k), buffer(k) as u64));
        crate::watch::tests::free(b);
        done(p, 2);
    }

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let bit = victory_shader::emit::VIEW_READS + slot;
        // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
        // holds the submission lock).
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// The same where title memory is not tracked (what Rosetta gets for
    /// all of it): the title's constant under the busy range still reaches
    /// the copy, and copy-back still tells the shader's write from it.
    #[test]
    fn untracked_title_writes_under_a_busy_range_are_copied() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        let len = 2 * PAGE;
        // Private memory without write watching is not the runtime's.
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } as usize;
        assert!(p != 0);
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no host GPU here
        };
        assert!(copy.tracked.is_none(), "plain private memory is not tracked");
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(p, len);
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our allocation.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert_eq!(copy.update(p + PAGE + 16, 16), 16, "the busy range is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        copy.write_host((p + 10) as u64, &[0xaa]);
        assert_eq!(copy.copy_back(p, len), PAGE, "the shader's page goes back");
        // SAFETY: bytes of our allocation.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x55, "the constant is not undone");
        }
        drop(copy);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Four 64 KiB blocks of plain memory with holes: the first committed
    /// whole, 8 KiB of the second, 4 KiB of the third and the last 4 KiB of
    /// the fourth, the rest reserved; what is committed filled with `fill`.
    /// No block is the runtime's memory, so nothing of it is tracked.
    fn holed(fill: u8) -> usize {
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc};
        // SAFETY: a fresh reservation, released by the test.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(p != 0);
        for (at, len) in [(0, PAGE), (PAGE, 8 << 10), (2 * PAGE, 4 << 10), (4 * PAGE - (4 << 10), 4 << 10)] {
            commit(p + at, len, fill);
        }
        p
    }

    /// Commits `len` bytes at `at` (in a reservation of ours), filled with
    /// `fill`.
    fn commit(at: usize, len: usize, fill: u8) {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
        // SAFETY: inside a reservation of the test's, ours to fill.
        unsafe {
            assert!(!VirtualAlloc(at as *const _, len, MEM_COMMIT, PAGE_READWRITE).is_null());
            (at as *mut u8).write_bytes(fill, len);
        }
    }

    /// The whole copy's bytes.
    fn held(copy: &Shadow) -> Vec<u8> {
        let (start, len) = copy.range();
        copy.read_host(start, len as usize / 4).iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// What title memory `[p, p + len)` reads as: its bytes, zeros where
    /// nothing is committed.
    fn read_as(p: usize, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for (i, page) in out.chunks_mut(4096).enumerate() {
            crate::host::read_into((p + i * 4096) as *const u8, page);
        }
        out
    }

    /// The steps of the tests below on a copy of [`holed`] memory, the
    /// copy's bytes after each refresh: the title commits a page in a gap
    /// and fills it, a shader writes the copy where nothing is committed,
    /// the title decommits a page, and commits all that was left.
    fn through_the_holes(mut after: impl FnMut(&Arc<Shadow>, usize, Vec<u8>)) -> bool {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        let p = holed(7);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
            return false; // no host GPU here
        };
        assert!(copy.tracked.is_none(), "plain memory is not tracked");
        let mut step = |copy: &Arc<Shadow>| {
            copy.update(p, len);
            let bytes = held(copy);
            assert_eq!(bytes, read_as(p, len), "the copy holds what title memory reads as");
            after(copy, p, bytes);
        };
        step(&copy);
        step(&copy);
        // The title commits a page where nothing was and fills it: nobody
        // is told (the runtime reports only its page mappings).
        commit(p + PAGE + (8 << 10), 4 << 10, 0x5a);
        step(&copy);
        // A shader writes the copy where title memory has nothing.
        copy.write_host((p + 3 * PAGE + 64) as u64, &[1, 2, 3, 4]);
        step(&copy);
        // SAFETY: a page of our reservation.
        assert_ne!(unsafe { VirtualFree((p + 4096) as *mut _, 4096, MEM_DECOMMIT) }, 0);
        step(&copy);
        commit(p + PAGE, 3 * PAGE, 0x33);
        step(&copy);
        // Part of a range only.
        commit(p + 4096, 4096, 0x44);
        copy.update(p + 100, 3 * 4096);
        assert_eq!(held(&copy)[100..100 + 3 * 4096], read_as(p, len)[100..100 + 3 * 4096]);
        drop(copy);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
        true
    }

    /// Where a copy found nothing committed it does not read again: it asks
    /// whether that is still so. Memory committed in a gap since reaches
    /// the copy at the next refresh (it would not if gaps were kept without
    /// asking), zeros come back where a shader wrote the copy, and a page
    /// decommitted reads as zeros.
    #[test]
    fn gaps_in_a_copys_memory_are_remembered_and_asked_about_again() {
        let fallbacks = || crate::host::FALLBACKS.with(|n| n.get());
        let k = 4 << 10;
        let (mut n, mut before) = (0, fallbacks());
        let ran = through_the_holes(|copy, p, bytes| {
            let gaps = copy.gaps.lock().unwrap().0.clone();
            let took = fallbacks() - std::mem::replace(&mut before, fallbacks());
            match n {
                0 => {
                    assert_eq!(gaps, [(p + PAGE + 2 * k, p + 2 * PAGE), (p + 2 * PAGE + k, p + 4 * PAGE - k)]);
                    assert!(took > 0, "found piece by piece");
                }
                1 => assert_eq!(took, 0, "no read tried where nothing was committed"),
                2 => {
                    assert!(bytes[PAGE + 2 * k..PAGE + 3 * k].iter().all(|&b| b == 0x5a), "the page committed since");
                    assert_eq!(gaps[0], (p + PAGE + 3 * k, p + 2 * PAGE), "what is left of the gap");
                }
                3 => {
                    assert_eq!(bytes[3 * PAGE + 64..3 * PAGE + 68], [0; 4], "zeros again");
                    assert_eq!(took, 0);
                }
                4 => assert_eq!(gaps[0], (p + k, p + 2 * k), "the page decommitted"),
                5 => assert_eq!(gaps, [(p + k, p + 2 * k)], "all else committed"),
                _ => unreachable!(),
            }
            n += 1;
        });
        assert!(!ran || n == 6);
    }

    /// With and without remembering gaps (`VICTORY_GAP_MEMO=0`) the copy
    /// holds the same bytes after every refresh.
    #[test]
    fn remembering_gaps_changes_no_byte_of_a_copy() {
        let run = |off: bool| {
            NO_GAP_MEMO.with(|o| o.set(off));
            let mut all = Vec::new();
            through_the_holes(|copy, _, bytes| {
                assert!(!off || copy.gaps.lock().unwrap().0.is_empty());
                all.push(bytes);
            });
            NO_GAP_MEMO.with(|o| o.set(false));
            all
        };
        let (with, without) = (run(false), run(true));
        assert!(with == without, "the same bytes at each of {} steps", with.len());
    }

    /// A copy whose memory has more gaps than are remembered remembers
    /// none, and is copied as before.
    #[test]
    fn a_copy_with_too_many_gaps_remembers_none() {
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc};
        let pages = 2 * (GAPS + 2);
        let len = pages * 4096;
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(p != 0);
        // Every other page committed.
        (0..pages).step_by(2).for_each(|i| commit(p + i * 4096, 4096, 9));
        register(p, p as u64, len as u64);
        if let Some(copy) = find(p as u64, len as u64) {
            for _ in 0..2 {
                copy.update(p, len);
                assert!(held(&copy) == read_as(p, len));
                let memo = copy.gaps.lock().unwrap();
                assert!(memo.0.is_empty() && memo.1);
            }
        }
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "each byte once where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Ranges each reaching past the last (root views at rising addresses)
    /// and one apart, a title write under them refreshed, shaders' writes
    /// across the ranges' ends: the bytes each copy-back wrote, title
    /// memory and the copy after, and the snapshot's segments and bytes at
    /// its largest. None without a host GPU.
    #[allow(clippy::type_complexity)]
    fn rising_ranges() -> Option<(Vec<usize>, Vec<u8>, Vec<u8>, (usize, usize))> {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return None;
        };
        // SAFETY: (all title reads and writes below) bytes of our pool.
        let title = |at: usize, byte: u8| unsafe { ((p + at) as *mut u8).write_volatile(byte) };
        (0..len).step_by(7).for_each(|at| title(at, (at % 251) as u8));
        copy.update(p, len);
        let mut ranges: Vec<(usize, usize)> = (0..6).map(|k| (1000 + k * 5000, 20_000)).collect();
        ranges.push((3 * PAGE - 100, 200));
        ranges.push((6000, 3000));
        for &(at, n) in &ranges {
            copy.writing(p + at, n, false);
        }
        let taken = {
            let before = copy.before.lock().unwrap();
            (before.0.len(), before.0.iter().map(|(_, b)| b.len()).sum())
        };
        // The title writes under the ranges; a refresh takes it for the
        // copy and the snapshot.
        title(12_000, 0x77);
        copy.update(p + 8192, 8192);
        // Shaders write across the ranges' ends, and beside the title's byte.
        copy.write_host((p + 20_990) as u64, &[0xa1; 20]);
        copy.write_host((p + 25_996) as u64, &[0xa2; 8]);
        copy.write_host((p + 12_001) as u64, &[0xa3]);
        copy.write_host((p + 3 * PAGE - 2) as u64, &[0xa4; 4]);
        // The title writes again while they run.
        title(30_000, 0x78);
        let written: Vec<usize> = ranges.iter().map(|&(at, n)| copy.copy_back(p + at, n)).collect();
        assert!(copy.before.lock().unwrap().0.is_empty(), "the snapshot goes with the last range");
        let memory = read_as(p, len);
        let copied = held(&copy);
        unregister(p, p as u64);
        Some((written, memory, copied, taken))
    }

    /// Snapshots take each declared byte once, as segments of their own,
    /// and what shaders wrote goes back as it does with one segment rebuilt
    /// over the ranges that touch (`VICTORY_SNAPSHOTS_APART=0`): the same
    /// bytes written, title memory and copy.
    #[test]
    fn snapshots_apart_copy_back_what_joined_ones_do() {
        let run = |joined: bool| {
            JOINED.with(|j| j.set(joined));
            let out = rising_ranges();
            JOINED.with(|j| j.set(false));
            out
        };
        let (Some(apart), Some(joined)) = (run(false), run(true)) else {
            return; // no host GPU here
        };
        // 1000 to 46000, and 200 bytes apart.
        assert_eq!(apart.3, (7, 45_200), "each byte once, a segment per range that added some");
        assert_eq!(joined.3, (2, 45_200), "one segment over the ranges that touch");
        assert_eq!(apart.0, joined.0, "bytes written back per range");
        assert!(apart.1 == joined.1, "title memory");
        assert!(apart.2 == joined.2, "the copy");
        // The shaders' bytes and the title's are all in title memory.
        let m = &apart.1;
        assert!(m[20_990..21_010].iter().all(|&b| b == 0xa1) && m[25_996..26_004].iter().all(|&b| b == 0xa2));
        assert_eq!([m[12_000], m[12_001], m[30_000]], [0x77, 0xa3, 0x78]);
        assert_eq!(m[3 * PAGE - 2..3 * PAGE + 2], [0xa4; 4]);
    }

    fn random(seed: &mut u64, n: usize) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % n.max(1) as u64) as usize
    }

    /// Runs `f` with this thread's copy-backs done range by range
    /// (`one_by_one`) or as batches on `threads` threads.
    fn copying_back<R>(one_by_one: bool, threads: usize, f: impl FnOnce() -> R) -> R {
        ONE_BY_ONE.with(|o| o.set(one_by_one));
        THREADS.with(|t| t.set(threads));
        let out = f();
        ONE_BY_ONE.with(|o| o.set(false));
        THREADS.with(|t| t.set(0));
        out
    }

    /// A pool of `pages` pages filled with bytes below 0x80 (what the tests'
    /// shaders write is 0x80 and above, so never the byte the title holds).
    fn patterned(pages: usize) -> usize {
        let p = crate::watch::tests::pool(pages);
        // SAFETY: our pool, which nothing else knows of yet.
        let bytes = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, pages * PAGE) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i * 7 + i / PAGE) as u8 & 0x7f;
        }
        p
    }

    fn bytes_at(p: usize, n: usize) -> Vec<u8> {
        // SAFETY: bytes of the caller's pool.
        (0..n).map(|i| unsafe { ((p + i) as *const u8).read_volatile() }).collect()
    }

    /// Three buffers of a pool, the title and its shaders at work in them,
    /// and the copy-back of the ranges declared: title memory after it,
    /// each copy after its next refresh, the ranges whose writes stayed out
    /// of title memory (offsets in the pool), and where the buffers are.
    /// None without a host GPU.
    #[allow(clippy::type_complexity)]
    fn declared_copy_back(seed: u64) -> Option<(Vec<u8>, Vec<Vec<u8>>, Vec<(usize, usize)>, [(usize, usize); 3])> {
        const PAGES: usize = 14;
        let mut seed = seed;
        let p = patterned(PAGES);
        // Beside a page's start, on one, and ending inside one.
        let buffers = [(0x30, 3 * PAGE + 0x1234), (4 * PAGE, 4 * PAGE), (8 * PAGE + 100, 5 * PAGE)];
        for (k, &(at, len)) in buffers.iter().enumerate() {
            register(p + k, (p + at) as u64, len as u64);
        }
        let copies: Vec<Arc<Shadow>> = buffers.iter().filter_map(|&(at, len)| find((p + at) as u64, len as u64)).collect();
        let finish = |copies: Vec<Arc<Shadow>>| {
            for (k, &(at, _)) in buffers.iter().enumerate() {
                unregister(p + k, (p + at) as u64);
            }
            drop(copies);
            crate::watch::tests::free(p);
        };
        if copies.len() < buffers.len() {
            finish(copies);
            return None; // no host GPU here
        }
        for (copy, &(at, len)) in copies.iter().zip(&buffers) {
            copy.update(p + at, len);
        }
        // The ranges declared: empty, short, a page, several; on a page of
        // title memory or anywhere; overlapping; one in eight marked (no
        // bytes kept), one in seven withheld.
        let mut back: Vec<(Arc<Shadow>, usize, usize, bool)> = Vec::new();
        let mut kept = Vec::new();
        for (copy, &(at, len)) in copies.iter().zip(&buffers) {
            for _ in 0..random(&mut seed, 12) {
                let n = match random(&mut seed, 6) {
                    0 => 0,
                    1 => random(&mut seed, 64),
                    2 => PAGE,
                    _ => random(&mut seed, 2 * PAGE + PAGE / 2),
                }
                .min(len);
                let mut from = random(&mut seed, len - n + 1);
                if random(&mut seed, 2) == 0 {
                    from = ((at + from).next_multiple_of(PAGE) - at).min(len - n);
                }
                let address = p + at + from;
                if back.iter().any(|b| Arc::ptr_eq(&b.0, copy) && (b.1, b.2) == (address, n)) {
                    continue;
                }
                let marked = random(&mut seed, 8) == 0 && copy.marked();
                copy.writing(address, n, marked);
                kept.push(!marked);
                back.push((copy.clone(), address, n, random(&mut seed, 7) != 0));
            }
        }
        // A range overlapping a withheld one of its copy is withheld too
        // (`crate::host::Host::execute`).
        let held_back: Vec<(usize, usize, usize)> =
            back.iter().filter(|b| !b.3).map(|(s, a, n, _)| (Arc::as_ptr(s) as usize, *a, *n)).collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = Arc::as_ptr(&b.0) as usize;
            b.3 = !held_back.iter().any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // The shaders write inside the ranges whose bytes are kept, at
        // their ends among others.
        for ((copy, address, n, _), _) in back.iter().zip(&kept).filter(|(b, kept)| **kept && b.2 > 0) {
            for _ in 0..random(&mut seed, 5) {
                let len = [1, 4, 64, 3000, PAGE + 5][random(&mut seed, 5)].min(*n);
                let at = match random(&mut seed, 4) {
                    0 => 0,
                    1 => n - len,
                    _ => random(&mut seed, n - len + 1),
                };
                let bytes: Vec<u8> = (0..len).map(|_| 0x80 | random(&mut seed, 128) as u8).collect();
                copy.write_host((address + at) as u64, &bytes);
            }
        }
        // The title writes meanwhile, anywhere (guarded pages among them).
        for _ in 0..random(&mut seed, 12) {
            let n = [1, 2, 100][random(&mut seed, 3)];
            let at = random(&mut seed, PAGES * PAGE - n);
            for i in 0..n {
                // SAFETY: a byte of our pool.
                unsafe { ((p + at + i) as *mut u8).write_volatile(random(&mut seed, 128) as u8) };
            }
        }
        copy_back_declared(&back);
        for copy in &copies {
            assert!(copy.pending.lock().unwrap().is_empty() && copy.before.lock().unwrap().0.is_empty(), "nothing left kept");
        }
        let title = bytes_at(p, PAGES * PAGE);
        let after: Vec<Vec<u8>> = copies
            .iter()
            .zip(&buffers)
            .map(|(copy, &(at, len))| {
                copy.update(p + at, len);
                held(copy)
            })
            .collect();
        // Where shaders' bytes may have stayed in the copies alone.
        let apart: Vec<(usize, usize)> =
            back.iter().zip(&kept).filter(|(b, kept)| !b.3 || !**kept).map(|(b, _)| (b.1 - p, b.1 - p + b.2)).collect();
        drop(back);
        finish(copies);
        Some((title, after, apart, buffers))
    }

    /// Whatever ranges a submission's lists declared (overlapping, empty,
    /// marked, withheld, their ends on and beside pages) and whatever the
    /// shaders and the title wrote: copied back as a batch on one thread or
    /// several, title memory ends byte for byte as copied back range by
    /// range (`VICTORY_COPY_BACK_BATCH=0`), and so does every copy after
    /// its next refresh, which holds title memory's bytes wherever the
    /// shaders' went back: no page counts as current that is not.
    #[test]
    fn declared_ranges_at_once_leave_what_one_by_one_leaves() {
        let mut changed = 0;
        for round in 0..12u64 {
            let seed = 0x9e37_79b9_7f4a_7c15 ^ (round + 1).wrapping_mul(0x2545_f491_4f6c_dd1d);
            let Some(wanted) = copying_back(true, 0, || declared_copy_back(seed)) else {
                return; // no host GPU here
            };
            for threads in [1, 4] {
                let done = copying_back(false, threads, || declared_copy_back(seed)).expect("a host GPU");
                assert!(done.0 == wanted.0, "round {round}, {threads} threads: title memory");
                // But where the shaders' bytes stay in the copies alone:
                // how long they stay is `crate::memory`'s to say, by state
                // other tests' submissions share (a submission that missed
                // memory makes the copies take title memory at once).
                let apart = |here: usize| wanted.2.iter().chain(&done.2).any(|&(from, to)| (from..to).contains(&here));
                for (k, (d, w)) in done.1.iter().zip(&wanted.1).enumerate() {
                    let at = wanted.3[k].0;
                    if let Some(i) = (0..d.len()).find(|&i| d[i] != w[i] && !apart(at + i)) {
                        panic!("round {round}, {threads} threads: copy {k} at {:#x}: {:#x} for {:#x}", at + i, d[i], w[i]);
                    }
                }
            }
            for (copy, &(at, _)) in wanted.1.iter().zip(&wanted.3) {
                for (i, &b) in copy.iter().enumerate() {
                    let here = at + i;
                    if !wanted.2.iter().any(|&(from, to)| (from..to).contains(&here)) {
                        assert_eq!(b, wanted.0[here], "round {round}: a copy's byte at {here:#x} is title memory's");
                    }
                }
            }
            changed += usize::from(wanted.0.iter().any(|&b| b >= 0x80));
        }
        assert!(changed > 6, "the shaders' bytes went back in most rounds ({changed})");
    }

    /// Times the copy-back of a 48 MiB buffer's declared ranges (megabyte
    /// ranges at rising addresses, each half over the last, a third of the
    /// pages written by shaders: a long run in one, scattered dwords in the
    /// next) range by range and as a batch on one thread and several.
    #[test]
    #[ignore = "synthetic release benchmark; run with --release --ignored --nocapture"]
    fn benchmark_copy_back_release() {
        const PAGES: usize = 768;
        let p = patterned(PAGES);
        let len = PAGES * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        let mut round = 0u8;
        let mut time = |one_by_one: bool, threads: usize| {
            let mut best = (f64::MAX, 0);
            for _ in 0..6 {
                round = round.wrapping_add(1);
                copy.update(p, len);
                let mut back = Vec::new();
                let mut at = 0x1230;
                while at + (1 << 20) <= len {
                    copy.writing(p + at, 1 << 20, false);
                    back.push((copy.clone(), p + at, 1 << 20, true));
                    at += 1 << 19;
                }
                for page in (1..PAGES - 1).step_by(3) {
                    copy.write_host((p + page * PAGE + 0x400) as u64, &vec![0x80 | round; 20 << 10]);
                    for k in 0..30 {
                        copy.write_host((p + (page + 1) * PAGE + k * 2000 + 16) as u64, &[0x80 | round; 4]);
                    }
                }
                let started = std::time::Instant::now();
                let written = copying_back(one_by_one, threads, || copy_back_declared(&back));
                let ms = started.elapsed().as_secs_f64() * 1e3;
                if ms < best.0 {
                    best = (ms, written);
                }
            }
            best
        };
        let one = time(true, 0);
        println!("range by range: {:.2} ms ({} KiB of pieces written)", one.0, one.1 >> 10);
        for threads in [1, 2, 4, 6] {
            let batch = time(false, threads);
            println!("as a batch on {threads} threads: {:.2} ms ({} KiB of pieces written)", batch.0, batch.1 >> 10);
        }
        // SAFETY: bytes of our pool.
        assert!(unsafe { ((p + PAGE + 0x400) as *const u8).read_volatile() } >= 0x80, "the shaders' bytes went back");
        unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// Times the copy-back of marked pages: 16 copies of 4 MiB, 20
    /// submissions a frame, each with 64 pages marked (4 of every copy: a
    /// long run, scattered dwords, one dword, a long run again), two of them
    /// written by the title meanwhile. Page by page and as a batch on one
    /// thread and several; with diagnostics, where the time goes.
    #[test]
    #[ignore = "synthetic release benchmark; run with --release --ignored --nocapture"]
    fn benchmark_marked_pages_release() {
        const COPIES: usize = 16;
        const EACH: usize = 64;
        let (Some(host), Some(b)) = (crate::host::host(), bitmap(false)) else {
            return; // no host GPU here
        };
        let p = patterned(COPIES * EACH);
        let len = EACH * PAGE;
        let copies: Vec<Arc<Shadow>> = (0..COPIES)
            .map(|k| {
                register(p + k, (p + k * len) as u64, len as u64);
                find((p + k * len) as u64, len as u64).expect("a copy")
            })
            .collect();
        assert!(copies.iter().all(|c| c.marks.is_some()));
        let mut round = 0usize;
        let mut frame = |one_by_one: bool, threads: usize, slow: bool| {
            MARKS_SLOW.with(|s| s.set(slow));
            crate::watch::tests::POOLS_BY_PAGE.with(|s| s.set(slow));
            let mut us = 0;
            for _submission in 0..20 {
                round += 1;
                let mut marks = Vec::new();
                let mut back = Vec::new();
                for (k, copy) in copies.iter().enumerate() {
                    let start = p + k * len;
                    copy.update(start, len);
                    copy.writing(start, len, true);
                    back.push((copy.clone(), start, len, true));
                    let byte = 0x80 | (round as u8 & 0x7f);
                    for i in 0..4 {
                        let page = start + (round * 4 + i * 17) % EACH * PAGE;
                        match i {
                            1 => (0..30).for_each(|j| copy.write_host((page + j * 2000 + 16) as u64, &[byte; 4])),
                            2 => copy.write_host((page + 0x8000) as u64, &[byte; 4]),
                            _ => copy.write_host((page + 0x400) as u64, &vec![byte; 20 << 10]),
                        }
                        marks.push(copy.mark_entry(page as u64).expect("a bit").0);
                        if k < 2 && i == 0 {
                            // SAFETY: a byte of our pool; the handler opens it.
                            unsafe { ((page + 9) as *mut u8).write_volatile(round as u8 & 0x7f) };
                        }
                    }
                }
                host.between_submissions(|| {
                    for &n in &marks {
                        // SAFETY: a dword of the mapped bitmap; no submission runs.
                        unsafe { *(b.data as *mut u32).add(n as usize / 32) |= 1 << (n % 32) };
                    }
                    let started = std::time::Instant::now();
                    copying_back(one_by_one, threads, || copy_back_marks(&[], false));
                    us += started.elapsed().as_micros();
                });
                copy_back_declared(&back);
            }
            MARKS_SLOW.with(|s| s.set(false));
            crate::watch::tests::POOLS_BY_PAGE.with(|s| s.set(false));
            us as f64 / 1e3
        };
        for (what, one_by_one, threads, slow) in [
            ("page by page, marks read dword by dword", true, 0, true),
            ("as batches on 1 thread, marks read dword by dword", false, 1, true),
            ("as batches on 4 threads, marks read dword by dword", false, 4, true),
            ("page by page", true, 0, false),
            ("as batches on 1 thread", false, 1, false),
            ("as batches on 2 threads", false, 2, false),
            ("as batches on 4 threads", false, 4, false),
            ("as batches, threads as in the game", false, 0, false),
        ] {
            frame(one_by_one, threads, slow);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, take};
                for c in [C::MarksScanUs, C::MarksFindUs, C::MarksPrepUs, C::MarkCompareUs, C::MarkOpenUs, C::MarkPutUs]
                    .into_iter()
                    .chain([C::MarkCloseUs, C::MarkAgainUs, C::MarkReadUs, C::MarkWriteUs, C::MarkPieces, C::MarkChanged])
                    .chain([C::SavedPages, C::MarkedKept, C::MarksWords, C::PoolAsks, C::ProtectCalls, C::OpenPages, C::BackSpread])
                {
                    take(c);
                }
            }
            let ms: Vec<f64> = (0..5).map(|_| frame(one_by_one, threads, slow)).collect();
            let best = ms.iter().copied().fold(f64::MAX, f64::min);
            println!("{what}: {best:.2} ms a frame of 1,280 marked pages (of 5: {ms:.2?})");
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, take};
                let ms = |c: C| take(c) as f64 / 5e3;
                let n = |c: C| take(c) as f64 / 5.0;
                println!(
                    "    bitmaps {:.2} ms ({:.0} dwords), finding {:.2}, pieces made {:.2}, compare {:.2}, opening {:.2}, \
                     put {:.2} (threads: reading {:.2}, writing {:.2}), closing and taking {:.2}, again {:.2}; {:.0} pieces, \
                     {:.0} changed, {:.0} saved, {:.0} kept the title's; {:.0} pool askings, {:.0} protect calls, {:.0} pages \
                     opened; {:.0} passes on helpers",
                    ms(C::MarksScanUs),
                    n(C::MarksWords),
                    ms(C::MarksFindUs),
                    ms(C::MarksPrepUs),
                    ms(C::MarkCompareUs),
                    ms(C::MarkOpenUs),
                    ms(C::MarkPutUs),
                    ms(C::MarkReadUs),
                    ms(C::MarkWriteUs),
                    ms(C::MarkCloseUs),
                    ms(C::MarkAgainUs),
                    n(C::MarkPieces),
                    n(C::MarkChanged),
                    n(C::SavedPages),
                    n(C::MarkedKept),
                    n(C::PoolAsks),
                    n(C::ProtectCalls),
                    n(C::OpenPages),
                    n(C::BackSpread),
                );
            }
        }
        // What the steps of a page cost on their own, in a pool no copy
        // tracks: the kernel's read of 64 KiB into plain memory and into a
        // copy; a 20 KiB write into a page just made read-only and writable
        // again (as a guarded page opened for our write is), and again; the
        // two protection changes.
        {
            use crate::back::Title;
            use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
            const N: usize = 512;
            let q = patterned(N);
            let mut plain = vec![0u8; PAGE];
            let source = vec![0x5au8; 20 << 10];
            let us = |started: std::time::Instant| started.elapsed().as_secs_f64() * 1e6 / N as f64;
            let started = std::time::Instant::now();
            (0..N).for_each(|k| assert!(TitleMemory.read(q + k * PAGE, plain.as_mut_ptr(), PAGE)));
            let read_plain = us(started);
            let started = std::time::Instant::now();
            for k in 0..N {
                // SAFETY: a page of the first copy's mapping.
                let page = unsafe { copies[0].data.add(k % EACH * PAGE) };
                assert!(TitleMemory.read(q + k * PAGE, page, PAGE));
            }
            let read_copy = us(started);
            let started = std::time::Instant::now();
            for k in 0..N {
                let mut old = 0;
                // SAFETY: a page of our pool; only its protection changes.
                unsafe {
                    VirtualProtect((q + k * PAGE) as *const _, PAGE, PAGE_READONLY, &mut old);
                    VirtualProtect((q + k * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old);
                }
            }
            let protect = us(started);
            let started = std::time::Instant::now();
            (0..N).for_each(|k| TitleMemory.write(q + k * PAGE + 0x400, source.as_ptr(), source.len()));
            let write_first = us(started);
            let started = std::time::Instant::now();
            (0..N).for_each(|k| TitleMemory.write(q + k * PAGE + 0x400, source.as_ptr(), source.len()));
            let write_again = us(started);
            println!(
                "a page's steps alone, microseconds: 64 KiB read into plain memory {read_plain:.2}, into a copy {read_copy:.2}; \
                 read-only and writable again {protect:.2}; 20 KiB written after that {write_first:.2}, written again {write_again:.2}"
            );
            crate::watch::tests::free(q);
        }
        for (k, copy) in copies.into_iter().enumerate() {
            unregister(p + k, (p + k * len) as u64);
            drop(copy);
        }
        crate::watch::tests::free(p);
    }

    /// Copies whose ranges overlap in title memory (a buffer inside
    /// another) are not done as a batch: the second reads what the first
    /// wrote, in the order declared, as before.
    #[test]
    fn copies_sharing_memory_go_back_one_by_one() {
        let run = |one_by_one: bool| {
            copying_back(one_by_one, 4, || {
                let p = patterned(2);
                register(p, p as u64, 2 * PAGE as u64);
                register(p + 1, (p + 256) as u64, 4096);
                let (Some(outer), Some(inner)) = (find(p as u64, 2 * PAGE as u64), find((p + 256) as u64, 4096)) else {
                    unregister(p, p as u64);
                    unregister(p + 1, (p + 256) as u64);
                    return None; // no host GPU here
                };
                assert!(!Arc::ptr_eq(&outer, &inner));
                outer.update(p, 2 * PAGE);
                inner.update(p + 256, 4096);
                let back =
                    vec![(inner.clone(), p + 256, 4096, true), (outer.clone(), p, 2 * PAGE, true), (inner.clone(), p + 300, 8, true)];
                for (copy, address, len, _) in &back {
                    copy.writing(*address, *len, false);
                }
                // Both write the byte at 300; each one of its own.
                inner.write_host((p + 300) as u64, &[0x91, 0x92]);
                outer.write_host((p + 300) as u64, &[0xa1]);
                outer.write_host((p + PAGE + 5) as u64, &[0xa3]);
                if !one_by_one {
                    assert!(declared_at_once(&back).is_none(), "no batch of pieces sharing memory");
                }
                copy_back_declared(&back);
                let title = bytes_at(p, 2 * PAGE);
                outer.update(p, 2 * PAGE);
                inner.update(p + 256, 4096);
                let copies = (held(&outer), held(&inner));
                unregister(p, p as u64);
                unregister(p + 1, (p + 256) as u64);
                drop((outer, inner, back));
                crate::watch::tests::free(p);
                Some((title, copies))
            })
        };
        let (Some(wanted), Some(done)) = (run(true), run(false)) else {
            return; // no host GPU here
        };
        // The outer copy's byte went over the inner's, and the inner copy's
        // second range, declared last, put its own there again.
        assert_eq!(wanted.0[300..302], [0x91, 0x92], "in the order declared");
        assert_eq!(wanted.0[PAGE + 5], 0xa3);
        assert!(done == wanted);
    }

    /// Marked pages of two copies (whole pages and the parts at a copy's
    /// ends), some written by the title after the copy took them: copied
    /// back as a batch, title memory and the copies end as page by page.
    #[allow(clippy::type_complexity)]
    fn marked_copy_back(seed: u64) -> Option<(Vec<u8>, Vec<Vec<u8>>, Vec<(usize, usize)>, [(usize, usize); 2])> {
        const PAGES: usize = 12;
        let host = crate::host::host()?;
        let b = bitmap(false)?;
        let mut seed = seed;
        let p = patterned(PAGES);
        let buffers = [(0x500, 5 * PAGE), (6 * PAGE, 5 * PAGE + 0x800)];
        for (k, &(at, len)) in buffers.iter().enumerate() {
            register(p + k, (p + at) as u64, len as u64);
        }
        let copies: Vec<Arc<Shadow>> = buffers.iter().filter_map(|&(at, len)| find((p + at) as u64, len as u64)).collect();
        let out = (copies.len() == buffers.len() && copies.iter().all(|c| c.marks.is_some())).then(|| {
            for (copy, &(at, len)) in copies.iter().zip(&buffers) {
                copy.update(p + at, len);
                copy.writing(p + at, len, true);
            }
            // The shaders write through views of both copies; the title
            // writes some pages meanwhile (their bytes from before saved).
            let mut marks = Vec::new();
            let mut unmarked = Vec::new();
            for (copy, &(at, len)) in copies.iter().zip(&buffers) {
                for _ in 0..2 + random(&mut seed, 10) {
                    let n = [1, 4, 700, PAGE][random(&mut seed, 4)];
                    let from = random(&mut seed, len - n + 1);
                    let bytes: Vec<u8> = (0..n).map(|_| 0x80 | random(&mut seed, 128) as u8).collect();
                    copy.write_host((p + at + from) as u64, &bytes);
                    // Marked, but for one write in six (a write no view
                    // marked stays in the copy).
                    if random(&mut seed, 6) != 0 {
                        let (first, last) = ((p + at + from) / PAGE, (p + at + from + n - 1) / PAGE);
                        marks.extend((first..=last).map(|page| copy.mark_entry((page * PAGE).max(p + at) as u64).expect("a bit").0));
                    } else {
                        unmarked.push((at + from, at + from + n));
                    }
                }
            }
            for _ in 0..random(&mut seed, 8) {
                let n = [1, 3, 200][random(&mut seed, 3)];
                let at = random(&mut seed, PAGES * PAGE - n);
                for i in 0..n {
                    // SAFETY: a byte of our pool.
                    unsafe { ((p + at + i) as *mut u8).write_volatile(random(&mut seed, 128) as u8) };
                }
            }
            host.between_submissions(|| {
                for &n in &marks {
                    // SAFETY: a dword of the mapped bitmap; no submission runs.
                    unsafe { *(b.data as *mut u32).add(n as usize / 32) |= 1 << (n % 32) };
                }
                copy_back_marks(&[], false);
            });
            let title = bytes_at(p, PAGES * PAGE);
            let after: Vec<Vec<u8>> = copies
                .iter()
                .zip(&buffers)
                .map(|(copy, &(at, len))| {
                    // Ends what the copies keep for marked pages.
                    copy.copy_back(p + at, len);
                    copy.update(p + at, len);
                    held(copy)
                })
                .collect();
            (title, after, unmarked, buffers)
        });
        for (k, &(at, _)) in buffers.iter().enumerate() {
            unregister(p + k, (p + at) as u64);
        }
        drop(copies);
        crate::watch::tests::free(p);
        out
    }

    #[test]
    fn marked_pages_at_once_leave_what_one_by_one_leaves() {
        let mut changed = 0;
        for round in 0..12u64 {
            let seed = 0x1234_5678_9abc_def1 ^ (round + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let Some(wanted) = copying_back(true, 0, || marked_copy_back(seed)) else {
                return; // no host GPU here, or no mark bits left
            };
            // On one thread and several; the marks read back the quick way
            // (pages opened unread, guarded again unasked) and the slow.
            for (threads, slow) in [(1, false), (4, false), (4, true)] {
                MARKS_SLOW.with(|s| s.set(slow));
                crate::watch::tests::POOLS_BY_PAGE.with(|s| s.set(slow));
                let done = copying_back(false, threads, || marked_copy_back(seed));
                MARKS_SLOW.with(|s| s.set(false));
                crate::watch::tests::POOLS_BY_PAGE.with(|s| s.set(false));
                let done = done.expect("a host GPU");
                assert!(done.0 == wanted.0, "round {round}, {threads} threads: title memory");
                // But where a write no view marked stays in its copy alone,
                // until something makes the copy take the page again
                // (`crate::memory`, by state other tests share).
                let apart = |here: usize| wanted.2.iter().any(|&(from, to)| (from..to).contains(&here));
                for (k, (d, w)) in done.1.iter().zip(&wanted.1).enumerate() {
                    let at = wanted.3[k].0;
                    if let Some(i) = (0..d.len()).find(|&i| d[i] != w[i] && !apart(at + i)) {
                        panic!("round {round}, {threads} threads: copy {k} at {:#x}: {:#x} for {:#x}", at + i, d[i], w[i]);
                    }
                }
            }
            changed += usize::from(wanted.0.iter().any(|&b| b >= 0x80));
        }
        assert!(changed > 6, "the shaders' bytes went back in most rounds ({changed})");
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// A refresh leaves a block the title has just written open
    /// (`crate::watch::rest_ms`) and copies it again when asked again; a
    /// range shaders are about to write is settled first, so that their
    /// bytes are still told from what the title writes meanwhile.
    #[test]
    fn a_resting_block_is_copied_again_and_settled_before_shaders_write_it() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        let tracked = copy.tracked.as_ref().expect("pool memory is tracked");
        assert_eq!(copy.update(p, len), len);
        crate::watch::tests::rest(60_000);
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe { title(40).write_volatile(0x11) };
        assert_eq!(copy.update_as(p, len, Take::Resting), PAGE, "the page written");
        assert_eq!(host(40), 0x11);
        assert!(tracked.rested(p, p + len) && !copy.clean(), "left open and marked");
        // Written again while it rests: no fault, and the next refresh has it.
        unsafe { title(41).write_volatile(0x22) };
        assert_eq!(copy.update_as(p, len, Take::Resting), PAGE, "copied again while it rests");
        assert_eq!(host(41), 0x22);
        unsafe { title(42).write_volatile(0x33) };
        // Shaders are to write the range: taken for good, with that write.
        copy.writing(p, len, true);
        assert!(!tracked.rested(p, p + len) && copy.clean(), "settled");
        assert_eq!(host(42), 0x33);
        // The shaders write a byte of page 0; the title another meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        unsafe { title(50).write_volatile(0x44) };
        let bits = b.data as *mut u32;
        // SAFETY: a dword of the mapped bitmap.
        unsafe { *bits.add(base / 32) |= 1 << (base % 32) };
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte goes back");
        assert_eq!((read(40), read(41), read(42), read(50)), (0x11, 0x22, 0x33, 0x44), "the title's bytes stay");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        crate::watch::tests::rest(0);
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// Memory of our own holding `n` whole 4 MiB windows outside any buffer
    /// object: the allocation, and the first window's start. None without a
    /// host GPU.
    fn windows(n: usize) -> Option<(usize, u64)> {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        host()?;
        let bytes = (n + 1) * WINDOW as usize;
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert!(p != 0);
        Some((p, (p as u64).next_multiple_of(WINDOW)))
    }

    /// The windows of the allocation at `p` (`n` windows) go, with what is
    /// remembered of them, and it is freed.
    fn done(p: usize, n: usize) {
        let within = (p as u64, (p + (n + 1) * WINDOW as usize) as u64);
        released().forget(within.0, within.1);
        release_idle(LATER, 0, Some(within));
        released().forget(within.0, within.1);
        crate::watch::tests::free(p);
    }

    /// Long after the tests' stamps (they are stamped at time 0: no pass
    /// moves the time on in tests).
    const LATER: crate::idle::Stamp = (1 << 40, 1 << 40);

    fn registered(c: &Arc<Shadow>) -> bool {
        all().iter().any(|a| Arc::ptr_eq(a, c))
    }

    /// Whether the copy behind `held` goes once the page tables naming it
    /// are rewritten and the last submission's batch is dropped (on a
    /// thread of its own).
    fn freed(held: &std::sync::Weak<Shadow>) -> bool {
        host().expect("checked").between_submissions(crate::memory::rewrite_stale);
        (0..200).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            held.strong_count() == 0
        })
    }

    /// A window nothing asked for in the idle time (seconds and frames) is
    /// released, and its host buffer goes once nothing holds it; one whose
    /// shaders' writes are not copied back stays. Asked for again, a new
    /// window is made, holding the title's bytes as they are now, and it
    /// counts as made again.
    #[test]
    fn idle_windows_go_and_come_back_with_current_bytes() {
        let Some((p, w)) = windows(1) else { return };
        let within = Some((w, w + WINDOW));
        // SAFETY: (title writes below) bytes of our allocation.
        unsafe { ((w + 16) as *mut u32).write_volatile(1) };
        let copy = find(w + 16, 4).expect("a window");
        assert_eq!(copy.range(), (w, WINDOW), "a loose window");
        copy.update(w as usize, 4096);
        assert_eq!(copy.read_host(w + 16, 1), [1]);
        assert_eq!(release_idle((5_000, 1 << 40), 30, within), (0, 0), "asked for 5 s ago");
        assert_eq!(release_idle((1 << 40, 5), 30, within), (0, 0), "asked for 5 frames ago");
        copy.writing(w as usize, 64, false);
        assert_eq!(release_idle(LATER, 30, within), (0, 0), "shaders' writes not copied back");
        copy.copy_back(w as usize, 64);
        assert!(registered(&copy));
        assert_eq!(release_idle(LATER, 30, within), (1, WINDOW), "idle");
        assert!(!registered(&copy), "no longer found");
        let held = Arc::downgrade(&copy);
        drop(copy);
        assert!(freed(&held), "its memory returned");
        // The title writes meanwhile; the next use makes a new window.
        // SAFETY: bytes of our allocation.
        unsafe { ((w + 16) as *mut u32).write_volatile(2) };
        let again = find(w + 16, 4).expect("a new window");
        assert!(registered(&again));
        assert!(!released().made((w, WINDOW)), "counted as made again");
        again.update(w as usize, 4096);
        assert_eq!(again.read_host(w + 16, 1), [2], "the title's bytes as they are now");
        drop(again);
        done(p, 1);
    }

    /// A window a list recorded and not yet run uses is not released: the
    /// list's submission brings it up to date, and other lookups find the
    /// same copy meanwhile. Once the list has run it goes, and its buffer
    /// with the list's reset.
    #[test]
    fn windows_of_lists_in_flight_stay_until_the_lists_have_run() {
        let Some((p, w)) = windows(1) else { return };
        let host = host().expect("checked");
        let within = Some((w, w + WINDOW));
        let copy = find(w, 64).expect("a window");
        let list = host.new_list().expect("a list");
        copy.refresh(&list, w, 64);
        list.close();
        assert_eq!(release_idle(LATER, 30, within), (0, 0), "a list not yet run uses it");
        assert!(Arc::ptr_eq(&find(w + 32, 4).expect("the window"), &copy), "still the copy found");
        // The title writes, then the list is submitted: its copy has it.
        // SAFETY: bytes of our allocation.
        unsafe { ((w + 8) as *mut u32).write_volatile(7) };
        host.execute(&[&list]);
        assert_eq!(copy.read_host(w + 8, 1), [7], "brought up to date for the list");
        assert_eq!(release_idle(LATER, 30, within), (1, WINDOW), "the list has run");
        assert!(!registered(&copy));
        let held = Arc::downgrade(&copy);
        drop(copy);
        host.between_submissions(crate::memory::rewrite_stale);
        assert!(held.strong_count() > 0, "the list holds it until reset");
        list.reset();
        assert!(freed(&held), "freed with the list's reset");
        // A list reset before it ran lets go too.
        let copy = find(w, 64).expect("a new window");
        copy.refresh(&list, w, 64);
        copy.gpu_writes(&list, w, 64, false);
        assert_eq!(release_idle(LATER, 30, within), (0, 0));
        list.close();
        list.reset();
        assert_eq!(release_idle(LATER, 30, within), (1, WINDOW));
        drop((copy, list));
        done(p, 1);
    }

    /// A window a view heap slot views is not released (shaders indexing
    /// the heap may read it); once the slot views something else, it goes.
    #[test]
    fn windows_heap_slots_view_stay_until_the_slot_changes() {
        let Some((p, w)) = windows(1) else { return };
        let within = Some((w, w + WINDOW));
        let Some(heap) = crate::descriptor::host::HostHeap::new(false, 4) else {
            done(p, 1);
            return;
        };
        let view = crate::descriptor::View::Constant(w + 256, 256);
        heap.viewed.set(0, Some(&view), true);
        let copy = find(w + 256, 256).expect("the window");
        let at = Arc::as_ptr(&copy) as usize;
        assert!(crate::bindless::shown(&[at].into_iter().collect()).contains(&at));
        assert_eq!(release_idle(LATER, 30, within), (0, 0), "a slot views it");
        assert!(registered(&copy));
        heap.viewed.set(0, None, false);
        assert_eq!(release_idle(LATER, 30, within), (1, WINDOW), "no slot views it");
        assert!(!registered(&copy));
        drop(copy);
        done(p, 1);
    }

    /// Shaders finding no copy at a released window's memory make it again
    /// before the next submission, and it is never released again; a miss
    /// where nothing was released makes nothing.
    #[test]
    fn released_windows_shaders_miss_are_made_again_and_kept() {
        let Some((p, w)) = windows(2) else { return };
        let within = Some((w, w + 2 * WINDOW));
        let copy = find(w, 4).expect("a window");
        assert_eq!(release_idle(LATER, 30, within), (1, WINDOW));
        assert!(!registered(&copy));
        // SAFETY: bytes of our allocation.
        unsafe { ((w + 0x1_0000) as *mut u32).write_volatile(9) };
        note_missed(w + 0x1_0000, w + 0x2_0000);
        note_missed(w + WINDOW, w + WINDOW + 0x1_0000);
        // Other tests' misses are made again by whichever test comes first.
        make_missed_again();
        let again = LOOSE.write().unwrap().get(&(w, WINDOW)).cloned().expect("made again");
        assert!(!Arc::ptr_eq(&again, &copy), "a new one");
        assert!(!LOOSE.write().unwrap().contains_key(&(w + WINDOW, WINDOW)), "never released: not made");
        again.update(w as usize + 0x1_0000, 4);
        assert_eq!(again.read_host(w + 0x1_0000, 1), [9], "from title memory");
        assert!(released().pinned((w, WINDOW)));
        assert_eq!(release_idle(LATER, 30, within), (0, 0), "kept");
        assert!(registered(&again));
        drop((copy, again));
        done(p, 2);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }
}
