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
use crate::watch::PAGE;

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
    pending: Mutex<std::collections::HashSet<(usize, usize)>>,
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

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, std::collections::HashSet<usize>);

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
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;

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
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
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
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
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
        if let Some((taken, before)) = segments.iter().find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &before[from - taken..to - taken];
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
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
                    if changed {
                        let _writing = crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
                t.disarm();
            }
        }
        written
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
            }
            // Equal to title memory now: taken, unless it holds the page in part.
            if from == page && to == page + PAGE {
                taken.push(page / PAGE);
            }
        }
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
                let _writing = crate::watch::writing(from, to - from);
                for (at, bytes) in &writes[i..j] {
                    crate::host::write_mapped(*at as *mut u8, bytes.as_ptr(), bytes.len());
                    written += bytes.len();
                }
                i = j;
            }
        }
        if !taken.is_empty() {
            t.taken(&taken);
        }
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
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
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
                    bytes += self.update_free(at, a);
                }
                if e > a.max(at) {
                    bytes += self.update_busy(a.max(at), e);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
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
    fn update_busy(&self, from: usize, to: usize) -> usize {
        let runs = match &self.tracked {
            Some(t) => t.take(from, to),
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
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes: the pages changed since taken, or all of it where the
    /// memory is not tracked.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if let Some(t) = &self.tracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let (runs, unmapped) = t.take_split(from, to);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
            // Cut to the range: next to it shaders may be writing the copy.
            let mut copied: usize = runs.into_iter().map(|(a, b)| self.copy(a, b)).sum();
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
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            let into = if memory && memory_withheld { &mut held } else { &mut marked };
            while x != 0 {
                into.push((w * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
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
            continue;
        }
        match by_copy.iter_mut().find(|(c, _)| Arc::ptr_eq(c, &copy)) {
            Some((_, list)) => list.push(page),
            None => by_copy.push((copy, vec![page])),
        }
    }
    for (copy, mut list) in by_copy {
        list.sort_unstable();
        list.dedup();
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

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
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
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
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

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

#[track_caller]
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
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
        let there = LOOSE.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&(start, len));
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
        .lock()
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
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
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
        let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
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
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
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
            .lock()
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
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
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
        let again = LOOSE.lock().unwrap().get(&(w, WINDOW)).cloned().expect("made again");
        assert!(!Arc::ptr_eq(&again, &copy), "a new one");
        assert!(!LOOSE.lock().unwrap().contains_key(&(w + WINDOW, WINDOW)), "never released: not made");
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
