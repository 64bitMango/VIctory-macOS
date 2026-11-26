//! What shaders wrote into host copies goes back into title memory piece by
//! piece (`crate::shadow`), a submission's pieces at once and on several
//! threads: the helpers kept for it ([`spread`]), what a piece's bytes were
//! before ([`Before`]), and the three passes over the pieces ([`compare`],
//! [`put`], [`again`]). Plain std, so its tests run everywhere; title memory
//! is reached through [`Title`], and telling the change trackers is the
//! caller's (between the passes).
//!
//! A piece is at most 64 KiB of one copy, inside one 64 KiB page of title
//! memory. For each byte of it, with `h` the copy's byte, `o` the byte it
//! held before the shaders ran and `t` title memory's now: where `h != o`
//! the shaders wrote it and title memory takes `h`; elsewhere the copy
//! takes `t` (the title may have written it meanwhile). Only runs of bytes
//! that change title memory are written.
//!
//! What the passes rely on (the caller sees to it, `crate::shadow`):
//! - no two pieces of a batch share a byte of a copy or of title memory
//!   (nor show the same memory at two addresses), so the pieces can be done
//!   in any order and at once, and each is read and written once;
//! - no submission runs, so nothing but the batch writes the copies;
//! - what a piece's bytes were before does not change during the batch;
//! - title memory a piece is written to was opened for our writes before
//!   [`put`] and is given back to its trackers only after it.
//!
//! The title's threads run meanwhile. A piece is read, merged and written by
//! one thread without a pause, as before, and less of it is written (runs,
//! not the whole piece), so a title write is undone only where it lands in
//! a run (bytes the shaders wrote too) or in a gap joined into one, between
//! that read and that write.

// Without the host GPU only its tests use it.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Condvar, Mutex};

use crate::segments::Segment;

/// The unit compared: 64 KiB of title memory, cut at its multiples.
pub(crate) const PIECE: usize = 64 << 10;

/// Runs of a piece are written as one where at most this many bytes that
/// need no write lie between them (a write is a call into the kernel, which
/// costs what copying some kilobytes does).
const JOIN: usize = 64;

/// A piece is written in at most this many runs: with more, those less than
/// a 4 KiB page apart are joined, and if that leaves too many still, one
/// run from its first changed byte to its last is written (the whole piece
/// was, before).
const RUNS: usize = 8;

/// The helper threads: kept, since a submission's batch is short (making
/// threads for each cost more than it saved) and there are some twenty
/// submissions a frame.
struct Helpers {
    state: Mutex<State>,
    /// Signalled when there is a job.
    work: Condvar,
    /// Signalled when no helper is inside the job any more.
    done: Condvar,
    /// One job at a time.
    turn: Mutex<()>,
}

struct State {
    job: Option<Job>,
    /// Counts jobs: a helper joins each once.
    job_number: u64,
    /// Helpers the job still wants, and those inside it.
    wanted: usize,
    inside: usize,
    /// Helper threads made.
    made: usize,
    /// Whether a helper panicked inside the job.
    panicked: bool,
}

/// The work of a job: called once by each thread that joins it.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn() + Sync));

// SAFETY: the work is `Sync`, and the thread that set the job keeps it alive
// until no helper is inside it (`Helpers::run`).
unsafe impl Send for Job {}

static HELPERS: Helpers = Helpers {
    state: Mutex::new(State {
        job: None,
        job_number: 0,
        wanted: 0,
        inside: 0,
        made: 0,
        panicked: false,
    }),
    work: Condvar::new(),
    done: Condvar::new(),
    turn: Mutex::new(()),
};

impl Helpers {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `work` here and on up to `helpers` helper threads at once, and
    /// returns when all of them have left it.
    fn run(&'static self, helpers: usize, work: &(dyn Fn() + Sync)) {
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: only the lifetime changes. Helpers call the work between
        // joining the job and leaving it, both under the state lock, and
        // this function does not return before the job is taken away and
        // no helper is inside it.
        let job = Job(unsafe {
            std::mem::transmute::<*const (dyn Fn() + Sync + '_), *const (dyn Fn() + Sync + 'static)>(work)
        });
        {
            let mut s = self.state();
            while s.made < helpers {
                let made = std::thread::Builder::new()
                    .name("victory copy-back".into())
                    .spawn(move || self.help());
                if made.is_err() {
                    break;
                }
                s.made += 1;
            }
            s.job = Some(job);
            s.job_number += 1;
            s.wanted = helpers.min(s.made);
            s.panicked = false;
        }
        self.work.notify_all();
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        let mut s = self.state();
        s.job = None;
        s.wanted = 0;
        while s.inside > 0 {
            s = self.done.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        let panicked = s.panicked;
        drop(s);
        if let Err(panic) = mine {
            std::panic::resume_unwind(panic);
        }
        assert!(!panicked, "a copy-back helper thread panicked");
    }

    /// A helper thread: joins each job that wants it, once.
    fn help(&self) {
        let mut joined = 0;
        loop {
            let job = {
                let mut s = self.state();
                loop {
                    if let Some(job) = s.job
                        && s.job_number != joined
                        && s.wanted > 0
                    {
                        s.wanted -= 1;
                        s.inside += 1;
                        joined = s.job_number;
                        break job;
                    }
                    s = self.work.wait(s).unwrap_or_else(|e| e.into_inner());
                }
            };
            // SAFETY: the job's thread waits until we have left it.
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job.0)() }));
            let mut s = self.state();
            s.inside -= 1;
            s.panicked |= ran.is_err();
            if s.inside == 0 {
                self.done.notify_all();
            }
        }
    }
}

/// Runs `f(i)` for each `i` below `n`, each once: on this thread and on up
/// to `threads - 1` helper threads kept for it (`crate::texture::
/// in_parallel` makes its threads anew each time). Returns when all are
/// done.
pub(crate) fn spread(n: usize, threads: usize, f: impl Fn(usize) + Sync) {
    let threads = threads.min(n);
    if threads < 2 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    HELPERS.run(threads - 1, &|| {
        loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
    });
}

/// How many threads a batch may use, this one among them: a quarter of the
/// CPUs, four at most (as `crate::texture::in_parallel`: the title's own
/// threads need the rest), or `VICTORY_COPY_BACK_THREADS` (1: all of it on
/// the submitting thread).
pub(crate) fn threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        let asked = std::env::var("VICTORY_COPY_BACK_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        asked.map_or((cpus / 4).clamp(1, 4), |n| n.clamp(1, 16))
    })
}

/// Title memory, as the passes reach it.
pub(crate) trait Title: Sync {
    /// Copies `n` bytes of title memory at `at` to `to`: false if a page of
    /// it cannot be read (some bytes may have been copied).
    fn read(&self, at: usize, to: *mut u8, n: usize) -> bool;
    /// Copies `n` bytes at `from` into title memory at `at`, leaving pages
    /// that cannot be written alone.
    fn write(&self, at: usize, from: *const u8, n: usize);
}

/// What a piece's bytes were before the submission's shaders ran.
pub(crate) enum Before<'a> {
    /// Kept by the copy's snapshot: its segments, and where the piece
    /// starts in the copy (the segments hold all of it).
    Kept(&'a [Segment], usize),
    /// Kept when the title first wrote the piece's page since the copy took
    /// it: those bytes, and where the piece starts in them.
    Saved(Box<[u8]>, usize),
    /// What title memory holds: the title has not written the piece's page
    /// since the copy took it.
    Title,
}

/// A piece of a batch.
pub(crate) struct Piece<'a> {
    /// Where it is in title memory, in its copy's mapping, and its length.
    pub(crate) at: usize,
    pub(crate) host: *mut u8,
    pub(crate) n: usize,
    pub(crate) before: Before<'a>,
    /// Whether it is a whole page of title memory (which its copy may take
    /// as current once it is made equal to title memory).
    pub(crate) whole: bool,
    /// Which of the caller's copies it is of.
    pub(crate) copy: u32,
    state: AtomicU8,
    /// Bytes written into title memory, and in how many writes.
    written: AtomicU32,
    writes: AtomicU32,
}

// SAFETY: `host` is the piece's own bytes of a copy's mapping, which only
// the thread doing the piece touches (no two pieces share a byte).
unsafe impl Send for Piece<'_> {}
// SAFETY: as above; the rest is shared plain data and atomics.
unsafe impl Sync for Piece<'_> {}

/// The shaders changed the piece: title memory is to be opened for it, then
/// read, merged and written ([`put`]).
const PUT: u8 = 1;
/// The shaders changed nothing of it, and the copy takes title memory's
/// bytes ([`put`], with no write).
const SYNC: u8 = 2;
/// The piece's bytes in the copy are title memory's.
const SYNCED: u8 = 4;
/// Something of it was written into title memory.
const WROTE: u8 = 8;

impl<'a> Piece<'a> {
    pub(crate) fn new(at: usize, host: *mut u8, n: usize, before: Before<'a>, whole: bool, copy: u32) -> Piece<'a> {
        debug_assert!(n > 0 && at / PIECE == (at + n - 1) / PIECE);
        Piece {
            at,
            host,
            n,
            before,
            whole,
            copy,
            state: AtomicU8::new(0),
            written: AtomicU32::new(0),
            writes: AtomicU32::new(0),
        }
    }

    fn is(&self, flag: u8) -> bool {
        self.state.load(Relaxed) & flag != 0
    }

    /// After [`compare`]: whether title memory is to be opened for it.
    pub(crate) fn opens(&self) -> bool {
        self.is(PUT)
    }

    /// After [`put`]: whether its bytes in the copy are title memory's.
    pub(crate) fn synced(&self) -> bool {
        self.is(SYNCED)
    }

    /// After [`put`]: whether something of it was written into title memory.
    pub(crate) fn wrote(&self) -> bool {
        self.is(WROTE)
    }

    /// After [`put`]: the bytes written into title memory, and the writes.
    pub(crate) fn written(&self) -> (u64, u64) {
        (u64::from(self.written.load(Relaxed)), u64::from(self.writes.load(Relaxed)))
    }

    /// The copy's bytes of the piece.
    ///
    /// # Safety
    /// Only the thread doing the piece may hold the slice.
    #[allow(clippy::mut_from_ref)]
    unsafe fn host(&self) -> &mut [u8] {
        // SAFETY: the piece's own bytes of a live mapping (the caller's).
        unsafe { std::slice::from_raw_parts_mut(self.host, self.n) }
    }
}

/// Time the threads of a pass spent, in nanoseconds (summed over them, so
/// it can exceed the time the pass took): reading title memory and merging,
/// and writing it.
#[derive(Default)]
pub(crate) struct Spent {
    pub(crate) reading_ns: AtomicU64,
    pub(crate) writing_ns: AtomicU64,
}

/// A thread's working memory for a piece: title memory as read, a piece's
/// bytes from before gathered from several segments, its runs.
type Working = (Vec<u8>, Vec<u8>, Vec<(u32, u32)>);

std::thread_local! {
    static WORKING: std::cell::RefCell<Working> = const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

/// The first pass: finds the pieces the shaders changed, by comparing the
/// copy with what it held before. Nothing is written. For [`Before::Title`]
/// pieces what was before is title memory itself: they are read for it, or
/// (`unread`) taken for changed without a look, since [`put`] reads them
/// anyway and a page a shader marked as written is nearly always changed
/// (the read here was a fifth of a marked page's time).
pub(crate) fn compare(pieces: &[Piece], title: &impl Title, threads: usize, unread: bool) {
    // Nothing to look at: no pass.
    if unread && pieces.iter().all(|p| matches!(p.before, Before::Title)) {
        pieces.iter().for_each(|p| p.state.store(PUT, Relaxed));
        return;
    }
    spread(pieces.len(), threads, |i| {
        let piece = &pieces[i];
        // SAFETY: this thread does the piece.
        let host = unsafe { piece.host() };
        WORKING.with_borrow_mut(|(read, gathered, _)| {
            let state = match &piece.before {
                Before::Kept(segments, from) => {
                    let old = crate::segments::held(segments, *from, from + piece.n, gathered);
                    if *host != *old { PUT } else { 0 }
                }
                Before::Saved(bytes, from) => {
                    if *host != bytes[*from..from + piece.n] { PUT } else { SYNC }
                }
                Before::Title if unread => PUT,
                Before::Title => {
                    read.resize(PIECE, 0);
                    if !title.read(piece.at, read.as_mut_ptr(), piece.n) {
                        0
                    } else if *host != read[..piece.n] {
                        PUT
                    } else {
                        SYNCED
                    }
                }
            };
            piece.state.store(state, Relaxed);
        });
    });
}

/// The second pass, over the pieces [`compare`] found changed (title memory
/// open for them, [`Piece::opens`]) or to be taken from title memory: each
/// is read from title memory, merged, and its runs written, by one thread
/// without a pause. A piece that cannot be read is left alone. With
/// `spent`, the threads' time is added to it.
pub(crate) fn put(pieces: &[Piece], title: &impl Title, threads: usize, spent: Option<&Spent>) {
    let live: Vec<usize> = (0..pieces.len()).filter(|&i| pieces[i].is(PUT | SYNC)).collect();
    spread(live.len(), threads, |k| {
        let piece = &pieces[live[k]];
        let started = spent.map(|_| std::time::Instant::now());
        // SAFETY: this thread does the piece.
        let host = unsafe { piece.host() };
        WORKING.with_borrow_mut(|(read, gathered, runs)| {
            read.resize(PIECE, 0);
            if !title.read(piece.at, read.as_mut_ptr(), piece.n) {
                return;
            }
            let read = &read[..piece.n];
            if piece.is(SYNC) {
                host.copy_from_slice(read);
                piece.state.fetch_or(SYNCED, Relaxed);
                return;
            }
            let old = match &piece.before {
                Before::Kept(segments, from) => crate::segments::held(segments, *from, from + piece.n, gathered),
                Before::Saved(bytes, from) => &bytes[*from..from + piece.n],
                Before::Title => read,
            };
            runs.clear();
            merge(host, old, read, runs);
            fewer(runs);
            let merged = started.map(|_| std::time::Instant::now());
            let mut written = 0;
            for &(from, n) in runs.iter() {
                // The copy holds the merged bytes: title memory's between
                // the shaders'.
                title.write(piece.at + from as usize, host[from as usize..].as_ptr(), n as usize);
                written += n;
            }
            piece.written.store(written, Relaxed);
            piece.writes.store(runs.len() as u32, Relaxed);
            piece.state.fetch_or(if runs.is_empty() { SYNCED } else { SYNCED | WROTE }, Relaxed);
            if let (Some(spent), Some(started), Some(merged)) = (spent, started, merged) {
                spent.reading_ns.fetch_add((merged - started).as_nanos() as u64, Relaxed);
                spent.writing_ns.fetch_add(merged.elapsed().as_nanos() as u64, Relaxed);
            }
        });
    });
}

/// The third pass, once the copies' trackers have taken the whole pages
/// made equal to title memory (guarded them: the title's writes from then
/// on are seen): those pages are read into the copies again. A title write
/// that landed in such a page between its read in [`put`] and its guard is
/// in the copy now; without this pass it stayed out of the copy until the
/// title next wrote the page.
pub(crate) fn again(pieces: &[Piece], title: &impl Title, threads: usize) {
    let taken: Vec<usize> = (0..pieces.len()).filter(|&i| pieces[i].whole && pieces[i].synced()).collect();
    spread(taken.len(), threads, |k| {
        let piece = &pieces[taken[k]];
        // A page that cannot be read any more keeps what the copy holds.
        title.read(piece.at, piece.host, piece.n);
    });
}

/// Merges a piece: `host` is the copy's bytes, `old` what they were before
/// the shaders ran, `title` title memory's now (the same slice as `old`
/// where title memory is what the copy held before). The copy takes title
/// memory's bytes where the shaders wrote nothing, and `runs` lists (offset,
/// bytes) what title memory is to take from the copy after this: every byte
/// the shaders wrote that title memory does not hold, runs at most [`JOIN`]
/// bytes apart as one (the bytes between are title memory's own).
///
/// The copy is written only where it changes: its memory is the GPU's
/// buffer, and writing all of a piece into it cost more than reading the
/// piece from title memory did (8 of 11 us a page, Wine on an M5 Pro).
pub(crate) fn merge(host: &mut [u8], old: &[u8], title: &[u8], runs: &mut Vec<(u32, u32)>) {
    /// Bytes compared at once where nothing changed, and where something
    /// did.
    const MANY: usize = 1024;
    const FEW: usize = 64;
    let n = host.len();
    assert!(old.len() == n && title.len() == n);
    // Title memory is what the copy held before: it wrote nothing.
    let before = std::ptr::eq(old, title);
    let mut note = |from: usize, to: usize| match runs.last_mut() {
        Some((at, len)) if (*at + *len) as usize + JOIN >= from => *len = to as u32 - *at,
        _ => runs.push((from as u32, (to - from) as u32)),
    };
    let mut i = 0;
    while i < n {
        let many = (i + MANY).min(n);
        if host[i..many] == old[i..many] {
            // The shaders wrote nothing here: title memory's bytes.
            if !before && host[i..many] != title[i..many] {
                host[i..many].copy_from_slice(&title[i..many]);
            }
            i = many;
            continue;
        }
        while i < many {
            let end = (i + FEW).min(many);
            let (h, o, t) = (&host[i..end], &old[i..end], &title[i..end]);
            if h == o {
                if !before && h != t {
                    host[i..end].copy_from_slice(t);
                }
            } else if before || t == o {
                // The title wrote nothing here: what differs is the
                // shaders', from their first byte to their last, and the
                // copy holds title memory's bytes beside them already.
                let first = h.iter().zip(o).position(|(h, o)| h != o).unwrap_or(0);
                let last = h.iter().zip(o).rposition(|(h, o)| h != o).unwrap_or(0);
                note(i + first, i + last + 1);
            } else {
                // Both wrote here: byte by byte.
                for k in i..end {
                    if host[k] == old[k] {
                        host[k] = title[k];
                    } else if host[k] != title[k] {
                        note(k, k + 1);
                    }
                }
            }
            i = end;
        }
    }
}

/// Joins a piece's runs until there are at most [`RUNS`].
fn fewer(runs: &mut Vec<(u32, u32)>) {
    if runs.len() <= RUNS {
        return;
    }
    let mut joined: Vec<(u32, u32)> = Vec::with_capacity(RUNS + 1);
    for &(at, len) in runs.iter() {
        match joined.last_mut() {
            Some((a, n)) if *a + *n + 4096 >= at => *n = at + len - *a,
            _ => joined.push((at, len)),
        }
    }
    if joined.len() > RUNS {
        let (first, last) = (joined[0], joined[joined.len() - 1]);
        joined = vec![(first.0, last.0 + last.1 - first.0)];
    }
    *runs = joined;
}

/// Takes the bits set in `bits` (dwords of a bitmap, bit 0 of the first
/// its bit 0): each is cleared and its number added to `out`, in order.
/// Dwords that are 0 are passed sixteen at a time and not written (a
/// submission marks a few dwords of thousands).
pub(crate) fn take_bits(bits: &mut [u32], out: &mut Vec<u32>) {
    for (k, some) in bits.chunks_mut(16).enumerate() {
        if some.iter().fold(0, |any, &word| any | word) == 0 {
            continue;
        }
        for (i, word) in some.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                out.push(((k * 16 + i) * 32) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
}

/// The bits set in `map`, or in `other` too, that `seen` does not hold yet:
/// their numbers are added to `out`, in order, and `seen` takes them. All
/// three are as long. Dwords with nothing new are passed sixteen at a time.
pub(crate) fn new_bits(map: &[u32], other: Option<&[u32]>, seen: &mut [u32], out: &mut Vec<u32>) {
    assert!(map.len() == seen.len() && other.is_none_or(|o| o.len() == seen.len()));
    let mut at = 0;
    while at < seen.len() {
        let end = (at + 16).min(seen.len());
        let (a, s) = (&map[at..end], &mut seen[at..end]);
        let b = other.map(|o| &o[at..end]);
        let new = match b {
            Some(b) => a.iter().zip(b).zip(s.iter()).fold(0, |any, ((&a, &b), &s)| any | ((a | b) & !s)),
            None => a.iter().zip(s.iter()).fold(0, |any, (&a, &s)| any | (a & !s)),
        };
        if new != 0 {
            for (i, s) in s.iter_mut().enumerate() {
                let mut fresh = (a[i] | b.map_or(0, |b| b[i])) & !*s;
                *s |= fresh;
                while fresh != 0 {
                    out.push(((at + i) * 32) as u32 + fresh.trailing_zeros());
                    fresh &= fresh - 1;
                }
            }
        }
        at = end;
    }
}

/// Sorts `ranges` (from, to) and joins those that overlap or touch; empty
/// ones go.
pub(crate) fn union(ranges: &mut Vec<(usize, usize)>) {
    ranges.retain(|&(from, to)| from < to);
    ranges.sort_unstable();
    let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match joined.last_mut() {
            Some((_, end)) if *end >= from => *end = (*end).max(to),
            _ => joined.push((from, to)),
        }
    }
    *ranges = joined;
}

/// Calls `f(from, to)` with the pieces of `[from, to)` of a copy whose
/// first byte is at title address `start`, cut at multiples of [`PIECE`] of
/// title memory.
pub(crate) fn pieces(start: usize, from: usize, to: usize, mut f: impl FnMut(usize, usize)) {
    let mut at = from;
    while at < to {
        let n = (PIECE - (start + at) % PIECE).min(to - at);
        f(at, at + n);
        at += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::UnsafeCell;

    fn random(seed: &mut u64, n: usize) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % n.max(1) as u64) as usize
    }

    /// Every index is done once, whatever the threads, job after job, and
    /// from several submitting threads at once.
    #[test]
    fn spread_does_each_index_once() {
        std::thread::scope(|scope| {
            for t in 0..3 {
                scope.spawn(move || {
                    for round in 0..300 {
                        let n = (round * 7 + t) % 97;
                        let done: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                        spread(n, 1 + round % 5, |i| {
                            done[i].fetch_add(1, Relaxed);
                        });
                        assert!(done.iter().all(|d| d.load(Relaxed) == 1), "round {round}, {n} items");
                    }
                });
            }
        });
    }

    /// A panic in the work reaches the thread that spread it, after every
    /// helper has left the job; the helpers go on with the next.
    #[test]
    fn a_panic_in_spread_work_is_passed_on() {
        let caught = std::panic::catch_unwind(|| {
            spread(64, 4, |i| assert!(i != 13, "the thirteenth"));
        });
        assert!(caught.is_err());
        let done = AtomicU32::new(0);
        spread(64, 4, |_| {
            done.fetch_add(1, Relaxed);
        });
        assert_eq!(done.load(Relaxed), 64);
    }

    /// The merge as it was written byte by byte (`Shadow::copy_back_or_not`):
    /// title memory with the shaders' bytes put in, the copy taking title
    /// memory's in turn; whether title memory changed.
    fn merge_by_bytes(host: &mut [u8], old: &[u8], title: &mut [u8]) -> bool {
        let mut changed = false;
        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
            if *h != *o && *t != *h {
                *t = *h;
                changed = true;
            }
            *h = *t;
        }
        changed
    }

    /// Whatever the shaders and the title wrote, at whatever length and
    /// alignment: the merge leaves the copy as the byte-by-byte one does,
    /// and title memory with its runs written is the byte-by-byte one's;
    /// no run holds only bytes that need no write at its ends.
    #[test]
    fn merged_runs_give_what_the_byte_by_byte_merge_gives() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..4000 {
            let n = match round % 4 {
                0 => 1 + random(&mut seed, 40),
                1 => 1 + random(&mut seed, 700),
                2 => PIECE,
                _ => 1 + random(&mut seed, PIECE),
            };
            let old: Vec<u8> = (0..n).map(|i| (i * 13 + round) as u8).collect();
            let (mut host, mut title) = (old.clone(), old.clone());
            // The shaders' writes: single bytes, dwords, long runs, some at
            // the piece's ends, some of the value held already.
            for _ in 0..random(&mut seed, 12) {
                let len = [1, 4, 4, 16, 300, 5000][random(&mut seed, 6)].min(n);
                let at = match random(&mut seed, 4) {
                    0 => 0,
                    1 => n - len,
                    _ => random(&mut seed, n - len + 1),
                };
                for b in &mut host[at..at + len] {
                    *b = if random(&mut seed, 8) == 0 { *b } else { random(&mut seed, 256) as u8 };
                }
            }
            // The title's, some over the shaders' and some of the same value.
            for _ in 0..random(&mut seed, 6) {
                let len = [1, 3, 8, 200][random(&mut seed, 4)].min(n);
                let at = random(&mut seed, n - len + 1);
                for (k, b) in title[at..at + len].iter_mut().enumerate() {
                    *b = if random(&mut seed, 4) == 0 { host[at + k] } else { random(&mut seed, 256) as u8 };
                }
            }
            let (mut host_wanted, mut title_wanted) = (host.clone(), title.clone());
            let changed = merge_by_bytes(&mut host_wanted, &old, &mut title_wanted);
            let mut runs = Vec::new();
            let before = title.clone();
            merge(&mut host, &old, &before, &mut runs);
            assert!(host == host_wanted, "round {round}: the copy");
            assert_eq!(!runs.is_empty(), changed, "round {round}");
            for joined in [false, true] {
                let mut runs = runs.clone();
                if joined {
                    fewer(&mut runs);
                    assert!(runs.len() <= RUNS);
                }
                let mut title = before.clone();
                let mut end = 0;
                for &(at, len) in &runs {
                    let (at, len) = (at as usize, len as usize);
                    assert!(len > 0 && at >= end && at + len <= n, "round {round}: runs in order, inside");
                    assert!(joined || end == 0 || at > end + JOIN, "round {round}: runs closer are one");
                    end = at + len;
                    let needs = |i: usize| before[i] != title_wanted[i];
                    assert!(needs(at) && needs(end - 1), "round {round}: a run starts and ends on a byte to write");
                    title[at..end].copy_from_slice(&host[at..end]);
                }
                assert!(title == title_wanted, "round {round}: title memory");
            }
        }
    }

    /// Title memory of a test: bytes from address `BASE` on, with pages
    /// that cannot be read or written.
    struct Memory {
        bytes: UnsafeCell<Vec<u8>>,
        unmapped: Vec<usize>,
    }
    const BASE: usize = 0x4000_0000;

    // SAFETY: the tests' pieces share no byte of it.
    unsafe impl Sync for Memory {}

    impl Memory {
        fn mapped(&self, at: usize, n: usize) -> bool {
            !self.unmapped.iter().any(|&page| page * PIECE < at - BASE + n && at - BASE < (page + 1) * PIECE)
        }
        fn at(&self, at: usize) -> *mut u8 {
            // SAFETY: inside the bytes (the tests' addresses are).
            unsafe { (*self.bytes.get()).as_mut_ptr().add(at - BASE) }
        }
    }

    impl Title for Memory {
        fn read(&self, at: usize, to: *mut u8, n: usize) -> bool {
            if !self.mapped(at, n) {
                return false;
            }
            // SAFETY: inside the bytes; the destination is the caller's.
            unsafe { std::ptr::copy_nonoverlapping(self.at(at), to, n) };
            true
        }
        fn write(&self, at: usize, from: *const u8, n: usize) {
            if self.mapped(at, n) {
                // SAFETY: as above.
                unsafe { std::ptr::copy_nonoverlapping(from, self.at(at), n) };
            }
        }
    }

    /// A copy of a test: where its title memory starts (an offset from
    /// `BASE`), its bytes, its snapshot, and the ranges declared (offsets in
    /// the copy; some empty, some not kept).
    struct Copy {
        start: usize,
        host: Vec<u8>,
        segments: Vec<Segment>,
        ranges: Vec<(usize, usize)>,
    }

    /// Copy-back as `Shadow::copy_back_or_not` did it: range by range, by
    /// 64 KiB piece, a whole piece once, the whole piece written.
    fn one_by_one(copies: &mut [Copy], title: &Memory) {
        for copy in copies.iter_mut() {
            let mut compared = std::collections::HashSet::new();
            let mut gathered = Vec::new();
            for &(from, to) in &copy.ranges {
                if from >= to || !crate::segments::covered(&copy.segments, from, to) {
                    continue;
                }
                pieces(BASE + copy.start, from, to, |at, end| {
                    let n = end - at;
                    if n == PIECE && !compared.insert(at) {
                        return;
                    }
                    let old = crate::segments::held(&copy.segments, at, end, &mut gathered);
                    let host = &mut copy.host[at..end];
                    let mut read = vec![0u8; n];
                    let address = BASE + copy.start + at;
                    if *host != *old && title.read(address, read.as_mut_ptr(), n) && merge_by_bytes(host, old, &mut read) {
                        title.write(address, read.as_ptr(), n);
                    }
                });
            }
        }
    }

    /// The same as a batch: the union of each copy's ranges, in pieces, on
    /// `threads` threads.
    fn at_once(copies: &mut [Copy], title: &Memory, threads: usize) {
        let mut all = Vec::new();
        for (k, copy) in copies.iter_mut().enumerate() {
            let mut kept: Vec<(usize, usize)> = copy
                .ranges
                .iter()
                .copied()
                .filter(|&(from, to)| from < to && crate::segments::covered(&copy.segments, from, to))
                .collect();
            union(&mut kept);
            let host = copy.host.as_mut_ptr();
            for (from, to) in kept {
                pieces(BASE + copy.start, from, to, |at, end| {
                    // SAFETY: inside the copy's bytes.
                    let bytes = unsafe { host.add(at) };
                    let before = Before::Kept(&copy.segments, at);
                    all.push(Piece::new(BASE + copy.start + at, bytes, end - at, before, end - at == PIECE, k as u32));
                });
            }
        }
        compare(&all, title, threads, false);
        put(&all, title, threads, None);
        again(&all, title, threads);
    }

    /// Copies with random declared ranges (overlapping, touching, empty,
    /// some not kept, their ends on and beside piece ends), random shader
    /// and title writes (runs at piece edges among them) and unmapped pages:
    /// done at once on one thread or several, title memory and the copies
    /// end byte for byte as done range by range on one.
    #[test]
    fn a_batch_on_threads_leaves_what_ranges_one_by_one_leave() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        const PAGES: usize = 24;
        for round in 0..60 {
            let mut memory: Vec<u8> = (0..PAGES * PIECE).map(|i| (i * 7 + round) as u8).collect();
            // Copies side by side, not at page ends.
            let mut copies = Vec::new();
            let mut at = random(&mut seed, 3 * PIECE);
            while at + PIECE < PAGES * PIECE && copies.len() < 4 {
                let len = (1 + random(&mut seed, 6 * PIECE)).min(PAGES * PIECE - at);
                let host = memory[at..at + len].to_vec();
                let mut copy = Copy { start: at, host, segments: Vec::new(), ranges: Vec::new() };
                for _ in 0..random(&mut seed, 14) {
                    let n = match random(&mut seed, 6) {
                        0 => 0,
                        1 => random(&mut seed, 64),
                        2 => PIECE,
                        _ => random(&mut seed, 3 * PIECE),
                    }
                    .min(len);
                    // On a page of title memory, or anywhere.
                    let mut from = random(&mut seed, len - n + 1);
                    if random(&mut seed, 2) == 0 {
                        from = (from + at).next_multiple_of(PIECE).saturating_sub(at).min(len - n);
                    }
                    copy.ranges.push((from, from + n));
                    // Kept, but for one in eight (a marked range keeps none).
                    if n > 0 && random(&mut seed, 8) != 0 {
                        let host = &copy.host;
                        crate::segments::cover_apart(&mut copy.segments, from, from + n, |a, b| &host[a..b], Vec::with_capacity);
                    }
                }
                // The shaders write the copy inside its ranges (and at their
                // very ends), the title writes its memory anywhere.
                for &(from, to) in copy.ranges.clone().iter().filter(|r| r.0 < r.1) {
                    for _ in 0..random(&mut seed, 5) {
                        let n = [1, 4, 64, 3000, PIECE + 5][random(&mut seed, 5)].min(to - from);
                        let a = match random(&mut seed, 4) {
                            0 => from,
                            1 => to - n,
                            _ => from + random(&mut seed, to - from - n + 1),
                        };
                        for b in &mut copy.host[a..a + n] {
                            *b = random(&mut seed, 256) as u8;
                        }
                    }
                }
                for _ in 0..random(&mut seed, 8) {
                    let n = [1, 2, 100][random(&mut seed, 3)].min(len);
                    let a = at + random(&mut seed, len - n + 1);
                    for b in &mut memory[a..a + n] {
                        *b = random(&mut seed, 256) as u8;
                    }
                }
                copies.push(copy);
                at += len + random(&mut seed, 2 * PIECE);
            }
            let unmapped: Vec<usize> = (0..PAGES).filter(|_| random(&mut seed, 12) == 0).collect();
            let clone = |copies: &[Copy]| -> Vec<Copy> {
                copies
                    .iter()
                    .map(|c| Copy { start: c.start, host: c.host.clone(), segments: c.segments.clone(), ranges: c.ranges.clone() })
                    .collect()
            };
            let title = |bytes: &[u8]| Memory { bytes: UnsafeCell::new(bytes.to_vec()), unmapped: unmapped.clone() };
            let (mut wanted, wanted_title) = (clone(&copies), title(&memory));
            one_by_one(&mut wanted, &wanted_title);
            let wanted_title = wanted_title.bytes.into_inner();
            for threads in [1, 2, 4, 7] {
                let (mut done, done_title) = (clone(&copies), title(&memory));
                at_once(&mut done, &done_title, threads);
                assert!(done_title.bytes.into_inner() == wanted_title, "round {round}, {threads} threads: title memory");
                for (k, (d, w)) in done.iter().zip(&wanted).enumerate() {
                    assert!(d.host == w.host, "round {round}, {threads} threads: copy {k}");
                }
            }
        }
    }

    /// Pieces of marked pages (`Shadow::copy_back_marked`): where the title
    /// wrote nothing, what differs from title memory is the shaders' and
    /// goes back; where it wrote, what differs from the bytes kept at its
    /// first write does, and the copy takes the title's others. At once on
    /// threads, title memory and the copy end as page by page with each
    /// changed page written whole, and the pieces say what was done.
    #[test]
    fn marked_pieces_at_once_leave_what_pages_one_by_one_leave() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        const PAGES: usize = 40;
        for round in 0..40 {
            let memory: Vec<u8> = (0..PAGES * PIECE).map(|i| (i * 11 + round) as u8).collect();
            let mut host = memory.clone();
            let mut title = memory.clone();
            // (from, to, the bytes kept if the title wrote the page).
            type Marked = (usize, usize, Option<Box<[u8]>>);
            let mut marked: Vec<Marked> = Vec::new();
            for page in 0..PAGES {
                if random(&mut seed, 3) == 0 {
                    continue;
                }
                // Whole, or in part as at a copy's ends.
                let (from, to) = match random(&mut seed, 4) {
                    0 => (page * PIECE + 1 + random(&mut seed, PIECE - 1), (page + 1) * PIECE),
                    1 => (page * PIECE, page * PIECE + 1 + random(&mut seed, PIECE - 1)),
                    _ => (page * PIECE, (page + 1) * PIECE),
                };
                for _ in 0..random(&mut seed, 5) {
                    let n = [1, 4, 500, PIECE][random(&mut seed, 4)].min(to - from);
                    let a = if random(&mut seed, 3) == 0 { to - n } else { from + random(&mut seed, to - from - n + 1) };
                    for b in &mut host[a..a + n] {
                        *b = random(&mut seed, 256) as u8;
                    }
                }
                let saved = (random(&mut seed, 3) == 0).then(|| {
                    for _ in 0..1 + random(&mut seed, 4) {
                        let n = [1, 8, 300][random(&mut seed, 3)];
                        let a = page * PIECE + random(&mut seed, PIECE - n + 1);
                        for b in &mut title[a..a + n] {
                            *b = random(&mut seed, 256) as u8;
                        }
                    }
                    memory[page * PIECE..(page + 1) * PIECE].to_vec().into_boxed_slice()
                });
                marked.push((from, to, saved));
            }
            let unmapped: Vec<usize> = (0..PAGES).filter(|_| random(&mut seed, 15) == 0).collect();
            // Page by page.
            let (mut host_wanted, mut title_wanted) = (host.clone(), title.clone());
            let mut wrote_wanted = Vec::new();
            for (from, to, saved) in &marked {
                let (from, to) = (*from, *to);
                if unmapped.contains(&(from / PIECE)) {
                    wrote_wanted.push(false);
                    continue;
                }
                let differs = match saved {
                    None => {
                        let differs = host_wanted[from..to] != title_wanted[from..to];
                        title_wanted[from..to].copy_from_slice(&host_wanted[from..to]);
                        differs
                    }
                    Some(before) => {
                        let page = from / PIECE * PIECE;
                        merge_by_bytes(&mut host_wanted[from..to], &before[from - page..to - page], &mut title_wanted[from..to])
                    }
                };
                wrote_wanted.push(differs);
            }
            for threads in [1, 3, 6] {
                let mut host = host.clone();
                let title = Memory { bytes: UnsafeCell::new(title.clone()), unmapped: unmapped.clone() };
                let pieces: Vec<Piece> = marked
                    .iter()
                    .map(|(from, to, saved)| {
                        let before = match saved {
                            Some(bytes) => Before::Saved(bytes.clone(), from % PIECE),
                            None => Before::Title,
                        };
                        // SAFETY: inside the copy's bytes.
                        let bytes = unsafe { host.as_mut_ptr().add(*from) };
                        Piece::new(BASE + from, bytes, to - from, before, to - from == PIECE, 0)
                    })
                    .collect();
                // Read for the first pass, or (every other time) not.
                let unread = threads == 3;
                compare(&pieces, &title, threads, unread);
                put(&pieces, &title, threads, None);
                for (k, piece) in pieces.iter().enumerate() {
                    let there = !unmapped.contains(&((piece.at - BASE) / PIECE));
                    assert_eq!(piece.wrote(), wrote_wanted[k], "round {round}, piece {k}");
                    assert_eq!(piece.synced(), there, "round {round}, piece {k}: equal to title memory if it is there");
                    assert!(!piece.wrote() || piece.opens(), "written only where opened");
                    let untouched = matches!(piece.before, Before::Title) && !unread && !piece.wrote();
                    assert!(!(untouched && piece.opens()), "read first, a piece with nothing to write is not opened");
                    let (bytes, writes) = piece.written();
                    assert!(bytes as usize <= piece.n && (bytes > 0) == piece.wrote() && writes as usize <= RUNS);
                }
                again(&pieces, &title, threads);
                assert!(piece_copies(&pieces) == [0]);
                drop(pieces);
                assert!(title.bytes.into_inner() == title_wanted, "round {round}, {threads} threads: title memory");
                assert!(host == host_wanted, "round {round}, {threads} threads: the copy");
            }
        }
    }

    /// A title write that lands in a page after its piece was read and
    /// before the page is guarded: reading the page again brings it to the
    /// copy, a whole page only (a piece of one is not taken by its copy).
    #[test]
    fn taken_pages_read_again_hold_a_title_write_made_meanwhile() {
        let memory: Vec<u8> = (0..2 * PIECE).map(|i| i as u8 & 0x7f).collect();
        let mut host = memory.clone();
        (host[10], host[PIECE + 10]) = (0x81, 0x82);
        let title = Memory { bytes: UnsafeCell::new(memory.clone()), unmapped: Vec::new() };
        let segments: Vec<Segment> = vec![(0, memory.clone())];
        let bytes = host.as_mut_ptr();
        let whole = Piece::new(BASE, bytes, PIECE, Before::Kept(&segments, 0), true, 0);
        // SAFETY: inside the copy's bytes.
        let second = unsafe { bytes.add(PIECE) };
        let part = Piece::new(BASE + PIECE, second, PIECE - 1, Before::Kept(&segments, PIECE), false, 0);
        let pieces = [whole, part];
        compare(&pieces, &title, 2, false);
        put(&pieces, &title, 2, None);
        // SAFETY: bytes of the test's title memory.
        unsafe {
            assert_eq!((title.at(BASE + 10).read(), title.at(BASE + PIECE + 10).read()), (0x81, 0x82));
            title.at(BASE + 20).write(0x55);
            title.at(BASE + PIECE + 20).write(0x56);
        }
        again(&pieces, &title, 2);
        drop(pieces);
        assert_eq!((host[10], host[20]), (0x81, 0x55), "the taken page holds the title's write");
        assert_eq!((host[PIECE + 10], host[PIECE + 20]), (0x82, 20), "the part of a page is as it was read");
    }

    /// The copies a batch's pieces are of.
    fn piece_copies(pieces: &[Piece]) -> Vec<u32> {
        let mut copies: Vec<u32> = pieces.iter().map(|p| p.copy).collect();
        copies.sort_unstable();
        copies.dedup();
        copies
    }

    /// The bits of a sparse bitmap, taken sixteen dwords at a time, are the
    /// bits found dword by dword, and the bitmap is left clear; new bits of
    /// one map or two are those a dword-by-dword pass finds, and are seen
    /// from then on.
    #[test]
    fn bits_are_found_as_dword_by_dword() {
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for round in 0..300 {
            let words = [0, 1, 15, 16, 17, 100, 1000][round % 7];
            let sparse = |seed: &mut u64| -> Vec<u32> {
                (0..words)
                    .map(|_| match random(seed, 40) {
                        0 => random(seed, 1 << 31) as u32 * 2 + 1,
                        1 => 1 << random(seed, 32),
                        _ => 0,
                    })
                    .collect()
            };
            let mut bits = sparse(&mut seed);
            let wanted: Vec<u32> = (0..words as u32 * 32).filter(|&n| bits[n as usize / 32] >> (n % 32) & 1 != 0).collect();
            let mut taken = Vec::new();
            take_bits(&mut bits, &mut taken);
            assert_eq!(taken, wanted, "round {round}");
            assert!(bits.iter().all(|&w| w == 0));
            let (a, b, mut seen) = (sparse(&mut seed), sparse(&mut seed), sparse(&mut seed));
            for other in [None, Some(&b[..])] {
                let mut seen_now = seen.clone();
                let both = |i: usize| a[i] | other.map_or(0, |o| o[i]);
                let wanted: Vec<u32> = (0..words as u32 * 32)
                    .filter(|&n| (both(n as usize / 32) & !seen[n as usize / 32]) >> (n % 32) & 1 != 0)
                    .collect();
                let mut new = Vec::new();
                new_bits(&a, other, &mut seen_now, &mut new);
                assert_eq!(new, wanted, "round {round}");
                assert!((0..words).all(|i| seen_now[i] == seen[i] | both(i)));
                new.clear();
                new_bits(&a, other, &mut seen_now, &mut new);
                assert!(new.is_empty(), "seen from then on");
            }
            seen.clear();
        }
    }

    #[test]
    fn unions_and_pieces() {
        let mut ranges = vec![(50, 60), (10, 20), (20, 30), (25, 40), (70, 70), (58, 59), (90, 80)];
        union(&mut ranges);
        assert_eq!(ranges, [(10, 40), (50, 60)]);
        let mut cut = Vec::new();
        pieces(PIECE - 10, 5, 2 * PIECE + 30, |from, to| cut.push((from, to)));
        assert_eq!(cut, [(5, 10), (10, PIECE + 10), (PIECE + 10, 2 * PIECE + 10), (2 * PIECE + 10, 2 * PIECE + 30)]);
        pieces(0, 7, 7, |_, _| panic!("no piece of an empty range"));
    }
}
