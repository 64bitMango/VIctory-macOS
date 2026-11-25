//! Submissions of the title's that return before the host GPU has run them
//! (`VICTORY_ASYNC_SUBMIT=1`; off by default, every submission is then
//! waited for as before).
//!
//! A waited submission costs the title's thread the GPU's time: it stands
//! in `Host::execute` until the host fence moves. An unwaited one returns
//! once its lists are with the host queue, and the next submission's buffer
//! copies are brought up to date while the GPU still runs it.
//!
//! Which go unwaited. Only the title's own (a queue's piece,
//! `crate::queue`), and of those only one whose debts after its run can be
//! paid late with the result a waited one leaves ([`Facts::waited`]):
//! - one that owes nothing: no shader of its lists writes a buffer copy,
//!   no list has checks or work for after the run (read-backs are such
//!   work), none writes at the bottom of the pipe, no upload list of ours
//!   runs with it and no diagnostics time or dump it;
//! - one whose shaders write buffer copies only in ranges its lists
//!   declared (`Step::GpuWrite`, marked or not), and that owes nothing else
//!   ("Shader writes" below; `VICTORY_ASYNC_SHADER_WRITES=0` makes these
//!   wait again).
//!
//! Everything else is waited for, counted by reason. Our own submissions
//! (present, uploads, captures, the ambient and twin passes) always wait:
//! they read or free what they made as soon as the call returns.
//!
//! One at a time. At most one submission is unwaited; it keeps its host
//! lists (and through them every buffer, view table and descriptor slot
//! they name: a title list reset or released meanwhile leaves them to it,
//! `crate::list::restart`), and nothing retired is released while it runs.
//! What a copy keeps for its copy-back (the ranges being written, the
//! snapshot, the armed pages) and the views' written-pages bitmap are per
//! copy and per host, not per submission: two in flight could not tell
//! their marks apart, and the second's refresh of a range the first writes
//! would have to wait for the first anyway.
//!
//! When it is finished: waited for, then what a waited submission does
//! after its wait (`Host::settle`: marked pages and declared ranges copied
//! back, the views its shaders read made), retired objects released, its
//! lists given back, then the Signals kept behind it. By
//! - the next host submission, the title's or ours, before anything of that
//!   one is prepared or submitted (so two submissions' results reach title
//!   memory in the order they were made, and a cap of one needs no count);
//! - the making of a view heap (`Host::quiet`), which must find the GPU
//!   idle;
//! - the title asking a fence for an event at a value not reached yet, or
//!   signalling a fence from the CPU (the Signals kept come first);
//! - PresentX;
//! - a poll (a fence's value read, the held-work timer every 2 ms, which
//!   runs while one whose shaders write is in flight) that finds the GPU
//!   done with it.
//!
//! A queue's Signal does not finish it: the Signal is kept with it and the
//! fence moves when it is finished, so no fence (nor the memory the title
//! reads it at) says more than what has been finished. There is no shutdown
//! path: the host device lives as long as the process.
//!
//! Meanwhile the next submission may bring up to date the buffer ranges its
//! lists ask for (`Host::prewarm`), around what the unwaited one writes.
//! The bytes the title changed since are bytes the unwaited one does not
//! read (on the console the GPU reads title memory itself, so the title
//! leaves alone what a list reads until a fence says it ran). The steps in
//! their order do it all again after the finish and find little left. It
//! must not write title memory, run its steps, sync or make views in the
//! heaps, upload textures, give up textures, release anything retired or
//! reset a list or prelude: all of that comes after the finish, as before.
//!
//! # Shader writes
//!
//! A is the unwaited submission, B the next one, T title memory, h a host
//! copy of it.
//!
//! What A may still be writing. Bytes of h inside the ranges A's lists
//! declared, which the copy holds from A's steps until A's copy-back
//! (`Shadow::pending`), and nothing else: a declared range is the reach of
//! the view the shader writes through, with or without marks (marks say
//! afterwards which pages of it were written; before the finish only the
//! range is known). Two classes have no such bound and wait:
//! shaders that take unordered views from the heap themselves (their reach
//! is every such view of the heap; pages they mark were not armed, so a
//! title write there before the copy-back would keep the title's bytes and
//! lose theirs: the longer the flight, the more are lost), and programs
//! that use title memory (any page of any copy, at addresses they compute;
//! their checks and work for after the run must be done before anything
//! else sees the copies).
//!
//! What B may do meanwhile. Bring up to date any byte of any copy outside
//! A's declared ranges: A's shaders neither write those bytes nor (see
//! above) read what the title changed since. Inside them nothing: title
//! memory copied there would overwrite what the GPU is writing, and would
//! be taken into the snapshot as if A had started from it
//! (`Shadow::update_busy` is right only for ranges of lists not yet
//! submitted). Those refreshes are left to B's steps, after A's finish
//! (counted: "left for after the finish"). A page a refresh holds only in
//! part is copied in part and stays changed, so the trackers' state of
//! pages A writes is not touched either.
//!
//! A title write to a page A writes. To other bytes of the page: kept, as
//! when a waited submission runs. Unmarked ranges are compared with the
//! snapshot taken at A's steps (after their refresh, before A was
//! submitted; it covers every declared byte): only bytes where h differs
//! from it are the GPU's and go into T; every other byte of T stays the
//! title's, and h takes it. Marked pages were armed at A's steps: the
//! title's first write to one faults and the page's bytes from before are
//! kept (`crate::watch::Tracked::saved`), which is the old side of the
//! comparison in its place; a page the title did not write is still equal
//! to h as it was before A, and T itself is the old side. Either way the
//! result is what the waited order leaves: A's bytes where A wrote, the
//! title's later bytes elsewhere.
//! To the same bytes A writes: the GPU's win, as when the title writes
//! them while a waited submission runs. A title that writes (or reads)
//! bytes a list's shaders write before something told it that the list ran
//! races the GPU on the console, where ExecuteCommandLists returns at
//! once; every such something is ordered behind the finish here (below).
//!
//! The copy-back at the finish is the waited submission's own code
//! (`Host::settle`) on the same state, so T after it, and h where the next
//! submission's shaders read it, are byte for byte what the waited order
//! gives wherever the title kept to that rule. Two differences remain,
//! neither where a shader or the title is owed other bytes:
//! - h may hold newer title bytes than the waited order in pieces the GPU
//!   wrote (the title's writes of the flight come with the comparison);
//!   they are bytes the next refresh brings anyway, and the page counts as
//!   taken with them;
//! - the waited order drops what shaders wrote in a marked page the title
//!   had written and no refresh had taken whole
//!   (`Shadow::copy_back_marked`'s shortcut, counted as "kept the
//!   title's"). An early refresh, cut around A's ranges instead of B's,
//!   may take such a page whole before B's steps arm it, and B's shaders'
//!   bytes there then go back. Never the other way: every page the waited
//!   order finds taken when it arms it, this one does too.
//!
//! What the title relies on. Not on memory being written when
//! ExecuteCommandLists returns: the console's returns before the GPU has
//! started. It learns that the GPU ran a list by a fence a queue Signal
//! moved (an event, the value read, the value's word in memory), by a word
//! a list wrote at the bottom of the pipe (which it, or a list held on a
//! wait, reads), by a read-back or query we complete, or by having
//! presented. All of these come after the finish: Signals are kept behind
//! A; a submission with bottom-of-pipe writes or after-work waits, and a
//! later one's are made after its own wait, which follows A's finish;
//! PresentX finishes A. So title reads of what A writes are not trapped,
//! and need not be. What does read title memory with no such order is our
//! own code: every read of buffer memory we make on the title's behalf
//! (vertex, index, constant and indirect data, which shaders read from the
//! copies) is a step of a submission or work after its run, both behind
//! A's finish. At recording time we read descriptor records and the
//! title's own arguments, which shaders do not write; only diagnostics
//! (command notes, capture's "before" bytes) read buffers there, and may
//! show bytes from before A. A list held on a wait for a memory value
//! reads it in the pump: a value A's shaders write is found by the timer's
//! poll once A is finished.
//!
//! Marks are consumed once: the bitmap is read and cleared by A's
//! copy-back alone, under the submission lock, before B is submitted; B's
//! early refreshes never read it.
// shortcut: submissions that write at the bottom of the pipe wait. The
// words they write are what lists held on other queues wait for
// (`crate::hold`): unwaited, every such list would start from the 2 ms
// timer instead of the pump that ran its writer, and a later submission of
// the writer's queue would pass it. The rule (the writes made at the
// finish, the finish forced while any queue holds work) is designed, not
// made.
// shortcut: shaders writing through the heap's own views wait. Unwaited
// they need each unordered buffer view of the bound heap taken as a
// declared, marked range at the list's steps (kept from early refreshes,
// its pages armed): designed, not made.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the title's submissions may go unwaited
/// (`VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_ASYNC_SUBMIT"))
}

/// Whether those whose shaders write declared ranges of buffer copies may
/// too: unless `VICTORY_ASYNC_SHADER_WRITES=0`.
pub(crate) fn shader_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::switch::off("VICTORY_ASYNC_SHADER_WRITES"))
}

/// Why a submission of the title's was waited for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// Its shaders write declared ranges of buffer copies, and such
    /// submissions wait (`VICTORY_ASYNC_SHADER_WRITES=0`).
    ShaderWrites,
    /// Checks or work for after the run (read-backs, title memory written
    /// at computed addresses).
    After,
    /// Bottom-of-pipe writes: the title, or a list held on a wait, reads
    /// them with no call of ours.
    BottomOfPipe,
    /// A list of ours uploads textures with it.
    Uploads,
    /// Diagnostics time its lists or dump what they drew.
    Diagnostics,
    /// The host did not take its fence signal.
    Host,
    /// A shader writes through views it takes from the heap itself: no
    /// range bounds what it writes.
    HeapWrites,
    /// A program uses title memory: it writes any copy, at addresses it
    /// computes.
    TitleMemory,
}

/// The reasons, in the order the first that holds is named.
const REASONS: usize = 8;

/// What a submission owes once the GPU is done with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    /// Shaders write ranges of buffer copies its lists declared.
    pub(crate) declared_writes: bool,
    /// Shaders write through views they take from the heap themselves.
    pub(crate) heap_writes: bool,
    /// A program uses title memory.
    pub(crate) title_memory: bool,
    pub(crate) after: bool,
    pub(crate) bottom_of_pipe: bool,
    pub(crate) uploads: bool,
    pub(crate) diagnostics: bool,
}

impl Facts {
    /// The facts of two lists of one submission.
    pub(crate) fn with(self, other: Facts) -> Facts {
        Facts {
            declared_writes: self.declared_writes || other.declared_writes,
            heap_writes: self.heap_writes || other.heap_writes,
            title_memory: self.title_memory || other.title_memory,
            after: self.after || other.after,
            bottom_of_pipe: self.bottom_of_pipe || other.bottom_of_pipe,
            uploads: self.uploads || other.uploads,
            diagnostics: self.diagnostics || other.diagnostics,
        }
    }

    /// Every reason the submission must be waited for, the one named first;
    /// `declared`: whether shaders' writes to declared ranges may be copied
    /// back late ([`shader_writes`]).
    fn reasons(self, declared: bool) -> impl Iterator<Item = Waited> {
        [
            (self.title_memory, Waited::TitleMemory),
            (self.heap_writes, Waited::HeapWrites),
            (self.declared_writes && !declared, Waited::ShaderWrites),
            (self.after, Waited::After),
            (self.bottom_of_pipe, Waited::BottomOfPipe),
            (self.uploads, Waited::Uploads),
            (self.diagnostics, Waited::Diagnostics),
        ]
        .into_iter()
        .filter_map(|(holds, why)| holds.then_some(why))
    }

    /// Why the submission must be waited for (the first reason that
    /// holds), and whether it is the only one; or none: it may go unwaited.
    pub(crate) fn waited(self, declared: bool) -> Option<(Waited, bool)> {
        let mut reasons = self.reasons(declared);
        let first = reasons.next()?;
        Some((first, reasons.next().is_none()))
    }

    /// Whether its shaders write buffer copies: what they wrote is copied
    /// back when it is finished.
    pub(crate) fn writes(self) -> bool {
        self.declared_writes || self.heap_writes || self.title_memory
    }
}

/// What finished an unwaited submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum By {
    /// The title's next submission, after its buffer copies were brought up
    /// to date.
    Title,
    /// A submission of ours.
    Ours,
    /// The making of a view heap.
    Heap,
    /// The title asked a fence for an event.
    FenceWait,
    Present,
    /// A poll that found the GPU done with it.
    Done,
}

/// The unwaited submission, if any, and the Signals kept behind it (fence,
/// value) in the order the queues made them.
pub(crate) struct Slot<S> {
    held: Option<(S, Vec<(usize, u64)>)>,
}

impl<S> Slot<S> {
    pub(crate) const fn new() -> Self {
        Slot { held: None }
    }

    /// `submission` goes unwaited. The one before is finished: the caller
    /// finishes it before it submits.
    pub(crate) fn leave(&mut self, submission: S) {
        debug_assert!(self.held.is_none(), "two unwaited submissions");
        self.held = Some((submission, Vec::new()));
    }

    pub(crate) fn get(&self) -> Option<&S> {
        self.held.as_ref().map(|(s, _)| s)
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut S> {
        self.held.as_mut().map(|(s, _)| s)
    }

    /// A queue's Signal: kept behind the unwaited submission (true), or for
    /// the caller to make now (none is unwaited).
    pub(crate) fn signal(&mut self, fence: usize, value: u64) -> bool {
        match &mut self.held {
            Some((_, signals)) => {
                signals.push((fence, value));
                true
            }
            None => false,
        }
    }

    /// The submission is finished: it and the Signals to make now, in
    /// order. Taken only then: a Signal made while it is being finished is
    /// still kept behind it.
    pub(crate) fn finished(&mut self) -> Option<(S, Vec<(usize, u64)>)> {
        self.held.take()
    }
}

/// Finishes the unwaited submission, if there is one: whether there was.
/// Not on a thread that holds the submission lock (it has finished it).
pub(crate) fn finish(by: By) -> bool {
    #[cfg(windows)]
    return crate::host::finish_unwaited(by);
    #[cfg(not(windows))]
    {
        let _ = by;
        false
    }
}

/// Counters since the last frame-rate line.
static TITLE: AtomicU64 = AtomicU64::new(0);
static LEFT: AtomicU64 = AtomicU64::new(0);
/// Of those unwaited, the ones whose shaders write buffer copies.
static LEFT_WRITING: AtomicU64 = AtomicU64::new(0);
static WAITED: [AtomicU64; REASONS] = [const { AtomicU64::new(0) }; REASONS];
/// Waited for one reason alone: what lifting that reason would leave
/// unwaited.
static ALONE: [AtomicU64; REASONS] = [const { AtomicU64::new(0) }; REASONS];
static FINISHED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Microseconds finishing (all of it, of that waiting for the GPU, and of
/// the rest copying back what shaders wrote), microseconds and bytes of
/// copies brought up to date meanwhile, Signals kept behind unwaited
/// submissions.
static FINISH_US: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static PREWARM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static SIGNALS: AtomicU64 = AtomicU64::new(0);
/// Refreshes left for after the finish because the unwaited submission's
/// shaders write there: how many ranges, their bytes.
static DEFERRED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Microseconds the GPU ran an unwaited submission while the title's
/// thread did something else: from the return to the finish's start or
/// the GPU's end, and of that the early refreshes.
static OVERLAP_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A submission of the title's was made: waited for (why, and whether for
/// that alone), or not (`writes`: its shaders write buffer copies).
pub(crate) fn note(waited: Option<(Waited, bool)>, writes: bool) {
    TITLE.fetch_add(1, Relaxed);
    match waited {
        Some((why, alone)) => {
            WAITED[why as usize].fetch_add(1, Relaxed);
            if alone {
                ALONE[why as usize].fetch_add(1, Relaxed);
            }
        }
        None => {
            LEFT.fetch_add(1, Relaxed);
            if writes {
                LEFT_WRITING.fetch_add(1, Relaxed);
            }
        }
    };
}

/// An unwaited submission was finished: by what, how long it took, how
/// much of that was waiting for the GPU and how much copying back.
pub(crate) fn note_finished(by: By, all: std::time::Duration, waiting: std::time::Duration, back: std::time::Duration) {
    FINISHED[by as usize].fetch_add(1, Relaxed);
    FINISH_US[0].fetch_add(all.as_micros() as u64, Relaxed);
    FINISH_US[1].fetch_add(waiting.as_micros() as u64, Relaxed);
    FINISH_US[2].fetch_add(back.as_micros() as u64, Relaxed);
}

/// Buffer copies were brought up to date while a submission ran: how long
/// it took, the bytes copied, and the ranges (and their bytes) left for
/// after the finish.
pub(crate) fn note_prewarm(took: std::time::Duration, bytes: usize, left: (usize, usize)) {
    PREWARM[0].fetch_add(took.as_micros() as u64, Relaxed);
    PREWARM[1].fetch_add(bytes as u64, Relaxed);
    DEFERRED[0].fetch_add(left.0 as u64, Relaxed);
    DEFERRED[1].fetch_add(left.1 as u64, Relaxed);
}

/// The GPU ran an unwaited submission for `flight` while the title's
/// thread was elsewhere, `refreshing` of it in the early refreshes.
pub(crate) fn note_overlap(flight: std::time::Duration, refreshing: std::time::Duration) {
    OVERLAP_US[0].fetch_add(flight.as_micros() as u64, Relaxed);
    OVERLAP_US[1].fetch_add(refreshing.as_micros() as u64, Relaxed);
}

/// A Signal was kept behind an unwaited submission.
pub(crate) fn note_signal() {
    SIGNALS.fetch_add(1, Relaxed);
}

/// The frame-rate line's part, per frame over `frames`, and the counters
/// start again; empty with the switch off.
pub(crate) fn take_line(frames: f64) -> String {
    if !on() {
        return String::new();
    }
    let per = |n: &AtomicU64| n.swap(0, Relaxed) as f64 / frames;
    let ms = |n: &AtomicU64| n.swap(0, Relaxed) as f64 / 1e3 / frames;
    let mb = |n: &AtomicU64| n.swap(0, Relaxed) as f64 / 1e6 / frames;
    let (title, left, writing) = (per(&TITLE), per(&LEFT), per(&LEFT_WRITING));
    let w: [f64; REASONS] = std::array::from_fn(|i| per(&WAITED[i]));
    let a: [f64; REASONS] = std::array::from_fn(|i| per(&ALONE[i]));
    let by = [0, 1, 2, 3, 4, 5].map(|i| per(&FINISHED[i]));
    let (finish, waiting, back) = (ms(&FINISH_US[0]), ms(&FINISH_US[1]), ms(&FINISH_US[2]));
    let (prewarm, prewarm_mb) = (ms(&PREWARM[0]), mb(&PREWARM[1]));
    let (deferred, deferred_mb) = (per(&DEFERRED[0]), mb(&DEFERRED[1]));
    let (flight, refreshing) = (ms(&OVERLAP_US[0]), ms(&OVERLAP_US[1]));
    let signals = per(&SIGNALS);
    use Waited::*;
    let n = |why: Waited| format!("{:.1} ({:.1} for that alone)", w[why as usize], a[why as usize]);
    format!(
        "; unwaited {left:.1} of {title:.1} title submissions/frame ({writing:.1} of them with shader writes; \
         waited for: title-memory programs {}, heap-view writes {}, shader writes {}, after-work {}, \
         bottom-of-pipe {}, uploads {}, diagnostics {}, host {:.1}), finished by title submission {:.1}, \
         ours {:.1}, view heap {:.1}, fence wait {:.1}, present {:.1}, poll {:.1}; {finish:.1} ms finishing \
         ({waiting:.1} waiting for the GPU, {back:.1} copying back), {prewarm:.1} ms {prewarm_mb:.1} MB \
         refreshed meanwhile, {deferred:.1} refreshes {deferred_mb:.1} MB left for after the finish (shaders \
         in flight write there), the GPU ran {flight:.1} ms beside the title's thread ({refreshing:.1} of it \
         those refreshes), {signals:.1} signals kept",
        n(TitleMemory),
        n(HeapWrites),
        n(ShaderWrites),
        n(After),
        n(BottomOfPipe),
        n(Uploads),
        n(Diagnostics),
        w[Host as usize],
        by[0],
        by[1],
        by[2],
        by[3],
        by[4],
        by[5]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submission_that_owes_nothing_goes_unwaited() {
        assert_eq!(Facts::default().waited(true), None);
        assert_eq!(Facts::default().waited(false), None);
    }

    /// The hazards of an unwaited submission: a read-back or a
    /// bottom-of-pipe write seen before it is finished, shader writes with
    /// no bound on where they land. Each alone makes it wait.
    #[test]
    fn whatever_cannot_be_paid_late_makes_it_wait() {
        let f = Facts::default();
        let alone = |f: Facts| f.waited(true);
        assert_eq!(alone(Facts { heap_writes: true, ..f }), Some((Waited::HeapWrites, true)));
        assert_eq!(alone(Facts { title_memory: true, ..f }), Some((Waited::TitleMemory, true)));
        assert_eq!(alone(Facts { after: true, ..f }), Some((Waited::After, true)));
        assert_eq!(alone(Facts { bottom_of_pipe: true, ..f }), Some((Waited::BottomOfPipe, true)));
        assert_eq!(alone(Facts { uploads: true, ..f }), Some((Waited::Uploads, true)));
        assert_eq!(alone(Facts { diagnostics: true, ..f }), Some((Waited::Diagnostics, true)));
    }

    /// Shader writes to declared ranges are copied back at the finish:
    /// alone they do not make a submission wait, unless the switch says so;
    /// with anything else owed it waits for that.
    #[test]
    fn declared_shader_writes_alone_go_unwaited() {
        let writes = Facts { declared_writes: true, ..Facts::default() };
        assert!(writes.writes());
        assert_eq!(writes.waited(true), None);
        assert_eq!(writes.waited(false), Some((Waited::ShaderWrites, true)));
        assert_eq!(Facts { after: true, ..writes }.waited(true), Some((Waited::After, true)));
        assert_eq!(Facts { after: true, ..writes }.waited(false), Some((Waited::ShaderWrites, false)));
        // Writes with no bound wait whatever was declared beside them.
        assert_eq!(Facts { heap_writes: true, ..writes }.waited(true), Some((Waited::HeapWrites, true)));
        let memory = Facts { title_memory: true, after: true, ..writes };
        assert_eq!(memory.waited(true), Some((Waited::TitleMemory, false)));
    }

    #[test]
    fn one_list_that_owes_makes_its_submission_wait() {
        let plain = Facts::default();
        let writes = Facts { bottom_of_pipe: true, ..plain };
        assert_eq!(plain.with(plain).waited(true), None);
        assert_eq!(plain.with(writes).waited(true), Some((Waited::BottomOfPipe, true)));
        assert_eq!(writes.with(plain).waited(true), Some((Waited::BottomOfPipe, true)));
        let both = writes.with(Facts { heap_writes: true, ..plain });
        assert_eq!(both.waited(true), Some((Waited::HeapWrites, false)), "the first reason, not alone");
        let declared = plain.with(Facts { declared_writes: true, ..plain });
        assert_eq!(declared.waited(true), None);
        assert!(declared.writes() && !plain.writes());
    }

    /// Every reason has its counter, and a submission waited for one
    /// reason alone is counted there too.
    #[test]
    fn reasons_have_their_counters() {
        for why in [
            Waited::ShaderWrites,
            Waited::After,
            Waited::BottomOfPipe,
            Waited::Uploads,
            Waited::Diagnostics,
            Waited::Host,
            Waited::HeapWrites,
            Waited::TitleMemory,
        ] {
            assert!((why as usize) < REASONS);
        }
    }

    /// A fence is not signalled before the unwaited work it stands for is
    /// finished: Signals made while it runs, and while it is being
    /// finished, come out with it, in order; later ones are made at once.
    #[test]
    fn signals_stay_behind_the_unwaited_submission() {
        let mut slot = Slot::new();
        assert!(!slot.signal(7, 1), "none unwaited: signalled at once");
        slot.leave("a");
        assert!(slot.signal(7, 2));
        assert!(slot.signal(9, 1));
        assert_eq!(slot.get(), Some(&"a"), "still there while it is finished");
        assert!(slot.signal(7, 3), "made during the finish: still behind it");
        assert_eq!(slot.finished(), Some(("a", vec![(7, 2), (9, 1), (7, 3)])));
        assert!(!slot.signal(7, 4));
        assert_eq!(slot.finished(), None);
    }

    #[test]
    fn the_next_unwaited_submission_starts_with_no_signals() {
        let mut slot = Slot::new();
        slot.leave(1);
        slot.signal(7, 1);
        slot.finished();
        slot.leave(2);
        assert_eq!(slot.finished(), Some((2, vec![])));
    }

    #[test]
    fn counts_reach_the_line_once() {
        // The switch is off in tests unless the environment sets it.
        if !on() {
            assert_eq!(take_line(1.0), "");
        }
    }
}
