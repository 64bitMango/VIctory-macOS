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
//! `crate::queue`), and of those only one that owes nothing once the GPU is
//! done with it ([`Facts::waited`]): no shader of its lists writes a buffer
//! copy (no declared range, no write through a view heap, no program using
//! title memory), no list has checks or work for after the run (read-backs
//! are such work), none writes at the bottom of the pipe, no upload list of
//! ours runs with it and no diagnostics time or dump it. Everything else is
//! waited for, counted by reason. Our own submissions (present, uploads,
//! captures, the ambient and twin passes) always wait: they read or free
//! what they made as soon as the call returns.
//!
//! One at a time. At most one submission is unwaited; it keeps its host
//! lists (and through them every buffer, view table and descriptor slot
//! they name: a title list reset or released meanwhile leaves them to it,
//! `crate::list::restart`), and nothing retired is released while it runs.
//!
//! When it is finished: waited for, then what a waited submission does
//! after its wait (the views its shaders read made, retired objects
//! released, its lists given back), then the Signals kept behind it. By
//! - the next host submission, the title's or ours, before anything of that
//!   one is prepared or submitted (so two submissions' results reach title
//!   memory in the order they were made, and a cap of one needs no count);
//! - the making of a view heap (`Host::quiet`), which must find the GPU
//!   idle;
//! - the title asking a fence for an event at a value not reached yet, or
//!   signalling a fence from the CPU (the Signals kept come first);
//! - PresentX;
//! - a poll (a fence's value read, the held-work timer every 2 ms) that
//!   finds the GPU done with it.
//!
//! A queue's Signal does not finish it: the Signal is kept with it and the
//! fence moves when it is finished, so no fence (nor the memory the title
//! reads it at) says more than what has been finished. A wait for a memory
//! value cannot be for it (it writes none), and there is no shutdown path:
//! the host device lives as long as the process.
//!
//! Meanwhile the next submission may bring up to date the buffer ranges its
//! lists ask for (`Host::prewarm`): the unwaited one writes no buffer copy,
//! and the bytes the title changed since are bytes it does not read (on the
//! console the GPU reads title memory itself, so the title leaves alone
//! what a list reads until a fence says it ran). The steps in their order do
//! it all again after the finish and find little left. It must not write
//! title memory, run its steps, sync or make views in the heaps, upload
//! textures, give up textures, release anything retired or reset a list or
//! prelude: all of that comes after the finish, as before.
// shortcut: submissions whose shaders write buffer copies, or that write at
// the bottom of the pipe, wait. Those need the copy-back to stand a title
// write to a page between the return and the finish (marks no list declared
// keep the title's bytes there) and a refresh that goes around what the GPU
// is writing; and a write the title polls for with no call of ours.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the title's submissions may go unwaited
/// (`VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_ASYNC_SUBMIT"))
}

/// Why a submission of the title's was waited for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// Its shaders write buffer copies: what they wrote goes back to title
    /// memory.
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
}

/// What a submission owes once the GPU is done with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) shader_writes: bool,
    pub(crate) after: bool,
    pub(crate) bottom_of_pipe: bool,
    pub(crate) uploads: bool,
    pub(crate) diagnostics: bool,
}

impl Facts {
    /// The facts of two lists of one submission.
    pub(crate) fn with(self, other: Facts) -> Facts {
        Facts {
            shader_writes: self.shader_writes || other.shader_writes,
            after: self.after || other.after,
            bottom_of_pipe: self.bottom_of_pipe || other.bottom_of_pipe,
            uploads: self.uploads || other.uploads,
            diagnostics: self.diagnostics || other.diagnostics,
        }
    }

    /// Why the submission must be waited for (the first reason that
    /// holds), or none: it may go unwaited.
    pub(crate) fn waited(self) -> Option<Waited> {
        [
            (self.shader_writes, Waited::ShaderWrites),
            (self.after, Waited::After),
            (self.bottom_of_pipe, Waited::BottomOfPipe),
            (self.uploads, Waited::Uploads),
            (self.diagnostics, Waited::Diagnostics),
        ]
        .into_iter()
        .find_map(|(holds, why)| holds.then_some(why))
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
static WAITED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static FINISHED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Microseconds finishing (all of it, and of that waiting for the GPU),
/// microseconds and bytes of copies brought up to date meanwhile, Signals
/// kept behind unwaited submissions.
static FINISH_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static PREWARM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static SIGNALS: AtomicU64 = AtomicU64::new(0);

/// A submission of the title's was made: waited for (why), or not.
pub(crate) fn note(waited: Option<Waited>) {
    TITLE.fetch_add(1, Relaxed);
    match waited {
        Some(why) => WAITED[why as usize].fetch_add(1, Relaxed),
        None => LEFT.fetch_add(1, Relaxed),
    };
}

/// An unwaited submission was finished: by what, how long it took and how
/// much of that was waiting for the GPU.
pub(crate) fn note_finished(by: By, all: std::time::Duration, waiting: std::time::Duration) {
    FINISHED[by as usize].fetch_add(1, Relaxed);
    FINISH_US[0].fetch_add(all.as_micros() as u64, Relaxed);
    FINISH_US[1].fetch_add(waiting.as_micros() as u64, Relaxed);
}

/// Buffer copies were brought up to date while a submission ran.
pub(crate) fn note_prewarm(took: std::time::Duration, bytes: usize) {
    PREWARM[0].fetch_add(took.as_micros() as u64, Relaxed);
    PREWARM[1].fetch_add(bytes as u64, Relaxed);
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
    let (title, left) = (per(&TITLE), per(&LEFT));
    let waited = [0, 1, 2, 3, 4, 5].map(|i| per(&WAITED[i]));
    let by = [0, 1, 2, 3, 4, 5].map(|i| per(&FINISHED[i]));
    let (finish, waiting, prewarm) = (ms(&FINISH_US[0]), ms(&FINISH_US[1]), ms(&PREWARM[0]));
    let prewarm_mb = PREWARM[1].swap(0, Relaxed) as f64 / 1e6 / frames;
    let signals = per(&SIGNALS);
    format!(
        "; unwaited {left:.1} of {title:.1} title submissions/frame (waited for: shader writes {:.1}, \
         after-work {:.1}, bottom-of-pipe {:.1}, uploads {:.1}, diagnostics {:.1}, host {:.1}), finished by \
         title submission {:.1}, ours {:.1}, view heap {:.1}, fence wait {:.1}, present {:.1}, poll {:.1}; \
         {finish:.1} ms finishing ({waiting:.1} waiting for the GPU), {prewarm:.1} ms {prewarm_mb:.1} MB \
         refreshed meanwhile, {signals:.1} signals kept",
        waited[0], waited[1], waited[2], waited[3], waited[4], waited[5], by[0], by[1], by[2], by[3], by[4], by[5]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submission_that_owes_nothing_goes_unwaited() {
        assert_eq!(Facts::default().waited(), None);
    }

    /// The hazards of an unwaited submission: a read-back or a
    /// bottom-of-pipe write seen before it is finished, shader writes not
    /// yet back in title memory. Each alone makes it wait.
    #[test]
    fn whatever_is_owed_after_the_run_makes_it_wait() {
        let f = Facts::default();
        assert_eq!(Facts { shader_writes: true, ..f }.waited(), Some(Waited::ShaderWrites));
        assert_eq!(Facts { after: true, ..f }.waited(), Some(Waited::After));
        assert_eq!(Facts { bottom_of_pipe: true, ..f }.waited(), Some(Waited::BottomOfPipe));
        assert_eq!(Facts { uploads: true, ..f }.waited(), Some(Waited::Uploads));
        assert_eq!(Facts { diagnostics: true, ..f }.waited(), Some(Waited::Diagnostics));
    }

    #[test]
    fn one_list_that_owes_makes_its_submission_wait() {
        let plain = Facts::default();
        let writes = Facts { bottom_of_pipe: true, ..plain };
        assert_eq!(plain.with(plain).waited(), None);
        assert_eq!(plain.with(writes).waited(), Some(Waited::BottomOfPipe));
        assert_eq!(writes.with(plain).waited(), Some(Waited::BottomOfPipe));
        let both = writes.with(Facts { shader_writes: true, ..plain });
        assert_eq!(both.waited(), Some(Waited::ShaderWrites), "the first reason");
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
