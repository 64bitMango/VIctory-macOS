//! Unwaited submissions, tier one (`VICTORY_UNWAITED=1`; off by default) and
//! their checker (`VICTORY_UNWAITED_CHECK=1`, usable with the switch off
//! too). The design, its invariants I1 to I9 and the checker's rules C1 to
//! C7 are in `docs/unwaited.md`.
//!
//! A title submission is handed to the host queue and the call returns: it
//! is a *flight*. It is *settled* (waited for, then everything
//! `Host::execute` does after its wait) before anything can see its results
//! or disturb its inputs. This module holds what the queues, the fences and
//! the checker need of a flight without the host's types: whether one is in
//! the air, what it will tell at its settle (bottom-of-pipe words, queue
//! Signals kept behind it), the counters of the frame line, and the
//! checker's switch and report. The flight itself, its launch and its settle
//! are `crate::host::unwaited`.

// Without the host GPU, or without diagnostics, parts of it have no caller.
#![cfg_attr(any(not(windows), not(feature = "diagnostics")), allow(dead_code))]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use crate::queue::Written;

/// Whether title submissions go unwaited (`VICTORY_UNWAITED=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_UNWAITED"))
}

/// What the checker does with a violation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Check {
    Off,
    /// Ends the run with the violation (`VICTORY_UNWAITED_CHECK=1`).
    Abort,
    /// Logs the first twenty of each rule and counts the rest
    /// (`VICTORY_UNWAITED_CHECK=log`).
    Log,
}

fn check_mode() -> Check {
    static MODE: OnceLock<Check> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_UNWAITED_CHECK").ok().as_deref() {
        None | Some("0") => Check::Off,
        Some("log") => Check::Log,
        Some(_) => Check::Abort,
    })
}

/// Whether the checker is on (for this thread, in tests).
pub(crate) fn checking() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::CHECK.with(|c| c.get()) {
        return on;
    }
    check_mode() != Check::Off
}

/// Whether the checker also reads every refreshed range again
/// (`VICTORY_UNWAITED_CHECK_REFRESH=1`: about what the refreshes cost).
pub(crate) fn checking_refreshes() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::CHECK.with(|c| c.get()) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    checking() && *ON.get_or_init(|| crate::switch::on("VICTORY_UNWAITED_CHECK_REFRESH"))
}

/// A flight is in the air: set at its launch, cleared when its settle has
/// told everything it owed.
static IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Whether a flight is in the air. Never with the switch off.
#[inline]
pub(crate) fn in_flight() -> bool {
    IN_FLIGHT.load(std::sync::atomic::Ordering::Acquire)
}

std::thread_local! {
    /// This thread is settling the flight: its writes of host copies and of
    /// title memory are the landing, not a disturbance.
    static SETTLING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the thread as settling until dropped.
pub(crate) struct Settling(bool);

pub(crate) fn settling() -> Settling {
    Settling(SETTLING.with(|s| s.replace(true)))
}

impl Drop for Settling {
    fn drop(&mut self) {
        SETTLING.with(|s| s.set(self.0));
    }
}

/// What settled a flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum By {
    /// The next title piece, before it prepares anything.
    Title,
    /// A submission of ours (`Host::execute`).
    Ours,
    /// The making of a view heap (`Host::quiet`).
    Heap,
    Present,
    /// The title asked a fence for an event, or signalled one from the CPU.
    Fence,
    /// `CopyPageMappingsBatchX`.
    Mapping,
    /// The pump found a queue's head blocked.
    Pump,
    /// The poll found the GPU done with it.
    Poll,
    /// The host refused its fence signal: settled at once.
    Host,
}
const BYS: usize = 9;

/// Why a title piece was waited for by the waited code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// Diagnostics time its lists or trace the frame.
    Diagnostics,
}

/// What a flight will tell at its settle.
struct Tell {
    seq: u64,
    /// Bottom-of-pipe writes, in order: address, what, bytes.
    writes: Vec<(usize, Written, usize)>,
    /// Queue Signals kept behind it, in order: fence, value.
    signals: Vec<(usize, u64)>,
    /// Waits the pump counted as met by a word to come: address, bytes,
    /// the value predicted (checker rule C5).
    predicted: Vec<(u64, u8, u64)>,
}

static TELL: Mutex<Option<Tell>> = Mutex::new(None);

fn tell() -> std::sync::MutexGuard<'static, Option<Tell>> {
    TELL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Flights launched so far, and settled so far.
static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static SETTLES: AtomicU64 = AtomicU64::new(0);

/// A flight is launched (the caller holds the submission lock): its number.
/// `writes` are its lists' bottom-of-pipe writes.
pub(crate) fn launched(writes: Vec<(usize, Written, usize)>) -> u64 {
    let mut tell = tell();
    let seq = LAUNCHES.fetch_add(1, Relaxed) + 1;
    if checking() {
        if tell.is_some() {
            violation("C2", format_args!("flight {seq} launched with a flight in the air (I1)"));
        }
        let settled = SETTLES.load(Relaxed);
        if settled + 1 != seq {
            violation("C2", format_args!("flight {seq} launched with {settled} settled (I1)"));
        }
    }
    *tell = Some(Tell { seq, writes, signals: Vec::new(), predicted: Vec::new() });
    IN_FLIGHT.store(true, std::sync::atomic::Ordering::Release);
    COUNTS[C::Launched as usize].fetch_add(1, Relaxed);
    seq
}

/// The flight `seq` has landed and its bottom-of-pipe words are written:
/// `signal` makes each kept Signal, in order, and the flight is over. Under
/// the lock that decides whether a new Signal is kept, so that none is made
/// before these nor kept behind a flight that is gone.
pub(crate) fn landed(seq: u64, read: &dyn Fn(u64, u8) -> Option<u64>, mut signal: impl FnMut(usize, u64)) {
    let mut tell = tell();
    let Some(told) = tell.take() else {
        violation("C2", format_args!("flight {seq} settled twice or never launched"));
        return;
    };
    if checking() {
        let settled = SETTLES.load(Relaxed);
        if told.seq != seq || settled + 1 != seq {
            violation("C2", format_args!("flight {seq} settled out of order (in the air: {}, settled: {settled})", told.seq));
        }
        for &(address, bytes, value) in &told.predicted {
            let now = read(address, bytes);
            if now != Some(value) {
                violation(
                    "C5",
                    format_args!("a wait was counted as met by a word flight {seq} would write at {address:#x}: predicted {value:#x}, memory holds {now:x?}"),
                );
            }
        }
    }
    #[cfg(test)]
    let told = {
        let mut told = told;
        if tests::broken(tests::Break::SignalsReversed) {
            told.signals.reverse();
        }
        told
    };
    for (fence, value) in told.signals {
        signal(fence, value);
    }
    SETTLES.store(seq, Relaxed);
    IN_FLIGHT.store(false, std::sync::atomic::Ordering::Release);
}

/// A queue's Signal: kept behind the flight in the air (true: it is made at
/// the settle, where the fence is let go too), or for the caller to make now
/// (false: there is none). The caller keeps the fence alive first.
pub(crate) fn keep_signal(fence: usize, value: u64) -> bool {
    if !in_flight() {
        return false;
    }
    #[cfg(test)]
    if tests::broken(tests::Break::SignalsAtOnce) {
        return false;
    }
    match tell().as_mut() {
        Some(told) => {
            told.signals.push((fence, value));
            COUNTS[C::Signals as usize].fetch_add(1, Relaxed);
            true
        }
        None => false,
    }
}

/// The value the Signals kept behind the flight will leave `fence` at.
pub(crate) fn pending_fence(fence: usize) -> Option<u64> {
    if !in_flight() {
        return None;
    }
    tell().as_ref()?.signals.iter().rev().find(|s| s.0 == fence).map(|s| s.1)
}

/// What a memory value will be once pending bottom-of-pipe writes are made.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Word {
    /// No pending write touches it.
    Untouched,
    /// It will hold this.
    Value(u64),
    /// A timestamp will be written there, or memory that cannot be read
    /// holds the rest of it: known only after the settle.
    Unknown,
}

/// The `bytes` (at most 8) at `address` after `writes` are made in order
/// over `now` (what memory holds; none if it cannot be read).
pub(crate) fn overlaid(writes: &[(usize, Written, usize)], address: u64, bytes: u8, now: Option<u64>) -> Word {
    let n = usize::from(bytes.min(8));
    // Per byte: 0 untouched, 1 known, 2 unknown.
    let (mut state, mut value) = ([0u8; 8], [0u8; 8]);
    for &(at, what, len) in writes {
        let (at, len) = (at as u64, if len == 4 { 4u64 } else { 8 });
        for k in 0..n as u64 {
            let b = address.wrapping_add(k);
            if b >= at && b - at < len {
                match what {
                    Written::Value(v) => {
                        state[k as usize] = 1;
                        value[k as usize] = v.to_le_bytes()[(b - at) as usize];
                    }
                    _ => state[k as usize] = 2,
                }
            }
        }
    }
    if state[..n].iter().all(|&s| s == 0) {
        return Word::Untouched;
    }
    if state[..n].contains(&2) {
        return Word::Unknown;
    }
    if state[..n].contains(&0) {
        let Some(now) = now else { return Word::Unknown };
        let now = now.to_le_bytes();
        for k in 0..n {
            if state[k] == 0 {
                value[k] = now[k];
            }
        }
    }
    Word::Value(u64::from_le_bytes(value))
}

/// A wait's memory value as it will be when the flight in the air has
/// settled (the pump's `read`, `docs/unwaited.md` 4.3): what memory holds
/// unless a pending word covers it. A value that cannot be known yet reads
/// as memory does now, and the pump settles the flight if that blocks.
pub(crate) fn read_through(address: u64, bytes: u8, now: Option<u64>) -> Option<u64> {
    if !in_flight() {
        return now;
    }
    let mut tell = tell();
    let Some(told) = tell.as_mut() else { return now };
    match overlaid(&told.writes, address, bytes, now) {
        Word::Value(v) => {
            if now != Some(v) {
                COUNTS[C::Overlaid as usize].fetch_add(1, Relaxed);
                if checking() && told.predicted.len() < 4096 {
                    told.predicted.push((address, bytes, v));
                }
            }
            Some(v)
        }
        Word::Untouched | Word::Unknown => now,
    }
}

/// Host lists of settled flights, given back (`crate::list::recycle`) by
/// [`recycle`] on a thread that holds no lock of ours: resetting them frees
/// what title lists left them, under locks of its own.
#[cfg(windows)]
static RECYCLE: Mutex<Vec<std::sync::Arc<crate::host::HostList>>> = Mutex::new(Vec::new());

/// A settled flight's lists, for [`recycle`].
#[cfg(windows)]
pub(crate) fn recycle_later(lists: Vec<std::sync::Arc<crate::host::HostList>>) {
    RECYCLE.lock().unwrap_or_else(|e| e.into_inner()).extend(lists);
    WANTS_PUMP.store(true, Relaxed);
}

/// Gives back the lists of settled flights. Not under the submission lock.
#[cfg(windows)]
pub(crate) fn recycle() {
    let lists = std::mem::take(&mut *RECYCLE.lock().unwrap_or_else(|e| e.into_inner()));
    lists.into_iter().for_each(crate::list::recycle);
}

/// A settle landed words or moved fences that held work may wait for, on a
/// thread that could not pump: the poll pumps.
static WANTS_PUMP: AtomicBool = AtomicBool::new(false);

pub(crate) fn take_wants_pump() -> bool {
    WANTS_PUMP.swap(false, Relaxed)
}

/// The poll: a thread that, while a flight is in the air or a settle left
/// something to do, asks every millisecond whether the host GPU is done
/// with the flight (`crate::queue::poll_unwaited`). Started at the first
/// launch.
#[cfg(windows)]
pub(crate) fn start_poll() {
    static POLL: std::sync::Once = std::sync::Once::new();
    POLL.call_once(|| {
        let _ = std::thread::Builder::new().name("victory unwaited".into()).spawn(|| loop {
            let busy = in_flight() || WANTS_PUMP.load(Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(if busy { 1 } else { 4 }));
            if busy {
                crate::queue::poll_unwaited();
            }
        });
    });
}

/// The counters of the frame line.
#[derive(Clone, Copy)]
pub(crate) enum C {
    Launched,
    WaitedDiagnostics,
    /// Flights by write footprint: none, declared ranges, unbounded.
    WritesNone,
    WritesDeclared,
    WritesUnbounded,
    /// Flights by read footprint: exact ranges, heaps' hulls, every copy.
    ReadsExact,
    ReadsHulls,
    ReadsAll,
    /// Microseconds from launch to the start of the settle.
    FlightUs,
    /// Microseconds a settle stood waiting for the host GPU, and settles
    /// that found it still running their flight.
    WaitUs,
    StillRunning,
    /// Settles that found the GPU done, and the microseconds it had been
    /// done for where the poll saw it finish.
    FoundDone,
    LateUs,
    /// Microseconds of the settles after their wait.
    AfterUs,
    /// What the next piece's stages could have run beside the flight
    /// (microseconds; `docs/unwaited.md` 7): heap sync, uploads, refreshes
    /// and title copies, and the three together.
    WouldSync,
    WouldUploads,
    WouldRefresh,
    WouldAll,
    Signals,
    Overlaid,
    /// The checker: microseconds, bytes compared, ranges it could not check.
    CheckUs,
    CheckBytes,
    Unchecked,
    Violations,
}
const COUNTERS: usize = C::Violations as usize + 1;

static COUNTS: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];
static SETTLED_BY: [AtomicU64; BYS] = [const { AtomicU64::new(0) }; BYS];

pub(crate) fn add(counter: C, n: u64) {
    COUNTS[counter as usize].fetch_add(n, Relaxed);
}

/// A flight was settled by `by`.
pub(crate) fn note_settled(by: By) {
    SETTLED_BY[by as usize].fetch_add(1, Relaxed);
}

/// A title piece went to the waited code.
pub(crate) fn note_waited(why: Waited) {
    match why {
        Waited::Diagnostics => add(C::WaitedDiagnostics, 1),
    }
}

/// Adds the time since `started` to the checker's cost when dropped.
pub(crate) struct Checked(std::time::Instant);

pub(crate) fn checked() -> Checked {
    Checked(std::time::Instant::now())
}

impl Drop for Checked {
    fn drop(&mut self) {
        add(C::CheckUs, self.0.elapsed().as_micros() as u64);
    }
}

/// Frames presented since the last frame line.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// A frame was presented.
pub(crate) fn frame_done() {
    FRAMES.fetch_add(1, Relaxed);
}

/// The frame line's part per frame since the last one, and the counters
/// start again; none with the switch and the checker both off.
pub(crate) fn take_line() -> Option<String> {
    if !on() && !checking() {
        return None;
    }
    let frames = (FRAMES.swap(0, Relaxed) as f64).max(1.0);
    let per = |c: C| COUNTS[c as usize].swap(0, Relaxed) as f64 / frames;
    let ms = |c: C| COUNTS[c as usize].swap(0, Relaxed) as f64 / 1e3 / frames;
    let by: [f64; BYS] = std::array::from_fn(|i| SETTLED_BY[i].swap(0, Relaxed) as f64 / frames);
    let mut line = String::new();
    if on() {
        line.push_str(&format!(
            "unwaited: {:.1} title pieces/frame launched, {:.1} waited for diagnostics; settled by next piece {:.1}, ours {:.1}, \
             view heap {:.1}, present {:.1}, fence {:.1}, page mapping {:.1}, pump {:.1}, poll {:.1}, host {:.1}; flights writing \
             nothing {:.1}, declared ranges {:.1}, unbounded {:.1}; reading exact ranges {:.1}, hulls {:.1}, every copy {:.1}; \
             {:.1} ms from launch to settle, {:.1} ms settles stood waiting for the GPU ({:.1} found it still running), {:.1} found it done ({:.1} ms \
             after the poll saw it done), {:.1} ms of settles after the wait; would have overlapped: heap sync {:.1} ms, uploads {:.1} ms, \
             refreshes and title copies {:.1} ms, the three {:.1} ms; {:.1} signals kept, {:.1} waits met by a word to come",
            per(C::Launched),
            per(C::WaitedDiagnostics),
            by[By::Title as usize],
            by[By::Ours as usize],
            by[By::Heap as usize],
            by[By::Present as usize],
            by[By::Fence as usize],
            by[By::Mapping as usize],
            by[By::Pump as usize],
            by[By::Poll as usize],
            by[By::Host as usize],
            per(C::WritesNone),
            per(C::WritesDeclared),
            per(C::WritesUnbounded),
            per(C::ReadsExact),
            per(C::ReadsHulls),
            per(C::ReadsAll),
            ms(C::FlightUs),
            ms(C::WaitUs),
            per(C::StillRunning),
            per(C::FoundDone),
            ms(C::LateUs),
            ms(C::AfterUs),
            ms(C::WouldSync),
            ms(C::WouldUploads),
            ms(C::WouldRefresh),
            ms(C::WouldAll),
            per(C::Signals),
            per(C::Overlaid),
        ));
    }
    if checking() {
        if !line.is_empty() {
            line.push_str("; ");
        }
        line.push_str(&format!(
            "unwaited checker: {:.1} ms/frame, {:.1} MB compared, {:.1} ranges it could not check (unbounded writers, pages not \
             marked lately), {} violations so far",
            ms(C::CheckUs),
            per(C::CheckBytes) / 1e6,
            per(C::Unchecked),
            COUNTS[C::Violations as usize].load(Relaxed),
        ));
    }
    Some(line)
}

/// The checker found invariant `rule` broken: ends the run naming it
/// (`VICTORY_UNWAITED_CHECK=1`), or logs it (`=log`). Tests catch it.
pub(crate) fn violation(rule: &str, what: std::fmt::Arguments) {
    let n = COUNTS[C::Violations as usize].fetch_add(1, Relaxed);
    let text = format!("unwaited checker, rule {rule} (docs/unwaited.md): {what}");
    #[cfg(test)]
    if tests::CAUGHT.with(|c| c.borrow_mut().as_mut().map(|caught| caught.push(text.clone())).is_some()) {
        return;
    }
    if check_mode() == Check::Log {
        if n < 20 {
            tracing::error!("{text}");
        }
        return;
    }
    #[cfg(windows)]
    crate::host::fail(format_args!("{text}"));
    #[cfg(not(windows))]
    panic!("{text}");
}

/// Checker rules C1, C6 and C7: `what` (said by `detail`) is being done
/// while a flight is in the air, by a thread that is not settling it.
#[inline]
pub(crate) fn not_beside(rule: &str, what: &str, detail: impl FnOnce() -> String) {
    if in_flight() && checking() && !SETTLING.with(std::cell::Cell::get) {
        violation(rule, format_args!("{what} while a flight is in the air: {}", detail()));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    std::thread_local! {
        /// The checker switched on or off for this thread's test.
        pub(crate) static CHECK: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        /// Violations caught instead of ending the run.
        pub(crate) static CAUGHT: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
    }

    /// A protection undone for this thread's test (the deliberate breaks of
    /// `docs/unwaited.md` 11).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Break {
        None,
        /// Bottom-of-pipe words are written at the launch.
        WordsAtLaunch,
        /// A Signal is made at once with a flight in the air.
        SignalsAtOnce,
        /// The Signals kept are made last first.
        SignalsReversed,
        /// The flight does not hold its lists.
        ListsNotHeld,
        /// The settle does not wait for the host GPU.
        NoWait,
    }

    std::thread_local! {
        pub(crate) static BREAK: std::cell::Cell<Break> = const { std::cell::Cell::new(Break::None) };
    }

    /// A counter's value now (it starts again at each frame line).
    pub(crate) fn count(counter: C) -> u64 {
        COUNTS[counter as usize].load(Relaxed)
    }

    pub(crate) fn broken(what: Break) -> bool {
        BREAK.with(|b| b.get() == what)
    }

    /// Runs `f` with the checker on for this thread: what it found.
    pub(crate) fn caught<R>(f: impl FnOnce() -> R) -> (R, Vec<String>) {
        let before = (CHECK.with(|c| c.replace(Some(true))), CAUGHT.with(|c| c.replace(Some(Vec::new()))));
        let r = f();
        CHECK.with(|c| c.set(before.0));
        let found = CAUGHT.with(|c| c.replace(before.1)).unwrap_or_default();
        (r, found)
    }

    #[test]
    fn a_pending_value_is_read_through_and_a_timestamp_is_not_known() {
        let writes = [(0x1000, Written::Value(0x1122_3344_5566_7788), 8), (0x2000, Written::Value(7), 4), (0x3000, Written::GpuTimestamp, 8)];
        assert_eq!(overlaid(&writes, 0x1000, 8, Some(0)), Word::Value(0x1122_3344_5566_7788));
        assert_eq!(overlaid(&writes, 0x1000, 4, None), Word::Value(0x5566_7788));
        assert_eq!(overlaid(&writes, 0x1004, 4, None), Word::Value(0x1122_3344));
        // A four-byte write under an eight-byte wait: the rest is memory's.
        assert_eq!(overlaid(&writes, 0x2000, 8, Some(0xaaaa_bbbb_cccc_dddd)), Word::Value(0xaaaa_bbbb_0000_0007));
        assert_eq!(overlaid(&writes, 0x2000, 8, None), Word::Unknown);
        assert_eq!(overlaid(&writes, 0x3000, 8, Some(1)), Word::Unknown);
        assert_eq!(overlaid(&writes, 0x3004, 4, Some(1)), Word::Unknown);
        assert_eq!(overlaid(&writes, 0x4000, 8, Some(1)), Word::Untouched);
        assert_eq!(overlaid(&writes, 0x0ffc, 4, Some(1)), Word::Untouched);
    }

    #[test]
    fn later_writes_win_and_only_the_bytes_written_change() {
        let writes = [(0x1000, Written::Value(1), 8), (0x1000, Written::Value(2), 4), (0x1004, Written::SystemTimestamp, 4)];
        assert_eq!(overlaid(&writes, 0x1000, 4, Some(9)), Word::Value(2));
        // The low half is known; the high half was overwritten by a 4-byte
        // write taken as 4 bytes of a timestamp.
        assert_eq!(overlaid(&writes, 0x1000, 8, Some(9)), Word::Unknown);
        let writes = [(0x1000, Written::GpuTimestamp, 8), (0x1000, Written::Value(5), 8)];
        assert_eq!(overlaid(&writes, 0x1000, 8, None), Word::Value(5));
    }

    /// With no flight the overlay is memory itself, and nothing is kept.
    #[test]
    fn without_a_flight_nothing_is_kept_or_predicted() {
        if in_flight() {
            return;
        }
        assert_eq!(read_through(0x1000, 8, Some(3)), Some(3));
        assert_eq!(read_through(0x1000, 8, None), None);
        assert!(pending_fence(1).is_none() || in_flight());
    }

    #[test]
    fn the_line_is_made_only_with_a_switch_on() {
        if !on() && !checking() {
            assert!(take_line().is_none());
        }
        let (line, _) = caught(take_line);
        assert!(line.is_some_and(|l| l.contains("unwaited checker")));
    }
}
