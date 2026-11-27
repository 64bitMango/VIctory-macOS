//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Queues the lists' host lists behind what the queue holds, and runs what
/// may run (`crate::hold`; decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    #[cfg(not(windows))]
    let _ = this;
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let mut held = held();
        for &l in lists {
            for segment in crate::list::segments(l) {
                let waits = segment.waits();
                held.push(this as usize, Item::Segment(segment, waits));
            }
        }
        account().submissions += 1;
        pump(&mut held, Why::Submission);
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// What ran held work: the title's submission, a queue's Signal or Wait, a
/// CPU fence call, PresentX, or our timer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Why {
    #[cfg(windows)]
    Submission,
    #[cfg(windows)]
    Signal,
    #[cfg(windows)]
    Wait,
    Fence,
    Present,
    #[cfg(windows)]
    Timer,
}

#[cfg(windows)]
use crate::hold::Item;

/// Every queue's held work.
#[cfg(windows)]
type Holding = crate::hold::Held<std::sync::Arc<crate::host::HostList>>;

#[cfg(windows)]
static HELD: std::sync::Mutex<Holding> = std::sync::Mutex::new(crate::hold::Held::new());

#[cfg(windows)]
fn held() -> std::sync::MutexGuard<'static, Holding> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs held work that may run now (`why` for the frame line).
pub(crate) fn pump_held(why: Why) {
    #[cfg(windows)]
    pump(&mut held(), why);
    #[cfg(not(windows))]
    let _ = why;
}

/// The same unless another thread is at it (calls that poll: a fence's
/// value, our timer).
pub(crate) fn pump_held_if_free(why: Why) {
    #[cfg(windows)]
    if let Ok(mut held) = HELD.try_lock() {
        pump(&mut held, why);
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// How long a queue's head may wait before it runs anyway.
// shortcut: a safety valve against a wait nothing satisfies (a write we do
// not make), which would hold the title's fences until its 30 s watchdog
// ends the run; counted and logged, remove once runs show it never fires.
#[cfg(windows)]
const VALVE: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(windows)]
fn pump(held: &mut Holding, why: Why) {
    // Title pieces that return before the host GPU has run them
    // (`VICTORY_UNWAITED=1`; without it this function is as it was).
    if crate::unwaited::on() {
        return pump_unwaited(held, why);
    }
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    let read = read_value;
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| crate::fence::value(f as This);
    let started = std::time::Instant::now();
    let mut ran = std::time::Duration::ZERO;
    let mut run = |_: usize, segments: Vec<std::sync::Arc<crate::host::HostList>>| {
        let lists: Vec<&crate::host::HostList> = segments.iter().map(|s| &**s).collect();
        let at = std::time::Instant::now();
        host.execute(&lists);
        ran += at.elapsed();
        drop(lists);
        segments.into_iter().for_each(crate::list::recycle);
    };
    let mut signal = |f: usize, value: u64| {
        crate::fence::set(f as This, value);
        crate::fence::let_go(f as This);
    };
    let pumped = held.pump(&mut crate::hold::World {
        read: &read,
        fence: &fence,
        run: &mut run,
        signal: &mut signal,
    });
    if pumped.segments > 0 || pumped.unknown > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pumped.pieces;
        a.segments += pumped.segments;
        a.unknown += pumped.unknown;
        for (queue, waited) in pumped.waited {
            a.released.push((kind(queue), waited, why));
        }
    }
    // Once work stays held, a timer tests it every 2 ms: memory the title
    // writes from the CPU, or the valve, may let it go with no call of ours.
    if held.blocked(&read).is_empty() {
        return;
    }
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory held work".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(2));
            pump_held_if_free(Why::Timer);
        });
    });
}

/// [`pump`] with `VICTORY_UNWAITED` (`docs/unwaited.md` 4.3): a piece is
/// left on the host queue as the flight (`Host::execute_unwaited`, which
/// settles the flight before it first); a wait is tested against memory as
/// it will be once the flight's bottom-of-pipe words are written, and a
/// queue's fence wait against the fence as the Signals kept behind the
/// flight will leave it, since whatever they release is prepared after the
/// settle; a Signal reached while a flight is in the air is kept behind it.
/// A head still blocked with a flight in the air settles the flight (a
/// timestamp it may wait for is known only then) and is tested again.
///
/// KEEP IN STEP WITH [`pump`], whose copy this is.
#[cfg(windows)]
fn pump_unwaited(held: &mut Holding, why: Why) {
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    let read = |address: u64, bytes: u8| crate::unwaited::read_through(address, bytes, read_value(address, bytes));
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| {
        let now = crate::fence::value(f as This);
        crate::unwaited::pending_fence(f).map_or(now, |kept| kept.max(now))
    };
    loop {
        let started = std::time::Instant::now();
        let mut ran = std::time::Duration::ZERO;
        let mut run = |_: usize, segments: Vec<std::sync::Arc<crate::host::HostList>>| {
            let at = std::time::Instant::now();
            host.execute_unwaited(&segments);
            ran += at.elapsed();
            // The flight holds them until its settle.
            segments.into_iter().for_each(crate::list::recycle);
        };
        let mut signal = |f: usize, value: u64| {
            if crate::unwaited::keep_signal(f, value) {
                return;
            }
            crate::fence::set(f as This, value);
            crate::fence::let_go(f as This);
        };
        let pumped = held.pump(&mut crate::hold::World {
            read: &read,
            fence: &fence,
            run: &mut run,
            signal: &mut signal,
        });
        if pumped.segments > 0 || pumped.unknown > 0 {
            let mut a = account();
            a.pump += started.elapsed().saturating_sub(ran);
            a.pieces += pumped.pieces;
            a.segments += pumped.segments;
            a.unknown += pumped.unknown;
            for (queue, waited) in pumped.waited {
                a.released.push((kind(queue), waited, why));
            }
        }
        if held.blocked(&read).is_empty() {
            return;
        }
        if !crate::unwaited::in_flight() {
            break;
        }
        host.settle_flight(crate::unwaited::By::Pump);
        // Not settled (this thread is inside a submission): the poll does.
        if crate::unwaited::in_flight() {
            break;
        }
    }
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory held work".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(2));
            pump_held_if_free(Why::Timer);
        });
    });
}

/// The poll of `crate::unwaited`: settles the flight in the air if the host
/// GPU is done with it, runs held work its words or fences let go, and
/// gives back the lists of settled flights. Nothing if a thread is pumping.
#[cfg(windows)]
pub(crate) fn poll_unwaited() {
    if let Ok(mut held) = HELD.try_lock() {
        let settled = crate::host::host().is_some_and(|host| host.settle_flight_if_done(crate::unwaited::By::Poll));
        if crate::unwaited::take_wants_pump() || settled {
            pump(&mut held, Why::Timer);
        }
    }
    crate::unwaited::recycle();
}

/// Settles the flight in the air (`VICTORY_UNWAITED`), waiting for the host
/// GPU: before a call that must find its results, or the host idle. The
/// caller holds no lock of ours.
pub(crate) fn settle_unwaited(by: crate::unwaited::By) {
    #[cfg(windows)]
    if crate::unwaited::in_flight()
        && let Some(host) = crate::host::host()
    {
        host.settle_flight(by);
    }
    #[cfg(not(windows))]
    let _ = by;
}

/// The value a wait tests (4 or 8 bytes), if the memory can be read.
#[cfg(windows)]
fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
}

/// A queue's type, for logs (0 graphics, 2 compute, 3 copy). Titles do not
/// release queues that hold work.
#[cfg(windows)]
fn kind(queue: usize) -> u32 {
    self::queue(queue as This).kind
}

/// What queues did with held work since the last presented frame.
#[cfg(windows)]
#[derive(Default)]
struct Account {
    submissions: u32,
    pieces: u32,
    segments: u32,
    /// Segments that waited: queue type, how long, what ran them.
    released: Vec<(u32, std::time::Duration, Why)>,
    unknown: u32,
    forced: u32,
    mappings: u32,
    /// Time in the pump outside host submissions.
    pump: std::time::Duration,
}

#[cfg(windows)]
fn account() -> std::sync::MutexGuard<'static, Account> {
    static ACCOUNT: std::sync::OnceLock<std::sync::Mutex<Account>> = std::sync::OnceLock::new();
    ACCOUNT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame line of held work: what waited, how long and what released it,
/// and what is still held at the present, with the waits that do not hold.
#[cfg(windows)]
fn account_frame(frame: u64) {
    let (cuts, early) = crate::list::take_cuts();
    let (fence_waits, fence_waited) = crate::fence::take_waited();
    let a = std::mem::take(&mut *account());
    let blocked = held().blocked(&read_value);
    if a.released.is_empty() && blocked.is_empty() && a.forced == 0 && a.unknown == 0 && a.mappings == 0 {
        return;
    }
    let longest = a.released.iter().map(|r| r.1).max().unwrap_or_default();
    let mut by = [0u32; 6];
    for r in &a.released {
        by[r.2 as usize] += 1;
    }
    tracing::info!(
        frame,
        submissions = a.submissions,
        pieces = a.pieces,
        segments = a.segments,
        cuts,
        early_resets = early,
        waited = a.released.len(),
        longest_ms = longest.as_secs_f64() * 1e3,
        pump_ms = a.pump.as_secs_f64() * 1e3,
        fence_waits,
        fence_wait_ms = fence_waited.as_secs_f64() * 1e3,
        "held work this frame (submissions; host submissions, segments; lists cut at waits; lists reset or released \
         while held, each keeping its own host list; segments that waited, released by \
         submission/signal/wait/fence/present/timer {by:?}; time in the pump outside host submissions; events the          title waited on for fence values and their summed waits); held at the present: {}",
        blocked.len()
    );
    tracing::debug!(
        frame,
        "held work released this frame (queue type, ms, by): {:?}",
        a.released.iter().map(|(k, d, w)| (k, d.as_secs_f64() * 1e3, w)).collect::<Vec<_>>()
    );
    for (queue, items, waited, unmet) in &blocked {
        let level_warn = blocked.len() > 1;
        let unmet: Vec<String> = unmet
            .iter()
            .map(|w| format!("{:#x} cmp {} {:#x} (is {:?})", w.address, w.comparison, w.reference, read_value(w.address, w.bytes)))
            .collect();
        if level_warn {
            tracing::warn!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "two or more queues hold work at the present");
        } else {
            tracing::info!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "a queue holds work at the present");
        }
    }
    if a.forced > 0 || a.unknown > 0 || a.mappings > 0 {
        tracing::warn!(
            frame,
            forced = a.forced,
            unknown = a.unknown,
            mappings = a.mappings,
            "held work: run without its waits (valve), waits with an unknown comparison (taken as holding), \
             page mappings done ahead of held work"
        );
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they run, so it signals as Signal does.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// At once, or behind the work the queue holds (its fence, and the memory
/// the title reads it at, move only once that work has run).
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let mut held = held();
        if held.holds_work(this as usize) {
            crate::fence::keep(fence);
            held.push(this as usize, Item::Signal(fence as usize, value));
        } else if crate::unwaited::in_flight() {
            // Behind the flight (`VICTORY_UNWAITED`): made at its settle,
            // or now if it was settled meanwhile.
            crate::fence::keep(fence);
            if !crate::unwaited::keep_signal(fence as usize, value) {
                crate::fence::set(fence, value);
                crate::fence::let_go(fence);
            }
        } else {
            crate::fence::set(fence, value);
        }
        // Another queue may wait for this fence.
        pump(&mut held, Why::Signal);
    }
    #[cfg(not(windows))]
    {
        let _ = this;
        crate::fence::set(fence, value);
    }
    S_OK
}
/// The queue's later work waits until the fence reaches `value`.
///
/// shortcut: the fence is not kept alive while the queue waits for it;
/// the title makes no queue wait for a fence.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() && crate::fence::value(fence) < value {
        let mut held = held();
        held.push(this as usize, Item::FenceWait(fence as usize, value));
        pump(&mut held, Why::Wait);
    }
    #[cfg(not(windows))]
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(windows)]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// `VICTORY_FRAME_RATE_LOG=1`, in every build (diagnostics or not): every
/// 5 s a line with the frames presented so far and over the last 5 s, so a
/// build without diagnostics can be timed against one with them.
#[cfg(windows)]
fn start_frame_rate_log() {
    use std::time::{Duration, Instant};
    if !matches!(std::env::var("VICTORY_FRAME_RATE_LOG").as_deref().map(str::trim), Ok("1")) {
        return;
    }
    let started = std::thread::Builder::new().name("frame-rate-log".into()).spawn(|| {
        let mut before = (presented(), Instant::now());
        loop {
            std::thread::sleep(Duration::from_secs(5));
            let now = (presented(), Instant::now());
            let ms = now.1.duration_since(before.1).as_secs_f64() * 1e3;
            let frames = now.0 - before.0;
            let per_frame = if frames > 0 { ms / frames as f64 } else { 0.0 };
            tracing::info!(presented = now.0, frames, per_frame_ms = per_frame, "frame rate");
            before = now;
        }
    });
    if let Err(error) = started {
        tracing::warn!(%error, "could not start the frame-rate log");
    }
}

/// Count returns separately from entries so a quiet render thread can be
/// distinguished from a present call that has yet to return.
#[cfg(all(windows, feature = "diagnostics"))]
static PRESENT_RETURNS: AtomicU64 = AtomicU64::new(0);

#[cfg(all(windows, feature = "diagnostics"))]
fn start_progress_log() {
    use std::time::{Duration, Instant};
    if !crate::switch::on("VICTORY_PROGRESS_LOG") {
        return;
    }
    let result = std::thread::Builder::new().name("render-progress".into()).spawn(|| {
        let mut previous = PRESENT_RETURNS.load(Ordering::Relaxed);
        let mut last_progress = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs(15));
            let returned = PRESENT_RETURNS.load(Ordering::Relaxed);
            if returned != previous {
                previous = returned;
                last_progress = Instant::now();
            }
            let (cache_hits, cache_misses) = crate::shader_cache::counts();
            let (working_set_mb, commit_mb, system_load_pct, system_avail_mb) = memory_mb();
            tracing::info!(
                working_set_mb, commit_mb, system_load_pct, system_avail_mb,
                presents_started = presented(),
                presents_returned = returned,
                seconds_without_observed_progress = last_progress.elapsed().as_secs(),
                draws = crate::draw::drawn(),
                dispatches = crate::draw::dispatched(),
                pipelines_pending = crate::pipeline::making(),
                cache_hits, cache_misses,
                "render progress"
            );
        }
    });
    if let Err(error) = result {
        tracing::warn!(%error, "could not start render progress log");
    }
}

/// Wine's view of memory for the progress log: this process's working set
/// and committed private bytes, then the system's load and available
/// physical memory, in MiB (load in percent). Wine reports these values
/// itself; the macOS footprint can differ. Use an external watcher as the
/// reference.
#[cfg(all(windows, feature = "diagnostics"))]
fn memory_mb() -> (u64, u64, u32, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const MB: u64 = 1 << 20;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    // SAFETY: zeroed is a valid MEMORYSTATUSEX; its length is set before use.
    let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    s.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut s) };
    (
        c.WorkingSetSize as u64 / MB,
        c.PagefileUsage as u64 / MB,
        s.dwMemoryLoad,
        s.ullAvailPhys / MB,
    )
}

/// Every plane's resources. Only the start of a plane description is
/// known (token, count, resources), so the planes after the first are
/// found by their stride: the first offset past the first plane where the
/// same frame token is followed by a count of 1 to 16 and a list whose
/// first entry is one of our resources. All reads there are checked.
#[cfg(windows)]
fn plane_resources(planes: u32, plane: *const Plane) -> Vec<Vec<This>> {
    let read_u64 = |at: usize| {
        let mut b = [0u8; 8];
        crate::host::read_into(at as *const u8, &mut b).then(|| u64::from_le_bytes(b))
    };
    let list = |at: usize| -> Option<Vec<This>> {
        let count = read_u64(at + 8)? as u32;
        let resources = read_u64(at + 16)? as usize;
        if !(1..=16).contains(&count) || resources == 0 {
            return None;
        }
        let all: Option<Vec<This>> = (0..count as usize)
            .map(|i| read_u64(resources + 8 * i).map(|r| r as This))
            .collect();
        all.filter(|r| r.iter().all(|&r| crate::resource::is_resource(r)))
    };
    let base = plane as usize;
    if planes == 0 || base == 0 {
        return Vec::new();
    }
    let Some(first) = list(base) else {
        return Vec::new();
    };
    let mut all = vec![first];
    if planes > 1
        && let Some(token) = read_u64(base)
        && let Some(stride) = (24..=256)
            .step_by(8)
            .find(|&o| read_u64(base + o) == Some(token) && list(base + o).is_some())
    {
        for k in 1..planes.min(4) as usize {
            match list(base + k * stride) {
                Some(r) => all.push(r),
                None => break,
            }
        }
    }
    all
}

/// Logs what the title presents whenever it changes: each plane's
/// resources as (width, height, host format) (#15, #21: on Series S the
/// first plane is the 1920x1080 scene, the second the larger interface).
#[cfg(windows)]
fn log_planes(frame: u64, planes: u32, found: &[Vec<This>], shown: usize) {
    use std::sync::Mutex;
    /// A resource's width, height and host format.
    type Shown = (u64, u32, i32);
    static LAST: Mutex<Option<(u32, Vec<Vec<Shown>>)>> = Mutex::new(None);
    let sizes: Vec<Vec<Shown>> = found
        .iter()
        .map(|plane| {
            plane
                .iter()
                .map(|&r| {
                    crate::resource::host_resource(r).map_or((0, 0, -1), |t| {
                        // SAFETY: a live resource.
                        let d = unsafe { t.GetDesc() };
                        (d.Width, d.Height, d.Format.0)
                    })
                })
                .collect()
        })
        .collect();
    let now = Some((planes, sizes));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last != now {
        let (planes, sizes) = now.as_ref().expect("just set");
        tracing::info!(
            frame,
            planes,
            found = sizes.len(),
            shown,
            resources = ?sizes,
            "presented planes: each plane's resources as (width, height, format); the first resource of plane `shown` is shown"
        );
        *last = now;
    }
}

/// What a present shows: the plane whose first resource is the picture,
/// and the plane, if any, whose first resource is blended over it.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shown {
    plane: usize,
    overlay: Option<usize>,
}

/// What this present shows, from the planes found and the overrides
/// `VICTORY_PRESENT_PLANE=<n>` and `VICTORY_PRESENT_COMPOSITE=0|1`.
#[cfg(windows)]
fn shown_planes(found: &[Vec<This>]) -> Shown {
    static PLANE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let want = *PLANE.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_PLANE").ok().and_then(|v| v.parse().ok())
    });
    pick_planes(want, crate::composite::forced(), found.len())
}

/// What to show of `planes` planes. The title lists its planes from the
/// bottom up: plane 0 is the picture and a second plane is an overlay
/// drawn over it, at whatever size it has (#21: on Series S plane 0 is the
/// 1920x1080 scene and plane 1 the 3840x2160 interface, mostly
/// transparent). So plane 0 is shown, with plane 1 blended over it when
/// there are exactly two; one plane (Series X) is shown as it is.
///
/// `want` (`VICTORY_PRESENT_PLANE`) shows another plane when in range.
/// `composite` (`VICTORY_PRESENT_COMPOSITE`) forces the blend on (the
/// other of two planes over the shown one) or off; left unset, only plane
/// 0 of two gets plane 1 over it.
#[cfg(any(windows, test))]
fn pick_planes(want: Option<usize>, composite: Option<bool>, planes: usize) -> Shown {
    let plane = want.filter(|&w| w < planes).unwrap_or(0);
    let two = planes == 2;
    let overlay = match composite {
        Some(false) => None,
        Some(true) => two.then_some(1 - plane),
        None => (two && plane == 0).then_some(1),
    };
    Shown { plane, overlay }
}

/// Shows the first resource of one plane in our window, with another
/// plane's blended over it when the title presents two (`pick_planes`).
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // What may run before the frame is shown does.
    // shortcut: PresentX does not wait behind work the queue still holds (it
    // shows the frame as far as it ran); the frame line names such work.
    pump_held(Why::Present);
    // The flight in the air (`VICTORY_UNWAITED`) has run before the frame
    // is shown, and what its words let go too.
    if crate::unwaited::in_flight() {
        settle_unwaited(crate::unwaited::By::Present);
        pump_held(Why::Present);
        settle_unwaited(crate::unwaited::By::Present);
    }
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    account_frame(frame);
    #[cfg(feature = "diagnostics")]
    crate::frame::presented();
    #[cfg(all(windows, feature = "diagnostics"))]
    if frame == 1 {
        start_progress_log();
    }
    #[cfg(windows)]
    if frame == 1 {
        start_frame_rate_log();
    }
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    // The draws run so far, for the frame recorder's world frame
    // (`VICTORY_RECORD_AT=w<n>`): read where `capture::presented` reads
    // them, so that both tell the same frame as the world's first.
    #[cfg(windows)]
    let drawn = crate::draw::drawn();
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::presented(frame);
        if crate::capture::listed(frame) || crate::capture::has_targets() {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    let found = plane_resources(planes, plane_parameters);
    #[cfg(windows)]
    let Shown {
        plane: shown,
        overlay: overlay_plane,
    } = shown_planes(&found);
    #[cfg(windows)]
    log_planes(frame, planes, &found, shown);
    #[cfg(windows)]
    if let Some(&resource) = found.get(shown).and_then(|p| p.first())
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        // The overlay plane's first resource, blended over the picture.
        let other = overlay_plane
            .and_then(|o| found.get(o)?.first().copied())
            .filter(|&o| o != resource)
            .and_then(|o| Some((o, crate::resource::host_resource(o)?)));
        let overlay_state = other.as_ref().map(|(o, _)| {
            crate::resource::host_transition(*o, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before)
        });
        let overlay = other.as_ref().zip(overlay_state).map(|((_, t), s)| (t, s));
        let shown = host.present(&texture, state, overlay, frame);
        if let Some(((o, _), s)) = other.as_ref().zip(overlay_state) {
            crate::resource::host_transition(*o, s);
        }
        crate::resource::host_transition(resource, state);
        // The frame recorder's end of a frame (`VICTORY_RECORD`): the
        // picture is in `state` again.
        crate::replay::record::present(frame, Some((&texture, state)), drawn);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    } else {
        crate::replay::record::present(frame, None, drawn);
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    #[cfg(all(windows, feature = "diagnostics"))]
    PRESENT_RETURNS.fetch_add(1, Ordering::Relaxed);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. the title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once.
///
/// shortcut: done ahead of work the queue still holds (counted in the frame
/// line); queue it as an item like Signal if that count is not 0.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    // A queue operation on the console: behind the flight in the air
    // (`VICTORY_UNWAITED`), whose lists read the pages as they were mapped.
    if crate::unwaited::in_flight() {
        settle_unwaited(crate::unwaited::By::Mapping);
    }
    #[cfg(windows)]
    if held().holds_work(this as usize) {
        account().mappings += 1;
    }
    #[cfg(not(windows))]
    let _ = this;
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed")
        });
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// What gets shown. It only weighs the plane count and the overrides,
    /// so it runs on every host.
    #[test]
    fn plane_zero_is_shown_with_a_second_plane_over_it() {
        let shown = |plane, overlay| Shown { plane, overlay };
        // Series X: one plane, shown alone whatever is asked.
        for composite in [None, Some(false), Some(true)] {
            assert_eq!(pick_planes(None, composite, 1), shown(0, None));
            assert_eq!(pick_planes(Some(1), composite, 1), shown(0, None));
        }
        // Series S: the scene, then the interface over it.
        assert_eq!(pick_planes(None, None, 2), shown(0, Some(1)));
        assert_eq!(pick_planes(Some(0), None, 2), shown(0, Some(1)));
        assert_eq!(pick_planes(None, Some(false), 2), shown(0, None));
        assert_eq!(pick_planes(None, Some(true), 2), shown(0, Some(1)));
        // Another plane named: alone unless the blend is asked for.
        assert_eq!(pick_planes(Some(1), None, 2), shown(1, None));
        assert_eq!(pick_planes(Some(1), Some(true), 2), shown(1, Some(0)));
        assert_eq!(pick_planes(Some(1), Some(false), 2), shown(1, None));
        // A plane past those found falls back to plane 0.
        assert_eq!(pick_planes(Some(5), None, 2), shown(0, Some(1)));
        // Other counts: plane 0 alone (one overlay at most is blended).
        assert_eq!(pick_planes(None, None, 3), shown(0, None));
        assert_eq!(pick_planes(None, Some(true), 3), shown(0, None));
        assert_eq!(pick_planes(None, None, 0), shown(0, None));
    }

    /// Two planes 64 bytes apart (as a longer plane description would
    /// place them), each listing one of our textures, are both found; a
    /// list naming something else is not taken for a plane. This reads
    /// title memory through the host, so it runs where a host exists.
    #[cfg(windows)]
    #[test]
    fn planes_after_the_first_are_found_by_their_stride() {
        let texture = || {
            let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
            // SAFETY: a non-zero size; leaked, as title memory outlives a test.
            let address = unsafe { std::alloc::alloc_zeroed(layout) } as u64;
            let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x105, 0, 0];
            let iid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
            let mut out = std::ptr::null();
            let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
            assert_eq!(made, S_OK);
            out
        };
        let (a, b) = (texture(), texture());
        let (list_a, list_b) = (Box::new([a]), Box::new([b]));
        let mut planes = [0u64; 16];
        planes[0] = 7;
        planes[1] = 1;
        planes[2] = list_a.as_ptr() as u64;
        planes[8] = 7;
        planes[9] = 1;
        planes[10] = list_b.as_ptr() as u64;
        let found = plane_resources(2, planes.as_ptr().cast());
        assert_eq!(found, vec![vec![a], vec![b]]);
        // The second plane names a list of something that is not ours.
        let bogus = Box::new([planes.as_ptr() as This]);
        planes[10] = bogus.as_ptr() as u64;
        assert_eq!(plane_resources(2, planes.as_ptr().cast()), vec![vec![a]]);
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    /// `pump_unwaited` (`VICTORY_UNWAITED`), on queues' work of its own: a
    /// piece that waits for a word a flight will write at the bottom of the
    /// pipe, and one behind a queue's wait for a fence the flight's Signal
    /// will move, are run behind the flight: when they are prepared the
    /// word is in memory and the fence has moved. A piece that waits for a
    /// timestamp the flight will write cannot be told beforehand: the pump
    /// settles the flight and tests it again.
    #[cfg(windows)]
    #[test]
    fn work_waiting_for_what_a_flight_will_tell_runs_behind_it() {
        use crate::hold::ValueWait;
        use std::sync::{Arc, Mutex};
        let Some(host) = crate::host::host() else { return };
        let null = std::ptr::null();
        let queue = |kind: u32| {
            let mut q = null;
            assert_eq!(create(null, &QueueDesc { kind, priority: 0, flags: 0, nodes: 0 }, &QUEUE_IIDS[0], &mut q), S_OK);
            q as usize
        };
        let (qa, qb) = (queue(0), queue(2));
        let mut fence = null;
        assert_eq!(crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), S_OK);
        let fence = fence as usize;
        let words = Box::leak(Box::new([0u64; 3])).as_mut_ptr() as usize;
        let word = |i: usize| words + 8 * i;
        // SAFETY: (all reads) our leaked words.
        let read = |at: usize| unsafe { (at as *const u64).read_volatile() };
        let log: Arc<Mutex<Vec<(&'static str, u64)>>> = Default::default();
        let list = |write: Option<(usize, Written)>, says: &'static str, sees: Box<dyn Fn() -> u64 + Send>| {
            let list = host.new_list().expect("a host list");
            let at_preparation = log.clone();
            list.defer(move || at_preparation.lock().unwrap().push((says, sees())));
            if let Some((at, what)) = write {
                list.defer_write(at as *mut u8, what, 8);
            }
            list.close();
            Arc::new(list)
        };
        let (w0, w1, w2) = (word(0), word(1), word(2));
        let first = list(Some((w0, Written::Value(5))), "first prepared", Box::new(|| 0));
        first.defer_write(w2 as *mut u8, Written::GpuTimestamp, 8);
        let second = list(Some((w1, Written::Value(6))), "second prepared, the word it waited for", Box::new(move || read(w0)));
        let third = list(None, "third prepared, the fence its queue waited for", Box::new(move || crate::fence::value(fence as This)));
        let fourth = list(None, "fourth prepared, the timestamp it waited for is there", Box::new(move || u64::from(read(w2) != 0)));
        let wait = |address: usize, comparison: u32, reference: u64| ValueWait { address: address as u64, bytes: 8, comparison, reference, mask: u64::MAX };
        let mut held: Holding = crate::hold::Held::new();
        held.push(qa, Item::Segment(first, Vec::new()));
        crate::fence::keep(fence as This);
        held.push(qa, Item::Signal(fence, 7));
        // GREATER_EQUAL 5, then a queue's wait for the fence.
        held.push(qb, Item::Segment(second, vec![wait(w0, 7, 5)]));
        held.push(qb, Item::FenceWait(fence, 7));
        held.push(qb, Item::Segment(third, Vec::new()));
        pump_unwaited(&mut held, Why::Submission);
        assert!(held.is_empty(), "everything ran from one pump");
        // NOT_EQUAL 0 of the timestamp a flight will write.
        let stamped = list(Some((w2, Written::GpuTimestamp)), "stamped prepared", Box::new(|| 0));
        // SAFETY: w2 is an aligned timestamp slot in the test allocation.
        unsafe { (w2 as *mut u64).write_volatile(0) };
        held.push(qa, Item::Segment(stamped, Vec::new()));
        held.push(qa, Item::Segment(fourth, vec![wait(w2, 6, 0)]));
        pump_unwaited(&mut held, Why::Submission);
        assert!(held.is_empty(), "the pump settled the flight for the timestamp and ran what waited");
        settle_unwaited(crate::unwaited::By::Present);
        crate::unwaited::recycle();
        assert_eq!((read(w0), read(w1)), (5, 6));
        assert_eq!(crate::fence::value(fence as This), 7);
        assert_eq!(
            *log.lock().unwrap(),
            [
                ("first prepared", 0),
                ("second prepared, the word it waited for", 5),
                ("third prepared, the fence its queue waited for", 7),
                ("stamped prepared", 0),
                ("fourth prepared, the timestamp it waited for is there", 1),
            ]
        );
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// the title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
