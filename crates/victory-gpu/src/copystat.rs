//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
#[derive(Clone, Copy)]
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time in the kernel's write-watch answers, in the view check of the
    /// guard, in protecting; bytes and time of copies of memory neither
    /// watched nor guarded (copied whole).
    KernelUs,
    ViewQueryUs,
    ProtectUs,
    WholeBytes,
    WholeUs,
    /// Snapshots of ranges shaders write (bytes, time); page faults while
    /// preparing submissions and while copying back.
    SnapshotBytes,
    SnapshotUs,
    FaultsPrep,
    FaultsBack,
    /// Pages marked changed because a write landed on another mapping of
    /// them.
    AliasMarks,
    /// Time in the guard handler; protection changes making pages read-only
    /// (calls, pages) and giving write access back (pages).
    HandlerUs,
    ProtectCalls,
    ProtectPages,
    OpenPages,
    /// Tracked pages written (distinct, per frame), and of those the ones
    /// written the frame before too.
    WrittenPages,
    WrittenAgain,
    /// In change checks: time asking the runtime for other mappings, and
    /// waiting for the guard lock.
    AliasUs,
    GuardWaitUs,
    /// Preparing submissions: refresh batches done (copies and change
    /// checks); time in each kind of step (a step that flushes the batch
    /// counts that too); texture uploads for heaps; state preludes.
    FlushUs,
    StepRefreshUs,
    StepCopyUs,
    StepWorkUs,
    StepGpuWriteUs,
    StepTextureUs,
    StepHeapUs,
    StepMemoryUs,
    UploadsUs,
    PreludeUs,
    /// Copying back: bytes of copies compared with their snapshots, and
    /// the time.
    CompareBytes,
    CompareUs,
    /// Copying back pages shaders marked as written through views: pages
    /// compared, bytes written back, the time; pages that kept the title's
    /// bytes (it wrote them, their bytes from before unknown); pages whose
    /// bytes the title's first write saved.
    MarkedPages,
    MarkedBytes,
    MarkedUs,
    MarkedKept,
    SavedPages,
    /// Title copies into memory shown at two addresses, which make every
    /// range of the submission due again.
    AliasedCopies,
    /// Preparing submissions: descriptors the title copied into its heaps
    /// as bytes, made on the host first.
    SyncHeapsUs,
    /// Of those: heap slots looked at (every slot of each page written).
    SyncSlots,
    /// Time in copies redone piece by piece after meeting an unmapped page.
    FallbackUs,
    /// Title threads writing a view into a shader-visible heap: time
    /// waiting for the heaps' lock (held alone by passes over the slots;
    /// with `VICTORY_HEAPS_SHARED=0` by heap syncs and other writers too).
    MirrorWaitUs,
    /// Copies' remembered gaps (memory with nothing committed,
    /// `crate::shadow::Shadow::copy`): gaps found still uncommitted and
    /// zeroed with no read tried, their bytes; gaps forgotten because
    /// something was committed there.
    GapsKept,
    GapBytes,
    GapsDropped,
    /// Of the snapshot bytes: those copied again from an older snapshot of
    /// the same submission (a segment rebuilt as a range grew it).
    SnapshotAgain,
    /// Heap syncs: time reading the records of the pages written and
    /// finding those whose slot is to be made again, and how many were.
    SyncLookUs,
    SyncMade,
    /// Of the slots heap syncs made, by why: nothing was made there; the
    /// title's record names another view; the record names no view any
    /// more (`crate::descriptor::host::Due`).
    SyncFirst,
    SyncRewrote,
    SyncStale,
    /// And by what was made: nothing (the slot made null), a constant
    /// buffer's view, a placed texture's left null until read.
    SyncNulled,
    SyncConstant,
    SyncLeft,
    /// Waits for locks the recording threads and the submitting thread
    /// share (`crate::contend`): how often the heaps' lock was taken when a
    /// title thread writing a view asked ([`C::MirrorWaitUs`] is the time);
    /// time and times for a heap's slot lock, of heap syncs and of others;
    /// for the views table; for what a heap's slots view; for the copies'
    /// registries.
    HeapsContended,
    KeptWaitUs,
    KeptContended,
    ViewsWaitUs,
    ViewsContended,
    ViewedWaitUs,
    ViewedContended,
    CopiesWaitUs,
    CopiesContended,
    /// Time the heaps' lock was held alone by passes over the slots
    /// (`crate::descriptor::heaps_for_pass`).
    HeapsAloneUs,
    /// Views the title recorded that the views table held already, and new
    /// ones.
    RecordFound,
    RecordNew,
    /// A heap's slot locks (`crate::descriptor::host::HostHeap::made`), by
    /// who asked (title threads writing views, draws reading tables, the
    /// submitting thread): slots found made without a lock; locks taken;
    /// time waited and times one was found taken; and the time waited by
    /// who held it.
    SlotQuickTitle,
    SlotQuickDraw,
    SlotQuickSync,
    SlotTakesTitle,
    SlotTakesDraw,
    SlotTakesSync,
    SlotWaitTitleUs,
    SlotWaitDrawUs,
    SlotWaitSyncUs,
    SlotContendedTitle,
    SlotContendedDraw,
    SlotContendedSync,
    SlotBehindTitleUs,
    SlotBehindDrawUs,
    SlotBehindSyncUs,
    /// Heap syncs: runs of slots made under one hold of their lock; of the
    /// slots made null for a record that names no view, those whose record
    /// is a view of a released resource, zeros, other bytes; those made
    /// null so in the two frames before too; those that showed nothing
    /// already; such slots whose null view was kept (`VICTORY_NULL_ONCE`);
    /// and of the slots made for a new record, those a draw read through a
    /// table in the last eight frames.
    SyncRuns,
    SyncStaleReleased,
    SyncStaleZeros,
    SyncStaleOther,
    SyncStaleAgain,
    SyncStaleNothing,
    NullKept,
    SyncTabled,
    /// Write faults that found their block open already (another thread's
    /// fault opened it) and took no lock; blocks an asking left open and
    /// told written because they rested (`crate::watch::rest_ms`); the
    /// submissions whose refreshes could leave blocks so, and those that
    /// could not (their shaders write title memory at addresses they compute).
    GuardFaultsOpen,
    RestedPages,
    RestingSubmissions,
    ExactSubmissions,
    /// Heap syncs: time taking the changed pages of the heaps (guarding
    /// them again), asking whether a page is mapped, making what the views
    /// of a run's slots need, and making the run's slots under its lock.
    SyncTakeUs,
    SyncProbeUs,
    SyncPrepareUs,
    SyncMakeUs,
    /// The submitting thread's own waits (heap syncs, slots shaders read)
    /// for what a heap's slots view and for the views table.
    ViewedSyncWaitUs,
    ViewedSyncContended,
    ViewsSyncWaitUs,
    ViewsSyncContended,
    /// Of the records naming no view that are neither ours nor zeros
    /// ([`C::SyncStaleOther`]): a render target or depth view's record; a
    /// current view's entry and generation under another tag; a title
    /// address in the first eight bytes; the rest.
    SyncOtherTarget,
    SyncOtherTagged,
    SyncOtherAddress,
    SyncOtherRest,
    /// Table entries draws read that name no view: a view of a released
    /// resource, zeros, other bytes.
    DrawNoneReleased,
    DrawNoneZeros,
    DrawNoneOther,
    /// Not a counter: how many there are.
    Count,
}
const COUNTERS: usize = C::Count as usize;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

/// Snapshot bytes per host copy (its title start and length).
static SNAPSHOTS: Mutex<Option<HashMap<(u64, u64), u64>>> = Mutex::new(None);

/// A snapshot of `bytes` of the copy of `[start, start + len)`.
pub(crate) fn snapshot(start: u64, len: u64, bytes: u64) {
    *SNAPSHOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry((start, len))
        .or_default() += bytes;
}

/// Examples for the next line ([`sample`]).
static SAMPLES: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// An example for the next statistics line, if it has fewer than twelve.
pub(crate) fn sample(text: impl FnOnce() -> String) {
    let mut samples = SAMPLES.lock().unwrap_or_else(|e| e.into_inner());
    if samples.len() < 12 {
        samples.push(text());
    }
}

/// Our process's working set, bytes.
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.WorkingSetSize as u64
}

/// Our process's hard page faults so far (read from disk or the page
/// file), from the system's process list: `HardFaultCount` at +0x10 of
/// each entry, the process id at +0x50.
fn hard_faults() -> u64 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    const PROCESSES: u32 = 5;
    let id = std::process::id() as usize;
    let mut buf = vec![0u8; 4 << 20];
    let mut ret = 0;
    // SAFETY: a buffer of that length.
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) } < 0 {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap_or_default()));
        }
        if next == 0 || at + next + 0x58 > buf.len() {
            return 0;
        }
        at += next;
    }
}

pub(crate) fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// What else logs a line with the counters: called with the frames they are
/// over ([`also_log`]).
static MORE: std::sync::OnceLock<fn(f64)> = std::sync::OnceLock::new();

/// Has `more` log its own per-frame line whenever the copy statistics are
/// logged, given the frames since it was last called: for counters kept
/// where this crate cannot see them (the loader's emulated instructions).
pub fn also_log(more: fn(f64)) {
    let _ = MORE.set(more);
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB ({:.0} into memory \
         shown twice); {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks ({:.1} ms); {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
         guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
         {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
         asking for other mappings, {:.1} ms waiting for the guard lock; preparing: {:.1} ms in \
         refresh batches, steps (each with the batch it flushes): refresh {:.1} ms, copy {:.1} ms, \
         work {:.1} ms, GPU writes {:.1} ms, textures {:.1} ms, heaps {:.1} ms, title memory {:.1} ms; \
         heap texture uploads {:.1} ms, state preludes {:.1} ms, heap slots the title copied {:.1} ms ({:.0} slots), title threads waiting to write views {:.1} ms; copying back compared {:.1} MB          in {:.1} ms; marked pages: {:.0} compared, {:.1} MB written back in {:.1} ms, {:.0} kept the title's, {:.0} saved; working set {:.0} MB, \
         {:.0} hard faults",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::AliasedCopies as usize],
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        ms(c[C::FallbackUs as usize]),
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        ms(c[C::KernelUs as usize]),
        ms(c[C::ViewQueryUs as usize]),
        ms(c[C::ProtectUs as usize]),
        mb(c[C::WholeBytes as usize]),
        ms(c[C::WholeUs as usize]),
        mb(c[C::SnapshotBytes as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::FaultsPrep as usize],
        c[C::FaultsBack as usize],
        c[C::AliasMarks as usize],
        ms(c[C::HandlerUs as usize]),
        c[C::ProtectCalls as usize],
        c[C::ProtectPages as usize],
        c[C::OpenPages as usize],
        c[C::WrittenPages as usize],
        c[C::WrittenAgain as usize],
        ms(c[C::AliasUs as usize]),
        ms(c[C::GuardWaitUs as usize]),
        ms(c[C::FlushUs as usize]),
        ms(c[C::StepRefreshUs as usize]),
        ms(c[C::StepCopyUs as usize]),
        ms(c[C::StepWorkUs as usize]),
        ms(c[C::StepGpuWriteUs as usize]),
        ms(c[C::StepTextureUs as usize]),
        ms(c[C::StepHeapUs as usize]),
        ms(c[C::StepMemoryUs as usize]),
        ms(c[C::UploadsUs as usize]),
        ms(c[C::PreludeUs as usize]),
        ms(c[C::SyncHeapsUs as usize]),
        c[C::SyncSlots as usize],
        ms(c[C::MirrorWaitUs as usize]),
        mb(c[C::CompareBytes as usize]),
        ms(c[C::CompareUs as usize]),
        c[C::MarkedPages as usize],
        mb(c[C::MarkedBytes as usize]),
        ms(c[C::MarkedUs as usize]),
        c[C::MarkedKept as usize],
        c[C::SavedPages as usize],
        working_set() as f64 / 1e6,
        {
            static HARD: AtomicU64 = AtomicU64::new(0);
            let hard = hard_faults();
            (hard - HARD.swap(hard, Relaxed).min(hard)) as f64 / frames
        },
    );
    let held: Vec<String> = crate::watch::held_times()
        .iter()
        .map(|(line, total, longest)| format!("watch.rs:{line} {:.1} ms (longest {longest:.1})", total / frames))
        .collect();
    tracing::info!("  guard lock held per frame: {}", held.join(", "));
    tracing::info!(
        "  prepare waste per frame: uncommitted gaps of copies passed with one query {:.1} ({:.2} MB zeroed, no read tried), {:.1} forgotten (committed since); snapshots {:.1} MB, {:.1} MB of them copied again as their segment grew; heap syncs (the heaps' lock held throughout, {:.1} ms) looked at {:.0} slots in {:.1} ms and made {:.0}: {:.0} for the first time, {:.0} the title rewrote, {:.0} whose record names no view any more; of them {:.0} made null, {:.0} constant buffers, {:.0} placed textures left until read",
        c[C::GapsKept as usize],
        mb(c[C::GapBytes as usize]),
        c[C::GapsDropped as usize],
        mb(c[C::SnapshotBytes as usize]),
        mb(c[C::SnapshotAgain as usize]),
        ms(c[C::SyncHeapsUs as usize]),
        c[C::SyncSlots as usize],
        ms(c[C::SyncLookUs as usize]),
        c[C::SyncMade as usize],
        c[C::SyncFirst as usize],
        c[C::SyncRewrote as usize],
        c[C::SyncStale as usize],
        c[C::SyncNulled as usize],
        c[C::SyncConstant as usize],
        c[C::SyncLeft as usize],
    );
    tracing::info!(
        "  lock waits per frame (time, times a thread found the lock taken): title threads writing views for the heaps' lock {:.1} ms, {:.0} (held alone by passes over the slots {:.1} ms); for what a heap's slots hold {:.1} ms, {:.0}; for the views table {:.1} ms, {:.0} ({:.0} views recorded were held already, {:.0} new); for what a heap's slots view {:.1} ms, {:.0}; for the copies' registries {:.1} ms, {:.0}",
        ms(c[C::MirrorWaitUs as usize]),
        c[C::HeapsContended as usize],
        ms(c[C::HeapsAloneUs as usize]),
        ms(c[C::KeptWaitUs as usize]),
        c[C::KeptContended as usize],
        ms(c[C::ViewsWaitUs as usize]),
        c[C::ViewsContended as usize],
        c[C::RecordFound as usize],
        c[C::RecordNew as usize],
        ms(c[C::ViewedWaitUs as usize]),
        c[C::ViewedContended as usize],
        ms(c[C::CopiesWaitUs as usize]),
        c[C::CopiesContended as usize],
    );
    let by = |first: C| [0, 1, 2].map(|k| c[first as usize + k]);
    let ([quick_t, quick_d, quick_s], [takes_t, takes_d, takes_s]) = (by(C::SlotQuickTitle), by(C::SlotTakesTitle));
    let ([wait_t, wait_d, wait_s], [times_t, times_d, times_s]) = (by(C::SlotWaitTitleUs), by(C::SlotContendedTitle));
    let [behind_t, behind_d, behind_s] = by(C::SlotBehindTitleUs);
    tracing::info!(
        "  slot locks per frame (title threads writing views / draws reading tables / the submitting thread): slots found made without a lock {quick_t:.0} / {quick_d:.0} / {quick_s:.0}; locks taken {takes_t:.0} / {takes_d:.0} / {takes_s:.0}; waited {:.1} / {:.1} / {:.1} ms, found taken {times_t:.0} / {times_d:.0} / {times_s:.0} times; of the waiting {:.1} ms behind title threads, {:.1} behind draws, {:.1} behind the submitting thread",
        ms(wait_t),
        ms(wait_d),
        ms(wait_s),
        ms(behind_t),
        ms(behind_d),
        ms(behind_s),
    );
    tracing::info!(
        "  heap sync slots per frame: made under {:.0} holds of a run's lock; records naming no view: {:.0} views of released resources, {:.0} zeros, {:.0} other bytes, {:.0} at slots made null so in the two frames before too, {:.0} at slots that showed nothing already ({:.0} null views kept, VICTORY_NULL_ONCE); new records at slots a draw read through a table in the last eight frames: {:.0}",
        c[C::SyncRuns as usize],
        c[C::SyncStaleReleased as usize],
        c[C::SyncStaleZeros as usize],
        c[C::SyncStaleOther as usize],
        c[C::SyncStaleAgain as usize],
        c[C::SyncStaleNothing as usize],
        c[C::NullKept as usize],
        c[C::SyncTabled as usize],
    );
    tracing::info!(
        "  write faults per frame: {:.0} opened a block, {:.0} found it open (no lock taken), {:.2} ms in the handler in all (waits for the guard lock included); rest of {} ms: {:.0} blocks left open at an asking (each copied once more, not made read-only), {:.0} protect calls for {:.0} blocks; submissions that rested {:.1}, that could not {:.1} (their shaders write title memory at computed addresses)",
        c[C::GuardFaults as usize],
        c[C::GuardFaultsOpen as usize],
        ms(c[C::HandlerUs as usize]),
        crate::watch::rest_ms(),
        c[C::RestedPages as usize],
        c[C::ProtectCalls as usize],
        c[C::ProtectPages as usize],
        c[C::RestingSubmissions as usize],
        c[C::ExactSubmissions as usize],
    );
    if let Some(more) = MORE.get() {
        more(frames);
    }
    tracing::info!(
        "  heap sync time per frame: taking the heaps' changed pages {:.1} ms, asking whether pages are mapped {:.1} ms, making what views need {:.1} ms, making slots under their runs' locks {:.1} ms; the submitting thread waited {:.1} ms ({:.0} times) for what a heap's slots view and {:.1} ms ({:.0}) for the views table",
        ms(c[C::SyncTakeUs as usize]),
        ms(c[C::SyncProbeUs as usize]),
        ms(c[C::SyncPrepareUs as usize]),
        ms(c[C::SyncMakeUs as usize]),
        ms(c[C::ViewedSyncWaitUs as usize]),
        c[C::ViewedSyncContended as usize],
        ms(c[C::ViewsSyncWaitUs as usize]),
        c[C::ViewsSyncContended as usize],
    );
    tracing::info!(
        "  records that are neither ours nor zeros per frame: at slots heap syncs made null {:.0} render target or depth records, {:.0} a current view's entry under another tag, {:.0} with a title address first, {:.0} others; table entries draws read that name no view: {:.0} views of released resources, {:.0} zeros, {:.0} other bytes",
        c[C::SyncOtherTarget as usize],
        c[C::SyncOtherTagged as usize],
        c[C::SyncOtherAddress as usize],
        c[C::SyncOtherRest as usize],
        c[C::DrawNoneReleased as usize],
        c[C::DrawNoneZeros as usize],
        c[C::DrawNoneOther as usize],
    );
    for sample in std::mem::take(&mut *SAMPLES.lock().unwrap_or_else(|e| e.into_inner())) {
        tracing::info!("    for example: {sample}");
    }
    let mut snapshots: Vec<((u64, u64), u64)> =
        SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).take().into_iter().flatten().collect();
    snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let snapshots: Vec<String> = snapshots
        .iter()
        .take(6)
        .map(|((start, len), bytes)| format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames))
        .collect();
    tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

/// A global allocator that counts the bytes Rust code holds (the
/// statistics' "our heap"; the binary installs it in diagnostics builds).
pub struct CountingAlloc;

/// Bytes [`CountingAlloc`] has handed out and not taken back.
pub(crate) static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocations of at least this many bytes name their caller, once per
/// power-of-two size class (who holds the big blocks of our heap).
const LARGE: usize = 32 << 20;

/// Logs the stack of a large allocation of `size` bytes, the first of its
/// size class.
fn note_large(size: usize) {
    static SEEN: std::sync::Mutex<u64> = std::sync::Mutex::new(0);
    std::thread_local!(static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
    let class = 1u64 << size.ilog2().min(63);
    if INSIDE.with(|i| i.replace(true)) {
        return;
    }
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let first = *seen & class == 0;
        *seen |= class;
        first
    };
    if first {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = trace.lines().filter(|l| l.contains("victory")).take(24).collect();
        tracing::info!(mb = size >> 20, "a large heap allocation (first of its size class): {}", frames.join(" | "));
    }
    INSIDE.with(|i| i.set(false));
}

// SAFETY: every call goes to the system allocator unchanged; only counted.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's layout, passed on.
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: as above.
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
        HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
        // SAFETY: a block this allocator handed out with this layout.
        unsafe { std::alloc::System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: as above.
        let q = unsafe { std::alloc::System.realloc(p, layout, size) };
        if !q.is_null() {
            HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
            HEAP_BYTES.fetch_add(size as u64, Relaxed);
            if size >= LARGE {
                note_large(size);
            }
        }
        q
    }
}
