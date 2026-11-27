//! A title submission that returns before the host GPU has run it, and its
//! settle (`crate::unwaited`, `docs/unwaited.md`; `VICTORY_UNWAITED=1`).
//!
//! KEEP IN STEP WITH `Host::execute` (`crate::host`): [`Host::launch_held`]
//! is its preparation and [`Host::settle_held`] is what it does after its
//! wait, copied statement for statement so that the waited path itself is
//! not restructured. A change to one is to be carried to the other; the
//! test `both_paths_leave_the_same_bytes` runs the same submissions through
//! both and compares title memory and the copies.
//!
//! What differs from the copy, and only that: nothing is traced or timed per
//! list (such frames go to the waited code); the lists are not waited for
//! where they are submitted; what the waited path holds until its end is
//! held by the [`Flight`] until its settle.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::*;
use crate::unwaited::{By, C, add};

/// The shared upload list a flight runs (`crate::bindless::uploads`): in
/// its holder, which nothing resets before the flight is settled (I6;
/// checker rule C6 in `bindless::uploads`).
struct UploadList(*const HostList);

// SAFETY: a host list is used by one thread at a time, here under the
// submission lock.
unsafe impl Send for UploadList {}

/// A submission handed to the host queue and not yet settled.
pub(super) struct Flight {
    seq: u64,
    /// The host fence value that says it has run, and whether the host took
    /// the signal for it (if not, nothing tells: it is settled at once).
    value: u64,
    signalled: bool,
    /// The title's lists, held until the settle: a title list reset
    /// meanwhile leaves them to us (`crate::list::restart`).
    lists: Vec<Arc<HostList>>,
    uploads: Option<UploadList>,
    /// What `Host::execute` holds until its end.
    kept: Box<dyn std::any::Any + Send>,
    launched: Instant,
}

/// The flight in the air. Read and written under the submission lock only.
static FLIGHT: Mutex<Option<Flight>> = Mutex::new(None);

fn flight() -> std::sync::MutexGuard<'static, Option<Flight>> {
    FLIGHT.lock().unwrap_or_else(|e| e.into_inner())
}

/// When the poll saw the host GPU done with the flight in the air.
static DONE_AT: Mutex<Option<Instant>> = Mutex::new(None);

impl HostList {
    /// A shader of the list writes through views it takes from the heap
    /// itself: where is known only once it has run.
    pub(crate) fn note_heap_writes(&self) {
        self.deferred().heap_writes = true;
    }
}

/// Counts a flight by what its lists may write and read
/// (`docs/unwaited.md` 2).
fn count_footprints(runs: &[&HostList]) {
    let (mut declared, mut unbounded, mut hulls, mut all) = (false, false, false, false);
    for list in runs {
        let d = list.deferred();
        unbounded |= d.heap_writes;
        for step in &d.copies {
            match step {
                Step::GpuWrite(..) => declared = true,
                Step::Heap(_) => hulls = true,
                Step::Memory(_, writes) => {
                    all = true;
                    unbounded |= writes.load(Ordering::Relaxed);
                }
                _ => {}
            }
        }
    }
    add(if unbounded { C::WritesUnbounded } else if declared { C::WritesDeclared } else { C::WritesNone }, 1);
    add(if all { C::ReadsAll } else if hulls { C::ReadsHulls } else { C::ReadsExact }, 1);
}

impl Host {
    /// [`Host::execute`] for a piece of the title's with `VICTORY_UNWAITED`:
    /// the flight in the air is settled, the piece is prepared and handed
    /// to the host queue, and the call returns before the GPU has run it.
    pub(crate) fn execute_unwaited(&self, runs: &[Arc<HostList>]) {
        // Frames diagnostics time or trace go to the waited code, which
        // settles the flight first.
        #[cfg(feature = "diagnostics")]
        if timing() || crate::capture::listed(crate::queue::presented() + 1) {
            crate::unwaited::note_waited(crate::unwaited::Waited::Diagnostics);
            let lists: Vec<&HostList> = runs.iter().map(|l| &**l).collect();
            return self.execute(&lists);
        }
        {
            let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let _holding = Holding::enter();
            let stood = self.settle_held(By::Title, true).unwrap_or_default();
            self.launch_held(&mut submitted, runs, stood);
        }
        crate::unwaited::recycle();
        crate::unwaited::start_poll();
    }

    /// Settles the flight in the air, if any, waiting for the host GPU.
    /// Not on a thread inside a submission (it has settled it).
    pub(crate) fn settle_flight(&self, by: By) {
        if !crate::unwaited::in_flight() || holds_submissions() {
            return;
        }
        {
            let _submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let _holding = Holding::enter();
            self.settle_held(by, true);
        }
        crate::unwaited::recycle();
    }

    /// Settles the flight in the air if the host GPU is done with it and no
    /// thread is submitting: whether it did (calls that poll).
    pub(crate) fn settle_flight_if_done(&self, by: By) -> bool {
        if !crate::unwaited::in_flight() || holds_submissions() {
            return false;
        }
        let Ok(_submitted) = self.submitted.try_lock() else {
            return false;
        };
        let _holding = Holding::enter();
        // The fence decides, never the event (I8).
        let value = flight().as_ref().map(|f| f.value);
        // SAFETY: our fence.
        if value.is_some_and(|v| unsafe { self.fence.GetCompletedValue() } >= v) {
            DONE_AT.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Instant::now);
        }
        self.settle_held(by, false).is_some()
    }

    /// The preparation and submission of [`Host::execute`], the caller
    /// holding the submission lock with no flight in the air; the lists are
    /// left on the host queue as the flight. `stood`: how long the settle
    /// just before stood waiting for the GPU (for what this preparation
    /// could have overlapped).
    pub(super) fn launch_held(&self, submitted: &mut u64, title: &[Arc<HostList>], stood: Duration) {
        let own: Vec<&HostList> = title.iter().map(|l| &**l).collect();
        let runs = &own[..];
        count_footprints(runs);
        let copies_before = PREPARING[0].load(Ordering::Relaxed);
        // ---- Host::execute, from its start to its submission ----
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first: the last submission has run.
        #[cfg(feature = "diagnostics")]
        let evicting = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PrepEvictUs);
        crate::resource::evict_unused();
        #[cfg(feature = "diagnostics")]
        drop(evicting);
        let sync_started = Instant::now();
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        crate::descriptor::sync_heaps();
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        let sync_time = sync_started.elapsed();
        // Copies of title memory outside buffers nothing asked for in a
        // while are released, after the heaps' slots are what the title
        // wrote.
        #[cfg(feature = "diagnostics")]
        let sweeping = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PrepSweepUs);
        crate::shadow::idle_sweep();
        #[cfg(feature = "diagnostics")]
        drop(sweeping);
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // And textures read through tables, uploaded by the same list.
        let textures: Vec<_> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Texture(staging, copy) => Some((staging.clone(), copy.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let uploads_started = Instant::now();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty() || !textures.is_empty())
            .then(|| crate::bindless::uploads(&heaps, &textures))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        let uploads_time = uploads_started.elapsed();
        let runs: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &runs[..];
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch {
            rest: crate::watch::rest_ms() != 0 && !runs.iter().any(|list| list.writes_memory()),
            ..Batch::default()
        };
        #[cfg(feature = "diagnostics")]
        if crate::watch::rest_ms() != 0 {
            use crate::copystat::{C, add};
            add(if batch.rest { C::RestingSubmissions } else { C::ExactSubmissions }, 1);
        }
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        #[cfg(feature = "diagnostics")]
        let flushing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PrepLastFlushUs);
        batch.flush();
        #[cfg(feature = "diagnostics")]
        drop(flushing);
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::PrepareUs, (prepared - started).as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        // ---- end of the copy ----
        let _ = &firsts;
        *submitted += 1;
        let value = *submitted;
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, value, 0);
        // SAFETY: closed lists; the fence and event are ours.
        let signalled = unsafe {
            self.queue.ExecuteCommandLists(&lists);
            self.queue.Signal(&self.fence, value).and_then(|()| self.fence.SetEventOnCompletion(value, self.event))
        };
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        // What this preparation could have run beside the flight before it,
        // had the stage been allowed to (R1 to R3).
        let refresh_time = Duration::from_micros(PREPARING[0].load(Ordering::Relaxed).saturating_sub(copies_before));
        let us = |d: Duration| d.as_micros() as u64;
        add(C::WouldSync, us(stood.min(sync_time)));
        add(C::WouldUploads, us(stood.min(uploads_time)));
        add(C::WouldRefresh, us(stood.min(refresh_time)));
        add(C::WouldAll, us(stood.min(sync_time + uploads_time + refresh_time)));
        let writes: Vec<(usize, crate::queue::Written, usize)> = runs.iter().flat_map(|list| list.deferred().writes.clone()).collect();
        // Tests: a protection undone (`crate::unwaited::tests::Break`).
        #[cfg(test)]
        if crate::unwaited::tests::broken(crate::unwaited::tests::Break::WordsAtLaunch) {
            for &(at, what, bytes) in &writes {
                what.write(at as *mut u8, bytes);
            }
        }
        #[cfg(test)]
        let title = if crate::unwaited::tests::broken(crate::unwaited::tests::Break::ListsNotHeld) { &[][..] } else { title };
        let seq = crate::unwaited::launched(writes);
        *DONE_AT.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *flight() = Some(Flight {
            seq,
            value,
            signalled: signalled.is_ok(),
            lists: title.to_vec(),
            uploads: uploads.as_ref().map(|u| UploadList(u.list() as *const HostList)),
            kept: Box::new((batch, heaps, textures)),
            launched: Instant::now(),
        });
        // The upload list's lock is let go on this thread; the list stays in
        // its holder, running, until the settle.
        drop(uploads);
        if let Err(e) = signalled {
            tracing::warn!(error = %e, "host submission not waited for");
            self.settle_held(By::Host, false);
        }
    }

    /// Settles the flight in the air, the caller holding the submission
    /// lock: waits for the host GPU (`wait`; without it only a flight the
    /// GPU is done with is settled), then does what [`Host::execute`] does
    /// after its wait, then makes the Signals kept behind the flight. None
    /// if nothing was settled, else how long it stood waiting for the GPU.
    pub(super) fn settle_held(&self, by: By, wait: bool) -> Option<Duration> {
        let flight = {
            let mut slot = flight();
            let f = slot.as_ref()?;
            // SAFETY: our fence.
            let done = !f.signalled || unsafe { self.fence.GetCompletedValue() } >= f.value;
            if !done && !wait {
                return None;
            }
            slot.take()?
        };
        let started = Instant::now();
        add(C::FlightUs, (started - flight.launched).as_micros() as u64);
        // The fence decides (its event may never be set, `Host::wait_for`).
        // SAFETY: our fence.
        let running = flight.signalled && unsafe { self.fence.GetCompletedValue() } < flight.value;
        #[cfg(test)]
        let running = running && !crate::unwaited::tests::broken(crate::unwaited::tests::Break::NoWait);
        if running {
            add(C::StillRunning, 1);
            self.wait_for(flight.value);
        } else {
            add(C::FoundDone, 1);
            if let Some(at) = DONE_AT.lock().unwrap_or_else(|e| e.into_inner()).take() {
                add(C::LateUs, at.elapsed().as_micros() as u64);
            }
        }
        let stood = started.elapsed();
        add(C::WaitUs, stood.as_micros() as u64);
        SUBMISSION_US[1].fetch_add(stood.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, flight.value, 0);
        // SAFETY: our fence.
        if crate::unwaited::checking() && flight.signalled && unsafe { self.fence.GetCompletedValue() } < flight.value {
            crate::unwaited::violation("C2", format_args!("flight {} is being landed before the host fence reached {} (I2)", flight.seq, flight.value));
        }
        let own: Vec<&HostList> = flight
            .uploads
            .iter()
            // SAFETY: the shared upload list, not reset since the launch.
            .map(|u| unsafe { &*u.0 })
            .chain(flight.lists.iter().map(|l| &**l))
            .collect();
        let runs = &own[..];
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after an unwaited submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, None)
            ));
        }
        // From here to the end this thread's writes are the landing.
        let _settling = crate::unwaited::settling();
        let landing = Instant::now();
        // ---- Host::execute, after its wait ----
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: crate::quick::Map<(usize, usize, usize), usize> = Default::default();
        // Shaders that read memory with no copy computed something else than
        // the console's: it stays out of title memory, where it broke the
        // title's own data (runs c1, c4), and so does what every program
        // using title memory wrote while any misses
        // (`crate::memory::withholding`). Only lists with such programs have
        // checks; all are asked before anything goes back. Other programs in
        // those lists write through bounded views from their own inputs:
        // their ranges go back (the imposter transforms beside the BVH
        // builder in one list, run light13).
        #[cfg(feature = "diagnostics")]
        let asking = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::BackMissedUs);
        let missed: Vec<bool> = runs
            .iter()
            .map(|list| list.deferred().missed.iter().any(|check| check()))
            .collect();
        #[cfg(feature = "diagnostics")]
        drop(asking);
        #[cfg(feature = "diagnostics")]
        let gathering = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::BackGatherUs);
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        for (list, &missed) in runs.iter().zip(&missed) {
            let deferred = list.deferred();
            let missed = missed || withholding && !deferred.missed.is_empty();
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _, memory) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    // A range both kinds declare is withheld whole.
                    let goes_back = !(missed && *memory);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= goes_back,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, goes_back));
                        }
                    }
                }
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held.iter().any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        #[cfg(feature = "diagnostics")]
        drop(gathering);
        crate::shadow::copy_back_marks(&withheld, withholding);
        // Texture views left null that shaders read: made for what follows.
        #[cfg(feature = "diagnostics")]
        let reading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::BackReadsUs);
        let read = crate::shadow::take_reads();
        if !read.is_empty() {
            crate::descriptor::make_read(&read);
        }
        #[cfg(feature = "diagnostics")]
        drop(reading);
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            let written = crate::shadow::copy_back_declared(&back);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        let after = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::BackAfterUs);
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        #[cfg(feature = "diagnostics")]
        drop(after);
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        // The lists have run: the copies they used may be released when idle.
        for list in runs {
            list.deferred().flying.clear();
        }
        // ---- end of the copy ----
        release_retired();
        release_later(Box::new((flight.kept, back, withheld)));
        self.log_messages();
        // Everything the flight owed is paid: the Signals kept behind it.
        let read = |address: u64, bytes: u8| {
            let mut value = [0u8; 8];
            read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
        };
        crate::unwaited::landed(flight.seq, &read, |fence, value| {
            crate::fence::set(fence as crate::object::This, value);
            crate::fence::let_go(fence as crate::object::This);
        });
        add(C::AfterUs, landing.elapsed().as_micros() as u64);
        crate::unwaited::note_settled(by);
        drop(own);
        crate::unwaited::recycle_later(flight.lists);
        Some(stood)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::This;
    use crate::queue::Written;
    use crate::unwaited::tests::{BREAK, Break, caught};
    use crate::watch::PAGE;
    use std::sync::atomic::AtomicBool;

    const PAGES: usize = 96;
    /// Bytes of the buffers a list copies between to keep the GPU busy.
    const HEAVY: u64 = 16 << 20;

    type Log = Arc<Mutex<Vec<&'static str>>>;

    /// A title buffer of tracked memory with its copy, the host buffer the
    /// "shaders" of its lists copy from (real GPU copies into the copy's
    /// buffer, and into the marks' bitmap), and what the title watches.
    struct Scene {
        p: usize,
        copy: Arc<crate::shadow::Shadow>,
        src: ID3D12Resource,
        heavy: [ID3D12Resource; 2],
        /// The page whose mark the GPU writes, and its dword in the bitmap.
        k: usize,
        mark: Option<(ID3D12Resource, u64)>,
        words: Box<[u64; 4]>,
        fence: This,
    }

    /// Where the title's own thread writes while submissions run (offset,
    /// last value): bytes no list writes, in the pages shaders write and
    /// elsewhere, each 4 KiB or more from the shaders' bytes. Of the marked
    /// range it writes the second page, which the shaders do not: a title
    /// write to a marked page between its refresh and its arming keeps the
    /// title's bytes and drops the shaders' there, waited for or not (the
    /// page is "changed" and nothing was saved), so its timing would decide
    /// the bytes. The page the shaders mark is written once, at a known
    /// point ([`MARKED_SPOT`]).
    fn spots(k: usize) -> [(usize, u8); 6] {
        [
            (2 * PAGE + 30000, 0x51),
            (3 * PAGE + 40000, 0x52),
            ((k + 1) * PAGE + 30000, 0x53),
            (5 * PAGE + 4000, 0x54),
            (7 * PAGE + 9, 0x55),
            (2 * PAGE + 8192, 0x56),
        ]
    }

    /// The title's one write into the page shaders mark: its offset in the
    /// page and its value.
    const MARKED_SPOT: (usize, u8) = (20000, 0x66);

    /// What the second list copies into title memory at its place in the
    /// queue's order.
    static COPIED: [u8; 32] = [0xc5; 32];

    impl Scene {
        fn new(host: &Host) -> Option<Scene> {
            let p = crate::watch::tests::pool(PAGES);
            let len = (PAGES * PAGE) as u64;
            crate::shadow::register(p, p as u64, len);
            let made = (|| {
                let copy = crate::shadow::find(p as u64, len)?;
                let (src, data) = host.shared_buffer(4096)?;
                // The page of 32..64 whose mark is the first bit of a dword:
                // the whole dword is this copy's, so a GPU write of it
                // touches no other test's marks.
                let found = (32..64).find_map(|k| {
                    let (bitmap, at, bit) = copy.test_mark((p + k * PAGE) as u64)?;
                    (bit == 1).then_some((k, (bitmap, at)))
                });
                let k = found.as_ref().map_or(40, |f| f.0);
                // SAFETY: the mapped buffer of 4096 bytes.
                unsafe {
                    for i in 0..1024 {
                        data.add(i).write(0x80 | (i as u8 & 0x3f));
                    }
                    data.add(1024).cast::<u32>().write(1);
                }
                let heavy = [host.shared_buffer(HEAVY)?.0, host.shared_buffer(HEAVY)?.0];
                let mut fence = std::ptr::null();
                crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence);
                Some(Scene { p, copy, src, heavy, k, mark: found.map(|f| f.1), words: Box::new([0; 4]), fence })
            })();
            if made.is_none() {
                crate::shadow::unregister(p, p as u64);
                crate::watch::tests::free(p);
            }
            made
        }

        fn word(&self, i: usize) -> u64 {
            // SAFETY: our own words, written by bottom-of-pipe writes.
            unsafe { std::ptr::from_ref(&self.words[i]).read_volatile() }
        }

        fn byte(&self, at: usize) -> u8 {
            // SAFETY: a byte of our pool.
            unsafe { ((self.p + at) as *const u8).read_volatile() }
        }

        /// The title writes [`MARKED_SPOT`].
        fn write_marked_spot(&self) {
            // SAFETY: a byte of our pool.
            unsafe { ((self.p + self.k * PAGE + MARKED_SPOT.0) as *mut u8).write_volatile(MARKED_SPOT.1) };
        }

        fn title(&self) -> Vec<u8> {
            (0..PAGES * PAGE).map(|at| self.byte(at)).collect()
        }

        /// The first list: reads the buffer, and its "shaders" write an
        /// unmarked declared range (pages 2 and 3) and a marked one (from
        /// page `k`), after `heavy` large copies; a word at the bottom of
        /// the pipe.
        fn first(&self, host: &Host, heavy: u32, log: &Log) -> Arc<HostList> {
            let list = host.new_list().expect("a host list");
            let p = self.p as u64;
            self.copy.refresh(&list, p, (PAGES * PAGE) as u64);
            self.copy.gpu_writes(&list, p + 2 * PAGE as u64, 2 * PAGE as u64, false);
            self.copy.gpu_writes(&list, p + (self.k * PAGE) as u64, 8 * PAGE as u64, self.mark.is_some());
            // SAFETY: an open list and live buffers; the ranges are inside them.
            unsafe {
                for _ in 0..heavy {
                    list.list.CopyBufferRegion(&self.heavy[1], 0, &self.heavy[0], 0, HEAVY);
                }
                list.list.CopyBufferRegion(&self.copy.buffer, self.copy.offset(p + 2 * PAGE as u64 + 100), &self.src, 0, 64);
                list.list.CopyBufferRegion(&self.copy.buffer, self.copy.offset(p + (self.k * PAGE) as u64 + 8), &self.src, 64, 64);
                if let Some((bitmap, at)) = &self.mark {
                    list.list.CopyBufferRegion(bitmap, *at, &self.src, 1024, 4);
                }
            }
            list.defer_write(std::ptr::from_ref(&self.words[0]) as *mut u8, Written::Value(0xa), 8);
            let log = log.clone();
            list.defer_after(move || log.lock().unwrap().push("first landed"));
            list.close();
            Arc::new(list)
        }

        /// The second list: a queue-ordered title copy, and "shaders" that
        /// write the first one's unmarked range again, 8 of its bytes too.
        fn second(&self, host: &Host, heavy: u32, log: &Log) -> Arc<HostList> {
            let list = host.new_list().expect("a host list");
            let p = self.p as u64;
            self.copy.refresh(&list, p, (PAGES * PAGE) as u64);
            list.defer_copy((self.p + 5 * PAGE + 16) as *mut u8, COPIED.as_ptr(), COPIED.len());
            self.copy.refresh(&list, p, (PAGES * PAGE) as u64);
            self.copy.gpu_writes(&list, p + 2 * PAGE as u64, 2 * PAGE as u64, false);
            // SAFETY: as in `first`.
            unsafe {
                for _ in 0..heavy {
                    list.list.CopyBufferRegion(&self.heavy[1], 0, &self.heavy[0], 0, HEAVY);
                }
                list.list.CopyBufferRegion(&self.copy.buffer, self.copy.offset(p + 2 * PAGE as u64 + 156), &self.src, 200, 64);
                list.list.CopyBufferRegion(&self.copy.buffer, self.copy.offset(p + 3 * PAGE as u64 + 7), &self.src, 300, 16);
            }
            list.defer_write(std::ptr::from_ref(&self.words[1]) as *mut u8, Written::Value(0xb), 8);
            list.defer_write(std::ptr::from_ref(&self.words[2]) as *mut u8, Written::Value(0xb2), 4);
            let log = log.clone();
            list.defer_after(move || log.lock().unwrap().push("second landed"));
            list.close();
            Arc::new(list)
        }

        /// The title's thread: writes its spots again and again until told
        /// to stop, then their last values.
        fn racer(&self, seed: u64, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<()> {
            let (p, k) = (self.p, self.k);
            std::thread::spawn(move || {
                let mut n = seed;
                // SAFETY: (all writes) bytes of the scene's pool, alive until joined.
                let write = |at: usize, v: u8| unsafe { ((p + at) as *mut u8).write_volatile(v) };
                while !stop.load(Ordering::Relaxed) {
                    n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let (at, _) = spots(k)[(n >> 33) as usize % 6];
                    write(at, (n >> 40) as u8 | 1);
                    match (n >> 20) % 4 {
                        0 => std::thread::yield_now(),
                        1 => std::thread::sleep(Duration::from_micros((n >> 50) % 300)),
                        _ => {}
                    }
                }
                for (at, last) in spots(k) {
                    write(at, last);
                }
            })
        }

        /// After everything: the copy brought up to date holds title
        /// memory's bytes (no change was lost by a tracker).
        fn copy_is_current(&self, host: &Host) -> bool {
            host.between_submissions(|| self.copy.update(self.p, PAGES * PAGE));
            self.copy.test_bytes(self.p, PAGES * PAGE) == self.title()
        }

        fn done(self) {
            let p = self.p;
            crate::fence::let_go(self.fence);
            drop(self.copy);
            crate::shadow::unregister(p, p as u64);
            crate::watch::tests::free(p);
        }
    }

    /// Waits until the host GPU has run everything submitted so far (tests
    /// that undo a wait must not free what the GPU still uses).
    fn idle(host: &Host) {
        let value = *host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: our fence.
        while unsafe { host.fence.GetCompletedValue() } < value {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The two lists through `Host::execute`.
    fn waited(scene: &Scene, host: &Host, heavy: u32, log: &Log) {
        let (a, b) = (scene.first(host, heavy, log), scene.second(host, heavy, log));
        scene.write_marked_spot();
        host.execute(&[&a]);
        crate::fence::set(scene.fence, 7);
        crate::fence::set(scene.fence, 9);
        host.execute(&[&b]);
    }

    /// The two lists as flights, with the submission lock held throughout so
    /// that no other test's submission settles them: what was not as
    /// expected at each point (`broke`: a protection undone meanwhile).
    fn flown(scene: &Scene, host: &Host, heavy: u32, log: &Log, broke: Break) -> Vec<String> {
        let mut wrong = Vec::new();
        let mut expect = |ok: bool, what: &str| {
            if !ok {
                wrong.push(what.to_owned());
            }
        };
        let (a, b) = (scene.first(host, heavy, log), scene.second(host, heavy, log));
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let _holding = Holding::enter();
            host.settle_held(By::Ours, true);
            BREAK.with(|x| x.set(broke));
            host.launch_held(&mut submitted, std::slice::from_ref(&a), Duration::ZERO);
            // The title writes the page the shaders mark, armed by now.
            scene.write_marked_spot();
            // Two queue Signals behind it, as `crate::queue` makes them.
            for value in [7, 9] {
                crate::fence::keep(scene.fence);
                if !crate::unwaited::keep_signal(scene.fence as usize, value) {
                    crate::fence::set(scene.fence, value);
                    crate::fence::let_go(scene.fence);
                }
            }
            expect(crate::unwaited::in_flight(), "a flight in the air after the launch");
            expect(scene.word(0) == 0, "the first word unwritten while its list is in the air");
            expect(scene.byte(2 * PAGE + 100) == 0, "the shaders' bytes not in title memory while their list is in the air");
            expect(crate::fence::value(scene.fence) == 0, "the fence unmoved while the list before its Signals is in the air");
            expect(crate::unwaited::pending_fence(scene.fence as usize) == Some(9), "the fence's value to come known to the pump");
            expect(Arc::strong_count(&a) == 2, "the flight holds its list");
            let stood = host.settle_held(By::Title, true);
            expect(stood.is_some(), "a flight to settle");
            expect(!crate::unwaited::in_flight(), "no flight after the settle");
            expect(scene.word(0) == 0xa, "the first word written by the settle");
            expect(scene.byte(2 * PAGE + 100) == 0x80, "the shaders' bytes in title memory after the settle");
            expect(crate::fence::value(scene.fence) == 9, "the fence at its last Signal after the settle");
            host.launch_held(&mut submitted, std::slice::from_ref(&b), stood.unwrap_or_default());
            expect(scene.word(1) == 0 && scene.word(2) == 0, "the second list's words unwritten while it is in the air");
            expect(scene.byte(5 * PAGE + 16) == 0xc5, "the second list's title copy made at its preparation, after the first landed");
            host.settle_held(By::Present, true);
            expect(scene.word(1) == 0xb && scene.word(2) == 0xb2, "the second list's words written by its settle");
            BREAK.with(|x| x.set(Break::None));
        }
        idle(host);
        crate::unwaited::recycle();
        wrong
    }

    /// The same two submissions, with shader writes (declared and marked),
    /// bottom-of-pipe words, Signals and a title thread writing the same
    /// pages throughout, through `Host::execute` and as flights: title
    /// memory, the words and the fence end the same, byte for byte, and
    /// the copies are current. Many times, with the title thread's timing
    /// and the GPU's load varied; every other round through
    /// `Host::execute_unwaited`, where other tests' submissions may settle
    /// the flights.
    #[test]
    fn both_paths_leave_the_same_bytes() {
        let Some(host) = host() else { return };
        let rounds = std::env::var("VICTORY_UNWAITED_TEST_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(12u64);
        // Rounds whose shaders marked a page, and whose flight was still
        // on the GPU when its settle began.
        let (mut marked, mut flying) = (0, 0);
        for round in 0..rounds {
            let (Some(w), Some(u)) = (Scene::new(host), Scene::new(host)) else { return };
            marked += u32::from(w.mark.is_some() && u.mark.is_some());
            let waits_before = crate::unwaited::tests::count(C::StillRunning);
            let heavy = (round % 4) as u32 * 2;
            let (log_w, log_u): (Log, Log) = Default::default();
            let stop = Arc::new(AtomicBool::new(false));
            let racers = [w.racer(round, stop.clone()), u.racer(round ^ 0x55, stop.clone())];
            waited(&w, host, heavy, &log_w);
            if round % 2 == 0 {
                let (wrong, found) = caught(|| flown(&u, host, heavy, &log_u, Break::None));
                assert!(wrong.is_empty(), "round {round}: {wrong:?}");
                assert!(found.is_empty(), "round {round}: the checker found {found:?}");
            } else {
                let (a, b) = (u.first(host, heavy, &log_u), u.second(host, heavy, &log_u));
                host.execute_unwaited(std::slice::from_ref(&a));
                u.write_marked_spot();
                // The Signals a queue makes behind it.
                for value in [7, 9] {
                    crate::fence::keep(u.fence);
                    if !crate::unwaited::keep_signal(u.fence as usize, value) {
                        crate::fence::set(u.fence, value);
                        crate::fence::let_go(u.fence);
                    }
                }
                host.execute_unwaited(std::slice::from_ref(&b));
                host.settle_flight(By::Present);
                // Another test's submission may have launched in between.
                idle(host);
                assert_eq!(u.word(1), 0xb, "round {round}: settled");
            }
            stop.store(true, Ordering::Relaxed);
            racers.into_iter().for_each(|r| r.join().unwrap());
            // The marked range starts where each buffer's mark bits put it:
            // the pages before it and the range itself are compared, and
            // nothing else was written.
            let (tw, tu) = (w.title(), u.title());
            let picture = |scene: &Scene, title: &[u8]| {
                let range = scene.k * PAGE..(scene.k + 8) * PAGE;
                let written: Vec<u8> = title[..32 * PAGE].iter().chain(&title[range.clone()]).copied().collect();
                let rest = (32 * PAGE..PAGES * PAGE).filter(|at| !range.contains(at)).all(|at| title[at] == 0);
                (written, rest)
            };
            let ((pw, rest_w), (pu, rest_u)) = (picture(&w, &tw), picture(&u, &tu));
            assert!(rest_w && rest_u, "round {round}: bytes written outside what the lists and the title write");
            if let Some(at) = (0..pw.len()).find(|&i| pw[i] != pu[i]) {
                panic!("round {round}: title memory differs at page {} (the marked range from 32 on) offset {:#x}: waited {:#04x}, unwaited {:#04x}", at / PAGE, at % PAGE, pw[at], pu[at]);
            }
            // What they must both hold: the shaders' bytes, the later
            // list's where both wrote, the title copy, the title's own.
            assert_eq!(tu[2 * PAGE + 100], 0x80, "round {round}");
            assert_eq!(tu[2 * PAGE + 156], 0x80 | (200 & 0x3f), "round {round}: the second list's byte over the first's");
            assert_eq!(tu[u.k * PAGE + 8], 0x80 | 64, "round {round}: the marked page's bytes");
            assert_eq!(tu[u.k * PAGE + MARKED_SPOT.0], MARKED_SPOT.1, "round {round}: the title's byte in the marked page");
            assert_eq!(tu[5 * PAGE + 16], 0xc5, "round {round}");
            for (at, last) in spots(u.k) {
                assert_eq!(tu[at], last, "round {round}: the title's own byte at {at:#x}");
            }
            assert_eq!((w.word(0), w.word(1), w.word(2)), (u.word(0), u.word(1), u.word(2)), "round {round}");
            assert_eq!((u.word(0), u.word(1), u.word(2)), (0xa, 0xb, 0xb2), "round {round}");
            assert_eq!((crate::fence::value(w.fence), crate::fence::value(u.fence)), (9, 9), "round {round}");
            assert_eq!(*log_w.lock().unwrap(), *log_u.lock().unwrap(), "round {round}: the order of landings");
            assert_eq!(*log_u.lock().unwrap(), ["first landed", "second landed"], "round {round}");
            assert!(w.copy_is_current(host) && u.copy_is_current(host), "round {round}: a copy missed a change");
            flying += u32::from(crate::unwaited::tests::count(C::StillRunning) > waits_before);
            w.done();
            u.done();
        }
        eprintln!("both_paths_leave_the_same_bytes: {rounds} rounds, {marked} with pages marked by the GPU, {flying} with a settle that found its flight still on the GPU");
        assert!(marked > 0, "no round had mark bits: the marked path was not run");
        assert!(rounds < 4 || flying > 0, "no settle ever found its flight still on the GPU");
    }

    /// Each protection undone in turn: the script's own expectations or the
    /// checker say so.
    #[test]
    fn protections_undone_are_found() {
        let Some(host) = host() else { return };
        let run = |broke: Break| {
            let scene = Scene::new(host)?;
            let log: Log = Default::default();
            let (wrong, found) = caught(|| flown(&scene, host, 2, &log, broke));
            scene.done();
            Some((wrong, found))
        };
        let Some((wrong, found)) = run(Break::None) else { return };
        assert!(wrong.is_empty() && found.is_empty(), "nothing undone: {wrong:?} {found:?}");
        let has = |list: &[String], what: &str| list.iter().any(|s| s.contains(what));
        let (wrong, _) = run(Break::WordsAtLaunch).unwrap();
        assert!(has(&wrong, "the first word unwritten while its list is in the air"), "{wrong:?}");
        let (wrong, _) = run(Break::SignalsAtOnce).unwrap();
        assert!(has(&wrong, "the fence unmoved while the list before its Signals is in the air"), "{wrong:?}");
        let (wrong, _) = run(Break::SignalsReversed).unwrap();
        assert!(has(&wrong, "the fence at its last Signal after the settle"), "{wrong:?}");
        let (wrong, _) = run(Break::ListsNotHeld).unwrap();
        assert!(has(&wrong, "the flight holds its list") && has(&wrong, "the first word written by the settle"), "{wrong:?}");
        let (_, found) = run(Break::NoWait).unwrap();
        assert!(has(&found, "rule C2") && has(&found, "before the host fence reached"), "{found:?}");
    }

    /// What must not happen beside a flight, done beside one: the checker
    /// names each (C1 a refresh and a declaration, C2 a second launch, C6 a
    /// release, C7 a queue-ordered title copy).
    #[test]
    fn what_is_done_beside_a_flight_is_found() {
        let Some(host) = host() else { return };
        let Some(scene) = Scene::new(host) else { return };
        let log: Log = Default::default();
        let (a, b) = (scene.first(host, 8, &log), scene.second(host, 0, &log));
        let ((), found) = caught(|| {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            let _holding = Holding::enter();
            host.settle_held(By::Ours, true);
            host.launch_held(&mut submitted, std::slice::from_ref(&a), Duration::ZERO);
            // A refresh beside the flight (the title wrote page 9).
            // SAFETY: a byte of the scene's pool.
            unsafe { ((scene.p + 9 * PAGE) as *mut u8).write_volatile(1) };
            scene.copy.update(scene.p + 9 * PAGE, PAGE);
            // A queue-ordered copy into title memory.
            copy_mapped((scene.p + 10 * PAGE) as *mut u8, COPIED.as_ptr(), 8, true);
            release_retired();
            crate::shadow::idle_sweep();
            // The next submission prepared and launched with no settle.
            host.launch_held(&mut submitted, std::slice::from_ref(&b), Duration::ZERO);
            host.settle_held(By::Ours, true);
        });
        idle(host);
        crate::unwaited::recycle();
        let has = |rule: &str, what: &str| found.iter().any(|s| s.contains(rule) && s.contains(what));
        assert!(has("rule C1", "a host copy is refreshed from title memory"), "{found:?}");
        assert!(has("rule C1", "a range of a host copy is declared written"), "{found:?}");
        assert!(has("rule C7", "a queue-ordered copy into title memory"), "{found:?}");
        assert!(has("rule C6", "retired host objects are released"), "{found:?}");
        assert!(has("rule C6", "idle copies of title memory are released"), "{found:?}");
        assert!(has("rule C2", "launched with a flight in the air"), "{found:?}");
        // The first flight was never landed: its word is still unwritten.
        assert_eq!(scene.word(0), 0, "the test itself shows the lost landing");
        drop((a, b));
        scene.done();
    }

    /// A flight after which the title makes no call is settled by the poll,
    /// soon after the GPU is done with it.
    #[test]
    fn a_flight_left_alone_is_settled_by_the_poll() {
        let Some(host) = host() else { return };
        let Some(scene) = Scene::new(host) else { return };
        let log: Log = Default::default();
        let a = scene.first(host, 2, &log);
        host.execute_unwaited(std::slice::from_ref(&a));
        let started = Instant::now();
        while scene.word(0) == 0 && started.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(scene.word(0), 0xa, "settled with no call of the title's");
        assert_eq!(scene.byte(2 * PAGE + 100), 0x80);
        idle(host);
        drop(a);
        scene.done();
    }
}
