//! The copy side of the checker of `crate::unwaited`
//! (`VICTORY_UNWAITED_CHECK=1`; `docs/unwaited.md` 10, rules C3 and C4),
//! which runs with submissions waited for or not. It keeps no bytes of its
//! own: what a range held before its submission is what the copy-back
//! itself compares with.
//!
//! C3, a copy-back over a title write: a byte the shaders changed in the
//! copy (it differs from what the copy held before; nothing else writes a
//! copy while its submission runs, rule C1) that the title changed too, to
//! something else than what is about to be written.
//! - Unmarked declared ranges: "before" is the copy's snapshot. Only pages
//!   the title wrote since the copy took them can hold a title write, and
//!   only where the copy held title memory's bytes at the declaration (the
//!   page was unchanged then, or a refresh under the range took it since)
//!   is the snapshot also what title memory held: those pages are read and
//!   compared, the others counted as unchecked.
//! - Marked pages: a page the title wrote while armed has its bytes from
//!   before that write saved by the tracker; they are "before". A page the
//!   title did not write has no title write to undo.
//!
//! C4, the trackers' word. The copy-back of a marked page the title has not
//! written since the copy took it writes every byte in which the copy
//! differs from title memory. So "unchanged since taken" must mean "title
//! memory equals the copy" there. Checked where it is made so (every page a
//! copy-back takes and reads again; with `VICTORY_UNWAITED_CHECK_REFRESH=1`
//! every page a refresh takes) and where it is relied on: a page shaders
//! marked in the last [`RECENT`] submissions is read and compared when it
//! is declared again, once per taking and again every [`AGAIN`]
//! submissions (each of a frame's twenty submissions declares the same
//! pages: all of them every time read gigabytes a frame). A title write
//! that races the check has set "changed" before it lands (the page is
//! guarded first), so a difference with "unchanged" still standing is a
//! title write the copy never saw.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::*;
use crate::unwaited::{C, add, violation};

/// Submissions a page stays "marked lately" for, and a page found equal to
/// title memory is taken as equal for (unless it is taken again first).
const RECENT: u64 = 256;
const AGAIN: u64 = 64;

/// What the checker keeps of a copy from its declarations to its copy-back.
#[derive(Default)]
struct Kept {
    /// The unmarked declared ranges, as offsets in the copy.
    covered: crate::ranges::Ranges,
    /// Pages (by address) of them where the snapshot is also what title
    /// memory held.
    judged: HashSet<usize>,
}

/// By copy (its address: held by its lists until the copy-back).
static KEPT: Mutex<Option<HashMap<usize, Kept>>> = Mutex::new(None);
/// Pages shaders marked lately (by address), and the submission they did.
static MARKED: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());
/// Pages found equal to title memory while unchanged (by address), and the
/// submission that found them so; taken out when taken again.
static EQUAL: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());
static SUBMISSION: AtomicU64 = AtomicU64::new(0);
/// Pages (by address) whose copies hold what shaders wrote and a submission
/// withheld from title memory (`crate::memory::drop_withheld`), and when:
/// there the copy differs from title memory by intent, for a while.
static WITHHELD: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

fn kept() -> std::sync::MutexGuard<'static, Option<HashMap<usize, Kept>>> {
    KEPT.lock().unwrap_or_else(|e| e.into_inner())
}

fn equal() -> std::sync::MutexGuard<'static, BTreeMap<usize, u64>> {
    EQUAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `[address, address + len)` was withheld: not judged for a while.
pub(crate) fn exempt(address: usize, len: usize) {
    let now_is = SUBMISSION.load(Relaxed);
    let mut withheld = WITHHELD.lock().unwrap_or_else(|e| e.into_inner());
    for page in (address / PAGE * PAGE..address.saturating_add(len)).step_by(PAGE) {
        withheld.insert(page, now_is);
    }
}

fn exempted(page: usize) -> bool {
    let oldest = SUBMISSION.load(Relaxed).saturating_sub(RECENT);
    WITHHELD.lock().unwrap_or_else(|e| e.into_inner()).get(&page).is_some_and(|&when| when >= oldest)
}

/// Title memory `[at, at + n)`, if all of it can be read.
fn title(at: usize, n: usize) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; n];
    crate::host::read_into(at as *const u8, &mut bytes).then_some(bytes)
}

/// The first byte the shaders changed (`host` against `before`) that the
/// title changed too (`now` against `before`, which title memory held as
/// well) to something else than the copy's byte: its offset.
fn undone(host: &[u8], before: &[u8], now: &[u8]) -> Option<usize> {
    if now == before || host == before {
        return None;
    }
    (0..host.len()).find(|&i| host[i] != before[i] && now[i] != before[i] && now[i] != host[i])
}

/// The 64 KiB pages (by address) title memory `[from, to)` touches.
fn pages_of(from: usize, to: usize) -> impl Iterator<Item = usize> {
    (from / PAGE * PAGE..to).step_by(PAGE)
}

impl Shadow {
    /// The copy's bytes of offsets `[from, to)`.
    fn check_host(&self, from: usize, to: usize) -> &[u8] {
        // SAFETY: inside the mapping (the callers cut to the copy); read
        // while no submission runs.
        unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) }
    }

    /// Whether the tracker stands for "title memory equals the copy" in the
    /// 64 KiB page at `page`: the copy holds it whole, it is tracked whole,
    /// not written since it was taken, not left open, nothing withheld.
    fn check_stands(&self, page: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        page >= start
            && page + PAGE <= end
            && self.tracked.as_ref().is_some_and(|t| {
                !t.changed(page / PAGE) && !t.rested(page, page + PAGE) && t.untracked(page, page + PAGE).is_empty()
            })
            && !exempted(page)
    }

    /// C4 for the whole pages in title memory `[from, to)`: where the
    /// tracker says unchanged, title memory equals the copy. `when` says
    /// what has just relied on it or made it so.
    pub(super) fn check_equal(&self, from: usize, to: usize, when: &str) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start).next_multiple_of(PAGE), to.min(end) / PAGE * PAGE);
        let now_is = SUBMISSION.load(Relaxed);
        let mut page = from;
        while page < to {
            if self.check_stands(page)
                && let Some(now) = title(page, PAGE)
            {
                add(C::CheckBytes, PAGE as u64);
                let host = self.check_host(page - start, page - start + PAGE);
                // Read a second time before it is called a violation: a page
                // mapped anew between its guard and the read is told to the
                // trackers a moment later.
                let differs = |now: &[u8]| !crate::fastcmp::eq(now, host) && self.check_stands(page);
                let now = if differs(&now) {
                    std::thread::yield_now();
                    title(page, PAGE).unwrap_or(now)
                } else {
                    now
                };
                if now == host {
                    equal().insert(page, now_is);
                } else if differs(&now) {
                    let i = (0..PAGE).find(|&i| now[i] != host[i]).unwrap_or(0);
                    violation(
                        "C4",
                        format_args!(
                            "{when}: the page at {page:#x} of copy {start:#x}+{:#x} is unchanged since taken by its tracker, but title memory differs from the copy, first at {:#x} (title {:02x?}, copy {:02x?}); a copy-back of a marked write there would put the copy's bytes over the title's",
                            self.len,
                            page + i,
                            &now[i..(i + 8).min(PAGE)],
                            &host[i..(i + 8).min(PAGE)]
                        ),
                    );
                }
            }
            page += PAGE;
        }
    }

    /// A refresh took the pages of title memory `[from, to)` again: whether
    /// they equal the copy is to be found anew, and under a range declared
    /// written (`under`, from `Shadow::update_busy`, which brings the same
    /// bytes into the snapshot) the snapshot is what title memory held.
    pub(super) fn check_retaken(&self, from: usize, to: usize, under: bool) {
        let whole = (from.next_multiple_of(PAGE)..to / PAGE * PAGE).step_by(PAGE);
        {
            let mut equal = equal();
            for page in whole.clone() {
                equal.remove(&page);
            }
        }
        if under && let Some(kept) = kept().as_mut().and_then(|k| k.get_mut(&(self as *const Shadow as usize))) {
            kept.judged.extend(whole);
        }
    }

    /// At a declaration (`Shadow::writing`, after the range's refresh):
    /// notes what C3 judges at the copy-back, and checks C4 for the pages
    /// of a marked range that shaders marked lately.
    pub(super) fn check_declared(&self, address: usize, len: usize, marked: bool) {
        let _cost = crate::unwaited::checked();
        let start = self.start as usize;
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        if !(marked && self.tracked.is_some()) {
            let mut all = kept();
            let kept = all.get_or_insert_with(HashMap::new).entry(self as *const Shadow as usize).or_default();
            for (a, b) in kept.covered.missing(from, to) {
                kept.judged.extend(pages_of(start + a, start + b).filter(|&page| self.check_stands(page)));
                kept.covered.insert(a, b, 0);
            }
            return;
        }
        let now_is = SUBMISSION.load(Relaxed);
        let lately: Vec<usize> = {
            let marks = MARKED.lock().unwrap_or_else(|e| e.into_inner());
            let equal = equal();
            marks
                .range((start + from) / PAGE * PAGE..start + to)
                .filter(|&(page, &when)| {
                    when + RECENT >= now_is && equal.get(page).is_none_or(|&found| found + AGAIN < now_is)
                })
                .map(|(&page, _)| page)
                .collect()
        };
        for page in lately {
            self.check_equal(page, page + PAGE, "before a submission whose shaders may write it");
        }
    }
}

/// Before the copy-back of the pages shaders marked (per copy, by address):
/// C3 for those the title wrote while they were armed, and all of them are
/// marked lately from now on.
pub(super) fn marked(by_copy: &[(Arc<Shadow>, Vec<usize>)]) {
    let _cost = crate::unwaited::checked();
    let now_is = SUBMISSION.load(Relaxed);
    let mut marks = MARKED.lock().unwrap_or_else(|e| e.into_inner());
    for (copy, pages) in by_copy {
        let (start, end) = (copy.start as usize, (copy.start + copy.len) as usize);
        for &page in pages {
            marks.insert(page, now_is);
            let Some(t) = &copy.tracked else { continue };
            let Some(saved) = t.saved_peek(page / PAGE) else {
                // Unchanged: no title write to undo. Changed and not armed:
                // the title's bytes are kept.
                continue;
            };
            let (a, b) = (page.max(start), (page + PAGE).min(end));
            if a >= b || saved.len() < b - page {
                continue;
            }
            let Some(now) = title(a, b - a) else { continue };
            add(C::CheckBytes, 2 * (b - a) as u64);
            let (host, before) = (copy.check_host(a - start, b - start), &saved[a - page..b - page]);
            if let Some(i) = undone(host, before, &now) {
                violation(
                    "C3",
                    format_args!(
                        "the copy-back of the marked page at {page:#x} of copy {start:#x}+{:#x} would overwrite a title write at {:#x}: it held {:#04x} before the submission, the title wrote {:#04x}, the shaders {:#04x}",
                        copy.len,
                        a + i,
                        before[i],
                        now[i],
                        host[i]
                    ),
                );
            }
        }
    }
    if marks.len() > 1 << 17 {
        marks.retain(|_, when| *when + RECENT >= now_is);
    }
}

/// Before the copy-back of the declared ranges `back` (copy, address,
/// bytes, whether it goes back), which ends the submission: C3 for the
/// unmarked ranges noted, and nothing is kept any more.
pub(super) fn declared(back: &[(Arc<Shadow>, usize, usize, bool)]) {
    let _cost = crate::unwaited::checked();
    SUBMISSION.fetch_add(1, Relaxed);
    let Some(all) = kept().take() else { return };
    let mut gathered = Vec::new();
    for (key, kept) in all {
        let Some((copy, ..)) = back.iter().find(|b| Arc::as_ptr(&b.0) as usize == key) else {
            continue;
        };
        let Some(t) = &copy.tracked else {
            add(C::Unchecked, 1);
            continue;
        };
        // Ranges withheld stay in the copy: nothing of this copy is judged.
        if back.iter().any(|b| Arc::as_ptr(&b.0) as usize == key && !b.3) {
            add(C::Unchecked, 1);
            continue;
        }
        let start = copy.start as usize;
        let snapshot = copy.before.lock().unwrap_or_else(|e| e.into_inner());
        for (a, b) in kept.covered.iter() {
            for page in pages_of(start + a, start + b) {
                // Not written by the title since the copy took it: nothing
                // of the title's to undo.
                if !t.changed(page / PAGE) {
                    continue;
                }
                let (s, e) = (a.max(page.saturating_sub(start)), b.min(page + PAGE - start));
                if !kept.judged.contains(&page) || !crate::segments::covered(&snapshot.0, s, e) {
                    add(C::Unchecked, 1);
                    continue;
                }
                let Some(now) = title(start + s, e - s) else { continue };
                add(C::CheckBytes, 2 * (e - s) as u64);
                let before = crate::segments::held(&snapshot.0, s, e, &mut gathered);
                let host = copy.check_host(s, e);
                if let Some(i) = undone(host, before, &now) {
                    violation(
                        "C3",
                        format_args!(
                            "the copy-back of a declared range of copy {start:#x}+{:#x} would overwrite a title write at {:#x}: it held {:#04x} before the submission, the title wrote {:#04x}, the shaders {:#04x}",
                            copy.len,
                            start + s + i,
                            before[i],
                            now[i],
                            host[i]
                        ),
                    );
                }
            }
        }
    }
}

/// After a copy-back took whole pages (`taken`, page numbers per copy) as
/// equal to title memory and read them again: C4 for them.
pub(super) fn taken(copies: &[&Shadow], taken: &[Vec<usize>]) {
    let _cost = crate::unwaited::checked();
    for (copy, pages) in copies.iter().zip(taken) {
        for &page in pages {
            copy.check_equal(page * PAGE, (page + 1) * PAGE, "after a copy-back took the page");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::undone;

    #[test]
    fn only_a_byte_both_wrote_differently_is_a_title_write_undone() {
        let before = [1u8, 2, 3, 4];
        // The shaders wrote byte 1, the title byte 2: nothing undone.
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 2, 7, 4]), None);
        // Both wrote byte 1, differently.
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 8, 3, 4]), Some(1));
        // Both wrote the same value.
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 9, 3, 4]), None);
        // Nobody wrote; only the title wrote; only the shaders wrote.
        assert_eq!(undone(&before, &before, &before), None);
        assert_eq!(undone(&before, &before, &[7, 7, 7, 7]), None);
        assert_eq!(undone(&[7, 7, 7, 7], &before, &before), None);
    }
}

#[cfg(test)]
impl Shadow {
    /// The views' written-pages bitmap with the place of the mark of the
    /// page of title address `address` in it: the byte offset of its dword
    /// and its bit (tests: a mark written by the GPU).
    pub(crate) fn test_mark(&self, address: u64) -> Option<(ID3D12Resource, u64, u32)> {
        let (bit, _) = self.mark_entry(address)?;
        let b = bitmap(false)?;
        Some((b._buffer.clone(), u64::from(bit / 32) * 4, 1 << (bit % 32)))
    }

    /// The copy's bytes of title memory `[address, address + n)` (tests).
    pub(crate) fn test_bytes(&self, address: usize, n: usize) -> Vec<u8> {
        let at = address - self.start as usize;
        assert!(at + n <= self.len as usize);
        self.check_host(at, at + n).to_vec()
    }
}

#[cfg(test)]
mod copies {
    use super::*;
    use crate::unwaited::tests::caught;

    /// A tracked pool of `pages` pages with its copy, brought up to date;
    /// none without a host GPU.
    fn pool(pages: usize) -> Option<(usize, Arc<Shadow>)> {
        let p = crate::watch::tests::pool(pages);
        register(p, p as u64, (pages * PAGE) as u64);
        let copy = find(p as u64, (pages * PAGE) as u64);
        if copy.is_none() {
            unregister(p, p as u64);
            crate::watch::tests::free(p);
        }
        let copy = copy?;
        copy.update(p, pages * PAGE);
        Some((p, copy))
    }

    fn done(p: usize, copy: Arc<Shadow>) {
        drop(copy);
        unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// C3: the shaders and the title both write a byte of a declared range
    /// while its submission runs; the copy-back would put the shaders' over
    /// the title's. Neighbouring bytes written by one of them are fine.
    #[test]
    fn a_copy_back_over_a_title_write_is_found() {
        let Some((p, copy)) = pool(2) else { return };
        let title = |at: usize, v: u8| unsafe { ((p + at) as *mut u8).write_volatile(v) };
        // Another test's flight may be in the air (rule C1 then speaks of
        // this test's declarations): only C3 is looked at.
        let caught = |f: &dyn Fn()| -> ((), Vec<String>) {
            let ((), found) = caught(f);
            ((), found.into_iter().filter(|s| s.contains("rule C3")).collect())
        };
        let ((), found) = caught(&|| {
            copy.writing(p, PAGE, false);
            copy.write_host((p + 10) as u64, &[0xaa]);
            title(5000, 0x11);
            copy_back_declared(&[(copy.clone(), p, PAGE, true)]);
        });
        assert!(found.is_empty(), "each wrote its own byte: {found:?}");
        copy.update(p, 2 * PAGE);
        let ((), found) = caught(&|| {
            copy.writing(p, PAGE, false);
            copy.write_host((p + 20) as u64, &[0xbb]);
            title(20, 0x22);
            copy_back_declared(&[(copy.clone(), p, PAGE, true)]);
        });
        assert!(found.len() == 1 && found[0].contains("rule C3") && found[0].contains(&format!("{:#x}", p + 20)), "{found:?}");
        done(p, copy);
    }

    /// What the checker costs: frames of 20 submissions over 16 copies of
    /// 64 pages, each submission declaring 55 MB / 20 of unmarked ranges and
    /// landing 96 marked pages (1,920 a frame, the same pages from frame to
    /// frame, as the world's are), with the checker off and on.
    #[test]
    #[ignore = "synthetic release benchmark; run with --release --ignored --nocapture"]
    fn benchmark_checker_release() {
        const COPIES: usize = 16;
        const EACH: usize = 64;
        const UNMARKED: usize = 172 << 10;
        let (Some(_), Some(b)) = (crate::host::host(), bitmap(false)) else { return };
        let len = EACH * PAGE;
        let p = crate::watch::tests::pool(COPIES * EACH);
        let copies: Vec<Arc<Shadow>> = (0..COPIES)
            .map(|k| {
                register(p + k, (p + k * len) as u64, len as u64);
                find((p + k * len) as u64, len as u64).expect("a copy")
            })
            .collect();
        let mut round = 0u8;
        let mut frame = || {
            let started = std::time::Instant::now();
            for submission in 0..20usize {
                round = round.wrapping_add(1);
                let mut back = Vec::new();
                for (k, copy) in copies.iter().enumerate() {
                    let start = p + k * len;
                    copy.update(start, len);
                    // Unmarked: the first pages; marked: the rest.
                    copy.writing(start, UNMARKED, false);
                    copy.writing(start + 4 * PAGE, len - 4 * PAGE, true);
                    back.push((copy.clone(), start, UNMARKED, true));
                    back.push((copy.clone(), start + 4 * PAGE, len - 4 * PAGE, true));
                    copy.write_host((start + 4096 + submission) as u64, &[0x80 | round; 64]);
                    for i in 0..6 {
                        let page = start + (4 + (submission * 3 + i * 7) % 60) * PAGE;
                        copy.write_host((page + 0x400) as u64, &[0x80 | round; 256]);
                        let bit = copy.mark_entry(page as u64).expect("a bit").0;
                        // SAFETY: a dword of the mapped bitmap.
                        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
                    }
                }
                copy_back_marks(&[], false);
                copy_back_declared(&back);
            }
            started.elapsed().as_secs_f64() * 1e3
        };
        let off: Vec<f64> = (0..6).map(|_| frame()).collect();
        let cost = |c: C| crate::unwaited::tests::count(c);
        let (us, bytes, unchecked) = (cost(C::CheckUs), cost(C::CheckBytes), cost(C::Unchecked));
        crate::unwaited::tests::REFRESHES.with(|r| r.set(false));
        let (on, found) = caught(|| (0..6).map(|_| frame()).collect::<Vec<f64>>());
        crate::unwaited::tests::REFRESHES.with(|r| r.set(true));
        assert!(found.is_empty(), "{found:?}");
        let frames = 5.0;
        // The first frame with the checker on has no pages marked lately.
        let mean = |v: &[f64]| v[1..].iter().sum::<f64>() / frames;
        eprintln!(
            "checker benchmark: {:.1} ms a frame without, {:.1} ms with (its own count: {:.1} ms a frame over all six, {:.1} MB compared, {:.0} pages or ranges unchecked a frame)",
            mean(&off),
            mean(&on),
            (cost(C::CheckUs) - us) as f64 / 1e3 / 6.0,
            (cost(C::CheckBytes) - bytes) as f64 / 1e6 / 6.0,
            (cost(C::Unchecked) - unchecked) as f64 / 6.0,
        );
        drop(copies);
        (0..COPIES).for_each(|k| unregister(p + k, (p + k * len) as u64));
        crate::watch::tests::free(p);
    }

    /// C4: a page whose tracker says "unchanged since taken" while title
    /// memory differs from the copy (a title write the copy never saw, as
    /// when a page was taken without being read again) is found where the
    /// copy-back would rely on it.
    #[test]
    fn a_stale_page_called_unchanged_is_found() {
        let Some((p, copy)) = pool(2) else { return };
        let ((), found) = caught(|| copy.check_equal(p, p + 2 * PAGE, "test"));
        assert!(found.is_empty(), "taken and equal: {found:?}");
        // The copy's page 1 goes stale behind the tracker's back.
        copy.write_host((p + PAGE + 77) as u64, &[0x7c]);
        let ((), found) = caught(|| copy.check_equal(p, p + 2 * PAGE, "test"));
        assert!(found.len() == 1 && found[0].contains("rule C4") && found[0].contains(&format!("{:#x}", p + PAGE + 77)), "{found:?}");
        // A page the title wrote is not "unchanged": nothing to find.
        unsafe { ((p + PAGE + 5) as *mut u8).write_volatile(1) };
        let ((), found) = caught(|| copy.check_equal(p, p + 2 * PAGE, "test"));
        assert!(found.is_empty(), "the tracker knows the page changed: {found:?}");
        done(p, copy);
    }
}
