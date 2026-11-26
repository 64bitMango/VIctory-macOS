//! The copy side of the checker of `crate::unwaited`
//! (`VICTORY_UNWAITED_CHECK=1`; `docs/unwaited.md` 10, rules C3 and C4),
//! which runs with submissions waited for or not.
//!
//! C3, a copy-back over a title write. When shaders are declared to write a
//! range, the checker keeps the copy's bytes and title memory's there. At
//! the copy-back, where the copy has changed (the shaders wrote: nothing
//! else writes a copy while its submission runs, rule C1) and title memory
//! has changed too, to something else than what is about to be written, the
//! copy-back is about to undo a title write. Unmarked ranges are kept
//! whole. Of marked ranges only the pages shaders marked in the last
//! [`RECENT`] submissions are kept (the ranges are the views' whole reach,
//! a gigabyte in the world; the pages written repeat from frame to frame);
//! pages marked for the first time in a while are counted as unchecked.
//!
//! C4, the trackers' word. The copy-back of a marked page the title has not
//! written since the copy took it writes every byte in which the copy
//! differs from title memory. So "unchanged since taken" must mean "title
//! memory equals the copy" there: checked when such a page is kept before
//! its submission, and when a copy-back (or, with
//! `VICTORY_UNWAITED_CHECK_REFRESH=1`, a refresh) has just taken pages.
//! A title write that races the check has set "changed" before it lands
//! (the page is guarded first), so a difference with "unchanged" still
//! standing is a title write the copy never saw.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::*;
use crate::unwaited::{C, add, violation};

/// Submissions a page stays "marked lately" for.
const RECENT: u64 = 256;

/// What the checker keeps of a copy from its declarations to its copy-back.
#[derive(Default)]
struct Kept {
    /// Unmarked declared ranges, apart: offset in the copy, the copy's
    /// bytes before, title memory's before.
    regions: Vec<(usize, Vec<u8>, Vec<u8>)>,
    covered: crate::ranges::Ranges,
    /// Marked pages (by address): title memory's bytes before, cut to the
    /// copy.
    pages: HashMap<usize, Vec<u8>>,
}

/// By copy (its address: held by its lists until the copy-back).
static KEPT: Mutex<Option<HashMap<usize, Kept>>> = Mutex::new(None);
/// Pages shaders marked lately (by address), and the submission they did.
static MARKED: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());
static SUBMISSION: AtomicU64 = AtomicU64::new(0);
/// Pages (by address) whose copies hold what shaders wrote and a submission
/// withheld from title memory (`crate::memory::drop_withheld`), and when:
/// there the copy differs from title memory by intent, for a while.
static WITHHELD: Mutex<BTreeMap<usize, u64>> = Mutex::new(BTreeMap::new());

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

fn kept() -> std::sync::MutexGuard<'static, Option<HashMap<usize, Kept>>> {
    KEPT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Title memory `[at, at + n)`, if all of it can be read.
fn title(at: usize, n: usize) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; n];
    crate::host::read_into(at as *const u8, &mut bytes).then_some(bytes)
}

/// The first byte where the copy changed (`host` against `before`) and
/// title memory changed too (`now` against `was`) to something else than
/// the copy's byte: its offset.
fn undone(host: &[u8], before: &[u8], now: &[u8], was: &[u8]) -> Option<usize> {
    if now == was || host == before {
        return None;
    }
    (0..host.len()).find(|&i| host[i] != before[i] && now[i] != was[i] && now[i] != host[i])
}

impl Shadow {
    /// The copy's bytes of offsets `[from, to)`.
    fn check_host(&self, from: usize, to: usize) -> &[u8] {
        // SAFETY: inside the mapping (the callers cut to the copy); read
        // while no submission runs.
        unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) }
    }

    /// Whether the tracker stands for "title memory equals the copy" in the
    /// 64 KiB page at `page`: it is tracked whole, not written since it was
    /// taken and not left open.
    fn check_stands(&self, page: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        page >= start
            && page + PAGE <= end
            && self.tracked.as_ref().is_some_and(|t| {
                !t.changed(page / PAGE) && !t.rested(page, page + PAGE) && t.untracked(page, page + PAGE).is_empty()
            })
            && !exempted(page)
    }

    /// A refresh under a range declared written (`Shadow::update_busy`)
    /// brought title memory `[from, to)` into the copy: what the checker
    /// keeps there is the copy's bytes now, as the snapshot's are.
    pub(super) fn check_patched(&self, from: usize, to: usize) {
        let _cost = crate::unwaited::checked();
        let start = self.start as usize;
        let (from, to) = (from.max(start), to.min(start + self.len as usize));
        if from >= to {
            return;
        }
        let mut all = kept();
        let Some(kept) = all.as_mut().and_then(|k| k.get_mut(&(self as *const Shadow as usize))) else {
            return;
        };
        let (a, b) = (from - start, to - start);
        for (at, before, was) in kept.regions.iter_mut() {
            let (s, e) = ((*at).max(a), (*at + before.len()).min(b));
            if s < e {
                let host = self.check_host(s, e);
                before[s - *at..e - *at].copy_from_slice(host);
                was[s - *at..e - *at].copy_from_slice(host);
            }
        }
        for (&page, was) in kept.pages.iter_mut() {
            let (s, e) = (page.max(from), (page + was.len()).min(to));
            if s < e {
                was[s - page..e - page].copy_from_slice(self.check_host(s - start, e - start));
            }
        }
    }

    /// C4 for the whole pages in title memory `[from, to)`: where the
    /// tracker says unchanged, title memory equals the copy. `when` says
    /// what has just relied on it or made it so.
    pub(super) fn check_equal(&self, from: usize, to: usize, when: &str) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start).next_multiple_of(PAGE), to.min(end) / PAGE * PAGE);
        let mut page = from;
        while page < to {
            if self.check_stands(page)
                && let Some(now) = title(page, PAGE)
            {
                add(C::CheckBytes, PAGE as u64);
                let host = self.check_host(page - start, page - start + PAGE);
                if now != host && self.check_stands(page) {
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

    /// At a declaration (`Shadow::writing`, after the range's refresh):
    /// keeps what C3 compares at the copy-back, and checks C4 for the
    /// marked pages kept.
    pub(super) fn check_declared(&self, address: usize, len: usize, marked: bool) {
        let _cost = crate::unwaited::checked();
        let start = self.start as usize;
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        let mut all = kept();
        let kept = all.get_or_insert_with(HashMap::new).entry(self as *const Shadow as usize).or_default();
        if !(marked && self.tracked.is_some()) {
            for (a, b) in kept.covered.missing(from, to) {
                match title(start + a, b - a) {
                    Some(was) => {
                        add(C::CheckBytes, 2 * (b - a) as u64);
                        kept.regions.push((a, self.check_host(a, b).to_vec(), was));
                    }
                    None => add(C::Unchecked, 1),
                }
                kept.covered.insert(a, b, 0);
            }
            return;
        }
        // The pages of the range marked lately.
        let oldest = SUBMISSION.load(Relaxed).saturating_sub(RECENT);
        let lately: Vec<usize> = {
            let marks = MARKED.lock().unwrap_or_else(|e| e.into_inner());
            marks
                .range((start + from) / PAGE * PAGE..start + to)
                .filter(|&(_, &when)| when >= oldest)
                .map(|(&page, _)| page)
                .collect()
        };
        for page in lately {
            if kept.pages.contains_key(&page) || !self.check_stands(page) {
                continue;
            }
            let (a, b) = (page.max(start), (page + PAGE).min(start + self.len as usize));
            let Some(was) = title(a, b - a) else { continue };
            add(C::CheckBytes, (b - a) as u64);
            let host = self.check_host(a - start, b - start);
            if was != host && self.check_stands(page) {
                let i = (0..was.len()).find(|&i| was[i] != host[i]).unwrap_or(0);
                violation(
                    "C4",
                    format_args!(
                        "before a submission whose shaders may write it: the page at {page:#x} of copy {start:#x}+{:#x} is unchanged since taken by its tracker, but title memory differs from the copy, first at {:#x} (title {:02x?}, copy {:02x?})",
                        self.len,
                        a + i,
                        &was[i..(i + 8).min(was.len())],
                        &host[i..(i + 8).min(host.len())]
                    ),
                );
                continue;
            }
            kept.pages.insert(page, was);
        }
    }
}

/// Before the copy-back of the pages shaders marked (per copy, by address):
/// C3 for those kept, and all of them are marked lately from now on.
pub(super) fn marked(by_copy: &[(Arc<Shadow>, Vec<usize>)]) {
    let _cost = crate::unwaited::checked();
    let now_is = SUBMISSION.load(Relaxed);
    let mut all = kept();
    let mut marks = MARKED.lock().unwrap_or_else(|e| e.into_inner());
    for (copy, pages) in by_copy {
        let start = copy.start as usize;
        let mut kept = all.as_mut().and_then(|k| k.get_mut(&(Arc::as_ptr(copy) as usize)));
        for &page in pages {
            marks.insert(page, now_is);
            let Some(was) = kept.as_mut().and_then(|k| k.pages.remove(&page)) else {
                add(C::Unchecked, 1);
                continue;
            };
            let a = page.max(start);
            let Some(now) = title(a, was.len()) else { continue };
            add(C::CheckBytes, 2 * was.len() as u64);
            let host = copy.check_host(a - start, a - start + was.len());
            // The page held title memory's bytes before (C4, checked when
            // it was kept): they are the copy's before as well.
            if let Some(i) = undone(host, &was, &now, &was) {
                violation(
                    "C3",
                    format_args!(
                        "the copy-back of the marked page at {page:#x} of copy {start:#x}+{:#x} would overwrite a title write at {:#x}: it held {:#04x} before the submission, the title wrote {:#04x}, the shaders {:#04x}",
                        copy.len,
                        a + i,
                        was[i],
                        now[i],
                        host[i]
                    ),
                );
            }
        }
    }
    if marks.len() > 1 << 17 {
        let oldest = now_is.saturating_sub(RECENT);
        marks.retain(|_, when| *when >= oldest);
    }
}

/// Before the copy-back of the declared ranges `back` (copy, address,
/// bytes, whether it goes back), which ends the submission: C3 for the
/// unmarked ranges kept, and nothing is kept any more.
pub(super) fn declared(back: &[(Arc<Shadow>, usize, usize, bool)]) {
    let _cost = crate::unwaited::checked();
    SUBMISSION.fetch_add(1, Relaxed);
    let Some(all) = kept().take() else { return };
    for (key, kept) in all {
        let Some((copy, ..)) = back.iter().find(|b| Arc::as_ptr(&b.0) as usize == key) else {
            continue;
        };
        // Ranges withheld stay in the copy: nothing of this copy is judged.
        if back.iter().any(|b| Arc::as_ptr(&b.0) as usize == key && !b.3) {
            add(C::Unchecked, kept.regions.len() as u64);
            continue;
        }
        let start = copy.start as usize;
        for (at, before, was) in &kept.regions {
            let Some(now) = title(start + at, was.len()) else { continue };
            add(C::CheckBytes, 2 * was.len() as u64);
            let host = copy.check_host(*at, at + before.len());
            if let Some(i) = undone(host, before, &now, was) {
                violation(
                    "C3",
                    format_args!(
                        "the copy-back of a declared range of copy {start:#x}+{:#x} would overwrite a title write at {:#x}: title memory held {:#04x} before the submission and {:#04x} now, the shaders wrote {:#04x} over {:#04x}",
                        copy.len,
                        start + at + i,
                        was[i],
                        now[i],
                        host[i],
                        before[i]
                    ),
                );
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
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 2, 7, 4], &before), None);
        // Both wrote byte 1, differently.
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 8, 3, 4], &before), Some(1));
        // Both wrote the same value.
        assert_eq!(undone(&[1, 9, 3, 4], &before, &[1, 9, 3, 4], &before), None);
        // Nobody wrote.
        assert_eq!(undone(&before, &before, &before, &before), None);
        // The copy's bytes before differ from title memory's (an unmarked
        // range whose copy was stale): only the copy's change counts.
        assert_eq!(undone(&[5, 9], &[5, 6], &[1, 7], &[1, 2]), Some(1));
        assert_eq!(undone(&[5, 6], &[5, 6], &[1, 7], &[1, 2]), None);
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
