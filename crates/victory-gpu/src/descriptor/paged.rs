//! A table of optional entries that gets its memory a page at a time, for
//! what a host heap keeps per slot (`HostHeap::kept`): a view heap has over
//! two million slots, and a title views host resources through a fraction of
//! them. A dense table writes every slot when the heap is made; this one
//! touches the memory of the slots that hold something.

/// Slots to a page: 4 KiB of the 16-byte entries a host heap keeps. A title
/// writes its slots in runs (tables) that can sit anywhere in the heap, and
/// a run takes the pages it touches whole, so a smaller page leaves more of
/// the heap unmade (runs of 64 slots at random places, 130,000 slots in a
/// 2.36 million slot heap: 7.7 MB at 256 slots, 13.9 at 512, 22.7 at
/// 1,024; slots in one stretch take the same at any size). A smaller page
/// also makes the table of pages, 16 bytes a page and made with the heap,
/// larger: 147 KB at 256 slots, 590 KB at 64.
const PAGE: usize = 256;

/// `len` slots, each empty or holding a `T`. A slot's page is made when
/// the slot first holds something, and stays made: what a slot holds is
/// released as in a dense table, by replacing it or with the table, and in
/// the same order (by slot).
// Used by its tests only since `Pages` took its place in the host heaps.
#[allow(dead_code)]
pub(crate) struct Paged<T> {
    pages: Vec<Option<Box<[Option<T>]>>>,
    len: usize,
}

#[allow(dead_code)]
impl<T> Paged<T> {
    /// `len` empty slots, with no page made.
    pub(crate) fn new(len: usize) -> Self {
        Self {
            pages: std::iter::repeat_with(|| None)
                .take(len.div_ceil(PAGE))
                .collect(),
            len,
        }
    }

    /// What slot `index` holds; none for an empty slot, a page not made, or
    /// a slot past the end.
    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        self.pages
            .get(index / PAGE)?
            .as_ref()?
            .get(index % PAGE)?
            .as_ref()
    }

    /// The slots that hold something, in order, with their indices: the
    /// pages made are walked, the others skipped.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.pages.iter().enumerate().flat_map(|(p, page)| {
            page.iter()
                .flat_map(|slots| slots.iter().enumerate())
                .filter_map(move |(i, slot)| slot.as_ref().map(|t| (p * PAGE + i, t)))
        })
    }

    /// Slot `index`, to write; none past the end. Its page is made only if
    /// `make` (the write is of something): with no page the slot is empty,
    /// and writing nothing there is no change, so none is returned.
    pub(crate) fn slot(&mut self, index: usize, make: bool) -> Option<&mut Option<T>> {
        if index >= self.len {
            return None;
        }
        let page = &mut self.pages[index / PAGE];
        if page.is_none() && !make {
            return None;
        }
        let first = index - index % PAGE;
        let page = page.get_or_insert_with(|| {
            std::iter::repeat_with(|| None)
                .take(PAGE.min(self.len - first))
                .collect()
        });
        Some(&mut page[index % PAGE])
    }
}

/// Times a thread found a page of a [`Pages`] taken when it had something
/// to put there, and the microseconds it waited, since last asked.
pub(crate) static WAITS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

type Page<T> = std::sync::Mutex<Option<Box<[Option<T>]>>>;

/// A [`Paged`] table for several threads at once: each page under a lock
/// of its own, so threads at slots of different pages do not wait for each
/// other (a host heap's one table of what its slots hold was taken for
/// every slot made, by the title's recording threads and the submitting
/// thread: 8,000 waits a frame). With `one` (`VICTORY_KEPT_PAGES=0`) every
/// use takes one lock first, as the one table did.
pub(crate) struct Pages<T> {
    pages: Box<[Page<T>]>,
    len: usize,
    one: Option<std::sync::Mutex<()>>,
}

impl<T> Pages<T> {
    /// `len` empty slots, with no page made.
    pub(crate) fn new(len: usize, one: bool) -> Self {
        Self {
            pages: (0..len.div_ceil(PAGE)).map(|_| std::sync::Mutex::new(None)).collect(),
            len,
            one: one.then(|| std::sync::Mutex::new(())),
        }
    }

    fn one(&self) -> Option<std::sync::MutexGuard<'_, ()>> {
        self.one.as_ref().map(|one| one.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Slot `index` holds `value` from now: what it held is returned.
    /// `then` is called with the page's lock held, with whether the slot
    /// holds something now; not for a slot that held nothing and gets
    /// nothing (its page is not made for that), nor past the end.
    pub(crate) fn replace(&self, index: usize, value: Option<T>, then: impl FnOnce(bool)) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let _one = self.one();
        let mut page = match self.pages[index / PAGE].try_lock() {
            Ok(page) => page,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            // Counted: how often, and for how long, a thread waits here.
            Err(std::sync::TryLockError::WouldBlock) => {
                let since = std::time::Instant::now();
                let page = self.pages[index / PAGE].lock().unwrap_or_else(|e| e.into_inner());
                WAITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                WAITS[1].fetch_add(since.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                page
            }
        };
        if page.is_none() && value.is_none() {
            return None;
        }
        let first = index - index % PAGE;
        let slots = page.get_or_insert_with(|| std::iter::repeat_with(|| None).take(PAGE.min(self.len - first)).collect());
        let holds = value.is_some();
        let old = std::mem::replace(&mut slots[index % PAGE], value);
        then(holds);
        old
    }

    /// `f` of what slot `index` holds (none: empty, no page, past the end).
    pub(crate) fn with<R>(&self, index: usize, f: impl FnOnce(Option<&T>) -> R) -> R {
        let Some(page) = self.pages.get(index / PAGE) else {
            return f(None);
        };
        let _one = self.one();
        let page = page.lock().unwrap_or_else(|e| e.into_inner());
        f(page.as_ref().and_then(|slots| slots.get(index % PAGE)).and_then(Option::as_ref))
    }

    /// `f` of each slot below `end` that holds something, in order, with
    /// its index; a page at a time under its lock.
    pub(crate) fn each(&self, end: usize, mut f: impl FnMut(usize, &T)) {
        let _one = self.one();
        for (p, page) in self.pages.iter().enumerate().take(end.div_ceil(PAGE)) {
            let page = page.lock().unwrap_or_else(|e| e.into_inner());
            for (i, slot) in page.iter().flat_map(|slots| slots.iter().enumerate()) {
                if let Some(held) = slot
                    && p * PAGE + i < end
                {
                    f(p * PAGE + i, held);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table with a lock a page holds what it is given as the one table
    /// does: a slot that held nothing and gets nothing makes no page and
    /// tells nobody, what a slot held comes back when it is replaced, and
    /// the slots that hold something are gone through in order, up to an
    /// end. From several threads at once too, each with slots of its own.
    #[test]
    fn a_table_with_a_lock_a_page_holds_what_it_is_given() {
        for one in [false, true] {
            let pages: Pages<usize> = Pages::new(3 * PAGE + 7, one);
            let mut told = Vec::new();
            assert_eq!(pages.replace(5, None, |holds| told.push(holds)), None);
            assert!(told.is_empty() && pages.with(5, |held| held.is_none()));
            assert_eq!(pages.replace(5, Some(50), |holds| told.push(holds)), None);
            assert_eq!(pages.replace(5, Some(51), |holds| told.push(holds)), Some(50));
            assert_eq!(pages.replace(5, None, |holds| told.push(holds)), Some(51));
            assert_eq!(pages.replace(6, None, |holds| told.push(holds)), None, "its page is made: told");
            assert_eq!(told, [true, true, false, false]);
            assert_eq!(pages.replace(3 * PAGE + 7, Some(1), |_| panic!("past the end")), None);
            assert!(pages.with(3 * PAGE + 7, |held| held.is_none()) && pages.with(usize::MAX, |held| held.is_none()));
            std::thread::scope(|scope| {
                for t in 0..4 {
                    let pages = &pages;
                    scope.spawn(move || {
                        for round in 0..200 {
                            for i in (t..3 * PAGE + 7).step_by(4) {
                                let old = pages.replace(i, Some(i * 1000 + round), |holds| assert!(holds));
                                assert_eq!(old, (round > 0).then(|| i * 1000 + round - 1));
                            }
                        }
                    });
                }
            });
            let mut seen = Vec::new();
            pages.each(2 * PAGE + 3, |i, held| seen.push((i, *held)));
            assert_eq!(seen, (0..2 * PAGE + 3).map(|i| (i, i * 1000 + 199)).collect::<Vec<_>>());
            assert_eq!(pages.with(3 * PAGE + 6, |held| held.copied()), Some((3 * PAGE + 6) * 1000 + 199));
            // Pages made by several threads at once, in fresh tables: each
            // keeps what every thread put there (as one thread puts it all).
            for _ in 0..200 {
                let pages: Pages<usize> = Pages::new(4 * PAGE, one);
                std::thread::scope(|scope| {
                    for t in 0..4 {
                        let pages = &pages;
                        scope.spawn(move || {
                            for i in (t..4 * PAGE).step_by(4 * PAGE / 16) {
                                assert_eq!(pages.replace(i, Some(i), |holds| assert!(holds)), None);
                            }
                        });
                    }
                });
                let mut seen = Vec::new();
                pages.each(usize::MAX, |i, held| seen.push((i, *held)));
                let mut alone = Vec::new();
                for t in 0..4 {
                    alone.extend((t..4 * PAGE).step_by(4 * PAGE / 16).map(|i| (i, i)));
                }
                alone.sort_unstable();
                assert_eq!(seen, alone, "what every thread put in the pages it made at once");
            }
        }
    }
    use std::sync::{Arc, Mutex};

    fn made<T>(table: &Paged<T>) -> usize {
        table.pages.iter().flatten().count()
    }

    #[test]
    fn a_table_makes_pages_only_for_what_it_holds() {
        let mut table = Paged::<u32>::new(10 * PAGE);
        assert_eq!(made(&table), 0);
        assert_eq!(table.get(5), None);
        // Writing nothing where nothing is held makes no page.
        assert!(table.slot(5, false).is_none());
        assert_eq!(made(&table), 0);
        *table.slot(5, true).unwrap() = Some(7);
        *table.slot(7 * PAGE + 3, true).unwrap() = Some(9);
        assert_eq!(made(&table), 2);
        assert_eq!(
            (table.get(5), table.get(7 * PAGE + 3)),
            (Some(&7), Some(&9))
        );
        // Slots of a made page are empty until written, as are the rest.
        assert_eq!(
            (table.get(6), table.get(PAGE), table.get(8 * PAGE)),
            (None, None, None)
        );
        // A page that is made takes writes of nothing: the old entry comes back.
        assert_eq!(table.slot(5, false).unwrap().take(), Some(7));
        assert_eq!(table.get(5), None);
        assert_eq!(made(&table), 2);
        // Replacing returns the old entry.
        let slot = table.slot(7 * PAGE + 3, true).unwrap();
        assert_eq!(slot.replace(10), Some(9));
        assert_eq!(table.get(7 * PAGE + 3), Some(&10));
    }

    #[test]
    fn slots_past_the_end_hold_nothing_and_make_no_page() {
        // The last page is short: its slots past the end do not exist.
        let end = 2 * PAGE + 10;
        let mut table = Paged::<u32>::new(end);
        assert!(table.slot(end, true).is_none());
        assert!(table.slot(end + 1, true).is_none());
        assert!(table.slot(3 * PAGE, true).is_none());
        assert!(table.slot(usize::MAX, true).is_none());
        assert_eq!(made(&table), 0);
        assert_eq!(table.pages[2], None);
        *table.slot(end - 1, true).unwrap() = Some(1);
        assert_eq!(table.pages[2].as_ref().map(|p| p.len()), Some(10));
        assert_eq!(
            (table.get(end - 1), table.get(end), table.get(usize::MAX)),
            (Some(&1), None, None)
        );
        // No slots, no pages.
        let mut none = Paged::<u32>::new(0);
        assert!(none.slot(0, true).is_none());
        assert_eq!(none.get(0), None);
    }

    /// Releases what a dense table's `Vec` does, in the same order.
    #[test]
    fn a_table_drops_its_entries_in_slot_order() {
        struct Entry(usize, Arc<Mutex<Vec<usize>>>);
        impl Drop for Entry {
            fn drop(&mut self) {
                self.1.lock().unwrap().push(self.0);
            }
        }
        let dropped = Arc::new(Mutex::new(Vec::new()));
        let mut table = Paged::new(5 * PAGE);
        let slots = [4 * PAGE + 1, 3, 2 * PAGE, PAGE + 5, 2];
        for &i in &slots {
            *table.slot(i, true).unwrap() = Some(Entry(i, dropped.clone()));
        }
        // Replaced by an entry that drops when the table does.
        let old = table
            .slot(3, true)
            .unwrap()
            .replace(Entry(3, dropped.clone()));
        assert!(dropped.lock().unwrap().is_empty());
        drop(old);
        assert_eq!(*dropped.lock().unwrap(), [3]);
        drop(table);
        assert_eq!(
            *dropped.lock().unwrap(),
            [3, 2, 3, PAGE + 5, 2 * PAGE, 4 * PAGE + 1]
        );
    }
}
