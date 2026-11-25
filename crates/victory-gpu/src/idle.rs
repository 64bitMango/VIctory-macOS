//! Which host copies of title memory outside buffers (`crate::shadow`, the
//! 4 MiB windows) are idle, and what is remembered of those released: plain
//! std, so its tests run everywhere.
//!
//! The title unmaps little of the memory under its windows, so they were
//! never freed: in the world they grew by about 0.45 GB a minute (7.3 GB to
//! 12.8 GB of copies in 20 minutes, the process from 24.6 GB to 33.8 GB).

use std::collections::{BTreeMap, BTreeSet};

/// A window nothing asked for in this many seconds, and this many presented
/// frames, is released (`VICTORY_IDLE_RELEASE_S`). Two minutes, as host
/// textures (`crate::resource::IDLE_MS`): released after 30 s, four windows
/// in ten were made again (549 of 1,350 in ten minutes of the world). Frames
/// as well as time, as there: in runs slower than a frame a second the
/// frames decide, so a slow run does not cycle what a scene comes back to.
pub(crate) const IDLE_S: u64 = 120;

/// The idle time in seconds from `VICTORY_IDLE_RELEASE` (`0`: nothing is
/// released) and `VICTORY_IDLE_RELEASE_S` (a number above zero, else
/// [`IDLE_S`]); none when off.
pub(crate) fn setting(release: Option<&str>, seconds: Option<&str>) -> Option<u64> {
    if release.is_some_and(|v| v.trim() == "0") {
        return None;
    }
    let seconds = seconds.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&s| s > 0);
    Some(seconds.unwrap_or(IDLE_S))
}

/// When something was asked for, or now: milliseconds and presented frames.
pub(crate) type Stamp = (u64, u64);

/// Whether what was last asked for at `asked` is idle at `now`: for
/// `seconds` seconds and as many presented frames.
pub(crate) fn idle(asked: Stamp, now: Stamp, seconds: u64) -> bool {
    now.0.saturating_sub(asked.0) >= seconds.saturating_mul(1000) && now.1.saturating_sub(asked.1) >= seconds
}

/// Windows remembered at most (the title's memory holds a few thousand).
const REMEMBERED: usize = 1 << 16;

/// What is remembered of windows released for idleness, by (start, bytes).
#[derive(Default)]
pub(crate) struct Released {
    /// Released and not made again since.
    gone: BTreeSet<(u64, u64)>,
    /// The longest of them.
    longest: u64,
    /// Made again because shaders read their memory at addresses they
    /// computed and found no copy: such reads ask for nothing, so these are
    /// never released again.
    pinned: BTreeSet<(u64, u64)>,
    /// Title memory (start, end) such shaders found no copy for since the
    /// last [`Released::wanted`].
    missed: BTreeMap<u64, u64>,
}

impl Released {
    pub(crate) const fn new() -> Released {
        Released { gone: BTreeSet::new(), longest: 0, pinned: BTreeSet::new(), missed: BTreeMap::new() }
    }

    /// The window `key` was released.
    pub(crate) fn release(&mut self, key: (u64, u64)) {
        if self.gone.len() >= REMEMBERED {
            self.gone.clear();
            self.longest = 0;
        }
        self.gone.insert(key);
        self.longest = self.longest.max(key.1);
    }

    /// A window `key` was made: whether one was released there.
    pub(crate) fn made(&mut self, key: (u64, u64)) -> bool {
        self.gone.remove(&key)
    }

    /// Whether `key` is never released ([`Released::wanted`] made it again).
    pub(crate) fn pinned(&self, key: (u64, u64)) -> bool {
        self.pinned.contains(&key)
    }

    /// How many windows are never released.
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn pins(&self) -> usize {
        self.pinned.len()
    }

    /// Shaders found no copy for title memory `[start, end)`.
    pub(crate) fn missed(&mut self, start: u64, end: u64) {
        if self.gone.is_empty() || self.missed.len() >= REMEMBERED {
            return;
        }
        let e = self.missed.entry(start).or_insert(end);
        *e = (*e).max(end);
    }

    /// The released windows holding memory shaders found no copy for since
    /// the last call: to be made again, and never released after.
    pub(crate) fn wanted(&mut self) -> Vec<(u64, u64)> {
        let mut wanted = BTreeSet::new();
        for (from, to) in std::mem::take(&mut self.missed) {
            let near = self.gone.range((from.saturating_sub(self.longest), 0)..(to, 0));
            wanted.extend(near.filter(|&&(start, len)| start + len > from).copied());
        }
        if self.pinned.len() + wanted.len() > REMEMBERED {
            self.pinned.clear();
        }
        self.pinned.extend(wanted.iter().copied());
        wanted.into_iter().collect()
    }

    /// Forgets everything about windows inside `[start, end)` (tests).
    #[cfg(test)]
    pub(crate) fn forget(&mut self, start: u64, end: u64) {
        let inside = |k: &(u64, u64)| !(start <= k.0 && k.0 + k.1 <= end);
        self.gone.retain(inside);
        self.pinned.retain(inside);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_variables_turn_it_off_and_set_the_time() {
        assert_eq!(setting(None, None), Some(IDLE_S), "on by default");
        assert_eq!(setting(Some("0"), Some("30")), None);
        assert_eq!(setting(Some(" 0 "), None), None);
        assert_eq!(setting(Some("1"), Some(" 45 ")), Some(45));
        assert_eq!(setting(None, Some("0")), Some(IDLE_S), "no idle time of zero");
        assert_eq!(setting(None, Some("soon")), Some(IDLE_S));
    }

    /// Idle only once both the seconds and as many frames have passed.
    #[test]
    fn idle_takes_the_time_and_the_frames() {
        let asked = (10_000, 50);
        assert!(!idle(asked, (39_999, 500), 30), "a millisecond short");
        assert!(!idle(asked, (400_000, 79), 30), "a frame short (a slow run)");
        assert!(idle(asked, (40_000, 80), 30));
        assert!(!idle(asked, (5_000, 20), 30), "asked for after `now` was read");
        assert!(idle((0, 0), (u64::MAX, u64::MAX), u64::MAX), "no overflow");
    }

    /// A window made where one was released counts once; one made anywhere
    /// else does not.
    #[test]
    fn windows_made_again_are_told_from_new_ones() {
        let mut r = Released::new();
        r.release((0x40_0000, 0x40_0000));
        assert!(!r.made((0x80_0000, 0x40_0000)), "never released");
        assert!(!r.made((0x40_0000, 0x80_0000)), "another window from the same start");
        assert!(r.made((0x40_0000, 0x40_0000)));
        assert!(!r.made((0x40_0000, 0x40_0000)), "counted once");
    }

    /// Memory shaders missed names the released windows holding any of it,
    /// each once, and those are pinned; misses elsewhere name nothing.
    #[test]
    fn missed_memory_names_the_released_windows_holding_it() {
        let mut r = Released::new();
        r.missed(0x40_0000, 0x41_0000);
        assert!(r.wanted().is_empty(), "nothing released yet");
        let (a, b, long) = ((0x40_0000, 0x40_0000), (0x80_0000, 0x40_0000), (0x100_0000, 0x80_0000));
        for key in [a, b, long] {
            r.release(key);
        }
        // A page in `a`, two in the long window's second half, one in none.
        r.missed(0x41_0000, 0x42_0000);
        r.missed(0x150_0000, 0x151_0000);
        r.missed(0x152_0000, 0x153_0000);
        r.missed(0xc0_0000, 0xc1_0000);
        assert_eq!(r.wanted(), [a, long]);
        assert!(r.pinned(a) && r.pinned(long) && !r.pinned(b));
        assert_eq!(r.pins(), 2);
        assert!(r.wanted().is_empty(), "taken");
        // A 256 MiB region with no copy at all names every window in it.
        r.missed(0, 1 << 28);
        assert_eq!(r.wanted(), [a, b, long]);
        // The range's end is not in it.
        let mut r = Released::new();
        r.release(b);
        r.missed(0x7f_0000, 0x80_0000);
        assert!(r.wanted().is_empty());
        r.missed(0xbf_0000, 0xc0_0000);
        assert_eq!(r.wanted(), [b]);
    }

    #[test]
    fn forgetting_takes_the_windows_inside_a_range() {
        let mut r = Released::new();
        let (a, b) = ((0x40_0000, 0x40_0000), (0x80_0000, 0x40_0000));
        r.release(a);
        r.release(b);
        r.missed(0, 1 << 28);
        r.wanted();
        r.forget(0x40_0000, 0x80_0000);
        assert!(!r.pinned(a) && r.pinned(b));
        assert!(!r.made(a) && r.made(b));
    }
}
