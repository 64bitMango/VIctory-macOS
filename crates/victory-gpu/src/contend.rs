//! Locks whose waits are counted (diagnostics): which of the locks the
//! title's recording threads and the submitting thread share makes them
//! wait, how long and how often, logged with the frame-rate line
//! (`crate::copystat`, "locks per frame"). Without the feature these are
//! the plain locks.
#![cfg_attr(not(windows), allow(dead_code))]

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// The locks counted.
#[derive(Clone, Copy)]
pub(crate) enum Lock {
    /// `crate::descriptor::SHADER_HEAPS`, by title threads writing a view.
    Heaps,
    /// What a host heap's slots hold (`crate::descriptor::host::HostHeap::kept`).
    Kept,
    /// The views table (`crate::descriptor::VIEWS`).
    Views,
    /// What a heap's slots view (`crate::bindless::Viewed`).
    Viewed,
    /// The registries of copies of title memory (`crate::shadow`).
    Copies,
}

thread_local! {
    /// This thread is in a heap sync: its waits for slot locks are counted
    /// apart from the title threads'.
    static SYNCING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Until dropped, this thread's slot lock waits are a heap sync's.
pub(crate) struct Syncing(bool);

pub(crate) fn syncing() -> Syncing {
    Syncing(SYNCING.with(|s| s.replace(true)))
}

impl Drop for Syncing {
    fn drop(&mut self) {
        SYNCING.with(|s| s.set(self.0));
    }
}

/// Whether this thread is in a heap sync.
#[cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]
pub(crate) fn in_sync() -> bool {
    SYNCING.with(|s| s.get())
}

/// A wait of this thread for `lock` since `since` is over.
#[cfg(all(windows, feature = "diagnostics"))]
fn waited(lock: Lock, since: std::time::Instant) {
    use crate::copystat::C;
    let (us, times) = match lock {
        Lock::Heaps => (C::MirrorWaitUs, C::HeapsContended),
        Lock::Kept => (C::KeptWaitUs, C::KeptContended),
        Lock::Views if in_sync() => (C::ViewsSyncWaitUs, C::ViewsSyncContended),
        Lock::Views => (C::ViewsWaitUs, C::ViewsContended),
        Lock::Viewed if in_sync() => (C::ViewedSyncWaitUs, C::ViewedSyncContended),
        Lock::Viewed => (C::ViewedWaitUs, C::ViewedContended),
        Lock::Copies => (C::CopiesWaitUs, C::CopiesContended),
    };
    crate::copystat::add(us, since.elapsed().as_micros() as u64);
    crate::copystat::add(times, 1);
}

/// Takes it at once if it is free; else waits, and counts the wait.
#[cfg(all(windows, feature = "diagnostics"))]
macro_rules! counted {
    ($lock:expr, $which:expr, $try:ident, $take:ident) => {
        match $lock.$try() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let since = std::time::Instant::now();
                let guard = $lock.$take().unwrap_or_else(|e| e.into_inner());
                waited($which, since);
                guard
            }
        }
    };
}

#[cfg(not(all(windows, feature = "diagnostics")))]
macro_rules! counted {
    ($lock:expr, $which:expr, $try:ident, $take:ident) => {{
        let _: Lock = $which;
        $lock.$take().unwrap_or_else(|e| e.into_inner())
    }};
}

/// `mutex`, locked.
pub(crate) fn lock<T>(mutex: &Mutex<T>, which: Lock) -> MutexGuard<'_, T> {
    counted!(mutex, which, try_lock, lock)
}

/// `lock`, held shared.
pub(crate) fn read<T>(lock: &RwLock<T>, which: Lock) -> RwLockReadGuard<'_, T> {
    counted!(lock, which, try_read, read)
}

/// `lock`, held alone.
pub(crate) fn write<T>(lock: &RwLock<T>, which: Lock) -> RwLockWriteGuard<'_, T> {
    counted!(lock, which, try_write, write)
}
