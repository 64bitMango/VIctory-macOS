//! A census of the title's physical-page views, logged with the per-frame
//! copy statistics (`victory_gpu::also_log_per_frame`): the views alive,
//! the runs they make (views that continue each other: the next one at the
//! end of the last, showing the next pages of the same section), and the
//! views mapped and unmapped per frame. Under Wine each view is one entry
//! in the host's list of views, which every map and unmap walks, and
//! `VirtualProtect` takes one view a call: the runs are what joining views
//! could bring both down to. After SlaktarnVI / avatarmanz e823263
//! (`view_runs`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::{PAGE, Physical, physical};

/// Views mapped and unmapped since the last line.
static CALLS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A view was mapped (from `map`): the census logs from then on (with
/// the `diagnostics` feature, as the copy statistics).
pub(super) fn mapped() {
    #[cfg(feature = "diagnostics")]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| victory_gpu::also_log_per_frame(log));
    }
    CALLS[0].fetch_add(1, Relaxed);
}

/// A view was unmapped.
pub(super) fn unmapped() {
    CALLS[1].fetch_add(1, Relaxed);
}

/// Views alive, and the runs of views that continue each other they make.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
pub(super) fn view_runs(physical: &Physical) -> (usize, usize) {
    let mut runs = 0;
    let mut next: Option<(usize, *const super::Section, usize)> = None;
    for (&at, (len, slice)) in &physical.views {
        let section = Arc::as_ptr(&slice.section);
        if next != Some((at, section, slice.index)) {
            runs += 1;
        }
        next = Some((at + len, section, slice.index + len / PAGE));
    }
    (physical.views.len(), runs)
}

/// Logs the census over `frames` frames.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
fn log(frames: f64) {
    let (views, runs) = view_runs(&physical());
    let [maps, unmaps] = [0, 1].map(|i| CALLS[i].swap(0, Relaxed) as f64 / frames.max(1.0));
    tracing::info!(
        "  title views: {views} alive in {runs} runs of views that continue each other; \
         {maps:.1} mapped and {unmaps:.1} unmapped per frame (protect calls: the line above)"
    );
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
        MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
        PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::super::{PAGE, Section, Slice};
    use super::*;

    /// Runs of views that continue each other count once; a gap, another
    /// section or pages out of order start a new run.
    #[test]
    fn views_that_continue_each_other_make_one_run() {
        // (No handle: dropping one closes nothing.)
        let section = || {
            Arc::new(Section {
                handle: std::ptr::null_mut(),
                size: 0,
                whole: std::sync::OnceLock::new(),
            })
        };
        let (a, b) = (section(), section());
        let mut p = Physical::default();
        let mut put = |at: usize, len: usize, s: &Arc<Section>, index: usize| {
            let slice = Slice {
                section: s.clone(),
                index,
            };
            p.views.insert(at, (len, slice));
        };
        put(0x10_0000, PAGE, &a, 0);
        put(0x10_0000 + PAGE, 2 * PAGE, &a, 1); // continues
        put(0x10_0000 + 3 * PAGE, PAGE, &a, 5); // out of order
        put(0x10_0000 + 4 * PAGE, PAGE, &b, 6); // another section
        put(0x10_0000 + 6 * PAGE, PAGE, &b, 7); // a gap
        put(0x10_0000 + 7 * PAGE, PAGE, &b, 8); // continues
        assert_eq!(view_runs(&p), (6, 4));
    }

    /// The host joins two placeholders left by unmapped views of a section
    /// into one (`MEM_COALESCE_PLACEHOLDERS`), and maps one view of both
    /// pages over it, which shows what the two showed. Joining views needs
    /// this; under Wine it tells whether the host has it.
    #[test]
    fn unmapped_views_placeholders_join_into_one_view() {
        const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
        // SAFETY (the whole test): a placeholder and a section of our own.
        unsafe {
            let process = GetCurrentProcess();
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                process,
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let view = |at: usize, offset: usize, len: usize| {
                MapViewOfFile3(
                    section,
                    process,
                    at as *const c_void,
                    offset as u64,
                    len,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
                .Value as usize
            };
            let unmap = |at: usize| {
                let v = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                };
                UnmapViewOfFile2(process, v, MEM_PRESERVE_PLACEHOLDER)
            };
            // One view a page, as the GPU's page mappings make them.
            assert_ne!(
                VirtualFree(base as *mut c_void, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                0
            );
            assert_eq!(view(base, 0, PAGE), base);
            assert_eq!(view(base + PAGE, PAGE, PAGE), base + PAGE);
            ((base + 8) as *mut u32).write(0x1234_5678);
            ((base + PAGE + 8) as *mut u32).write(0x9abc_def0);
            assert_ne!(unmap(base), 0);
            assert_ne!(unmap(base + PAGE), 0);
            let joined = VirtualFree(
                base as *mut c_void,
                2 * PAGE,
                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
            );
            assert_ne!(
                joined,
                0,
                "placeholders joined: {}",
                std::io::Error::last_os_error()
            );
            assert_eq!(
                view(base, 0, 2 * PAGE),
                base,
                "one view over the joined placeholder: {}",
                std::io::Error::last_os_error()
            );
            assert_eq!(((base + 8) as *const u32).read(), 0x1234_5678);
            assert_eq!(((base + PAGE + 8) as *const u32).read(), 0x9abc_def0);
            assert_ne!(unmap(base), 0);
            assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
            CloseHandle(section);
        }
    }
}
