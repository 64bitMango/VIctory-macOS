//! `VICTORY_LIMIT_EXITS=1` keys only the programs whose translation it
//! changes (`victory_shader::emit::limit_exits`): every other program keeps
//! its cache entry, its code and so its host pipelines. Whether a program
//! meets the pattern is worked out once (a translation, milliseconds) and
//! kept in the cache as its own entry (`shader.limit-exits-met`). The run
//! logs how many programs met it.

use std::collections::HashSet;
use std::sync::Mutex;

use victory_pipeline::{Cache, Converter};

/// `build` for `container` with the switch on: a build of its own if its
/// translation meets the pattern, `None` (keep `build`) otherwise or with
/// the switch off.
pub(super) fn build(
    cache: Option<&Cache>,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<victory_shader::emit::HostShader, String> + Send + Sync,
) -> Option<[u8; 32]> {
    if !victory_shader::emit::limit_exits::limit_exits() {
        return None;
    }
    let touch = |c: &[u8]| {
        let (_, met) = victory_shader::emit::limit_exits::meeting(|| translate(c));
        met
    };
    let met = match cache {
        Some(cache) => met_cached(cache, build, container, touch),
        None => touch(container),
    };
    note(container, met);
    met.then(|| keyed(build))
}

/// `build` for a program the switch changes.
fn keyed(build: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(b"loop limits at exits\n");
    *h.finalize().as_bytes()
}

/// The cache entry of whether a translation meets the pattern: the build
/// and the container in, one byte out.
struct Met<F> {
    touch: F,
}

impl<F: Fn(&[u8]) -> bool + Send + Sync> Converter for Met<F> {
    fn id(&self) -> &'static str {
        "shader.limit-exits-met"
    }
    fn version(&self) -> u32 {
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        Ok(vec![u8::from((self.touch)(&input[32..]))])
    }
}

/// Whether `container`'s translation under `build` meets the pattern,
/// worked out by `touch` once and then read from the cache.
fn met_cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    touch: impl Fn(&[u8]) -> bool + Send + Sync,
) -> bool {
    let input = [&build[..], container].concat();
    let converter = Met { touch };
    cache
        .get_or_convert(&converter, &input)
        .ok()
        .and_then(|path| crate::read_file(&path).ok())
        .and_then(|b| b.first().copied())
        .map_or_else(|| (converter.touch)(container), |b| b != 0)
}

/// Programs looked at and those that met the pattern, each once.
static SEEN: Mutex<Option<(HashSet<[u8; 32]>, u64)>> = Mutex::new(None);

/// Counts `container` once, and logs the counts when it is a new program
/// that meets the pattern.
fn note(container: &[u8], met: bool) {
    let id = *blake3::hash(container).as_bytes();
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let (all, hits) = seen.get_or_insert_with(Default::default);
    if !all.insert(id) {
        return;
    }
    if met {
        *hits += 1;
        tracing::info!(
            met = *hits,
            looked_at = all.len(),
            "VICTORY_LIMIT_EXITS: a program's loop limit kept at an exit"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A program that meets the pattern gets a build of its own; one that
    /// does not keeps its own. What a program meets is worked out once,
    /// then read from the cache.
    #[test]
    fn the_switch_keys_only_the_programs_it_changes() {
        let build = [7u8; 32];
        assert_ne!(keyed(&build), build);
        assert_eq!(keyed(&build), keyed(&build));
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let touch = |c: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            c == b"loop"
        };
        for _ in 0..2 {
            assert!(!met_cached(&cache, &build, b"plain", touch));
            assert!(met_cached(&cache, &build, b"loop", touch));
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2, "worked out once each");
    }
}
