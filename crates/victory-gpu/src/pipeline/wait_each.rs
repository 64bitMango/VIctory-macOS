//! A budget of waiting per pipeline over all frames
//! (`VICTORY_PIPELINE_WAIT_EACH_MS`; unset or 0: none, as before). Draws
//! wait for a pipeline that is not made yet within the frame's budget
//! (`make::wait`): one whose build outlasts that budget, or that is never
//! made, takes the whole of it again in every frame that binds it. With a
//! budget of its own, once it has used that up a draw with it is dropped at
//! once, as one past the frame's budget is, until a worker has made it. A
//! stall guard. After SlaktarnVI / avatarmanz e823263
//! (`PIPELINE_WAIT_EACH_MS`, 500 ms by default there).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The budget per pipeline, `None` for none. Read once.
fn each() -> Option<Duration> {
    static EACH: OnceLock<Option<Duration>> = OnceLock::new();
    *EACH.get_or_init(|| {
        let each = parse(std::env::var("VICTORY_PIPELINE_WAIT_EACH_MS").ok().as_deref());
        if let Some(each) = each {
            tracing::info!(ms = each.as_millis() as u64, "host pipeline wait budget per pipeline");
        }
        each
    })
}

/// `VICTORY_PIPELINE_WAIT_EACH_MS`'s value: milliseconds, `None` when it
/// is unset, unreadable or 0. Pure, for testing.
fn parse(value: Option<&str>) -> Option<Duration> {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&ms| ms != 0)
        .map(Duration::from_millis)
}

/// What is left of a pipeline's own budget once `waited_us` was waited for
/// it: `Duration::MAX` without one.
pub(super) fn left(waited_us: &AtomicU64) -> Duration {
    left_of(each(), waited_us.load(Ordering::Relaxed))
}

fn left_of(each: Option<Duration>, waited_us: u64) -> Duration {
    each.map_or(Duration::MAX, |each| each.saturating_sub(Duration::from_micros(waited_us)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_budget_unless_set() {
        assert_eq!(parse(None), None);
        assert_eq!(parse(Some("0")), None);
        assert_eq!(parse(Some("x")), None);
        assert_eq!(parse(Some(" 500 ")), Some(Duration::from_millis(500)));
        assert_eq!(left_of(None, 10_000_000), Duration::MAX, "none: as before");
    }

    #[test]
    fn a_pipeline_uses_its_budget_up_over_frames() {
        let each = Some(Duration::from_millis(500));
        assert_eq!(left_of(each, 0), Duration::from_millis(500));
        assert_eq!(left_of(each, 200_000), Duration::from_millis(300));
        assert_eq!(left_of(each, 500_000), Duration::ZERO);
        assert_eq!(left_of(each, 900_000), Duration::ZERO);
    }
}
