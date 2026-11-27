//! Loop-limit exits (`VICTORY_LIMIT_EXITS=1`; off by default). After
//! SlaktarnVI / avatarmanz e823263 (`LIMIT_EXITS`).
//!
//! A loop the console jumps back to without a test (its latch ends in
//! `s_branch` to the header) is cut at its iteration limit ([`super::loop_limit`])
//! by a return from inside the loop: the lane leaves the program there. The
//! host compiler then keeps every value the code after the loop might read
//! live around the whole loop, on every path. With the switch, the limit is
//! kept at an exit branch the loop already has and every iteration passes
//! (the latest branch on the dominator path from the latch up to the header
//! that leaves the loop one way and stays in it the other): a lane past the
//! limit leaves the loop there, at most one iteration later, and runs the
//! code after it. The fork measured one compute program at 374 host
//! registers with the return (spilling) and 144 through its exit (none).
//!
//! What a translation meets is recorded whether the switch is on or not
//! ([`meeting`]), so the shader cache can key only the programs whose
//! translation the switch changes.

use std::cell::Cell;
use std::sync::OnceLock;

/// Whether loop limits are kept at an existing exit (`VICTORY_LIMIT_EXITS`
/// `1`, `on`, `true` or `yes`). Read once; tests may override it per thread
/// ([`with_limit_exits`]).
pub fn limit_exits() -> bool {
    if let Some(on) = FORCE.get() {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse(std::env::var("VICTORY_LIMIT_EXITS").ok().as_deref()))
}

/// `VICTORY_LIMIT_EXITS`'s value: on for `1`, `on`, `true`, `yes`. Pure,
/// for testing.
fn parse(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true" | "yes")
    })
}

thread_local! {
    /// A per-thread override of [`limit_exits`] (tests).
    static FORCE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Whether the translations run in [`meeting`] met the pattern, while it
    /// runs on this thread.
    static MET: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Runs `f` with [`limit_exits`] as `on` on this thread.
pub fn with_limit_exits<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let was = FORCE.replace(Some(on));
    let out = f();
    FORCE.set(was);
    out
}

/// Runs `f` (translations) and says whether any of them has a loop whose
/// limit the switch moves to an exit: whether its output differs with the
/// switch on or off, whatever [`limit_exits`] holds. Every attempt a
/// translation makes counts.
pub fn meeting<T>(f: impl FnOnce() -> T) -> (T, bool) {
    let was = MET.replace(Some(false));
    let out = f();
    let met = MET.replace(was).unwrap_or(false);
    if was.is_some() && met {
        MET.set(Some(true));
    }
    (out, met)
}

/// Whether the pattern must be looked for: the switch is on, or a
/// [`meeting`] is listening on this thread.
pub(super) fn looking() -> bool {
    limit_exits() || MET.get().is_some()
}

/// A translation met the pattern.
pub(super) fn met() {
    MET.with(|m| {
        if m.get().is_some() {
            m.set(Some(true));
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_switch_is_off_unless_set() {
        use super::parse;
        assert!(!parse(None));
        assert!(!parse(Some("0")));
        assert!(!parse(Some("off")));
        assert!(!parse(Some("")));
        assert!(parse(Some("1")));
        assert!(parse(Some(" ON ")));
        assert!(parse(Some("yes")));
    }

    #[test]
    fn meeting_gives_what_was_met_inside() {
        super::met();
        assert!(!super::meeting(|| ()).1, "a meeting outside is not counted");
        let ((_, inner), outer) = super::meeting(|| super::meeting(super::met));
        assert!(inner);
        assert!(outer, "an outer meeting sees an inner one");
        assert!(!super::meeting(|| super::meeting(|| ())).1);
    }

    #[test]
    fn the_override_is_per_call() {
        let before = super::limit_exits();
        assert!(super::with_limit_exits(true, super::limit_exits));
        assert!(!super::with_limit_exits(false, super::limit_exits));
        assert_eq!(super::limit_exits(), before);
    }
}
