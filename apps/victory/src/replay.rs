//! `victory replay`: a captured frame's GPU work run again without the
//! title (docs/replay.md).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use victory_gpu::replay::play::{self, Detail, Options, Waits};

#[derive(Clone, Copy, ValueEnum)]
enum Timing {
    /// Each submission.
    Submissions,
    /// Each command too: by timestamps where the host's tell commands
    /// apart, else in pieces.
    Detailed,
    /// Each command by a timestamp after it.
    Stamps,
    /// Each command by sending the frame in pieces, each waited for.
    Pieces,
}

#[derive(Args)]
pub(crate) struct Replay {
    /// The folder given as VICTORY_RECORD (its last capture) or a capture.
    folder: PathBuf,
    /// How often the frame is run.
    #[arg(long, default_value_t = 5)]
    loops: u32,
    /// Do not put back, between runs, what a run changed.
    #[arg(long)]
    no_restore: bool,
    #[arg(long, value_enum, default_value_t = Timing::Submissions)]
    timing: Timing,
    /// Pipelines listed with detailed timing.
    #[arg(long, default_value_t = 20)]
    top: usize,
    /// Where the picture goes (default: replay.png in the capture).
    #[arg(long)]
    image: Option<PathBuf>,
    /// Run the frame unchanged too, turn by turn, and print both.
    #[arg(long)]
    compare: bool,
    /// Send all submissions at once and wait once at the end (the runtime,
    /// and a replay without this, wait for each).
    #[arg(long, conflicts_with = "one_submit")]
    no_waits: bool,
    /// One ExecuteCommandLists for the whole frame.
    #[arg(long)]
    one_submit: bool,
    /// Milliseconds the GPU stands idle after each submission, as it does
    /// while a title prepares the next.
    #[arg(long, default_value_t = 0.0)]
    pace: f64,
    /// Leave out what runs a pipeline whose name or DXIL hash has this in it
    /// (may be given several times).
    #[arg(long = "skip-shader")]
    skip_shader: Vec<String>,
    /// Take shaders from this folder where it has them (`<hash>.dxil`, by
    /// the hash the runtime names the title's shader by; see --retranslate).
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// With --compare: the folder the second run's shaders are from.
    #[arg(long)]
    shaders_b: Option<PathBuf>,
    /// Only translate the capture's shaders again, into this folder (for
    /// --shaders), with this build and the VICTORY_* switches of this run.
    /// The title's containers are those the recorder kept beside the capture.
    #[arg(long)]
    retranslate: Option<PathBuf>,
    /// Only write the frame's resources and the commands of these lists
    /// (`3..7`, or `5`).
    #[arg(long)]
    listing: Option<String>,
    /// Use the D3D12 debug layer, where there is one (slow).
    #[arg(long)]
    debug_layer: bool,
}

/// `3..7` (both included) or `5`.
fn range(text: &str) -> Option<(u32, u32)> {
    match text.split_once("..") {
        Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
        None => text.trim().parse().ok().map(|n| (n, n)),
    }
}

pub(crate) fn run(r: Replay) -> Result<()> {
    if let Some(text) = &r.listing {
        let lists = range(text).with_context(|| format!("--listing {text}"))?;
        return play::listing(&r.folder, lists, &mut |line| println!("{line}"));
    }
    // What a replay writes is made of the title's data, as the capture is.
    for out in r.retranslate.iter().chain(&r.image) {
        victory_core::ensure_outside_repo(out)?;
    }
    if let Some(into) = &r.retranslate {
        let done = play::retranslate(&r.folder, into)?;
        for line in &done.refused {
            println!("not translated: {line}");
        }
        println!(
            "{} of the capture's {} pipelines of the title's had their containers; {} shaders written to {} in {:.1} s",
            done.found,
            done.pipelines,
            done.written,
            into.display(),
            done.seconds
        );
        if done.found < done.pipelines {
            println!(
                "containers are kept in `containers` beside the capture by a run with VICTORY_RECORD set: those of shaders the title passed before that run's recorder was asked for are missing"
            );
        }
        return Ok(());
    }
    let options = Options {
        loops: r.loops.max(1),
        restore: !r.no_restore,
        detail: match r.timing {
            Timing::Submissions => Detail::Off,
            Timing::Detailed => Detail::Auto,
            Timing::Stamps => Detail::Stamps,
            Timing::Pieces => Detail::Pieces,
        },
        top: r.top,
        image: r.image,
        waits: match (r.no_waits, r.one_submit) {
            (_, true) => Waits::OneSubmit,
            (true, _) => Waits::None,
            _ => Waits::Each,
        },
        pace: r.pace,
        skip_shaders: r.skip_shader,
        shaders: r.shaders,
        shaders_b: r.shaders_b,
        compare: r.compare,
        debug: r.debug_layer,
    };
    let reports = play::replay(&r.folder, &options)?;
    for (what, report) in &reports {
        println!("{}", report.text(what));
    }
    if let [(_, before), (_, after)] = &reports[..] {
        println!(
            "{:+.2} ms ({:+.1} %)",
            after.median_ms() - before.median_ms(),
            (after.median_ms() / before.median_ms().max(1e-9) - 1.0) * 100.0
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranges() {
        assert_eq!(super::range("3..7"), Some((3, 7)));
        assert_eq!(super::range("5"), Some((5, 5)));
        assert_eq!(super::range("x"), None);
        assert_eq!(super::range("1..x"), None);
    }
}
