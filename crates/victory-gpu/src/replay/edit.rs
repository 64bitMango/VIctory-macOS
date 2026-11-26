//! Changes to a list's commands: the experiments of a replay, and lists cut
//! into pieces that are timed one by one.

use std::collections::HashSet;

use super::wire::{Cmd, Loc, NONE};

/// Drops what runs a pipeline of `psos`; how many commands that were.
pub fn skip_pipelines(cmds: Vec<Cmd>, psos: &HashSet<u32>) -> (Vec<Cmd>, u32) {
    let mut bound = NONE;
    let mut dropped = 0;
    let out = cmds
        .into_iter()
        .filter(|c| {
            if let Cmd::Pipeline { pso } = c {
                bound = *pso;
            }
            let drop = c.runs_shaders() && psos.contains(&bound);
            dropped += u32::from(drop);
            !drop
        })
        .collect();
    (out, dropped)
}

/// What a command that works is counted as.
pub fn kind(c: &Cmd, dispatches: &HashSet<u32>, mesh: &HashSet<u32>, bound: u32) -> &'static str {
    match c {
        Cmd::Barriers { .. } => "barriers",
        Cmd::Dispatch { .. } => "dispatches",
        Cmd::Indirect { signature, .. } if dispatches.contains(signature) => "indirect dispatches",
        Cmd::Indirect { .. } if mesh.contains(&bound) => "indirect mesh draws",
        Cmd::Indirect { .. } => "indirect draws",
        Cmd::DispatchMesh { .. } => "mesh draws",
        Cmd::Draw { .. } | Cmd::DrawIndexed { .. } => "draws",
        Cmd::ClearDepth { .. } | Cmd::ClearTarget { .. } | Cmd::Discard { .. } => "clears",
        Cmd::Resolve { .. } | Cmd::ResolveRegion { .. } => "resolves",
        Cmd::CopyBuffer { .. } => "copies of buffers",
        Cmd::CopyTexture {
            dst: Loc::Placed { .. },
            ..
        } => "copies of textures to buffers",
        Cmd::CopyTexture {
            src: Loc::Placed { .. },
            ..
        } => "copies of buffers to textures",
        Cmd::CopyTexture { .. } => "copies of textures",
        Cmd::CopyResource { .. } => "copies of whole resources",
        _ => "other",
    }
}

/// What a list has bound: the commands that bind it again, in an order that
/// holds (heaps, then root signatures before their arguments).
#[derive(Clone, Debug, Default)]
pub struct Bound {
    heaps: Option<Cmd>,
    rest: Vec<Cmd>,
}

impl Bound {
    /// Takes note of a command; false for one that binds nothing (it works,
    /// or changes resources' states).
    pub fn note(&mut self, c: &Cmd) -> bool {
        // What a command stands in the place of: an earlier one of its key.
        let same = |a: &Cmd, b: &Cmd| match (a, b) {
            (Cmd::Pipeline { .. }, Cmd::Pipeline { .. })
            | (Cmd::Topology { .. }, Cmd::Topology { .. })
            | (Cmd::Viewports { .. }, Cmd::Viewports { .. })
            | (Cmd::Scissors { .. }, Cmd::Scissors { .. })
            | (Cmd::BlendFactor { .. }, Cmd::BlendFactor { .. })
            | (Cmd::StencilRef { .. }, Cmd::StencilRef { .. })
            | (Cmd::DepthBounds { .. }, Cmd::DepthBounds { .. })
            | (Cmd::IndexBuffer { .. }, Cmd::IndexBuffer { .. })
            | (Cmd::Targets { .. }, Cmd::Targets { .. }) => true,
            (
                Cmd::RootSignature { graphics: a, .. },
                Cmd::RootSignature { graphics: b, .. },
            ) => a == b,
            (
                Cmd::Table {
                    graphics: a,
                    param: p,
                    ..
                },
                Cmd::Table {
                    graphics: b,
                    param: q,
                    ..
                },
            )
            | (
                Cmd::RootView {
                    graphics: a,
                    param: p,
                    ..
                },
                Cmd::RootView {
                    graphics: b,
                    param: q,
                    ..
                },
            ) => (a, p) == (b, q),
            (
                Cmd::Constants {
                    graphics: a,
                    param: p,
                    offset: o,
                    values: v,
                },
                Cmd::Constants {
                    graphics: b,
                    param: q,
                    offset: n,
                    values: w,
                },
            ) => (a, p, o, v.len()) == (b, q, n, w.len()),
            (
                Cmd::VertexBuffers {
                    start: a, count: n, ..
                },
                Cmd::VertexBuffers {
                    start: b, count: m, ..
                },
            ) => (a, n) == (b, m),
            _ => false,
        };
        match c {
            Cmd::Heaps { .. } => self.heaps = Some(c.clone()),
            Cmd::RootSignature { graphics, .. } => {
                // A new root signature has no arguments yet.
                self.rest.retain(|old| {
                    !matches!(
                        old,
                        Cmd::RootSignature { graphics: g, .. }
                        | Cmd::Table { graphics: g, .. }
                        | Cmd::Constants { graphics: g, .. }
                        | Cmd::RootView { graphics: g, .. } if g == graphics
                    )
                });
                self.rest.push(c.clone());
            }
            Cmd::Pipeline { .. }
            | Cmd::Topology { .. }
            | Cmd::Viewports { .. }
            | Cmd::Scissors { .. }
            | Cmd::BlendFactor { .. }
            | Cmd::StencilRef { .. }
            | Cmd::DepthBounds { .. }
            | Cmd::IndexBuffer { .. }
            | Cmd::Targets { .. }
            | Cmd::Table { .. }
            | Cmd::RootView { .. }
            | Cmd::Constants { .. }
            | Cmd::VertexBuffers { .. } => {
                self.rest.retain(|old| !same(old, c));
                self.rest.push(c.clone());
            }
            _ => return false,
        }
        true
    }

    /// The commands that bind again what is bound.
    pub fn again(&self) -> impl Iterator<Item = &Cmd> {
        self.heaps.iter().chain(&self.rest)
    }
}

/// A piece of a list: commands that can run as a list of their own.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub cmds: Vec<Cmd>,
    /// What its work is counted as, and the pipeline it runs ([`NONE`] for
    /// work that runs none).
    pub kind: &'static str,
    pub pso: u32,
    /// Its commands that work.
    pub works: u32,
}

/// Cuts a list into pieces: each holds the commands that work of one kind
/// with one pipeline bound, one after the other, with what is bound at its
/// start bound again in front and the barriers and bindings between them
/// in their places. Run in order, the pieces do what the list does.
pub fn pieces(cmds: &[Cmd], dispatches: &HashSet<u32>, mesh: &HashSet<u32>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut bound = Bound::default();
    let mut pso = NONE;
    // What came since the last command that works, and what was bound
    // before it.
    let mut since: Vec<Cmd> = Vec::new();
    let mut before = bound.clone();
    for c in cmds {
        if let Cmd::Pipeline { pso: p } = c {
            pso = *p;
        }
        let works = c.works();
        if !works {
            bound.note(c);
            since.push(c.clone());
            continue;
        }
        let runs = if c.runs_shaders() { pso } else { NONE };
        let k = kind(c, dispatches, mesh, pso);
        match out.last_mut() {
            Some(last) if last.kind == k && last.pso == runs => {
                last.cmds.append(&mut since);
                last.cmds.push(c.clone());
                last.works += 1;
            }
            _ => {
                let mut all: Vec<Cmd> = before.again().cloned().collect();
                all.append(&mut since);
                all.push(c.clone());
                out.push(Piece {
                    cmds: all,
                    kind: k,
                    pso: runs,
                    works: 1,
                });
            }
        }
        before = bound.clone();
    }
    // What follows the last work (barriers, mostly) still has to run.
    if !since.is_empty() {
        match out.last_mut() {
            Some(last) => last.cmds.append(&mut since),
            None => out.push(Piece {
                cmds: since,
                kind: "other",
                pso: NONE,
                works: 0,
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::wire::{Barrier, Va, slot};

    fn pipeline(pso: u32) -> Cmd {
        Cmd::Pipeline { pso }
    }

    fn draw(n: u32) -> Cmd {
        Cmd::Draw {
            vertices: n,
            instances: 1,
            first: 0,
            first_instance: 0,
        }
    }

    fn barrier(res: u32) -> Cmd {
        Cmd::Barriers {
            all: vec![Barrier::Uav { flags: 0, res }],
        }
    }

    fn root(graphics: bool, root: u32) -> Cmd {
        Cmd::RootSignature { graphics, root }
    }

    fn table(graphics: bool, param: u32, slot: u32) -> Cmd {
        Cmd::Table {
            graphics,
            param,
            heap: 1,
            slot,
        }
    }

    #[test]
    fn pipelines_are_skipped() {
        let copy = Cmd::CopyResource { dst: 1, src: 2 };
        let cmds = vec![
            pipeline(1),
            draw(3),
            pipeline(2),
            draw(4),
            copy.clone(),
            Cmd::Dispatch { groups: [1, 1, 1] },
            pipeline(1),
            draw(5),
        ];
        let (left, dropped) = skip_pipelines(cmds, &HashSet::from([2]));
        assert_eq!(dropped, 2);
        assert_eq!(
            left,
            [pipeline(1), draw(3), pipeline(2), copy, pipeline(1), draw(5)]
        );
    }

    #[test]
    fn what_is_bound_is_bound_again_in_order() {
        let mut b = Bound::default();
        let heaps = Cmd::Heaps { heaps: vec![1, 2] };
        let constants = |offset, values: Vec<u32>| Cmd::Constants {
            graphics: true,
            param: 2,
            offset,
            values,
        };
        let view = |param, offset| Cmd::RootView {
            graphics: false,
            kind: slot::CBV,
            param,
            va: Va::At { res: 3, offset },
        };
        for c in [
            root(true, 1),
            table(true, 0, 10),
            pipeline(7),
            heaps.clone(),
            table(true, 0, 11),
            table(true, 1, 12),
            constants(0, vec![1, 2]),
            constants(2, vec![3]),
            constants(0, vec![4, 5]),
            root(false, 2),
            view(0, 64),
            view(0, 128),
            pipeline(8),
        ] {
            assert!(b.note(&c));
        }
        assert!(!b.note(&draw(3)) && !b.note(&barrier(1)));
        let again: Vec<Cmd> = b.again().cloned().collect();
        assert_eq!(
            again,
            [
                heaps,
                root(true, 1),
                table(true, 0, 11),
                table(true, 1, 12),
                constants(2, vec![3]),
                constants(0, vec![4, 5]),
                root(false, 2),
                view(0, 128),
                pipeline(8),
            ]
        );
        // A new root signature drops its side's arguments, and only those.
        b.note(&root(true, 3));
        let again: Vec<Cmd> = b.again().cloned().collect();
        assert_eq!(
            again[1..],
            [root(false, 2), view(0, 128), pipeline(8), root(true, 3)]
        );
    }

    #[test]
    fn a_list_in_pieces_does_what_the_list_does() {
        let none = HashSet::new();
        let copy = Cmd::CopyResource { dst: 1, src: 2 };
        let cmds = vec![
            root(true, 1),
            pipeline(1),
            table(true, 0, 5),
            draw(1),
            table(true, 0, 6),
            draw(2),
            barrier(9),
            pipeline(2),
            draw(3),
            copy.clone(),
            draw(4),
            barrier(8),
        ];
        let all = pieces(&cmds, &none, &none);
        let shape: Vec<(&str, u32, u32)> = all.iter().map(|p| (p.kind, p.pso, p.works)).collect();
        assert_eq!(
            shape,
            [
                ("draws", 1, 2),
                ("draws", 2, 1),
                ("copies of whole resources", NONE, 1),
                ("draws", 2, 1)
            ]
        );
        assert_eq!(all[0].cmds, cmds[..6]);
        // The second starts with what the first had bound at its end; the
        // barrier and the new pipeline stand where they stood.
        assert_eq!(
            all[1].cmds,
            [
                root(true, 1),
                pipeline(1),
                table(true, 0, 6),
                barrier(9),
                pipeline(2),
                draw(3)
            ]
        );
        assert_eq!(
            all[2].cmds,
            [root(true, 1), table(true, 0, 6), pipeline(2), copy]
        );
        // What follows the last work stays with it.
        assert_eq!(all[3].cmds.last(), Some(&barrier(8)));
        // Every command that works or changes states is there once, in order.
        let once = |all: &[Cmd]| -> Vec<Cmd> {
            all.iter()
                .filter(|c| c.works() || matches!(c, Cmd::Barriers { .. }))
                .cloned()
                .collect()
        };
        let joined: Vec<Cmd> = all.iter().flat_map(|p| p.cmds.clone()).collect();
        assert_eq!(once(&joined), once(&cmds));
        // A list of barriers alone is one piece.
        let only = pieces(&[barrier(1)], &none, &none);
        assert_eq!((only.len(), only[0].works), (1, 0));
        assert!(pieces(&[], &none, &none).is_empty());
    }
}
