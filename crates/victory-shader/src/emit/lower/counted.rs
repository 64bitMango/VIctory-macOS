//! Dispatches counted in threads (the console's `DispatchX` with its "use
//! thread dimensions" bit): the host runs whole groups, so the threads of the
//! last group past the count have to leave by themselves.
//!
//! On the console such a dispatch launches only the threads below the count
//! on each axis: the others are never in EXEC, and a wave none of whose
//! threads is below it is not launched. Here every thread of a group
//! starts. A program made for such dispatches ([`Extra::thread_counts`])
//! reads, per axis with more than one thread in a group, a user-data dword
//! the host sets to the count ([`super::super::DISPATCH_THREADS`]; 0: all
//! of every group, which is what any other dispatch leaves there), and a
//! thread whose number on that axis (its group's times the group's size,
//! plus its own in the group) is not below it leaves at the end of the
//! entry block, before the program's first instruction.
//!
//! The entry block is what every thread runs alike: thread ids (also the
//! relabelling that makes a host lane its console lane, which wants every
//! lane of the host wave there), user data and handles. A thread is told by
//! the id the program itself sees (the relabelled one), so the threads left
//! are the console's, in the console's lanes; wave operations after the
//! entry block see only them, as the console's see only the lanes in EXEC.
//!
//! As halves of console waves ([`super::Halves`]) the program meets its
//! other half at group barriers. A host barrier some thread never reaches
//! is undefined in Metal; an M5 makes the rest meet (ce74732c: groups of 64
//! with any lanes gone, a whole 32-lane wave too), and the form already
//! relies on that where threads have ended. What the cut adds:
//! * A half may have no thread left at an exchange in a block every thread
//!   of the group runs (`full`), where the form stored no value for an
//!   absent half. With the cut every exchange that has such a value stores
//!   it in the entry block, which the threads that leave still run.
//! * The first lane of lanes all in EXEC is taken from the lower half, and
//!   `v_readlane_b32` from the half the lane is in, without a value for an
//!   absent half. Both hold when the threads that leave are the last of
//!   their console wave: the lower half then has a thread wherever the wave
//!   has one, its first is the wave's lane 0, and a lane that left was
//!   never launched on the console either (its register is no one's). That
//!   is so in a group of one dimension, so only such a program is cut as
//!   halves; another is made without the words, and the host says once
//!   that its last group runs whole (`victory-gpu`, `State::dispatch`).
//! * EXEC on entry read as a number is balloted in the block the program
//!   starts in, not in the entry block, where the threads that leave still
//!   are.

use super::super::bitcode::{Bin, Term, V, pred};
use super::super::{DISPATCH_THREADS, UserData};
use super::{Lower, Val};
use crate::ir::Program;
use crate::native::StageIo;
use crate::resource::{MEMORY, Resources, USER_DATA};

/// What [`super::super::Options`] asks of the lowering besides
/// [`super::Parts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Extra {
    /// Permit early depth testing for opaque pixel programs whose only side effect is read tracking.
    pub early_depth: bool,
    /// The threads of a compute group past a dispatch's counts of threads
    /// leave at once (see the module).
    pub thread_counts: bool,
    /// Buffer and global atomics stand between two fences
    /// ([`super::fence`]).
    pub atomic_fences: bool,
}

/// A program whose threads past a dispatch's counts leave
/// ([`Lower::leave_past_counts`]).
pub(super) struct Counted {
    /// The block the program's first instruction is in: the first that
    /// only the threads that stay run.
    pub start: usize,
    /// The user-data dwords the counts arrive in.
    words: Vec<u32>,
}

/// Whether a program reaches title memory by addresses it computes or
/// through the root signature's table in memory, or traces rays: the
/// console's ray-tracing structure builder and what reads its structures
/// (`Lower::root_memory`). Those are the programs made for dispatches
/// counted in threads.
pub(crate) fn by_address(p: &Program, res: &Resources) -> bool {
    res.scalar
        .iter()
        .any(|s| matches!(s.1.param, MEMORY | USER_DATA))
        || p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name().starts_with("image_bvh"))
}

impl Lower<'_> {
    /// See the module: ends the entry block with the test, and has the
    /// program's first block `first` start in a block of its own. Nothing
    /// changes in a program that is not asked to, is no compute program,
    /// has one thread a group, or runs as halves of console waves with
    /// groups of more than one dimension.
    pub(super) fn leave_past_counts(&mut self, first: usize) {
        let StageIo::Cs(cs) = self.x.native.io else {
            return;
        };
        if !self.extra.thread_counts || self.s.mesh.is_some() {
            return;
        }
        // (A first block that is a loop's header has its phis in the entry
        // block already; no program starts so.)
        if !self.x.p.blocks[first].preds.is_empty() {
            return;
        }
        let split = self.halves.as_ref().is_some_and(|h| !h.whole);
        if split && cs.threads[1..] != [1, 1] {
            return;
        }
        let axes: Vec<u8> = (0..3u8)
            .filter(|&c| cs.threads[usize::from(c)] > 1)
            .collect();
        if axes.is_empty() {
            return;
        }
        let e = self.s.entry;
        let zero = self.s.m.i32(0);
        let mut runs: Option<V> = None;
        let mut words = Vec::new();
        for (&c, index) in axes.iter().zip(self.free_user_words()) {
            words.push(index);
            let limit = self.user_at(UserData {
                index,
                slot: u8::MAX,
                root: DISPATCH_THREADS,
                dword: u32::from(c),
            });
            let Val::I32(group) = self.compute_id("groupId.i32", 94, c) else {
                unreachable!()
            };
            let Val::I32(thread) = self.thread_id(c) else {
                unreachable!()
            };
            let size = self.s.m.i32(cs.threads[usize::from(c)]);
            let at = self.s.m.bin(e, Bin::Mul, group, size);
            let at = self.s.m.bin(e, Bin::Add, at, thread);
            let all = self.s.m.cmp(e, pred::EQ, limit, zero);
            let below = self.s.m.cmp(e, pred::IULT, at, limit);
            let ok = self.s.m.bin(e, Bin::Or, all, below);
            runs = Some(match runs {
                Some(r) => self.s.m.bin(e, Bin::And, r, ok),
                None => ok,
            });
        }
        let runs = runs.expect("an axis");
        let (stop, go) = (self.abort_block(), self.s.m.new_block());
        self.s.m.terminate(e, Term::CondBr(runs, go, stop));
        self.block[first] = Some(go);
        self.counted = Some(Counted { start: go, words });
    }

    /// User-data dwords the title's root signature does not name, lowest
    /// first: the host's own words go there, within the user data the
    /// program has anyway where there is room.
    fn free_user_words(&self) -> impl Iterator<Item = u32> + use<> {
        let taken: Vec<(u32, u32)> = self
            .x
            .native
            .layout
            .iter()
            .map(|p| {
                let at = u32::from(p.user_data);
                (at, at + p.sgprs.max(1))
            })
            .collect();
        let read: Vec<u32> = self.user_reads.keys().copied().collect();
        (0..).filter(move |i| !taken.iter().any(|&(a, b)| (a..b).contains(i)) && !read.contains(i))
    }

    /// Whether the words [`Self::leave_past_counts`] reads are still the
    /// host's: nothing the program reads took their place.
    pub(super) fn counts_kept(&self) -> bool {
        self.counted.iter().flat_map(|c| &c.words).all(|i| {
            self.user_reads
                .get(i)
                .is_some_and(|u| u.root == DISPATCH_THREADS)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Lowered, Parts, lower_with_layout};
    use super::Extra;
    use crate::emit::DISPATCH_THREADS;
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;
    const V_LSHLREV_V1_2_V0: u32 = 0x3402_0082; // v_lshlrev_b32 v1, 2, v0
    const DS_WRITE_V1_V2: [u32; 2] = [0xD834_0000, 0x0000_0201]; // ds_write_b32 v1, v2
    const BCNT_S0: u32 = 0xBE80_1000; // s_bcnt1_i32_b64 s0, s[0:1]
    const V_MOV_V2_S0: u32 = 0x7E04_0200; // v_mov_b32 v2, s0

    /// A wave64 compute program of `threads` with 512 bytes of LDS and the
    /// root constants `layout` names, lowered for 32-lane host waves, with
    /// the cut or without.
    fn lowered(words: &[u32], threads: [u32; 3], layout: Vec<RootParam>, cut: bool) -> Lowered {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout,
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 2 << 1,
                threads,
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower_with_layout(
            &Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                atomic_locks: false,
                atomic_lock_rounds: false,
                atomic_umax: false,
                bias_as_gradients: false,
                narrow_waves: true,
                uav_size_level0: false,
                relax: Default::default(),
                view_marks: false,
                psv: None,
                dropped_rows: Vec::new(),
            },
            false,
            false,
            false,
            Extra {
                thread_counts: cut,
                ..Extra::default()
            },
        )
        .expect("lower")
    }

    /// The count words of a translation: (user-data index, axis).
    fn words(l: &Lowered) -> Vec<(u32, u32)> {
        l.user_data
            .iter()
            .filter(|u| u.root == DISPATCH_THREADS)
            .map(|u| (u.index, u.dword))
            .collect()
    }

    /// Each lane stores its id: no wave operation.
    fn plain() -> Vec<u32> {
        let mut w = vec![V_LSHLREV_V1_2_V0];
        w.extend([0xD834_0000, 0x0000_0001]); // ds_write_b32 v1, v0
        w.push(S_ENDPGM);
        w
    }

    /// The lanes with an even id, counted over the console's wave: a mask
    /// read as a number, which the halves exchange.
    fn counting() -> Vec<u32> {
        let mut w = vec![V_LSHLREV_V1_2_V0, 0x3604_0081]; // v_and_b32 v2, 1, v0
        w.extend([0xD4C2_0000, 0x0002_0480]); // v_cmp_eq_u32 s[0:1], 0, v2
        w.extend([BCNT_S0, V_MOV_V2_S0]);
        w.extend(DS_WRITE_V1_V2);
        w.push(S_ENDPGM);
        w
    }

    #[test]
    fn a_program_not_asked_to_reads_no_count() {
        for threads in [[64, 1, 1], [8, 8, 1]] {
            let l = lowered(&plain(), threads, vec![], false);
            assert_eq!(words(&l), []);
        }
    }

    #[test]
    fn a_count_word_per_axis_of_more_than_one_thread() {
        assert_eq!(words(&lowered(&plain(), [64, 1, 1], vec![], true)), [(0, 0)]);
        assert_eq!(words(&lowered(&plain(), [8, 8, 1], vec![], true)), [(0, 0), (1, 1)]);
        assert_eq!(words(&lowered(&plain(), [4, 1, 16], vec![], true)), [(0, 0), (1, 2)]);
        // One thread a group: every count is whole groups.
        assert_eq!(words(&lowered(&plain(), [1, 1, 1], vec![], true)), []);
    }

    #[test]
    fn the_count_words_go_where_the_title_has_none() {
        let constants = |user_data, sgprs| RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        };
        // Dwords 0-1 and 3-5 are the title's: 2, then 6.
        let layout = vec![constants(0, 2), constants(3, 3)];
        let l = lowered(&plain(), [8, 8, 1], layout, true);
        assert_eq!(words(&l), [(2, 0), (6, 1)]);
    }

    #[test]
    fn halves_of_one_dimension_are_cut_and_keep_their_form() {
        for threads in [64, 256] {
            let (with, without) = (
                lowered(&counting(), [threads, 1, 1], vec![], true),
                lowered(&counting(), [threads, 1, 1], vec![], false),
            );
            assert_eq!(words(&with), [(0, 0)]);
            assert_eq!((with.wave_size, with.exchanges), (Some(32), 1));
            assert_eq!((without.wave_size, without.exchanges), (Some(32), 1));
            assert_ne!(with.dxil, without.dxil);
        }
    }

    #[test]
    fn halves_of_more_dimensions_are_not_cut() {
        // 8 x 8 threads are one console wave in two host waves: the threads
        // past a count are not the last of the wave.
        let l = lowered(&counting(), [8, 8, 1], vec![], true);
        assert_eq!((l.wave_size, l.exchanges), (Some(32), 1));
        assert_eq!(words(&l), []);
        // One host wave holds the group: cut on both axes.
        let l = lowered(&counting(), [8, 4, 1], vec![], true);
        assert_eq!(l.wave_size, Some(32));
        assert_eq!(words(&l), [(0, 0), (1, 1)]);
    }
}
