//! Fences round the buffer atomics of the console's ray-tracing structure
//! builder and its like ([`Extra::atomic_fences`](super::Extra); after
//! Belobug 72e748ea and avatarmanz 30f46c92).
//!
//! The builder's threads take turns through atomics (a tree built from its
//! leaves up): what a thread wrote before its atomic has to be seen by the
//! thread the atomic lets on. The console's program waits for its writes
//! there (`s_waitcnt`, which the lowering leaves out: host stores have no
//! counters to wait for). Here such an atomic stands between two memory
//! fences over unordered-access memory, with no thread-group sync, so they
//! may stand in code only some lanes run: every lane in the block passes
//! them, whichever the atomic's own guard lets through. (The 64-bit image
//! atomics' locks use such fences in one lane's turn, on D3DMetal too.)

use super::super::bitcode::Attr;
use super::Lower;
use crate::native::{NativeProgram, ResourceType, RootType};

/// Whether a compute program writes through root views: with the programs
/// that reach title memory by address ([`super::by_address`]), those the
/// fences are for.
pub(crate) fn writes_root_views(native: &NativeProgram) -> bool {
    native
        .used
        .iter()
        .any(|u| u.root == RootType::Descriptor && u.kind == ResourceType::Uav)
}

impl Lower<'_> {
    /// Whether the instruction `name` stands between two fences: a buffer
    /// or global atomic of a compute program made with them.
    pub(super) fn fenced(&self, name: &str) -> bool {
        self.extra.atomic_fences
            && self.s.threads.is_some()
            && self.s.mesh.is_none()
            && (name.starts_with("buffer_atomic") || name.starts_with("global_atomic"))
    }

    /// Writes to unordered-access memory made before are seen, by every
    /// group, before those made after (DXIL's barrier of global UAV memory
    /// alone: no thread waits for another).
    pub(super) fn memory_fence(&mut self) {
        let (mode, void) = (self.s.m.i32(2), self.s.t.void);
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Extra, Lowered, Parts, lower_with_layout};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, range, resource};

    /// A compute program of 64 threads that adds 1 to the dword of its
    /// thread's number in a raw buffer (a table's first view), with a store
    /// to the same buffer first with `store`.
    fn lowered(store: bool, atomic_fences: bool) -> Lowered {
        let mut words: Vec<u32> = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xFA00_0000,
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0x7E04_0281, // v_mov_b32 v2, 1
        ];
        if store {
            // buffer_store_dword v2, off, s[8:11], 0 offset:256
            words.extend([0xE070_0100, 0x8002_0200]);
        }
        // buffer_atomic_add v2, v1, s[8:11], 0 offen
        words.extend([0xE0C8_1000, 0x8002_0201, 0xBF81_0000]);
        let param = |root, user_data, sgprs, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 16,
            layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let (p, st) = range::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = st.unwrap();
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
                narrow_waves: false,
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
                atomic_fences,
                ..Extra::default()
            },
        )
        .expect("lower")
    }

    #[test]
    fn an_atomic_stands_between_two_fences() {
        for store in [false, true] {
            assert_eq!(lowered(store, false).barriers, 0);
            // One before and one after the atomic; none round the store.
            assert_eq!(lowered(store, true).barriers, 2);
        }
    }

    #[test]
    fn programs_that_write_root_views_are_told() {
        let used = |root, kind| UsedResource {
            kind,
            root,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 0,
            slot: 0,
        };
        let native = |used| NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 16,
            layout: vec![],
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: vec![],
        };
        assert!(super::writes_root_views(&native(vec![used(RootType::Descriptor, ResourceType::Uav)])));
        assert!(!super::writes_root_views(&native(vec![
            used(RootType::Descriptor, ResourceType::Srv),
            used(RootType::Table, ResourceType::Uav),
        ])));
    }
}
