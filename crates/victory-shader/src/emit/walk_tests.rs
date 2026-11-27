//! Wave64 programs whose halves meet only once the per-lane branches around
//! their wave operations are taken by the whole console wave (`widened`,
//! `structure::structure_wide`): what each lane of the console wave ends
//! with, against the console's lockstep semantics worked out by hand. Each
//! program is made first as it was before (it keeps `WaveSize(64)`, which a
//! host of 32-lane waves does not run), then in halves with wave-wide
//! branches (`WaveSize(32)`). On Windows (under Wine: D3DMetal, 32-lane
//! waves) each runs on the hardware device; the scalar results every lane
//! stores must be the same in both halves of the wave.

use super::{HostShader, Options};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{dxbc, io, ir, resource, structure};

// Operand codes.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LITERAL: u32 = 255;
/// An inline integer constant 0..=64.
const fn k(n: u32) -> u32 {
    128 + n
}
/// VGPR `n` as a source.
const fn v(n: u32) -> u32 {
    256 + n
}

/// (`vsrc1` a VGPR, given as a source: [`v`].)
const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | (vsrc1 & 0xFF) << 9 | src0
}
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}
/// A compare writing VCC.
const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | (vsrc1 & 0xFF) << 9 | src0
}
/// A compare writing an SGPR pair (VOP3).
const fn vop3c(op: u32, sdst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | sdst, src0 | src1 << 9]
}
const fn sop1(op: u32, sdst: u32, src: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src
}
const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}
/// A branch `words` dwords past the next instruction.
const fn sopp(op: u32, words: i32) -> u32 {
    0xBF80_0000 | op << 16 | (words as u32 & 0xFFFF)
}

const V_MOV: u32 = 0x01;
const V_READFIRSTLANE: u32 = 0x02;
const V_LSHLREV: u32 = 0x1A;
const V_AND: u32 = 0x1B;
const V_XOR: u32 = 0x1D;
const V_ADD: u32 = 0x25;
const V_SUBREV: u32 = 0x27;
const V_CMP_LT_U32: u32 = 0xC1;
const V_CMP_EQ_U32: u32 = 0xC2;
const V_CMP_GT_U32: u32 = 0xC4;
const V_CMP_NE_U32: u32 = 0xC5;
const V_CMPX_GT_U32: u32 = 0xD4;
const S_MOV_B32: u32 = 0x03;
const S_MOV_B64: u32 = 0x04;
const S_BCNT1_B64: u32 = 0x10;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_ADD_I32: u32 = 0x02;
const S_AND_B64: u32 = 0x0F;
const S_ENDPGM: u32 = 0xBF81_0000;
/// `v_mbcnt_lo_u32_b32 v9, -1, 0`.
const V_MBCNT_LO_V9: [u32; 2] = [0xD765_0009, 0x0001_00C1];
const EXECZ: u32 = 0x08;
const EXECNZ: u32 = 0x09;

/// `buffer_store_dword v<data>, v1, s[8:11], 0 offen offset:<offset>`:
/// the dword at byte v1 + offset of the raw buffer.
const fn store(data: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0001 | data << 8]
}
/// `ds_write_b32 v<addr>, v<data>`.
const fn ds_write(addr: u32, data: u32) -> [u32; 2] {
    [0xD834_0000, addr | data << 8]
}
/// `ds_read_b32 v<dst>, v<addr>`.
const fn ds_read(dst: u32, addr: u32) -> [u32; 2] {
    [0xD8D8_0000, addr | dst << 24]
}

/// The table's buffer descriptor in s[8:11], and v1 the byte of the
/// thread's dword (4 × its id).
const PROLOGUE: [u32; 4] = [
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
    0xFA00_0000,
    vop2(V_LSHLREV, 1, k(2), v(0)),
];

/// `WaveActiveSum(v16)` as the console compiler expands it (as in the
/// lowering's tests): the sum in s12, EXEC as it was saved in s[0:1].
const WAVE_SUM_V16_S12: [u32; 19] = [
    0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
    0xD501_0002,
    0x0002_2080, // v_cndmask_b32 v2, 0, v16, s[0:1]
    0x4A04_04E9,
    0xDE54_C102, // v_add_nc_u32 v2, v2, v2 dpp8:[1,0,3,2,5,4,7,6]
    0x4A04_04E9,
    0x9360_1202, // v_add_nc_u32 v2, v2, v2 dpp8:[2,2,0,0,6,6,4,4]
    0x4A04_04E9,
    0x0009_2402, // v_add_nc_u32 v2, v2, v2 dpp8:[4,4,4,4,0,0,0,0]
    0x4A04_04FA,
    0xFF09_4002, // v_add_nc_u32 v2, v2, v2 row_mirror bound_ctrl:1
    0xD778_0003,
    0x0201_0102, // v_permlanex16_b32 v3, v2, 0, 0
    0x4A04_0503, // v_add_nc_u32 v2, v3, v2
    0xD760_0002,
    0x0001_4102, // v_readlane_b32 s2, v2, 32
    0x4A04_0402, // v_add_nc_u32 v2, s2, v2
    0xD760_000C,
    0x0001_0102, // v_readlane_b32 s12, v2, 0
];

/// `if (lanes) { body }`: EXEC saved in s[20:21] and narrowed to VCC, the
/// skip, `body`, EXEC back.
fn branch(lanes: &[u32], body: &[u32]) -> Vec<u32> {
    let mut w = lanes.to_vec();
    w.push(sop1(S_AND_SAVEEXEC_B64, 20, VCC));
    w.push(sopp(EXECZ, body.len() as i32));
    w.extend(body);
    w.push(sop1(S_MOV_B64, EXEC, 20));
    w
}

/// A count of the even lanes below 40 of those that are even: a lane
/// mask made and read as a number inside `if (id % 2 == 0)`. Each even
/// lane stores 20, the others nothing.
fn count_in_branch() -> Vec<u32> {
    let mut body = vop3c(V_CMP_GT_U32, 14, k(40), v(0)).to_vec();
    body.push(sop1(S_BCNT1_B64, 16, 14));
    body.push(vop1(V_MOV, 2, 16));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vop2(V_AND, 2, k(1), v(0)));
    w.push(vopc(V_CMP_EQ_U32, k(0), v(2)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// The first lane's 2 × id inside `if (id >= 20)`: 40 in every lane there
/// (lane 32's, 64, in the upper half alone). A lane count (`v_mbcnt`) in
/// the program asks for the console's lanes: without one, a first-lane read
/// is the host wave's, as HLSL's `WaveReadLaneFirst` allows.
fn first_lane_in_branch() -> Vec<u32> {
    let body = [
        vop2(V_LSHLREV, 3, k(1), v(0)),
        vop1(V_READFIRSTLANE, 16, v(3)),
        vop1(V_MOV, 2, 16),
        store(2, 0)[0],
        store(2, 0)[1],
    ];
    let mut w = PROLOGUE.to_vec();
    w.extend(V_MBCNT_LO_V9);
    w.push(vopc(V_CMP_LT_U32, k(19), v(0)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// A loop each lane goes round `id % 4 + 1` times (its exit per lane, by
/// EXEC), adding the lanes still in it to its sum and 1 to a scalar count:
/// the console's wave goes round 4 times, 64, 48, 32 and 16 lanes, and the
/// count is the wave's. Each lane stores its sum and the count (at +256).
fn loop_by_lane() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        vop2(V_AND, 3, k(3), v(0)),
        vop2(V_ADD, 3, k(1), v(3)),
        vop1(V_MOV, 4, k(0)),
        sop1(S_MOV_B32, 17, k(0)),
        sop1(S_MOV_B64, 12, EXEC),
        // the loop
        sop1(S_BCNT1_B64, 16, EXEC),
        vop2(V_ADD, 4, 16, v(4)),
        sop2(S_ADD_I32, 17, 17, k(1)),
        vop2(V_SUBREV, 3, k(1), v(3)),
        vopc(V_CMP_NE_U32, k(0), v(3)),
        sop2(S_AND_B64, EXEC, EXEC, VCC),
        sopp(EXECNZ, -7),
        sop1(S_MOV_B64, EXEC, 12),
        vop1(V_MOV, 5, 17),
    ]);
    w.extend(store(4, 0));
    w.extend(store(5, 256));
    w.push(S_ENDPGM);
    w
}

/// [`loop_by_lane`]'s loop with, inside it, `if (id % 2 == 0)` counting the
/// lanes there (a mask made in the branch, as [`count_in_branch`]): each
/// pass the even lanes still in the loop, 32, 16, 16 and none, added up by
/// each even lane while it goes round: 32 where id % 4 is 0 (one pass), 64
/// where it is 2 (three); odd lanes add nothing.
fn branch_in_loop_by_lane() -> Vec<u32> {
    let mut count = vop3c(V_CMP_GT_U32, 14, k(64), v(0)).to_vec();
    count.push(sop1(S_BCNT1_B64, 16, 14));
    count.push(vop2(V_ADD, 4, 16, v(4)));
    let mut body = branch(&[vopc(V_CMP_EQ_U32, k(0), v(5))], &count);
    body.extend([
        vop2(V_SUBREV, 3, k(1), v(3)),
        vopc(V_CMP_NE_U32, k(0), v(3)),
        sop2(S_AND_B64, EXEC, EXEC, VCC),
    ]);
    let back = -(body.len() as i32 + 1);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        vop2(V_AND, 3, k(3), v(0)),
        vop2(V_ADD, 3, k(1), v(3)),
        vop1(V_MOV, 4, k(0)),
        vop2(V_AND, 5, k(1), v(0)),
        sop1(S_MOV_B64, 12, EXEC),
    ]);
    w.extend(body);
    w.push(sopp(EXECNZ, back));
    w.push(sop1(S_MOV_B64, EXEC, 12));
    w.extend(store(4, 0));
    w.push(S_ENDPGM);
    w
}

/// Inside `if (id % 2 == 0)`: each lane writes its id to LDS and reads the
/// dword of lane id ^ 32, which the other half wrote (no barrier: the
/// console's wave runs in lockstep). Even lanes store id ^ 32.
fn lds_in_branch() -> Vec<u32> {
    let mut body = ds_write(1, 0).to_vec();
    body.extend([sop1(S_MOV_B32, 18, LITERAL), 128]);
    body.push(vop2(V_XOR, 6, 18, v(1)));
    body.extend(ds_read(7, 6));
    body.extend(store(7, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vop2(V_AND, 2, k(1), v(0)));
    w.push(vopc(V_CMP_EQ_U32, k(0), v(2)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// `WaveActiveSum(id)` inside `if (id < 50)`: 1225 in each such lane.
fn sum_in_branch() -> Vec<u32> {
    let mut body = vec![vop1(V_MOV, 16, v(0))];
    body.extend(WAVE_SUM_V16_S12);
    body.push(sop1(S_MOV_B64, EXEC, 0));
    body.push(vop1(V_MOV, 2, 12));
    body.extend(store(2, 0));
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMP_GT_U32, k(50), v(0)));
    w.extend(branch(&[], &body));
    w.push(S_ENDPGM);
    w
}

/// The lanes from 56 leave (`if (id >= 56) return;`), then
/// [`count_in_branch`]'s count for the rest: 20 in each even lane below
/// 56.
fn count_after_return() -> Vec<u32> {
    let rest = count_in_branch();
    let mut w = PROLOGUE.to_vec();
    w.push(vopc(V_CMPX_GT_U32, k(56), v(0)));
    // to the end: past the rest of `count_in_branch` but its own end
    let tail = &rest[PROLOGUE.len()..];
    w.push(sopp(EXECZ, tail.len() as i32 - 1));
    w.extend(tail);
    w
}

/// An LDS flag raised inside a branch (after avatarmanz/VIctory-7900XTX
/// PR #4): every lane clears the dword at LDS 0; inside `if (id < active)`
/// each lane there sets it to 1 and, unless `skip` (a scalar branch around
/// the rest), reads it back and sets it to 1 again; after the branch every
/// lane reads it and stores it plus one. The console's lanes all clear it
/// before any sets it, so each lane stores 2 if some lane is below `active`
/// and 1 if none is.
fn flag_in_branch(active: u32, skip: u32) -> Vec<u32> {
    let mut rest = ds_read(3, 9).to_vec();
    rest.push(vop1(V_MOV, 6, k(1)));
    rest.extend(ds_write(9, 6));
    let mut body = vec![vop1(V_MOV, 2, k(1))];
    body.extend(ds_write(9, 2));
    body.push(0xBF07_8001); // s_cmp_lg_u32 s1, 0
    body.push(sopp(0x05, rest.len() as i32)); // s_cbranch_scc1 past the rest
    body.extend(rest);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        sop1(S_MOV_B32, 0, LITERAL),
        active,
        sop1(S_MOV_B32, 1, k(skip)),
        vop1(V_MOV, 9, k(0)),
    ]);
    w.extend(ds_write(9, 9));
    w.extend(branch(&[vopc(V_CMP_GT_U32, 0, v(0))], &body));
    w.extend(ds_read(5, 9));
    w.push(vop2(V_ADD, 5, k(1), v(5)));
    w.extend(store(5, 0));
    w.push(S_ENDPGM);
    w
}

/// What lane `id` stores at each dword `n` of [`PROGRAMS`]' buffers, the
/// console's result (`None`: the lane stores nothing there).
type Want = fn(u32, u32) -> Option<u32>;

/// Name, program, LDS bytes, the dwords to read back, and what each holds.
fn programs() -> Vec<(&'static str, Vec<u32>, u32, u32, Want)> {
    vec![
        ("a count in a branch", count_in_branch(), 0, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(20)
        }),
        (
            "a first lane in a branch",
            first_lane_in_branch(),
            0,
            64,
            |id, n| (n == id && id >= 20).then_some(40),
        ),
        ("a loop by lane", loop_by_lane(), 0, 128, |id, n| {
            let sums = [64, 112, 144, 160];
            if n == id {
                Some(sums[(id % 4) as usize])
            } else {
                (n == id + 64).then_some(4)
            }
        }),
        ("LDS in a branch", lds_in_branch(), 512, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(id ^ 32)
        }),
        ("a sum in a branch", sum_in_branch(), 0, 64, |id, n| {
            (n == id && id < 50).then_some(1225)
        }),
        (
            "a count after a return",
            count_after_return(),
            0,
            64,
            |id, n| (n == id && id % 2 == 0 && id < 56).then_some(20),
        ),
        (
            "a branch in a loop by lane",
            branch_in_loop_by_lane(),
            0,
            64,
            |id, n| (n == id).then_some([32, 0, 64, 0][(id % 4) as usize]),
        ),
        ("an LDS flag in a branch", flag_in_branch(40, 0), 512, 64, |id, n| {
            (n == id).then_some(2)
        }),
        ("an LDS flag past a skip", flag_in_branch(40, 1), 512, 64, |id, n| {
            (n == id).then_some(2)
        }),
        ("an LDS flag one lane raises", flag_in_branch(1, 0), 512, 64, |id, n| {
            (n == id).then_some(2)
        }),
        ("an LDS flag no lane raises", flag_in_branch(0, 0), 512, 64, |id, n| {
            (n == id).then_some(1)
        }),
    ]
}

/// A wave64 compute program of one console wave (64 threads, ids in v0)
/// with four root constants in s0-s3 and a table in s4 whose first view is
/// a raw buffer.
fn native(words: &[u32], lds_bytes: u32) -> NativeProgram {
    native_group(words, lds_bytes, [64, 1, 1])
}

/// [`native`] for a group of `threads`: with more than one dimension the
/// thread ids x and y are in v0 and v1, else the flat id is in v0.
fn native_group(words: &[u32], lds_bytes: u32, threads: [u32; 3]) -> NativeProgram {
    let id_vgprs = if threads[1] > 1 { 2 } else { 1 };
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
        ],
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
            rsrc2: (lds_bytes / 512) << 15 | (id_vgprs - 1) << 11 | 5 << 1,
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The `WaveSize` a translation declares (PSV0's lane counts; 0 for none).
fn wave_size(dxil: &[u8]) -> u32 {
    let u = |o: usize| u32::from_le_bytes(dxil[o..o + 4].try_into().unwrap());
    for i in 0..u(28) as usize {
        let o = u(32 + 4 * i) as usize;
        if &dxil[o..o + 4] == b"PSV0" {
            return u(o + 8 + 4 + 16);
        }
    }
    panic!("no PSV0")
}

/// The program as it was made before (the halves apart: `WaveSize(64)`),
/// and with the walk: in halves, with wave-wide branches.
fn made(name: &str, words: &[u32], lds_bytes: u32) -> (HostShader, HostShader) {
    let native = native(words, lds_bytes);
    let p = ir::build(&native.code, true, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("raise");
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        narrow_waves: true,
        ..Options::default()
    };
    super::lower::take_wide();
    let before = super::emit_keeping(&c, &native, &p, &st, &io, &res, options, None).expect("emit");
    assert!(
        super::lower::take_wide().is_some(),
        "{name}: the halves met without the walk"
    );
    let walked = super::widened(&c, &native, &p, options, None)
        .unwrap_or_else(|| panic!("{name}: the walk"));
    (before, walked)
}

#[test]
fn the_test_programs_are_what_their_comments_say() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap_or_else(|e| panic!("{words:08x?}: {e}"))
            .iter()
            .map(|i| i.text(true))
            .collect()
    };
    assert_eq!(
        text(&loop_by_lane()[PROLOGUE.len()..]),
        [
            "v_and_b32 v3, 3, v0",
            "v_add_nc_u32 v3, 1, v3",
            "v_mov_b32 v4, 0",
            "s_mov_b32 s17, 0",
            "s_mov_b64 s[12:13], exec",
            "s_bcnt1_i32_b64 s16, exec",
            "v_add_nc_u32 v4, s16, v4",
            "s_add_i32 s17, s17, 1",
            "v_subrev_nc_u32 v3, 1, v3",
            "v_cmp_ne_u32 vcc, 0, v3",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz label_0014",
            "s_mov_b64 exec, s[12:13]",
            "v_mov_b32 v5, s17",
            "buffer_store_dword v4, v1, s[8:11], 0 offen",
            "buffer_store_dword v5, v1, s[8:11], 0 offen offset:256",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&lds_in_branch()[PROLOGUE.len()..]),
        [
            "v_and_b32 v2, 1, v0",
            "v_cmp_eq_u32 vcc, 0, v2",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_0034",
            "ds_write_b32 v1, v0",
            "s_mov_b32 s18, lit(0x00000080)",
            "v_xor_b32 v6, s18, v1",
            "ds_read_b32 v7, v6",
            "buffer_store_dword v7, v1, s[8:11], 0 offen",
            "s_mov_b64 exec, s[20:21]",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&[V_MBCNT_LO_V9[0], V_MBCNT_LO_V9[1], S_ENDPGM]),
        ["v_mbcnt_lo_u32_b32 v9, -1, 0", "s_endpgm"]
    );
    assert_eq!(
        text(&branch_in_loop_by_lane()[PROLOGUE.len() + 5..]),
        [
            "v_cmp_eq_u32 vcc, 0, v5",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_001C",
            "v_cmp_gt_u32 s[14:15], 64, v0",
            "s_bcnt1_i32_b64 s16, s[14:15]",
            "v_add_nc_u32 v4, s16, v4",
            "s_mov_b64 exec, s[20:21]",
            "v_subrev_nc_u32 v3, 1, v3",
            "v_cmp_ne_u32 vcc, 0, v3",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz label_0000",
            "s_mov_b64 exec, s[12:13]",
            "buffer_store_dword v4, v1, s[8:11], 0 offen",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&count_after_return()[PROLOGUE.len()..]),
        [
            "v_cmpx_gt_u32 exec, 56, v0",
            "s_cbranch_execz label_0034",
            "v_and_b32 v2, 1, v0",
            "v_cmp_eq_u32 vcc, 0, v2",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_0030",
            "v_cmp_gt_u32 s[14:15], 40, v0",
            "s_bcnt1_i32_b64 s16, s[14:15]",
            "v_mov_b32 v2, s16",
            "buffer_store_dword v2, v1, s[8:11], 0 offen",
            "s_mov_b64 exec, s[20:21]",
            "s_endpgm",
        ]
    );
}

/// Each program keeps the console's wave size as it was made before, and
/// runs in halves with the walk: `WaveSize(32)`, or none where only its LDS
/// kept it at 64.
#[test]
fn the_halves_meet_inside_branches_taken_wave_wide() {
    for (name, words, lds, _, _) in programs() {
        let (before, walked) = made(name, &words, lds);
        assert_eq!(wave_size(&before.dxil), 64, "{name}: before");
        assert_ne!(wave_size(&walked.dxil), 64, "{name}: walked");
    }
}

/// On the hardware device (D3DMetal under Wine: 32-lane waves), each
/// program as the walk makes it: every lane of both halves stores what the
/// console's lockstep wave gives it.
#[cfg(windows)]
#[test]
fn walked_programs_give_the_console_wave_s_results() {
    use super::host::{Dispatch, Host, Root, View};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); skipped");
            return;
        }
    };
    let mut ran = 0;
    for (name, words, lds, dwords, want) in programs() {
        let (_, shader) = made(name, &words, lds);
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{name}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        // A few groups at once, so that host waves of several groups run
        // side by side.
        for groups in [1u32, 8] {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud.clone()))],
                        views: &[View::RawUav(dwords)],
                        samplers: &[],
                        groups: [groups, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name}: dispatch failed: {e}"));
            for n in 0..dwords {
                // (every group stores the same: one lane's value)
                let expect: u32 = (0..64).filter_map(|id| want(id, n)).sum();
                assert_eq!(
                    out[0][n as usize], expect,
                    "{name}, {groups} groups, dword {n}"
                );
            }
        }
        ran += 1;
    }
    eprintln!(
        "ran on the hardware device: {ran} wave64 programs in halves with wave-wide branches"
    );
}

// ---- the first live lane of the console wave, with only its upper half live ----

const V_OR: u32 = 0x1C;
const V_MUL_U24: u32 = 0x0B;

/// [`branch`] with EXEC saved in s[`save`:`save` + 1], so that one can be
/// inside another.
fn branch_saving(save: u32, lanes: &[u32], body: &[u32]) -> Vec<u32> {
    let mut w = lanes.to_vec();
    w.push(sop1(S_AND_SAVEEXEC_B64, save, VCC));
    w.push(sopp(EXECZ, body.len() as i32));
    w.extend(body);
    w.push(sop1(S_MOV_B64, EXEC, save));
    w
}

/// VCC as `lo <= lane < hi`, the lane (in v4) being the thread's console
/// lane: `hi == 64` leaves the upper bound out.
fn lanes_between(lo: u32, hi: u32) -> Vec<u32> {
    let mut w = vec![vopc(V_CMP_LT_U32, k(lo - 1), v(4))];
    if hi < 64 {
        w.extend(vop3c(V_CMP_GT_U32, 14, k(hi), v(4)));
        w.push(sop2(S_AND_B64, VCC, VCC, 14));
    }
    w
}

/// The first lane's 2 × id, stored by each lane of the branch (`id` in
/// v0, 2 × id in v3).
fn read_first_and_store() -> Vec<u32> {
    let mut w = vec![
        vop2(V_LSHLREV, 3, k(1), v(0)),
        vop1(V_READFIRSTLANE, 16, v(3)),
        vop1(V_MOV, 2, 16),
    ];
    w.extend(store(2, 0));
    w
}

/// A first-lane read (of 2 × id, as in [`first_lane_in_branch`]) in
/// branches that leave the lanes `lo..hi` of each console wave: with
/// `nested`, inside `if (lane >= 16)` first, which both halves enter. A
/// lane count in the program asks for the console's lanes. Each lane in
/// the branch stores 2 × the id of the first of those lanes of its own
/// console wave.
fn first_lane_upper(lo: u32, hi: u32, nested: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend(V_MBCNT_LO_V9);
    w.push(vop2(V_AND, 4, k(63), v(0)));
    let inner = |body: &[u32]| branch_saving(22, &lanes_between(lo, hi), body);
    if nested {
        let body = inner(&read_first_and_store());
        w.extend(branch(&lanes_between(16, 64), &body));
    } else {
        w.extend(inner(&read_first_and_store()));
    }
    w.push(S_ENDPGM);
    w
}

/// What thread `id` of [`first_lane_upper`] stores.
fn first_lane_upper_wants(id: u32, lo: u32, hi: u32) -> u32 {
    let lane = id % 64;
    if (lo..hi).contains(&lane) {
        2 * (id - lane + lo)
    } else {
        0
    }
}

/// Each thread stores its id plus one (so that a thread that never stores
/// reads as 0) at the dword of its flat id: for a group of `[x, 1, 1]` its
/// id in v0; for one of `[x, y, 1]` its ids x and y in v0 and v1, stored
/// as `x | y << 8`. A lane count in the program asks for the console's
/// lanes.
fn thread_ids(shape: [u32; 3]) -> Vec<u32> {
    let mut w = PROLOGUE[..3].to_vec();
    w.extend(V_MBCNT_LO_V9);
    if shape[1] == 1 {
        w.push(vop2(V_ADD, 7, k(1), v(0)));
        w.push(vop2(V_LSHLREV, 1, k(2), v(0)));
    } else {
        // (the address, v1 last: it holds y until then)
        w.push(vop2(V_LSHLREV, 7, k(8), v(1)));
        w.push(vop2(V_OR, 7, v(7), 0));
        w.push(vop2(V_ADD, 7, k(1), v(7)));
        w.push(vop2(V_MUL_U24, 4, k(shape[0]), v(1)));
        w.push(vop2(V_ADD, 4, v(0), 4));
        w.push(vop2(V_LSHLREV, 1, k(2), v(4)));
    }
    w.extend(store(7, 0));
    w.push(S_ENDPGM);
    w
}

/// What the thread at dword `f` of [`thread_ids`] stores.
fn thread_ids_wants(f: u32, shape: [u32; 3]) -> u32 {
    if shape[1] == 1 {
        f + 1
    } else {
        ((f % shape[0]) | ((f / shape[0]) << 8)) + 1
    }
}

/// Name, program, group, and what each thread of the group stores at its
/// dword.
type Case = (&'static str, Vec<u32>, [u32; 3], fn(u32) -> u32);

fn upper_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for threads in [64, 128] {
        let g = [threads, 1, 1];
        cases.push((
            if threads == 64 {
                "a first lane, upper half only, 64"
            } else {
                "a first lane, upper half only, 128"
            },
            first_lane_upper(32, 64, false),
            g,
            |id| first_lane_upper_wants(id, 32, 64),
        ));
        cases.push((
            if threads == 64 {
                "a first lane, lanes 40..48, 64"
            } else {
                "a first lane, lanes 40..48, 128"
            },
            first_lane_upper(40, 48, false),
            g,
            |id| first_lane_upper_wants(id, 40, 48),
        ));
        cases.push((
            if threads == 64 {
                "a first lane, upper half only, nested, 64"
            } else {
                "a first lane, upper half only, nested, 128"
            },
            first_lane_upper(32, 64, true),
            g,
            |id| first_lane_upper_wants(id, 32, 64),
        ));
        cases.push((
            if threads == 64 {
                "a first lane, lanes 40..48, nested, 64"
            } else {
                "a first lane, lanes 40..48, nested, 128"
            },
            first_lane_upper(40, 48, true),
            g,
            |id| first_lane_upper_wants(id, 40, 48),
        ));
        cases.push((
            if threads == 64 {
                "thread ids, 64"
            } else {
                "thread ids, 128"
            },
            thread_ids(g),
            g,
            |f| f + 1,
        ));
    }
    cases.push(
        ("thread ids, 8 x 8", thread_ids([8, 8, 1]), [8, 8, 1], |f| {
            thread_ids_wants(f, [8, 8, 1])
        }),
    );
    cases.push((
        "thread ids, 16 x 8",
        thread_ids([16, 8, 1]),
        [16, 8, 1],
        |f| thread_ids_wants(f, [16, 8, 1]),
    ));
    cases
}

/// `words` for a group of `threads`, as the translation makes any program.
fn translated_in(name: &str, words: &[u32], threads: [u32; 3]) -> HostShader {
    let native = native_group(words, 0, threads);
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        narrow_waves: true,
        ..Options::default()
    };
    super::translate_program(&c, &native, options, None)
        .unwrap_or_else(|e| panic!("{name}: translate: {e}"))
}

#[test]
fn the_upper_half_test_programs_are_what_their_comments_say() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap_or_else(|e| panic!("{words:08x?}: {e}"))
            .iter()
            .map(|i| i.text(true))
            .collect()
    };
    assert_eq!(
        text(&first_lane_upper(40, 48, true)[PROLOGUE.len()..]),
        [
            "v_mbcnt_lo_u32_b32 v9, -1, 0",
            "v_and_b32 v4, 63, v0",
            "v_cmp_lt_u32 vcc, 15, v4",
            "s_and_saveexec_b64 s[20:21], vcc",
            "s_cbranch_execz label_0048",
            "v_cmp_lt_u32 vcc, 39, v4",
            "v_cmp_gt_u32 s[14:15], 48, v4",
            "s_and_b64 vcc, vcc, s[14:15]",
            "s_and_saveexec_b64 s[22:23], vcc",
            "s_cbranch_execz label_0044",
            "v_lshlrev_b32 v3, 1, v0",
            "v_readfirstlane_b32 s16, v3",
            "v_mov_b32 v2, s16",
            "buffer_store_dword v2, v1, s[8:11], 0 offen",
            "s_mov_b64 exec, s[22:23]",
            "s_mov_b64 exec, s[20:21]",
            "s_endpgm",
        ]
    );
    assert_eq!(
        text(&thread_ids([16, 8, 1])[3..]),
        [
            "v_mbcnt_lo_u32_b32 v9, -1, 0",
            "v_lshlrev_b32 v7, 8, v1",
            "v_or_b32 v7, v7, v0",
            "v_add_nc_u32 v7, 1, v7",
            "v_mul_u32_u24 v4, 16, v1",
            "v_add_nc_u32 v4, v0, v4",
            "v_lshlrev_b32 v1, 2, v4",
            "buffer_store_dword v7, v1, s[8:11], 0 offen",
            "s_endpgm",
        ]
    );
}

/// Every one runs in halves, not as the console's 64 lanes, which
/// D3DMetal does not run.
#[test]
fn the_upper_half_cases_run_in_halves() {
    for (name, words, threads, _) in upper_cases() {
        let shader = translated_in(name, &words, threads);
        assert_ne!(wave_size(&shader.dxil), 64, "{name}");
    }
}

/// On the hardware device (D3DMetal under Wine: 32-lane waves): the first
/// live lane of the console wave when only its upper half is live (in a
/// second console wave too), and every thread of a group given its own id.
#[cfg(windows)]
#[test]
fn the_upper_half_cases_give_the_console_wave_s_results() {
    use super::host::{Dispatch, Host, Root, View};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); skipped");
            return;
        }
    };
    let mut ran = 0;
    for (name, words, threads, want) in upper_cases() {
        let shader = translated_in(name, &words, threads);
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{name}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let n: u32 = threads.iter().product();
        // A few groups at once, as in the walked programs.
        for groups in [1u32, 8] {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud.clone()))],
                        views: &[View::RawUav(n)],
                        samplers: &[],
                        groups: [groups, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name}: dispatch failed: {e}"));
            // (every group stores the same)
            let expect: Vec<u32> = (0..n).map(want).collect();
            assert_eq!(out[0][..n as usize], expect[..], "{name}, {groups} groups");
        }
        ran += 1;
    }
    eprintln!("ran on the hardware device: {ran} upper-half and thread-id programs in halves");
}
