//! Wave64 programs in groups of several console waves, run as halves of
//! console waves in 32-lane host waves: cases from avatarmanz/VIctory-7900XTX
//! (its PRs #82 and #142), made as the translation makes any program
//! ([`super::translate_program`]) and, on Windows (under Wine: D3DMetal,
//! 32-lane waves), run on the hardware device against the console's
//! lockstep results worked out by hand.

use super::{HostShader, Options};
use crate::dxbc;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A wave64 compute program of `threads` (ids in v0) with four root
/// constants in s0-s3 and a table in s4 whose first view is a raw buffer,
/// its descriptor loaded into s[8:11] before `body`.
fn native(body: &[u32], threads: u32, lds_bytes: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
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
            rsrc2: (lds_bytes / 512) << 15 | 5 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` as the translation makes it for 32-lane host waves.
fn translated(native: &NativeProgram) -> HostShader {
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let options = Options {
        narrow_waves: true,
        ..Options::default()
    };
    super::translate_program(&c, native, options, None).expect("translate")
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

/// A group of two console waves (128 threads) whose lanes not below s0 leave
/// early as the console's compiler has them leave when the program's end is
/// too far for a branch: `s_cbranch_execnz` over a far jump to the end
/// (after the fork's PR #82). Then a count across the console wave: every
/// lane left stores 2 at its dword where the thread numbered s1 is among
/// the lanes left of its console wave, else 1.
fn far_early_out() -> Vec<u32> {
    let gate = [
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
        0xBF06_8010, // s_cmp_eq_u32 s16, 0
        0x8511_8281, // s_cselect_b32 s17, 1, 2
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xE070_1000,
        0x8002_0102, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, (to the end)
        0x820F_800F,                 // s_addc_u32 s15, s15, 0
        0xBE80_200E,                 // s_setpc_b64 s[14:15]
    ];
    let mut w = vec![
        0x7D88_0000,                     // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A,                     // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | jump.len() as u32, // s_cbranch_execnz past the jump
    ];
    w.extend_from_slice(&jump);
    w.extend_from_slice(&gate);
    w
}

/// What thread `t` of [`far_early_out`] stores, with `below` threads left
/// and thread `marked` the one the count looks for.
#[cfg(windows)]
fn far_early_out_wants(t: u32, below: u32, marked: u32) -> u32 {
    let seen = marked < below && marked / 64 == t / 64;
    match (t < below, seen) {
        (false, _) => 0,
        (true, false) => 1,
        (true, true) => 2,
    }
}

/// The prefix of console wave totals of the title's ray-tracing structure
/// builder (after the fork's PR #142): each console wave of a 256-thread
/// group writes its total, 10 + its index (read from its last lane, as the
/// builder's scan does), to LDS; then, under `if (id >> 6 != 0)`, a loop
/// over the waves before it adds their totals, reading the running sum's
/// first lane each pass, its lanes leaving at their own wave's index. Every
/// thread stores the sum of the waves before its own: 0, 10, 21, 33.
/// `by_lane`: each lane reads the total at its own id instead, which the
/// lanes of a wave do not hold alike.
fn wave_prefix(by_lane: bool) -> Vec<u32> {
    let address = if by_lane {
        0x3410_0082 // v_lshlrev_b32 v8, 2, v0
    } else {
        0x7E10_0210 // v_mov_b32 v8, s16
    };
    vec![
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0 (the console wave's index)
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0x4A12_0E8A, // v_add_nc_u32 v9, 10, v7
        0xD760_0013,
        0x0201_7F09, // v_readlane_b32 s19, v9, 63
        0x7E12_0213, // v_mov_b32 v9, s19
        0xD834_0000,
        0x0000_0908, // ds_write_b32 v8, v9
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 exec, 0, v7
        0xBF88_0011, // s_cbranch_execz to the end of the loop
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch to the body
        0x87FE_0614, // s_and_b64 exec, s[20:21], s[6:7] (the loop's top)
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // s_lshl_b32 s16, s18, 2 (the body)
        0x8112_8112, // s_add_i32 s18, s18, 1
        address,
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 exec, s18, v7
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF3, // s_cbranch_scc1 to the loop's top
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0801, // buffer_store_dword v8, v1, s[8:11], 0 offen
    ]
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
        text(&[far_early_out(), vec![S_ENDPGM]].concat()),
        [
            "v_cmp_gt_u32 vcc, s0, v0",
            "s_and_saveexec_b64 s[12:13], vcc",
            "s_cbranch_execnz label_0020",
            "s_getpc_b64 s[14:15]",
            "s_add_u32 s14, s14, lit(0x00000030)",
            "s_addc_u32 s15, s15, 0",
            "s_setpc_b64 s[14:15]",
            "v_cmp_eq_u32 vcc, s1, v0",
            "s_bcnt1_i32_b64 s16, vcc",
            "s_cmp_eq_u32 s16, 0",
            "s_cselect_b32 s17, 1, 2",
            "v_mov_b32 v1, s17",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[8:11], 0 offen",
            "s_endpgm",
        ]
    );
    let prefix = text(&[wave_prefix(false), vec![S_ENDPGM]].concat());
    for (at, want) in [
        (0, "v_lshrrev_b32 v7, 6, v0"),
        (3, "v_readlane_b32 s19, v9, 63"),
        (10, "v_cmpx_ne_u32 exec, 0, v7"),
        (17, "v_readfirstlane_b32 s17, v8"),
        (25, "v_cmpx_eq_u32 exec, s18, v7"),
        (27, "s_cbranch_scc1 label_0048"),
    ] {
        assert_eq!(prefix[at], want, "instruction {at}");
    }
}

/// Both run in halves: neither keeps the console's 64 lanes, which
/// D3DMetal does not run.
#[test]
fn the_fork_s_cases_run_in_halves() {
    for (name, words, threads, lds) in [
        ("a far early-out", far_early_out(), 128, 0),
        ("a prefix of wave totals", wave_prefix(false), 256, 512),
    ] {
        let shader = translated(&native(&words, threads, lds));
        assert_ne!(wave_size(&shader.dxil), 64, "{name}");
    }
    // Read by lane, the running sums differ between the halves: the
    // program keeps the console's 64 lanes.
    let shader = translated(&native(&wave_prefix(true), 256, 512));
    assert_eq!(wave_size(&shader.dxil), 64, "a prefix read by lane");
}

/// The root constants s0 and s1 of `shader`.
#[cfg(windows)]
fn constants(shader: &HostShader, s0: u32, s1: u32) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Constants && u.slot == 0 && u.dword <= 1 {
            ud[u.index as usize] = [s0, s1][u.dword as usize];
        }
    }
    ud
}

/// On the hardware device: each case gives every thread the console's
/// result.
#[cfg(windows)]
#[test]
fn the_fork_s_cases_give_the_console_s_results() {
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
    let run = |shader: &HostShader, s0: u32, s1: u32, dwords: u32| -> Vec<u32> {
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut shader.dxil.clone())
                .expect("DXIL validation failed");
        }
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(constants(shader, s0, s1)))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        out[0][..dwords as usize].to_vec()
    };
    let early = translated(&native(&far_early_out(), 128, 0));
    for (below, marked) in [
        (128u32, 40u32),
        (128, 100),
        (128, 63),
        (40, 50),
        (40, 10),
        (33, 32),
        (1, 0),
        (0, 5),
        (100, 70),
    ] {
        let got = run(&early, below, marked, 128);
        let want: Vec<u32> = (0..128)
            .map(|t| far_early_out_wants(t, below, marked))
            .collect();
        assert_eq!(
            got, want,
            "a far early-out: {below} left, thread {marked} looked for"
        );
    }
    let prefix = translated(&native(&wave_prefix(false), 256, 512));
    let got = run(&prefix, 0, 0, 256);
    let want: Vec<u32> = (0..256u32)
        .map(|t| (0..t >> 6).map(|w| 10 + w).sum())
        .collect();
    assert_eq!(got, want, "a prefix of wave totals");
    eprintln!("ran on the hardware device: a far early-out and a prefix of wave totals in halves");
}

/// A lane mask held as a number whose two dwords two instructions write
/// (the end of the ray-tracing structure builder's prefix sums over a
/// wave): `s[6:7]` is 0 and all ones, so the upper half's lanes take their
/// thread id and the lower half's 0. (`v_mbcnt` makes it a program of
/// console lanes, as the lane operations around such a mask do.)
fn numeric_mask() -> Vec<u32> {
    vec![
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBE87_03C1, // s_mov_b32 s7, -1
        0xD501_0006,
        0x001A_0080, // v_cndmask_b32 v6, 0, v0, s[6:7]
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xE070_1000,
        0x8002_0602, // buffer_store_dword v6, v2, s[8:11], 0 offen
    ]
}

/// The pair of a numeric mask written a dword at a time is known from the
/// use that reads it, so the lane tests the dword of its own half (before,
/// every lane tested the low dword by its lane number modulo 32).
#[test]
fn a_numeric_mask_written_a_dword_at_a_time_keeps_its_high_dword() {
    let code: Vec<u8> = [numeric_mask(), vec![S_ENDPGM]]
        .concat()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let text: Vec<String> = crate::gfx10::decode(&code)
        .expect("decode")
        .iter()
        .map(|i| i.text(true))
        .collect();
    assert_eq!(
        text[..4],
        [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "s_mov_b32 s6, 0",
            "s_mov_b32 s7, -1",
            "v_cndmask_b32 v6, 0, v0, s[6:7]",
        ]
    );
    let p = crate::ir::build(&code, true, 32).expect("ir::build");
    let st = crate::structure::structure(&p).expect("structure");
    assert_eq!(st.other_highs.len(), 1);
    let shader = translated(&native(&numeric_mask(), 64, 0));
    assert_eq!(wave_size(&shader.dxil), 32, "runs in halves");
}

/// On the hardware device: lanes 32 to 63 take their thread id, the others 0.
#[cfg(windows)]
#[test]
fn a_numeric_mask_selects_the_upper_half_on_the_device() {
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
    let shader = translated(&native(&numeric_mask(), 64, 0));
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(constants(&shader, 0, 0)))],
                views: &[View::RawUav(64)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    let want: Vec<u32> = (0..64).map(|l| if l < 32 { 0 } else { l }).collect();
    assert_eq!(out[0][..64], want[..]);
}

/// Before a loop: `s[2:3]` is the mask of the even threads, `v1` the sum
/// (0), `s4` the loop's count (0).
const MASK_LOOP_START: [u32; 7] = [
    0x360E_0081, // v_and_b32 v7, 1, v0
    0xD4C2_0002,
    0x0002_0E80, // v_cmp_eq_u32 s[2:3], 0, v7
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7E0A_0280, // v_mov_b32 v5, 0
    0xBE84_0380, // s_mov_b32 s4, 0
    0xD4C2_000C, // (v_cmp_eq_u32 s[12:13], 0, v0: thread 0, next word)
];

/// The loop's first instructions: a lane in the mask adds 1, another 2.
const MASK_LOOP_ADD: [u32; 3] = [
    0xD501_0002,
    0x0009_0282, // v_cndmask_b32 v2, 2, 1, s[2:3]
    0x4A02_0302, // v_add_nc_u32 v1, v2, v1
];

/// The loop's end (`back` words back to its first) after four rounds, and
/// the store of each thread's sum.
fn mask_loop_end(w: &mut Vec<u32>, back: u16) {
    w.extend([
        0x8004_8104, // s_add_u32 s4, s4, 1
        0xBF0A_8404, // s_cmp_lt_u32 s4, 4
        0xBF85_0000 | back.wrapping_neg() as u32, // s_cbranch_scc1 to the loop's first
        0x3410_0082, // v_lshlrev_b32 v8, 2, v0
        0xE070_1000,
        0x8002_0108, // buffer_store_dword v1, v8, s[8:11], 0 offen
    ]);
}

/// A mask the loop leaves alone, parked in lanes 0 and 1 of `v5` and read
/// back before the loop's end: the join at the loop's header takes in two
/// lane reads that nothing else reads as a mask (two water programs of the
/// title, after halfwave50 802a683).
fn parked_mask() -> Vec<u32> {
    let mut w = MASK_LOOP_START.to_vec();
    w.push(0x0002_0080);
    w.extend(MASK_LOOP_ADD);
    w.extend([
        0xD761_0005,
        0x0001_0002, // v_writelane_b32 v5, s2, 0
        0xD761_0005,
        0x0001_0203, // v_writelane_b32 v5, s3, 1
        0xBE82_0480, // s_mov_b64 s[2:3], 0
        0xD760_0002,
        0x0001_0105, // v_readlane_b32 s2, v5, 0
        0xD760_0003,
        0x0001_0305, // v_readlane_b32 s3, v5, 1
    ]);
    mask_loop_end(&mut w, 15);
    w
}

/// A branch inside the loop copies `s[source:source+1]` into the mask's
/// register in every round but the second: the back edge brings the join
/// after that branch. With `source` 6 the copy is the mask itself (saved in
/// `s[6:7]`); with 12 it is thread 0 alone, a join of two masks.
fn branched_mask(source: u8) -> Vec<u32> {
    let mut w = MASK_LOOP_START.to_vec();
    w.push(0x0002_0080);
    w.extend(MASK_LOOP_ADD);
    w.extend([
        0xBE86_0402,                   // s_mov_b64 s[6:7], s[2:3]
        0xBF06_8104,                   // s_cmp_eq_u32 s4, 1
        0xBF85_0001,                   // s_cbranch_scc1 over the copy
        0xBE82_0400 | u32::from(source), // s_mov_b64 s[2:3], s[source:]
    ]);
    mask_loop_end(&mut w, 10);
    w
}

/// What thread `t` of [`branched_mask`] stores: each round adds 1 where its
/// bit of the mask is set, else 2.
#[cfg(windows)]
fn branched_mask_wants(t: u32, source: u8) -> u32 {
    let even = t.is_multiple_of(2);
    let mut mask = even;
    let mut sum = 0;
    for round in 0..4 {
        sum += if mask { 1 } else { 2 };
        if round != 1 {
            mask = if source == 6 { mask } else { t == 0 };
        }
    }
    sum
}

#[test]
fn the_mask_loops_are_what_their_comments_say() {
    let text = |words: &[u32]| -> Vec<String> {
        let code: Vec<u8> = [words, &[S_ENDPGM]].concat().iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .expect("decode")
            .iter()
            .map(|i| format!("{:#x}: {}", i.offset, i.text(true)))
            .collect()
    };
    let p = text(&parked_mask());
    for want in [
        "0x0: v_and_b32 v7, 1, v0",
        "0x4: v_cmp_eq_u32 s[2:3], 0, v7",
        "0x18: v_cmp_eq_u32 s[12:13], 0, v0",
        "0x20: v_cndmask_b32 v2, 2, 1, s[2:3]",
        "0x28: v_add_nc_u32 v1, v2, v1",
        "0x2c: v_writelane_b32 v5, s2, 0",
        "0x34: v_writelane_b32 v5, s3, 1",
        "0x3c: s_mov_b64 s[2:3], 0",
        "0x40: v_readlane_b32 s2, v5, 0",
        "0x48: v_readlane_b32 s3, v5, 1",
        "0x50: s_add_u32 s4, s4, 1",
        "0x54: s_cmp_lt_u32 s4, 4",
        "0x58: s_cbranch_scc1 label_0020",
        "0x60: buffer_store_dword v1, v8, s[8:11], 0 offen",
    ] {
        assert!(p.iter().any(|l| l == want), "{want:?} not in {p:#?}");
    }
    let b = text(&branched_mask(6));
    for want in [
        "0x2c: s_mov_b64 s[6:7], s[2:3]",
        "0x30: s_cmp_eq_u32 s4, 1",
        "0x34: s_cbranch_scc1 label_003C",
        "0x38: s_mov_b64 s[2:3], s[6:7]",
        "0x44: s_cbranch_scc1 label_0020",
    ] {
        assert!(b.iter().any(|l| l == want), "{want:?} not in {b:#?}");
    }
    assert!(text(&branched_mask(12)).iter().any(|l| l == "0x38: s_mov_b64 s[2:3], s[12:13]"));
}

/// A lane-mask join at a loop's header whose back edge brings a mask
/// nothing else reads as one (lanes read back from a spill, the join of a
/// branch) is taken in as a mask, not refused as "a lane-mask join of an
/// untracked mask"; so is a back edge that brings a join of two masks.
#[test]
fn a_mask_that_comes_back_to_its_loop_through_a_join_or_a_spill_is_joined() {
    for (what, words) in [
        ("parked", parked_mask()),
        ("branched", branched_mask(6)),
        ("branched, two masks", branched_mask(12)),
    ] {
        let bytes = dxbc::build(&[]);
        let c = dxbc::Container::parse(&bytes).expect("container");
        let options = Options { narrow_waves: true, ..Options::default() };
        if let Err(e) = super::translate_program(&c, &native(&words, 64, 0), options, None) {
            panic!("{what}: {e}");
        }
    }
}

/// On the hardware device, in halves: each thread has the console's sum.
#[cfg(windows)]
#[test]
fn a_mask_joined_at_its_loop_gives_the_console_s_sums() {
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
    for (what, words, source) in [
        ("parked", parked_mask(), 6),
        ("branched", branched_mask(6), 6),
        ("branched, two masks", branched_mask(12), 12),
    ] {
        let shader = translated(&native(&words, 64, 0));
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{what}: compute_pipeline failed: {e}"));
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(constants(&shader, 0, 0)))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{what}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..64).map(|t| branched_mask_wants(t, source)).collect();
        assert_eq!(out[0][..64], want[..], "{what}");
    }
}
