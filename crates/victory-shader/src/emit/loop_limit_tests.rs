//! The iteration limit of a loop the console jumps back to without a test
//! (`s_branch`), kept at the loop's exit branch with `VICTORY_LIMIT_EXITS`
//! ([`super::limit_exits`]): a loop that ends counts exactly as on the
//! console, and one that never meets its exit leaves through that exit after
//! the limit (here with the sum of its first counts) instead of leaving the
//! program. After SlaktarnVI / avatarmanz e823263 (`loop_limit_tests.rs`).

use super::limit_exits::{meeting, with_limit_exits};
use super::{HostShader, Options};
use crate::dxbc;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `v_lshlrev_b32 v2, 2, v0` and `buffer_store_dword v1, v2, s[8:11], 0
/// offen`: each thread stores v1 at its dword.
const STORE_V1: [u32; 3] = [0x3404_0082, 0xE070_1000, 0x8002_0102];

/// `s_cmp_<op>_u32 s16, end`: an inline constant up to 64, else a literal.
fn cmp_s16(op: u32, end: u32) -> Vec<u32> {
    if end > 64 {
        vec![0xBF00_FF10 | op << 16, end]
    } else {
        vec![0xBF00_0010 | op << 16 | (128 + end) << 8]
    }
}

/// Each thread sums the counts `0..` until the scalar count reaches `end`,
/// then stores its sum. The loop's test is at its top; its back edge is an
/// `s_branch`, as the console's compiler lays out a rotated loop.
fn jump_back_loop(end: u32) -> Vec<u32> {
    let mut w = vec![
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBE90_0380, // s_mov_b32 s16, 0
    ];
    let test = cmp_s16(9, end); // LOOP: s_cmp_ge_u32 s16, end
    let back = -(test.len() as i32 + 4); // to LOOP, from after the s_branch
    w.extend(test);
    w.extend([
        0xBF85_0003,                          // s_cbranch_scc1 EXIT
        0x4A02_0210,                          // v_add_nc_u32 v1, s16, v1
        0x8110_8110,                          // s_add_i32 s16, s16, 1
        0xBF82_0000 | (back as u32 & 0xFFFF), // s_branch LOOP
    ]);
    w.extend(STORE_V1); // EXIT
    w
}

/// The same sum as a loop whose back edge is its test (`s_cbranch_scc1`):
/// the limit stays where it was, at the back edge.
fn tested_loop(end: u32) -> Vec<u32> {
    let mut w = vec![
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBE90_0380, // s_mov_b32 s16, 0
        0x4A02_0210, // LOOP: v_add_nc_u32 v1, s16, v1
        0x8110_8110, // s_add_i32 s16, s16, 1
    ];
    let test = cmp_s16(10, end); // s_cmp_lt_u32 s16, end
    let back = -(test.len() as i32 + 3);
    w.extend(test);
    w.push(0xBF85_0000 | (back as u32 & 0xFFFF)); // s_cbranch_scc1 LOOP
    w.extend(STORE_V1);
    w
}

/// A wave32 compute program of 32 threads (ids in v0) with four root
/// constants in s0-s3 and a table in s4 whose first view is a raw buffer,
/// its descriptor loaded into s[8:11] before `body`.
fn native(body: &[u32]) -> NativeProgram {
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
        wave64: false,
        vgprs: 16,
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
            rsrc2: 5 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `body` translated with the switch `on`, and whether it met the pattern.
fn translated(body: &[u32], on: bool) -> (HostShader, bool) {
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let n = native(body);
    let (shader, met) = with_limit_exits(on, || {
        meeting(|| super::translate_program(&c, &n, Options::default(), None))
    });
    (shader.expect("translate"), met)
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
    let jump = text(&[jump_back_loop(10), vec![S_ENDPGM]].concat());
    assert_eq!(jump[2], "s_cmp_ge_u32 s16, 10");
    assert!(jump[3].starts_with("s_cbranch_scc1"), "{}", jump[3]);
    assert_eq!(jump[4], "v_add_nc_u32 v1, s16, v1");
    assert_eq!(jump[5], "s_add_i32 s16, s16, 1");
    assert!(jump[6].starts_with("s_branch"), "{}", jump[6]);
    assert_eq!(jump[8], "buffer_store_dword v1, v2, s[8:11], 0 offen");
    let tested = text(&[tested_loop(100_000), vec![S_ENDPGM]].concat());
    assert!(tested[4].starts_with("s_cmp_lt_u32 s16, lit("), "{}", tested[4]);
    assert!(tested[5].starts_with("s_cbranch_scc1"), "{}", tested[5]);
}

/// A loop jumped back to meets the pattern whether the switch is on or not;
/// only with it on is its translation different. Looking for it leaves the
/// translation with the switch off as it is.
#[test]
fn a_loop_jumped_back_to_meets_the_switch() {
    let (off, met_off) = translated(&jump_back_loop(10), false);
    let (on, met_on) = translated(&jump_back_loop(10), true);
    assert!(met_off && met_on);
    assert_ne!(off.dxil, on.dxil, "the switch changes the translation");
    let bytes = dxbc::build(&[]);
    let c = dxbc::Container::parse(&bytes).expect("container");
    let n = native(&jump_back_loop(10));
    let plain = with_limit_exits(false, || {
        super::translate_program(&c, &n, Options::default(), None).expect("translate")
    });
    assert_eq!(plain.dxil, off.dxil, "looking changes nothing");
}

/// A loop whose back edge is its test keeps its limit there: it meets
/// nothing, and the switch leaves its translation as it is.
#[test]
fn a_loop_tested_at_its_back_edge_does_not_meet_the_switch() {
    let (off, met_off) = translated(&tested_loop(10), false);
    let (on, met_on) = translated(&tested_loop(10), true);
    assert!(!met_off && !met_on);
    assert_eq!(off.dxil, on.dxil);
}

/// On the hardware device: the 32 threads' stored values, or `None`
/// without one.
#[cfg(windows)]
fn run(shader: &HostShader) -> Option<Vec<u32>> {
    use super::host::{Dispatch, Host, Root, View};
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); not run");
            return None;
        }
    };
    let (root, pso) = host.compute_pipeline(shader).expect("pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][..32].to_vec())
}

/// A loop that ends counts as on the console, the switch on or off.
#[cfg(windows)]
#[test]
fn a_loop_jumped_back_to_counts_as_on_the_console() {
    for on in [false, true] {
        let Some(out) = run(&translated(&jump_back_loop(10), on).0) else {
            return;
        };
        assert_eq!(out, vec![45; 32], "0 + 1 + ... + 9 in every thread (switch {on})");
    }
}

/// A loop that never meets its exit: with the switch the lane leaves through
/// the loop's exit after the limit and stores the sum of the counts it ran;
/// without it the lane leaves the program, storing nothing.
#[cfg(windows)]
#[test]
fn a_loop_that_never_meets_its_exit_leaves_by_it_after_the_limit() {
    let limit = u64::from(super::loop_limit());
    let Some(on) = run(&translated(&jump_back_loop(1 << 30), true).0) else {
        return;
    };
    let sum = (limit * (limit - 1) / 2) as u32;
    assert_eq!(on, vec![sum; 32]);
    let Some(off) = run(&translated(&jump_back_loop(1 << 30), false).0) else {
        return;
    };
    assert_eq!(off, vec![0; 32]);
}
