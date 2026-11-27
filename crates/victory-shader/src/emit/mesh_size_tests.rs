//! The size of a mesh shader's output as D3DMetal and Metal count it
//! ([`super::container::metal_record`], [`super::container::metal_mesh_bytes`]):
//! the table of halfwave50's probe (GPL-3.0-or-later, from
//! halfwave50/VIctory-macOS-halfwave b50fd5b, `emit::mesh_size`), which made
//! pipelines of hand-assembled mesh shaders under Wine and read the total
//! D3DMetal 4.0 beta 3 printed when Metal refused one (an M4 Pro, 26
//! November 2026). Each row is a signature in the probe's notation, the
//! bytes of one vertex and of one primitive; rows starting with `#` are
//! headings. The headings name the fork's own switches where it measured
//! shapes they made.
//!
//! The notation, as far as these rows use it: elements in signature order,
//! `P` the position, `C<n>` clip and `K<n>` cull distances of n components,
//! `f`, `u`, `i`, `c`, `n`, `e` and `s` followed by n other elements of n
//! components (types and interpolation modes), `*k` k of them, `q` before
//! an element a per-primitive one, `+` before one an element in the
//! previous one's register, `@r` or `@r.c` its register and column, `:NAME`
//! its semantic, `^m` and `!m` its interpolation and written components;
//! `v=`, `p=`, `ps=` and `pd=` the counts and the pixel shader (none of
//! them changes a record's size).

use super::container::{SigElement, metal_mesh_bytes, metal_record};

#[rustfmt::skip]
const MEASURED: &[(&str, u32, u32)] = &[
    ("# the total is a*v + b*p: one signature at several counts", 0, 0),
    ("P f4*16 v=128", 272, 0),
    ("P f4*16 v=129", 272, 0),
    ("P f4*16 v=200", 272, 0),
    ("P f4*16 v=256", 272, 0),
    ("P f4*16 v=128 p=1", 272, 0),
    ("P f4*16 v=128 p=256", 272, 0),
    ("P f4*15 v=128", 256, 0),
    ("P f4*15 qf4 v=128 p=128", 256, 16),
    ("P f4*15 qf4 v=128 p=129", 256, 16),
    ("P f4*7 qf4*8 v=256 p=256", 128, 128),
    ("# what the pixel shader does with them", 0, 0),
    ("P f4*16 ps=decl", 272, 0),
    ("P f4*16 ps=none", 272, 0),
    ("P f3*16 ps=read", 216, 0),
    ("P f3*16 ps=none", 216, 0),
    ("# the position alone, and one row of each width after it", 0, 0),
    ("P", 16, 0),
    ("P f1", 20, 0),
    ("P f2", 24, 0),
    ("P f3", 36, 0),
    ("P f4", 32, 0),
    ("# runs of one width", 0, 0),
    ("P f1*2", 24, 0),
    ("P f1*3", 28, 0),
    ("P f1*4", 32, 0),
    ("P f1*5", 36, 0),
    ("P f2*2", 32, 0),
    ("P f2*3", 40, 0),
    ("P f2*4", 48, 0),
    ("P f3*2", 48, 0),
    ("P f3*3", 60, 0),
    ("P f3*4", 72, 0),
    ("P f3*5", 84, 0),
    ("P f3*8", 120, 0),
    ("P f4*2", 48, 0),
    ("P f4*3", 64, 0),
    ("P f4*8", 144, 0),
    ("# two widths, both orders", 0, 0),
    ("P f2 f4", 48, 0),
    ("P f4 f2", 40, 0),
    ("P f3 f4", 64, 0),
    ("P f4 f3", 48, 0),
    ("P f1 f4", 48, 0),
    ("P f4 f1", 36, 0),
    ("P f3 f1", 40, 0),
    ("P f1 f3", 36, 0),
    ("P f3 f2", 48, 0),
    ("P f2 f3", 36, 0),
    ("P f1 f2", 32, 0),
    ("P f2 f1", 28, 0),
    ("P f3 f4 f3 f4", 112, 0),
    ("P f3 f3 f4 f4", 80, 0),
    ("P f4 f4 f3 f3", 72, 0),
    ("P f3 f1 f3 f1", 64, 0),
    ("P f3 f3 f1 f1", 56, 0),
    ("P f2 f2 f4", 48, 0),
    ("P f2 f4 f2", 56, 0),
    ("P f1 f1 f1 f1 f4", 48, 0),
    ("P f1 f4 f1 f4", 80, 0),
    ("# registers: gaps, order in the signature against register order, the position's place", 0, 0),
    ("P f4@1 f4@2", 48, 0),
    ("P f4@1 f4@5", 48, 0),
    ("P f4@2 f4@1", 48, 0),
    ("P f3@2 f4@1", 64, 0),
    ("P f4@1 f3@2", 48, 0),
    ("f4@0 P@1 f3@2", 48, 0),
    ("f4@0 f3@1 P@2", 48, 0),
    ("f3@0 f4@1 P@2", 64, 0),
    ("P@2 f3@0 f4@1", 64, 0),
    ("# several elements in one register", 0, 0),
    ("P f3 +f1", 40, 0),
    ("P f2 +f2", 32, 0),
    ("P f1 +f1 +f1 +f1", 32, 0),
    ("P f1 +f2 +f1", 36, 0),
    ("P f2 +f1", 28, 0),
    ("P f3 +f1 f3 +f1", 64, 0),
    ("P f1@1.3 f3@1.0", 36, 0),
    ("P f3@1.1", 36, 0),
    ("P f1@1.3", 20, 0),
    ("P f2@1.2", 24, 0),
    ("# clip distances: how many, and in the first, a middle and the last register", 0, 0),
    ("P C1", 36, 0),
    ("P C2", 40, 0),
    ("P C3", 48, 0),
    ("P C4", 48, 0),
    ("P C4 f4", 64, 0),
    ("P C4 f3", 60, 0),
    ("P C1 f4", 64, 0),
    ("P C1 f3", 48, 0),
    ("P C4 f4*4", 112, 0),
    ("P f4*2 C4 f4*2", 112, 0),
    ("P f4*4 C4", 112, 0),
    ("P C4 f3*4", 96, 0),
    ("P f3*2 C4 f3*2", 108, 0),
    ("P f3*4 C4", 112, 0),
    ("P C1 f4*4", 112, 0),
    ("P f4*4 C1", 100, 0),
    ("P C4 f4*4 pd=0", 112, 0),
    ("P f4*4 C4 pd=0", 112, 0),
    ("C4@0 P@1 f4*4", 112, 0),
    ("P C4 C4 f4*2", 96, 0),
    ("# cull distances", 0, 0),
    ("P K4 f4*4", 96, 0),
    ("P f4*4 K4", 96, 0),
    ("P K1 f4*4", 96, 0),
    ("P C4 K4 f4*4", 128, 0),
    ("P C2 +K2 f4*4", 112, 0),
    ("# integers against floats, flat against interpolated", 0, 0),
    ("P u4", 32, 0),
    ("P u3", 36, 0),
    ("P u2", 24, 0),
    ("P u1", 20, 0),
    ("P i4", 32, 0),
    ("P i1", 20, 0),
    ("P u1 f4", 48, 0),
    ("P f4 u1", 36, 0),
    ("P u1 +u1 f4", 48, 0),
    ("P u1 +u1 +u1 f4", 48, 0),
    ("P u3 f4", 64, 0),
    ("P u3 u3", 48, 0),
    ("P c4", 32, 0),
    ("P c3", 36, 0),
    ("P c3 c3", 48, 0),
    ("P c3 f4", 64, 0),
    ("P n3 n3", 48, 0),
    ("P n3 f4", 64, 0),
    ("P f3 c3 f3 c3", 72, 0),
    ("P f3 f3 c3 c3", 72, 0),
    ("P f3 u3 f3 u3", 72, 0),
    ("P e3 e3", 48, 0),
    ("P s3 s3", 48, 0),
    ("P f3 +c1", 40, 0),
    ("P f2 +u2", 32, 0),
    ("# per-primitive rows", 0, 0),
    ("P f4 qf4", 32, 16),
    ("P f4 qf3", 32, 12),
    ("P f4 qf3*2", 32, 24),
    ("P f4 qf3 qf4", 32, 32),
    ("P f4 qf1", 32, 4),
    ("P f4 qu1", 32, 4),
    ("P f4 qf2 qf4", 32, 32),
    ("P f4 qf4*4", 32, 64),
    ("P f3*3 qf3*3", 60, 36),
    ("# the counts: vertices are rounded up to a multiple of 4, primitives are not", 0, 0),
    ("P f4*16 v=125", 272, 0),
    ("P f4*16 v=126", 272, 0),
    ("P f4*16 v=127", 272, 0),
    ("P f4*16 v=130", 272, 0),
    ("P f4*16 v=131", 272, 0),
    ("P f4*16 v=132", 272, 0),
    ("P f4*16 v=133", 272, 0),
    ("P f4*15 qf4 p=127", 256, 16),
    ("P f4*15 qf4 p=130", 256, 16),
    ("P f4*15 qf4 p=131", 256, 16),
    ("# where the 16 bytes of a shader with clip distances sit", 0, 0),
    ("P f1 C1", 40, 0),
    ("P f3 C1", 52, 0),
    ("P f2 C1", 44, 0),
    ("P f1 C4", 64, 0),
    ("P f3 C4", 64, 0),
    ("P f3 C4 f3", 84, 0),
    ("f3@0 P@1 C1@2", 52, 0),
    ("f3@0 C1@1 P@2", 52, 0),
    ("C1@0 f3@1 P@2", 48, 0),
    ("P f3 f3 C1 f1", 68, 0),
    ("P C1 C1", 40, 0),
    ("P C1 +C1", 40, 0),
    ("P C1 +C1 +C1 +C1", 48, 0),
    ("P C2 C2", 48, 0),
    ("P C4 C1", 52, 0),
    ("P C4 C4", 64, 0),
    ("P C4 C4 C4", 80, 0),
    ("P f3 K1", 40, 0),
    ("P f3 K4 f3", 84, 0),
    ("# clip distances: interpolation mode, and components never written (in another element, P f4!b, D3DMetal ends the process)", 0, 0),
    ("P C4^4 f4", 64, 0),
    ("P C4^1 f4", 64, 0),
    ("P C4^0 f4", 64, 0),
    ("P C4!1 f4", 64, 0),
    ("P C4!3 f4", 64, 0),
    ("P C4!0 f4", 64, 0),
    ("P f3 C4!1", 64, 0),
    ("# order: the signature's, the registers' or the names'", 0, 0),
    ("P f3:TEXCOORD0@2 f4:TEXCOORD1@1", 64, 0),
    ("P f3:TEXCOORD1@2 f4:TEXCOORD0@1", 64, 0),
    ("P f3:TEXCOORD1@1 f4:TEXCOORD0@2", 64, 0),
    ("P f4:TEXCOORD0@2 f3:TEXCOORD1@1", 48, 0),
    ("P f4:TEXCOORD1@2 f3:TEXCOORD0@1", 48, 0),
    ("P f4:TEXCOORD1@1 f3:TEXCOORD0@2", 48, 0),
    ("P f3:COLOR0 f4:TEXCOORD0", 64, 0),
    ("P f3:TEXCOORD0 f4:COLOR0", 64, 0),
    ("P f4:COLOR0 f3:TEXCOORD0", 48, 0),
    ("P f4:TEXCOORD0 f3:COLOR0", 48, 0),
    ("P f3:B5 f4:A7 f3:A2 f4:C0", 112, 0),
    ("P f3:A2 f3:B5 f4:A7 f4:C0", 80, 0),
    ("# one register, several elements", 0, 0),
    ("P f2@1 f1@1.3", 28, 0),
    ("P f2@1 f1@1.2", 28, 0),
    ("P f1@1 f2@1.2", 32, 0),
    ("P f2@1.2 f2@1.0", 32, 0),
    ("P f1@1.1 f1@1.3 f1@1.0 f1@1.2", 32, 0),
    ("# many small elements", 0, 0),
    ("P f1*52", 224, 0),
    ("P f1*60", 256, 0),
    ("P f2*30", 256, 0),
    ("P f1*16 f2*8 f3*4 f4*2", 224, 0),
    ("# shapes of 17 to 19 registers, each in three orders (the sizes a game on this port logged for them: 264 to 272, and 260 against a fit)", 0, 0),
    ("P@0 C4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17", 264, 0),
    ("P@0 C4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f4@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17", 272, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 C4@17", 272, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 C4@17", 272, 0),
    ("P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f3@9 f2@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C4@17", 272, 0),
    ("P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@9 f4@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C4@17", 272, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f2@15 f1@16 C1@17", 248, 0),
    ("P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@9 f1@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C1@17", 260, 0),
    ("C1@0 P@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17.2 u1@18", 252, 0),
    ("f3@0 P@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@10.2 u1@11 f4@12 f4@13 f4@14 f4@15 f4@16 f4@17 C1@18", 260, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f4@8 f4@9 f4@10 f4@11 f4@12 f3@13 f1@14.3 u1@15 C1@16", 252, 0),
    ("P@0 f3@1 f1@2.3 u1@3 f4@4 f4@5 f4@6 f4@7 f4@8 f4@9 f4@10 f4@11 f4@12 f4@13 f4@14 f4@15 C1@16", 260, 0),
    ("# the two two shapes as VICTORY_MERGE_MESH_ROWS=6 makes them : the first fits exactly", 0, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C4@17", 256, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 C4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 f2@17 f1@17.3", 260, 0),
    ("# the second with its four clip distances as an ordinary element (the 16 bytes go)", 0, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 f2@17 f1@17.3", 252, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f2@8 f2@9 f1@9.3 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f3@17", 252, 0),
    ("P C3 C1 f4", 80, 0),
    ("# the second as VICTORY_MERGE_MESH_ROWS=7 makes it", 0, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f3@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f4@14 f4@15 f2@16 f2@17 f1@17.3", 244, 0),
    ("# the two shapes as VICTORY_MERGE_MESH_ROWS=7 and 8 make them now: the three distances that never clip left out", 0, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C1@17", 244, 0),
    ("P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f2@7 f1@7.3 C1@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17", 248, 0),
];

/// A row's vertex and primitive elements.
fn parse(line: &str) -> Result<(Vec<SigElement>, Vec<SigElement>), String> {
    let (mut vertex, mut prims) = (Vec::new(), Vec::new());
    let mut row = 0u32;
    for token in line.split_whitespace() {
        if ["v=", "p=", "ps=", "pd="].iter().any(|p| token.starts_with(p)) {
            continue;
        }
        let mut t = token;
        let same = t.starts_with('+');
        t = t.trim_start_matches('+');
        let prim = t.starts_with('q');
        t = t.trim_start_matches('q');
        let mut chars = t.chars().peekable();
        let letter = chars.next().ok_or("an empty element")?;
        let kind = match letter {
            'P' => 3,
            'C' => 6,
            'K' => 7,
            'f' | 'u' | 'i' | 'c' | 'n' | 'e' | 's' => 0,
            other => return Err(format!("{other:?} in {token}")),
        };
        let number = |chars: &mut std::iter::Peekable<std::str::Chars>| {
            let mut n = String::new();
            while let Some(&c) = chars.peek().filter(|c| c.is_ascii_digit()) {
                n.push(c);
                chars.next();
            }
            n.parse::<u32>().ok()
        };
        let cols = match number(&mut chars) {
            Some(n) => n as u8,
            None if kind == 3 => 4,
            None => return Err(format!("no width in {token}")),
        };
        let mut count = 1;
        let mut at = None;
        while let Some(c) = chars.next() {
            match c {
                '*' => count = number(&mut chars).ok_or("a count")?,
                '@' => {
                    let r = number(&mut chars).ok_or("a register")?;
                    let col = if chars.peek() == Some(&'.') {
                        chars.next();
                        number(&mut chars).ok_or("a column")?
                    } else {
                        0
                    };
                    at = Some((r, col));
                }
                ':' => while chars.peek().is_some_and(|c| !"@^!".contains(*c)) {
                    chars.next();
                },
                '^' | '!' => {
                    number(&mut chars);
                }
                other => return Err(format!("{other:?} in {token}")),
            }
        }
        for _ in 0..count {
            let (r, col) = match at {
                Some(at) => at,
                None if same => (row.saturating_sub(1), 0),
                None => (row, 0),
            };
            row = row.max(r + 1);
            let e = SigElement {
                name: ["TEXCOORD", "SV_Position", "SV_ClipDistance", "SV_CullDistance"][match kind {
                    3 => 1,
                    6 => 2,
                    7 => 3,
                    _ => 0,
                }]
                .into(),
                index: 0,
                kind,
                comp: 9,
                interp: 2,
                row: Some(r),
                col: col as u8,
                cols,
                usage: (1 << cols) - 1,
                half: false,
            };
            if prim { prims.push(e) } else { vertex.push(e) }
        }
    }
    Ok((vertex, prims))
}

/// The record gives every size the probe measured.
#[test]
fn metal_records_give_every_measured_size() {
    let mut rows = 0;
    for &(line, vertex, primitive) in MEASURED.iter().filter(|m| !m.0.starts_with('#')) {
        let (v, p) = parse(line).unwrap_or_else(|e| panic!("{line}: {e}"));
        let prims = if p.is_empty() { 0 } else { metal_record(&p) };
        assert_eq!((metal_record(&v), prims), (vertex, primitive), "{line}");
        rows += 1;
    }
    assert!(rows >= 200, "{rows} measured rows");
}

/// Vertices count in fours, primitives singly (measured: `P f4*16` at 125
/// to 133 vertices is refused for 34816, 35904 and 36992 bytes; `P f4*15
/// qf4` at 127 primitives for 34800).
#[test]
fn mesh_bytes_round_vertices_up_to_fours() {
    let (v, _) = parse("P f4*16").unwrap();
    for (vertices, total) in [(125, 34816), (128, 34816), (129, 35904), (132, 35904), (133, 36992), (256, 69632)] {
        assert_eq!(metal_mesh_bytes(&v, &[], vertices, 128), total, "{vertices} vertices");
    }
    let (v, p) = parse("P f4*15 qf4").unwrap();
    assert_eq!(metal_mesh_bytes(&v, &p, 128, 127), 34800);
    assert_eq!(metal_mesh_bytes(&v, &p, 128, 129), 34832);
    assert_eq!(metal_mesh_bytes(&v, &[], 128, 128), super::container::METAL_MESH_BYTES);
}
