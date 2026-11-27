//! Packed 16-bit colour kept as R8G8B8A8_UNORM on the host.
//!
//! D3DMetal has no typed unordered store for B5G5R5A1_UNORM (86) or
//! B4G4R4A4_UNORM (115): an unordered view of such a texture stays null and
//! its writes are dropped (a compute pass filling one that multiplies the
//! sun's colour left direct sunlight black). Where the device reports no
//! render-target or typed-UAV support for one of them, a render target or
//! unordered-access texture in it is made R8G8B8A8_UNORM on the host
//! (`resource::target_format`): every channel keeps its value, with more
//! precision. Views naming the packed format read and write the wider one
//! (`descriptor::host`), which a shader sees as the same four channels.
//!
//! Bytes move between the two layouts here: a title's texels (2 bytes) are
//! widened where they go into such a texture (detiles, copies from a
//! buffer) and narrowed where they come out (read backs). Plain std, so its
//! tests run everywhere.
//!
//! After Belobug da580e4e, as halfwave50 ported it (273475d).

// Without the host GPU only its tests use it.
#![cfg_attr(not(windows), allow(dead_code))]

/// R8G8B8A8_UNORM, the host format of such a texture.
pub(crate) const RGBA8_UNORM: u32 = 28;

/// B5G5R5A1_UNORM and B4G4R4A4_UNORM.
pub(crate) const PACKED_16: [u32; 2] = [86, 115];
const B5G5R5A1: u32 = 86;

/// Whether `format` is one of [`PACKED_16`].
pub(crate) fn is_packed(format: u32) -> bool {
    PACKED_16.contains(&format)
}

/// An `n`-bit UNORM channel as 8 bits, rounded to nearest (bit replication
/// is exact for 1, 4 and 5 bits).
fn up(v: u16, bits: u32) -> u8 {
    match bits {
        1 => (v & 1) as u8 * 255,
        4 => (v & 0xf) as u8 * 17,
        _ => {
            let v = (v & 0x1f) as u8;
            v << 3 | v >> 2
        }
    }
}

/// An 8-bit UNORM channel as `bits` bits, rounded to nearest.
fn down(v: u8, bits: u32) -> u16 {
    let max = (1u32 << bits) - 1;
    ((u32::from(v) * max + 127) / 255) as u16
}

/// One texel of `format` (B in the low bits, A in the high ones) as R, G,
/// B, A bytes.
pub(crate) fn widen_texel(format: u32, t: u16) -> [u8; 4] {
    if format == B5G5R5A1 {
        [up(t >> 10, 5), up(t >> 5, 5), up(t, 5), up(t >> 15, 1)]
    } else {
        [up(t >> 8, 4), up(t >> 4, 4), up(t, 4), up(t >> 12, 4)]
    }
}

/// R, G, B, A bytes as one texel of `format`.
pub(crate) fn narrow_texel(format: u32, [r, g, b, a]: [u8; 4]) -> u16 {
    if format == B5G5R5A1 {
        down(b, 5) | down(g, 5) << 5 | down(r, 5) << 10 | down(a, 1) << 15
    } else {
        down(b, 4) | down(g, 4) << 4 | down(r, 4) << 8 | down(a, 4) << 12
    }
}

/// Texels of `format` (2 bytes each) into R8G8B8A8 (4 bytes each), as many
/// as both hold.
pub(crate) fn widen(format: u32, from: &[u8], to: &mut [u8]) {
    for (t, o) in from.as_chunks::<2>().0.iter().zip(to.as_chunks_mut::<4>().0) {
        o.copy_from_slice(&widen_texel(format, u16::from_le_bytes(*t)));
    }
}

/// R8G8B8A8 texels into texels of `format`, as many as both hold.
pub(crate) fn narrow(format: u32, from: &[u8], to: &mut [u8]) {
    for (t, o) in from.as_chunks::<4>().0.iter().zip(to.as_chunks_mut::<2>().0) {
        o.copy_from_slice(&narrow_texel(format, *t).to_le_bytes());
    }
}

/// `lines` lines of `width` texels of `format`, `from_pitch` bytes apart,
/// widened into lines `to_pitch` bytes apart (lines past the end of either
/// are left out, a last one cut short is widened as far as it goes).
pub(crate) fn widen_lines(format: u32, from: &[u8], from_pitch: usize, to: &mut [u8], to_pitch: usize, width: usize, lines: usize) {
    for line in 0..lines {
        let (f, t) = (line * from_pitch, line * to_pitch);
        if f >= from.len() || t >= to.len() {
            break;
        }
        let f = &from[f..(f + 2 * width).min(from.len())];
        let end = (t + 4 * width).min(to.len());
        widen(format, f, &mut to[t..end]);
    }
}

/// The row pitch of a footprint of `width` R8G8B8A8 texels: the host's
/// 256-byte multiple.
pub(crate) fn wide_pitch(width: u32) -> u32 {
    (width * 4).next_multiple_of(256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_texel_survives_a_widening_and_a_narrowing() {
        for format in PACKED_16 {
            for t in 0..=u16::MAX {
                assert_eq!(narrow_texel(format, widen_texel(format, t)), t, "{format} {t:#06x}");
            }
        }
    }

    #[test]
    fn channels_land_where_the_formats_name_them() {
        // B5G5R5A1: B in bits 0-4, G 5-9, R 10-14, A 15.
        assert_eq!(widen_texel(86, 0x001f), [0, 0, 255, 0]);
        assert_eq!(widen_texel(86, 0x03e0), [0, 255, 0, 0]);
        assert_eq!(widen_texel(86, 0x7c00), [255, 0, 0, 0]);
        assert_eq!(widen_texel(86, 0x8000), [0, 0, 0, 255]);
        assert_eq!(widen_texel(86, 0x0010), [0, 0, 132, 0]);
        // B4G4R4A4: B in bits 0-3, G 4-7, R 8-11, A 12-15.
        assert_eq!(widen_texel(115, 0x000f), [0, 0, 255, 0]);
        assert_eq!(widen_texel(115, 0x00f0), [0, 255, 0, 0]);
        assert_eq!(widen_texel(115, 0x0f00), [255, 0, 0, 0]);
        assert_eq!(widen_texel(115, 0xf000), [0, 0, 0, 255]);
        assert_eq!(widen_texel(115, 0x1234), [34, 51, 68, 17]);
    }

    /// A shader's 8-bit result goes back to the nearest packed value.
    #[test]
    fn narrowing_rounds_to_the_nearest_value() {
        for v in 0..=255u8 {
            let near5 = (0..32u16).min_by_key(|&q| (i32::from(v) * 31 - i32::from(q) * 255).abs()).unwrap();
            let near4 = (0..16u16).min_by_key(|&q| (i32::from(v) * 15 - i32::from(q) * 255).abs()).unwrap();
            assert_eq!(narrow_texel(86, [v, 0, 0, 0]) >> 10, near5, "{v}");
            assert_eq!(narrow_texel(115, [0, 0, v, 0]) & 0xf, near4, "{v}");
            assert_eq!(narrow_texel(86, [0, 0, 0, v]) >> 15, u16::from(v >= 128), "{v}");
        }
    }

    #[test]
    fn lines_widen_into_their_own_pitch() {
        let width = 3;
        let from: Vec<u8> = (0..2 * 8).map(|i| i as u8).collect(); // 2 lines, pitch 8
        let mut to = vec![0xaa; 256 + 12];
        widen_lines(115, &from, 8, &mut to, 256, width, 2);
        for line in 0..2 {
            for x in 0..width {
                let t = u16::from_le_bytes([from[line * 8 + 2 * x], from[line * 8 + 2 * x + 1]]);
                assert_eq!(to[line * 256 + 4 * x..][..4], widen_texel(115, t));
            }
        }
        assert!(to[12..256].iter().all(|&b| b == 0xaa), "the pitch's padding is left alone");
        let mut back = vec![0u8; 2 * width];
        narrow(115, &to[256..256 + 12], &mut back);
        assert_eq!(back, from[8..8 + 6]);
        assert_eq!(wide_pitch(3), 256);
        assert_eq!(wide_pitch(64), 256);
        assert_eq!(wide_pitch(65), 512);
    }
}
