//! The bytes a host copy held before a submission ran, where the
//! submission's shaders write (`crate::shadow`'s snapshots): segments
//! (offset in the copy, bytes), in order and disjoint. Plain std, so its
//! tests run everywhere.
//!
//! A byte is kept once, as the copy held it when the first range holding it
//! was declared, and the segments hold exactly the declared ranges' bytes.
//! Two ways to keep them, the same bytes either way:
//! - joined ([`cover_joined`]): ranges that overlap or touch share one
//!   segment, rebuilt whole whenever a range grows it. Dispatches whose
//!   ranges each reach a little past the last one's (root views a megabyte
//!   long at rising addresses) copied the segment again each time: 314 MB a
//!   frame for one 64 MB copy of which 60 MB were declared (run 22);
//! - apart ([`cover_apart`]): only the bytes no segment holds yet are
//!   taken, as segments of their own.

/// A segment: where it starts in the copy, and the bytes.
pub(crate) type Segment = (usize, Vec<u8>);

/// Keeps `[from, to)` (not empty) in one segment with every segment it
/// overlaps or touches: parts newly covered from the copy (`host` gives its
/// bytes now), the rest as it was taken. `room` gives an empty buffer for so
/// many bytes, `done` takes the buffers replaced. Returns the bytes of the
/// segment made (none if one segment held the range already), and how many
/// of them came from older segments.
pub(crate) fn cover_joined<'h>(
    segments: &mut Vec<Segment>,
    from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'h [u8],
    mut room: impl FnMut(usize) -> Vec<u8>,
    mut done: impl FnMut(Vec<u8>),
) -> (usize, usize) {
    // The segments the range overlaps or touches.
    let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
    let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
    if let [(at, bytes)] = &segments[first..last]
        && *at <= from
        && to <= at + bytes.len()
    {
        return (0, 0);
    }
    let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
    let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
    let mut grown = room(end - start);
    let (mut at, mut again) = (start, 0);
    for (taken, bytes) in segments.drain(first..last) {
        grown.extend_from_slice(host(at, taken));
        grown.extend_from_slice(&bytes);
        at = taken + bytes.len();
        again += bytes.len();
        done(bytes);
    }
    grown.extend_from_slice(host(at, end));
    segments.insert(first, (start, grown));
    (end - start, again)
}

/// Keeps the parts of `[from, to)` no segment holds yet, each as a segment
/// of its own, from the copy (`host` gives its bytes now); what segments
/// hold stays as it was taken. Returns the bytes taken.
pub(crate) fn cover_apart<'h>(
    segments: &mut Vec<Segment>,
    from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'h [u8],
    mut room: impl FnMut(usize) -> Vec<u8>,
) -> usize {
    let mut taken = 0;
    for (a, b) in missing(segments, from, to) {
        let mut bytes = room(b - a);
        bytes.extend_from_slice(host(a, b));
        let i = segments.partition_point(|(at, _)| *at < a);
        segments.insert(i, (a, bytes));
        taken += b - a;
    }
    taken
}

/// The parts of `[from, to)` no segment holds, in order.
pub(crate) fn missing(segments: &[Segment], from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = from;
    for (start, bytes) in &segments[segments.partition_point(|(s, bytes)| s + bytes.len() <= from)..] {
        if *start >= to {
            break;
        }
        if *start > at {
            out.push((at, *start));
        }
        at = at.max(start + bytes.len());
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// Whether the segments hold all of `[from, to)` (not empty): one segment,
/// or several that touch. Joined segments never touch, so there it is one.
pub(crate) fn covered(segments: &[Segment], from: usize, to: usize) -> bool {
    missing(segments, from, to).is_empty()
}

/// The bytes kept of `[from, to)`, which the segments hold ([`covered`]):
/// a segment's own where one holds them all, else gathered into `gathered`.
pub(crate) fn held<'a>(segments: &'a [Segment], from: usize, to: usize, gathered: &'a mut Vec<u8>) -> &'a [u8] {
    let first = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
    let (at, bytes) = &segments[first];
    if to <= at + bytes.len() {
        return &bytes[from - at..to - at];
    }
    gathered.clear();
    let mut next = from;
    for (at, bytes) in &segments[first..] {
        if next >= to {
            break;
        }
        let end = (at + bytes.len()).min(to);
        gathered.extend_from_slice(&bytes[next - at..end - at]);
        next = end;
    }
    gathered
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy's bytes: its offset and a counter, so that every byte tells
    /// where and when it was taken.
    fn copy(n: usize, time: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31) ^ time).collect()
    }

    fn joined(segments: &mut Vec<Segment>, host: &[u8], from: usize, to: usize) -> (usize, usize) {
        cover_joined(segments, from, to, |a, b| &host[a..b], Vec::with_capacity, drop)
    }

    fn apart(segments: &mut Vec<Segment>, host: &[u8], from: usize, to: usize) -> usize {
        cover_apart(segments, from, to, |a, b| &host[a..b], Vec::with_capacity)
    }

    /// In order, not empty, and disjoint.
    fn well_formed(segments: &[Segment]) -> bool {
        segments.iter().all(|s| !s.1.is_empty()) && segments.windows(2).all(|w| w[0].0 + w[0].1.len() <= w[1].0)
    }

    #[test]
    fn only_bytes_not_held_yet_are_taken_apart() {
        let host = copy(1000, 0);
        let mut s = Vec::new();
        assert_eq!(apart(&mut s, &host, 100, 200), 100);
        assert_eq!(apart(&mut s, &host, 150, 260), 60, "the part past the first");
        assert_eq!(apart(&mut s, &host, 100, 260), 0, "held by two that touch");
        assert_eq!(apart(&mut s, &host, 50, 400), 190, "before and after");
        assert_eq!(apart(&mut s, &host, 600, 700), 100);
        let starts: Vec<(usize, usize)> = s.iter().map(|(at, b)| (*at, b.len())).collect();
        assert_eq!(starts, [(50, 50), (100, 100), (200, 60), (260, 140), (600, 100)]);
        assert!(well_formed(&s));
        assert!(covered(&s, 50, 400) && covered(&s, 199, 201) && covered(&s, 600, 700));
        assert!(!covered(&s, 49, 60) && !covered(&s, 390, 601) && !covered(&s, 399, 401));
        assert_eq!(missing(&s, 0, 1000), [(0, 50), (400, 600), (700, 1000)]);
        let mut gathered = Vec::new();
        assert_eq!(held(&s, 120, 180, &mut gathered), &host[120..180]);
        assert_eq!(held(&s, 60, 390, &mut gathered), &host[60..390], "across four segments");
    }

    /// Joined, the same ranges cost the whole segment each time it grows.
    #[test]
    fn joined_segments_are_copied_again_as_they_grow() {
        let host = copy(1000, 0);
        let (mut j, mut a) = (Vec::new(), Vec::new());
        let (mut joined_bytes, mut again, mut apart_bytes) = (0, 0, 0);
        for k in 0..8 {
            let (from, to) = (100 + 50 * k, 200 + 50 * k);
            let (made, old) = joined(&mut j, &host, from, to);
            joined_bytes += made;
            again += old;
            apart_bytes += apart(&mut a, &host, from, to);
        }
        assert_eq!(apart_bytes, 450, "each byte once");
        assert_eq!(joined_bytes - again, 450);
        assert_eq!(again, 100 + 150 + 200 + 250 + 300 + 350 + 400);
        assert_eq!(j.len(), 1);
        assert_eq!(a.len(), 8);
    }

    /// Whatever ranges are declared, while the copy's bytes change between
    /// them (refreshes under declared ranges put the new bytes into the
    /// segments too), both ways hold the same bytes of the same ranges.
    #[test]
    fn apart_and_joined_hold_the_same_bytes() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        const N: usize = 4096;
        for round in 0..200 {
            let mut host = copy(N, round as u8);
            let (mut j, mut a): (Vec<Segment>, Vec<Segment>) = (Vec::new(), Vec::new());
            let mut declared = vec![false; N];
            for step in 0..40 {
                // Short ranges close together, as lists declare them, and
                // some long ones.
                let len = 1 + if random(4) == 0 { random(N / 2) } else { random(64) };
                let from = if random(3) == 0 { random(N - len) } else { (random(16) * 60).min(N - len) };
                let to = from + len;
                let (made, again) = joined(&mut j, &host, from, to);
                let taken = apart(&mut a, &host, from, to);
                let new = declared[from..to].iter().filter(|d| !**d).count();
                assert_eq!(taken, new, "apart takes each byte once");
                assert!(made == 0 && new == 0 || made - again >= new);
                declared[from..to].fill(true);
                assert!(well_formed(&j) && well_formed(&a));
                // The copy changes: a refresh under a declared range goes
                // into the segments holding it, as `Shadow::update_busy`
                // puts it; other bytes just change.
                let (p, q) = (random(N - 8), 1 + random(8));
                for at in p..p + q {
                    host[at] = host[at].wrapping_add(step as u8 + 1);
                }
                if random(2) == 0 {
                    for segments in [&mut j, &mut a] {
                        for (at, bytes) in segments.iter_mut() {
                            let (s, e) = ((*at).max(p), (*at + bytes.len()).min(p + q));
                            if s < e {
                                bytes[s - *at..e - *at].copy_from_slice(&host[s..e]);
                            }
                        }
                    }
                }
                // Both hold exactly the declared bytes, with the same values.
                let flat = |segments: &[Segment]| {
                    let mut all = vec![None; N];
                    for (at, bytes) in segments {
                        for (k, b) in bytes.iter().enumerate() {
                            assert!(all[at + k].replace(*b).is_none(), "held twice");
                        }
                    }
                    all
                };
                let (fj, fa) = (flat(&j), flat(&a));
                assert!(fj == fa, "round {round} step {step}");
                assert!(fj.iter().zip(&declared).all(|(b, d)| b.is_some() == *d));
                // And answer alike about any range.
                for _ in 0..8 {
                    let from = random(N - 1);
                    let to = from + 1 + random((N - from).min(200));
                    let one = j.iter().any(|(at, bytes)| *at <= from && to <= at + bytes.len());
                    assert_eq!(covered(&j, from, to), one, "joined segments never touch");
                    assert_eq!(covered(&a, from, to), one);
                    if one {
                        let (mut g, mut h) = (Vec::new(), Vec::new());
                        assert_eq!(held(&j, from, to, &mut g), held(&a, from, to, &mut h));
                    }
                }
            }
        }
    }
}
