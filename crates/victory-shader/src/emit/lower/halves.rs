//! Console waves split into halves on narrow host waves: which blocks of
//! the program the halves meet in, the exchanges between them, and what is
//! kept at the console's wave size when a wave operation has no exact
//! two-half form ([`kept_wide`]).

use super::*;

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
pub(super) fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

/// The blocks each lane runs exactly once: no cycle through them, and on
/// every path from the entry to an end. Every thread of a group meets at a
/// group barrier there.
pub(crate) fn once_blocks(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}

/// The blocks every lane of a console wave runs as often as the wave does:
/// no per-lane branch decides, directly or through the branches around it,
/// whether or how often a lane runs them (uniform branches and loops may).
/// Where a group is one console wave its threads meet at a group barrier
/// there: all take the wave's path, as often.
///
/// `live`: every lane still running, that is: a per-lane branch whose other
/// way is the program's end does not count (`if (...) return;` for some
/// lanes, then code for the rest). The threads that ended never reach a
/// barrier there, which Metal leaves undefined; Apple's GPUs (M5: groups
/// of 64 with any lanes gone, a whole host wave too, barriers in a loop)
/// make the rest meet.
pub(super) fn wave_blocks(x: &Parts, live: bool) -> Vec<bool> {
    let p = x.p;
    let succ = |b: usize| p.blocks[b].exit.successors();
    let per_lane = |q: usize| match x.st.exits[q] {
        LaneExit::Branch {
            when,
            taken,
            not_taken,
        } => {
            taken != not_taken
                && !x
                    .st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_) | Atom::Wave(_)))
        }
        LaneExit::Indirect => true,
        _ => false,
    };
    // A per-lane branch to the program's end only takes lanes out for good.
    let early = |q: usize| live && succ(q).iter().any(|&s| ends(p, s));
    undecided(p, |q| per_lane(q) && !early(q))
}

/// The scalar values every lane of a console wave holds alike where a
/// group is one console wave run in halves with branches taken wave-wide
/// ([`Structured::wide`]): the SGPRs it starts with, what scalar
/// instructions make of such values, what a wave's lane reads give (the
/// console wave's value in both halves), and the joins of such values
/// where the wave arrives as one (`together`: every predecessor a block
/// all its lanes run together). A join after a per-lane branch is each
/// lane's own: the lanes that skipped hold what the other way left.
/// Joins are all taken to hold first, then those that do not are taken out
/// until the rest do (loops).
pub(super) fn wave_same(p: &Program, st: &Structured, together: &[bool]) -> Set<Value> {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let scalar = |r: Reg| matches!(r, Reg::S(c) if c <= 105) || r == Reg::Scc;
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut joins: Vec<(usize, &Phi)> = order
        .iter()
        .flat_map(|&b| p.blocks[b].phis.iter().map(move |phi| (b, phi)))
        .filter(|(_, phi)| scalar(phi.reg) && !is_mask(phi.value))
        .collect();
    loop {
        let mut same: Set<Value> = joins.iter().map(|j| j.1.value).collect();
        for &(reg, v) in &p.inputs {
            if scalar(reg) {
                same.insert(v);
            }
        }
        // (a mask read as a number is the console's: a ballot of each half,
        // exchanged)
        same.extend(st.masks.keys().chain(st.highs.keys()).copied());
        for &b in &order {
            for inst in &p.blocks[b].insts {
                let name = inst.name();
                let made = match &inst.fields {
                    Fields::Sop { .. } | Fields::Smem { .. } => inst
                        .reads
                        .iter()
                        .filter(|r| r.0 != Reg::S(EXEC) && r.0 != Reg::S(EXEC + 1))
                        .all(|r| same.contains(&r.1)),
                    // (exchanged between the halves, or one lane's alone)
                    Fields::Valu(_) => matches!(name, "v_readfirstlane_b32" | "v_readlane_b32"),
                    _ => false,
                };
                if made {
                    same.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                }
            }
        }
        let before = joins.len();
        joins.retain(|&(b, phi)| {
            together[b]
                && p.blocks[b].preds.iter().all(|&q| together[q])
                && phi.args.iter().all(|a| same.contains(a))
        });
        if joins.len() == before {
            return same;
        }
    }
}

/// The values every lane of a console wave holds alike, of a compute
/// program whose group is several console waves run in halves (after
/// avatarmanz/VIctory-7900XTX PR #142): its `v_readfirstlane_b32` of such a
/// value is any lane's, so a host wave takes its own first lane where the
/// halves cannot meet. Lanes of one console wave hold a value alike when it
/// is made, lane by lane, of such values, constants and the console wave's
/// scalars, by an instruction every lane of the wave reaching it runs, or
/// under a condition the wave's lanes hold alike over an old value they
/// hold alike; read from LDS at such an address (shortcut: as the halves
/// form takes LDS in such groups, what a console wave writes is not read
/// back by itself before a barrier); or the thread id's x past the console
/// wave, `id >> 6` and up (`tid`, where console waves are runs of 64 ids
/// along x). A join counts where every predecessor is reached by whole
/// console waves: no branch on a condition the lanes of a wave may hold
/// differently decides it. The compare results whose lane bit is alike
/// (the conditions branches and EXEC test) are kept apart from the values
/// (`bits`: a mask is a number of the whole wave, not a lane's value).
/// Scalars are the console wave's (SGPRs, what scalar instructions and lane
/// reads make of them, masks read as numbers: [`wave_same`]), but for
/// joins: all joins are taken to hold first, and those that do not on that
/// footing are taken out until the rest do (loops, by the order the lanes
/// of a wave run them in, which is one order for all).
pub(super) fn wave_alike(
    p: &Program,
    st: &Structured,
    tid: Option<Value>,
    own: &Set<Value>,
) -> Set<Value> {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let scalar = |r: Reg| matches!(r, Reg::S(c) if c <= 105) || r == Reg::Scc;
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut joins: Vec<(usize, &Phi)> = order
        .iter()
        .flat_map(|&b| p.blocks[b].phis.iter().map(move |phi| (b, phi)))
        .filter(|(_, phi)| (scalar(phi.reg) || matches!(phi.reg, Reg::V(_))) && !is_mask(phi.value))
        .collect();
    let succ = |b: usize| p.blocks[b].exit.successors();
    loop {
        let mut same: Set<Value> = joins.iter().map(|j| j.1.value).collect();
        let mut bits: Set<Value> = Set::default();
        for &(reg, v) in &p.inputs {
            if scalar(reg) {
                same.insert(v);
            }
        }
        same.extend(st.masks.keys().chain(st.highs.keys()).copied());
        // The blocks whole console waves reach, by the bits known so far.
        let alike = |same: &Set<Value>, bits: &Set<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| match *a {
                Atom::Uniform(v) => same.contains(&v),
                Atom::Lane(v) => bits.contains(&v),
                Atom::Wave(_) => true,
                _ => false,
            })
        };
        // (passes until nothing is added: a guard may test a bit made
        // further on, round a loop)
        loop {
            let known = same.len() + bits.len();
            for &b in &order {
                for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                    let name = inst.name();
                    let reads_alike = |same: &Set<Value>| {
                        inst.reads
                            .iter()
                            .filter(|r| r.0 != Reg::S(EXEC) && r.0 != Reg::S(EXEC + 1))
                            .all(|r| same.contains(&r.1))
                    };
                    let guard_ok =
                        |same: &Set<Value>, bits: &Set<Value>| match st.guards[b][i] {
                            None => true,
                            // (the lanes the guard leaves out keep the old value)
                            Some(g) => {
                                alike(same, bits, g)
                                    && inst.writes.iter().filter(|w| matches!(w.0, Reg::V(_))).all(
                                        |w| inst.read(w.0).is_none_or(|old| same.contains(&old)),
                                    )
                            }
                        };
                    match &inst.fields {
                        Fields::Sop { .. } | Fields::Smem { .. } => {
                            if reads_alike(&same) {
                                same.extend(
                                    inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1),
                                );
                            }
                        }
                        // (exchanged between the halves, or one lane's alone;
                        // a waterfall loop's read is the lane's own)
                        Fields::Valu(_)
                            if matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") =>
                        {
                            same.extend(
                                inst.writes
                                    .iter()
                                    .filter(|w| scalar(w.0) && !own.contains(&w.1))
                                    .map(|w| w.1),
                            );
                        }
                        // an LDS read at an alike address reads one dword in every lane
                        Fields::Ds {
                            addr: Some(Operand::Vector { reg, .. }),
                            dst: Some(_),
                            data,
                            gds: false,
                            ..
                        } if name.starts_with("ds_read_b") && data.is_empty() => {
                            if inst.read(Reg::V(*reg)).is_some_and(|a| same.contains(&a))
                                && guard_ok(&same, &bits)
                            {
                                same.extend(
                                    inst.writes
                                        .iter()
                                        .filter(|w| matches!(w.0, Reg::V(_)))
                                        .map(|w| w.1),
                                );
                            }
                        }
                        Fields::Valu(v) => {
                            let ops = &v.src;
                            if v.dpp.is_some()
                                || v.sdwa.is_some()
                                || v.carry.is_some()
                                || ["lane", "mbcnt", "cndmask", "_co_", "_ci_", "addc", "subb"]
                                    .iter()
                                    .chain(&["movrel", "swap", "mac", "div_fmas"])
                                    .any(|n| name.contains(n))
                                || !guard_ok(&same, &bits)
                            {
                                continue;
                            }
                            // the thread id's x past the console wave: one value a
                            // wave (the shift count is its low five bits)
                            if let (
                                Some(tid),
                                "v_lshrrev_b32",
                                [
                                    Operand::Const(k @ 128..=192),
                                    Operand::Vector { reg, .. },
                                    ..,
                                ],
                            ) = (tid, name, &ops[..])
                                && (k - 128) & 31 >= 6
                                && inst.read(Reg::V(*reg)) == Some(tid)
                            {
                                same.extend(
                                    inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1),
                                );
                                continue;
                            }
                            let operands = ops.iter().all(|op| match *op {
                                Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                    reg.checked_add(k)
                                        .and_then(|r| inst.read(Reg::V(r)))
                                        .is_some_and(|x| same.contains(&x))
                                }),
                                Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                                    .all(|k| {
                                        code.checked_add(k)
                                            .and_then(|r| inst.read(Reg::S(r)))
                                            .is_some_and(|x| {
                                                same.contains(&x)
                                                    && (!is_mask(x) || bits.contains(&x))
                                            })
                                    }),
                                // (vccz, execz, scc, LDS direct; attributes)
                                Operand::Const(251..=254) | Operand::Attr { .. } => false,
                                _ => true,
                            });
                            // (every VGPR it reads besides: the old value of its
                            // destination, which a lane in EXEC keeps only where
                            // the result is 16 bits, or adds to as an accumulator)
                            let keeps = ["16", "dot", "mac", "fma_mix"]
                                .iter()
                                .any(|n| name.contains(n))
                                && !name.starts_with("v_pk_");
                            let implicit = inst.reads.iter().all(|r| {
                                let old = inst.writes.iter().any(|w| w.0 == r.0);
                                !matches!(r.0, Reg::V(_)) || old && !keeps || same.contains(&r.1)
                            });
                            if !operands || !implicit {
                                continue;
                            }
                            // A compare's bit is alike (`bits`); `v_cmpx`'s EXEC is
                            // no value of the lanes.
                            if name.starts_with("v_cmp") {
                                bits.extend(inst.writes.iter().map(|w| w.1));
                                if name.starts_with("v_cmpx") {
                                    continue;
                                }
                            }
                            same.extend(
                                inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1),
                            );
                        }
                        _ => {}
                    }
                }
            }
            if same.len() + bits.len() == known {
                break;
            }
        }
        let whole = undecided(p, |q| match st.exits[q] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => taken != not_taken && !alike(&same, &bits, when),
            LaneExit::Indirect => true,
            _ => succ(q).len() > 1,
        });
        let before = joins.len();
        joins.retain(|&(b, phi)| {
            p.blocks[b].preds.iter().all(|&q| whole[q]) && phi.args.iter().all(|a| same.contains(a))
        });
        if joins.len() == before {
            return same;
        }
    }
}

/// The blocks where the lanes of a console wave run in halves with
/// branches taken wave-wide may part on a scalar branch: a block all its
/// lanes run together ending in a branch on an SCC that is not
/// [`wave_same`].
pub(super) fn scalar_parting(p: &Program, st: &Structured, together: &[bool]) -> Vec<usize> {
    let same = wave_same(p, st, together);
    (0..p.blocks.len())
        .filter(|&b| together[b])
        .filter(|&b| match st.exits[b] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                taken != not_taken
                    && st
                        .bools
                        .support(when)
                        .iter()
                        .any(|a| matches!(a, Atom::Uniform(v) if !same.contains(v)))
            }
            _ => false,
        })
        .collect()
}

/// Whether block `b` only ends the program: nothing but the end in it, or
/// the far jump to such a block (the compiler's branch to an end out of a
/// branch offset's reach).
pub(super) fn ends(p: &Program, mut b: usize) -> bool {
    for _ in 0..4 {
        let block = &p.blocks[b];
        let idle = |i: &Inst| matches!(i.name(), "s_endpgm" | "s_nop" | "v_nop" | "s_waitcnt");
        match block.exit {
            Exit::End => return block.insts.iter().all(idle),
            Exit::Jump(to) => {
                let far = |i: &Inst| {
                    matches!(
                        i.name(),
                        "s_getpc_b64"
                            | "s_add_u32"
                            | "s_sub_u32"
                            | "s_addc_u32"
                            | "s_subb_u32"
                            | "s_setpc_b64"
                            | "s_branch"
                    )
                };
                if !block.insts.iter().all(|i| idle(i) || far(i)) {
                    return false;
                }
                b = to;
            }
            _ => return false,
        }
    }
    false
}

/// The blocks some cycle goes through.
pub(super) fn in_cycle(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    (0..n)
        .map(|b| {
            let mut seen = vec![false; n];
            let mut stack = p.blocks[b].exit.successors();
            while let Some(q) = stack.pop() {
                if !std::mem::replace(&mut seen[q], true) {
                    stack.extend(p.blocks[q].exit.successors());
                }
            }
            seen[b]
        })
        .collect()
}

/// The blocks that run, and as often, whichever way the branches ending the
/// blocks `decides` marks go: none of those decides it, directly or through
/// the branches around the block.
pub(super) fn undecided(p: &Program, decides: impl Fn(usize) -> bool) -> Vec<bool> {
    let n = p.blocks.len();
    let succ: Vec<Vec<usize>> = p.blocks.iter().map(|b| b.exit.successors()).collect();
    let pdom = postdominators(p);
    // `b` runs or not by the branch that ends `q`.
    let depends = |b: usize, q: usize| {
        succ[q].len() > 1 && succ[q].iter().any(|&s| pdom[s][b]) && (b == q || !pdom[q][b])
    };
    let decides: Vec<bool> = (0..n).map(decides).collect();
    let mut divergent: Vec<bool> = (0..n)
        .map(|b| (0..n).any(|q| decides[q] && depends(b, q)))
        .collect();
    loop {
        let mut changed = false;
        for b in 0..n {
            if !divergent[b] && (0..n).any(|q| divergent[q] && depends(b, q)) {
                divergent[b] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    divergent.iter().map(|d| !d).collect()
}

/// The `s_cbranch_execz` skips (byte offsets) the program computes the same
/// without: the wave takes one only when no lane is left, to save running
/// code that then does nothing, and per lane it is what keeps the lanes of
/// a wave apart. Without them every lane runs the code under EXEC, and the
/// halves of a wave can meet in it ([`Halves`]).
///
/// A skip stays where running its code with no lane could show: the code
/// has an end, a barrier or a message in it, or a loop a lane outside the
/// skip's EXEC could go round in (a count taken from lane 0's stale value
/// could run to the iteration limit), or leaves the skipped range other
/// than at the skip's target; it writes an SGPR or SCC that is read after
/// the target (other than a copy of what the register held, or a mask of
/// the same lanes); or it does not hand EXEC on within what it got.
///
/// `loops`: also the skips around such loops as no lane outside could go
/// round in. (Without them a loop's shape may be lost on what reads it: a
/// waterfall loop's first-lane reads are the lane's own only while the
/// lanes enter it by its skip.)
pub(crate) fn idle_skips(p: &Program, st: &Structured, loops: bool) -> Vec<usize> {
    let n = p.blocks.len();
    // Values some instruction reads, through the phis that pass them on.
    let mut live: Set<Value> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .flat_map(|i| i.reads.iter().map(|r| r.1))
        .collect();
    loop {
        let before = live.len();
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            if live.contains(&phi.value) {
                live.extend(phi.args.iter().copied());
            }
        }
        if live.len() == before {
            break;
        }
    }
    let alone = if p.wave64 {
        first_lane_tests(p, st)
    } else {
        Vec::new()
    };
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; n];
    for (k, &b) in order.iter().enumerate() {
        index[b] = k;
    }
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    let defs: Map<Value, &Inst> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .flat_map(|i| i.writes.iter().map(move |w| (w.1, i)))
        .collect();
    let joins: Map<Value, &Phi> = p
        .blocks
        .iter()
        .flat_map(|b| &b.phis)
        .map(|phi| (phi.value, phi))
        .collect();
    // What a scalar is a copy of: through `s_mov`, an SGPR parked in a VGPR
    // lane and read back, and a join of copies of one value.
    fn origin(
        st: &Structured,
        defs: &Map<Value, &Inst>,
        joins: &Map<Value, &Phi>,
        mut v: Value,
        depth: u32,
    ) -> Value {
        for _ in 0..64 {
            if let Some(&saved) = st.restored.get(&v) {
                v = saved;
                continue;
            }
            if let Some(phi) = joins.get(&v) {
                if depth == 0 {
                    break;
                }
                // (a loop's join of the value with itself is the value)
                let mut from = phi
                    .args
                    .iter()
                    .filter(|&&a| a != v)
                    .map(|&a| origin(st, defs, joins, a, depth - 1));
                match from.next() {
                    Some(first) if first != v && from.all(|a| a == first) => v = first,
                    _ => break,
                }
                continue;
            }
            let Some(inst) = defs.get(&v) else { break };
            let (
                "s_mov_b32" | "s_mov_b64",
                Fields::Sop {
                    dst: Some(Operand::Scalar { code: to, .. }),
                    src,
                    ..
                },
            ) = (inst.name(), &inst.fields)
            else {
                break;
            };
            let Some(&Operand::Scalar { code: from, .. }) = src.first() else {
                break;
            };
            let dword = (0..2).find(|&k| inst.write(Reg::S(to + k)) == Some(v));
            match dword.and_then(|k| inst.read(Reg::S(from + k))) {
                Some(a) => v = a,
                None => break,
            }
        }
        v
    }
    let origin = |v: Value| origin(st, &defs, &joins, v, 6);
    // Whether two scalars hold the same: copies of one value, or the same
    // dword of lane masks of the same lanes.
    let alike = |a: Value, b: Value| {
        let (a, b) = (origin(a), origin(b));
        let lanes = |v: Value| {
            let low = st.highs.get(&v).copied();
            st.masks.get(&low.unwrap_or(v)).map(|&m| (m, low.is_some()))
        };
        a == b || lanes(a).is_some_and(|m| Some(m) == lanes(b))
    };
    // Whether the skip that ends block `q` can go.
    let idle = |q: usize, branch: &Inst, taken: usize, not_taken: usize| {
        let block = &p.blocks[q];
        if taken == not_taken || p.blocks[taken].offset <= block.offset {
            return false;
        }
        let Some(&got) = branch.read(Reg::S(EXEC)).and_then(|v| st.masks.get(&v)) else {
            return false;
        };
        // The skip leaves one lane of a wave alone in the code.
        if alone
            .iter()
            .any(|&m| !st.bools.meet(&[(got, true), (m, false)]))
        {
            return false;
        }
        // No lane the skip's EXEC lacks is in the mask whose low dword is `v`.
        let within = |v: Value| {
            st.masks
                .get(&v)
                .is_some_and(|&m| !st.bools.meet(&[(m, true), (got, false)]))
        };
        // The skipped code: from the fall-through up to the target.
        let mut inside = vec![false; n];
        let mut stack = vec![not_taken];
        while let Some(b) = stack.pop() {
            if b == taken || std::mem::replace(&mut inside[b], true) {
                continue;
            }
            stack.extend(p.blocks[b].exit.successors());
        }
        // The lane masks joined in the skipped code that hold no lane the
        // skip's EXEC lacks, if the others do not (a loop's mask is what
        // entered it, less what has left): the most that can be taken so.
        let mut held: Vec<(&Phi, Bool)> = (0..n)
            .filter(|&b| inside[b])
            .flat_map(|b| &p.blocks[b].phis)
            .filter_map(|phi| Some((phi, *st.masks.get(&phi.value)?)))
            .collect();
        // No lane the skip's EXEC lacks, nor any of those masks hold.
        let outside = loop {
            let outside: Vec<(Bool, bool)> = std::iter::once((got, false))
                .chain(held.iter().map(|h| (h.1, false)))
                .collect();
            let before = held.len();
            held.retain(|(phi, _)| {
                phi.args.iter().all(|a| {
                    st.masks.get(a).is_some_and(|&m| {
                        let mut all = outside.clone();
                        all.push((m, true));
                        !st.bools.meet(&all)
                    })
                })
            });
            if held.len() == before {
                break outside;
            }
        };
        // A loop there goes round only in lanes the skip's EXEC holds: a
        // lane it lacks runs each block at most once, whatever the counts.
        // Such a lane, were it in the skipped code (so nothing the skip
        // tested counts), would not take the edge back: not with what the
        // edge's own branch tests, and what the branches tested that every
        // lane in the block has come through since the skip.
        let forward = |b: usize, s: usize| {
            if p.blocks[s].offset > p.blocks[b].offset {
                return true;
            }
            if !loops {
                return false;
            }
            let mut all = outside.clone();
            let side = |d: usize, to: usize| match st.exits[d] {
                LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } if taken != not_taken && (to == taken || to == not_taken) => {
                    Some((when, taken == to))
                }
                _ => None,
            };
            all.extend(side(b, s));
            let mut at = b;
            while at != q && idom[at] != at && idom[at] != usize::MAX && inside[idom[at]] {
                let d = idom[at];
                if p.blocks[at].preds == [d] {
                    all.extend(side(d, at));
                }
                at = d;
            }
            !st.bools.meet(&all)
        };
        let contained = (0..n).filter(|&b| inside[b]).all(|b| {
            let body = &p.blocks[b];
            b != q
                && b != 0
                && !matches!(body.exit, Exit::End | Exit::Indirect)
                && body.preds.iter().all(|&r| r == q || inside[r])
                && body.exit.successors().iter().all(|&s| forward(b, s))
                && !body.insts.iter().any(|i| {
                    matches!(i.name(), "s_barrier" | "s_sendmsg" | "s_sendmsghalt")
                        || i.name().starts_with("s_endpgm")
                })
        });
        if !contained || inside[q] {
            return false;
        }
        // Scalars written there and read past the target.
        let written = (0..n).filter(|&b| inside[b]).flat_map(|b| {
            let body = &p.blocks[b];
            let insts = body.insts.iter().flat_map(|i| i.writes.iter().copied());
            insts.chain(body.phis.iter().map(|phi| (phi.reg, phi.value)))
        });
        let scalars: Set<Value> = written
            .filter(|&(reg, _)| match reg {
                Reg::S(c) => c != EXEC && c != EXEC + 1,
                Reg::Scc => true,
                Reg::V(_) => false,
            })
            .map(|w| w.1)
            .collect();
        let leaks = (0..n).filter(|&b| !inside[b]).any(|b| {
            let body = &p.blocks[b];
            body.insts
                .iter()
                .any(|i| i.reads.iter().any(|r| scalars.contains(&r.1)))
                || body.phis.iter().any(|phi| {
                    live.contains(&phi.value)
                        && phi.args.iter().any(|a| scalars.contains(a))
                        && !phi.args.iter().all(|&a| alike(a, phi.args[0]))
                })
        });
        if leaks {
            return false;
        }
        // EXEC as the code hands it on holds no lane the skip's did not.
        p.blocks[taken]
            .phis
            .iter()
            .filter(|phi| phi.reg == Reg::S(EXEC))
            .flat_map(|phi| {
                phi.args
                    .iter()
                    .zip(&p.blocks[taken].preds)
                    .filter(|&(_, &r)| inside[r])
                    .map(|(&a, _)| a)
            })
            .all(within)
    };
    // (a block and its copies share the instruction: all must agree)
    let (mut skips, mut kept) = (Vec::new(), Set::default());
    for (q, block) in p.blocks.iter().enumerate() {
        let (Some(branch), Exit::Branch { taken, not_taken }) = (block.insts.last(), block.exit)
        else {
            continue;
        };
        if branch.name() != "s_cbranch_execz" {
            continue;
        }
        if idle(q, branch, taken, not_taken) {
            skips.push(branch.machine.offset);
        } else {
            kept.insert(branch.machine.offset);
        }
    }
    skips.retain(|at| !kept.contains(at));
    skips.sort_unstable();
    skips.dedup();
    skips
}

/// The uniform branches (by block: the block skipped and the target) that
/// skip one block of wave operations and of nothing that shows outside the
/// registers: no memory access, message, barrier or end. In a group of
/// several console waves one wave takes such a skip and another does not,
/// so the group cannot meet at a barrier in the block, and the halves of
/// the waves that run it cannot exchange.
///
/// The branch is sunk below the block: every lane at the branch runs the
/// block, and a lane whose wave skips then leaves by the skip's own edge
/// into the target, where each join takes for it what the branch's block
/// held. Such a lane has all its registers as with the skip taken before
/// the block (what the block wrote is read past it only through the
/// target's joins), and the block did nothing else. A lane whose wave does
/// not skip runs what it ran before, with the whole group at the block's
/// barriers. (The skips are whole waves': the condition is the wave's, the
/// same in both halves.)
pub(super) fn sunk_skips(x: &Parts) -> Map<usize, (usize, usize)> {
    let (p, st) = (x.p, x.st);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        rpo[b] = n;
    }
    let quiet = |i: &Inst| {
        let name = i.name();
        match &i.fields {
            Fields::Valu(_) => !name.contains("movrel"),
            Fields::Sop { .. } => ![
                "getpc", "setpc", "swappc", "rfe", "getreg", "setreg", "gpr_idx", "call",
            ]
            .iter()
            .any(|n| name.contains(n)),
            _ => matches!(name, "s_nop" | "s_branch") || name.starts_with("s_waitcnt"),
        }
    };
    let wave = |i: &Inst| {
        let name = i.name();
        name.contains("permlane")
            || name.contains("readlane")
            || name.contains("readfirstlane")
            || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            || reads_mask_as_data(st, i)
    };
    let mut sunk = Map::default();
    for &d in &order {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = st.exits[d]
        else {
            continue;
        };
        let atoms = st.bools.support(when);
        if atoms.is_empty() || !atoms.iter().all(|a| matches!(a, Atom::Uniform(_))) {
            continue;
        }
        for (body, to) in [(taken, not_taken), (not_taken, taken)] {
            let block = &p.blocks[body];
            if body != to
                && block.preds == [d]
                && block.phis.is_empty()
                && st.exits[body] == LaneExit::Jump(to)
                && rpo[d] < rpo[body]
                && rpo[body] < rpo[to]
                && block.insts.iter().all(quiet)
                && block.insts.iter().any(wave)
            {
                sunk.insert(d, (body, to));
            }
        }
    }
    sunk
}

/// Wave64: the masks that hold at most one lane of a console wave, the one
/// whose number is the first set bit of a mask (the console compiler's
/// once-per-wave block: `v_mbcnt` of all lanes, `s_ff1_i32_b64`,
/// `v_cmp[x]_eq` of the two).
///
/// The lanes of a host wave that make the compare together hold one
/// `s_ff1` result (a scalar instruction's, of a mask taken as a number
/// where they were all together: neither is in a loop the lanes could
/// have left in different passes), and each its own lane number, so at
/// most one of them passes; the two halves of a console wave hold the same
/// result where the mask is the console's ([`Lower::mask_dword`] exchanges
/// it or stops the form), so at most one lane of the console wave does,
/// the lane the console's compare passes.
pub(super) fn first_lane_tests(p: &Program, st: &Structured) -> Vec<Bool> {
    let cycle = in_cycle(p);
    let defs: Map<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    let def = |v: Value| {
        defs.get(&v)
            .map(|&(b, i)| (&p.blocks[b].insts[i], st.guards[b][i]))
    };
    let plain = |valu: &crate::gfx10::Valu| valu.sdwa.is_none() && valu.dpp.is_none();
    // `v`: the lane's number in its console wave, `v_mbcnt_hi(-1,
    // v_mbcnt_lo(-1, 0))`; with the conditions the two were written under.
    let lane_number = |v: Value| -> Option<[Option<Bool>; 2]> {
        let (hi, hi_guard) = def(v)?;
        let Fields::Valu(valu) = &hi.fields else {
            return None;
        };
        if hi.name() != "v_mbcnt_hi_u32_b32" || !plain(valu) {
            return None;
        }
        let (Some(&Operand::Const(193)), Some(&Operand::Vector { reg, .. })) =
            (valu.src.first(), valu.src.get(1))
        else {
            return None;
        };
        let (lo, lo_guard) = def(hi.read(Reg::V(reg))?)?;
        let Fields::Valu(valu) = &lo.fields else {
            return None;
        };
        let zero = valu
            .src
            .get(1)
            .is_some_and(|&op| scalar_constant(op) == Some(0));
        (lo.name() == "v_mbcnt_lo_u32_b32"
            && plain(valu)
            && matches!(valu.src.first(), Some(&Operand::Const(193)))
            && zero)
            .then_some([hi_guard, lo_guard])
    };
    let mut one = Vec::new();
    let insts = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| block.insts.iter().map(move |i| (b, i)));
    for (b, inst) in insts {
        let Fields::Valu(valu) = &inst.fields else {
            continue;
        };
        let name = inst.name();
        if !matches!(
            name,
            "v_cmp_eq_u32" | "v_cmp_eq_i32" | "v_cmpx_eq_u32" | "v_cmpx_eq_i32"
        ) || !plain(valu)
        {
            continue;
        }
        let (mut first, mut guards) = (false, None);
        for op in &valu.src {
            match *op {
                Operand::Vector { reg, .. } => {
                    guards = inst.read(Reg::V(reg)).and_then(lane_number);
                }
                Operand::Scalar { code, dwords: 1 } => {
                    let bit = inst.read(Reg::S(code));
                    first = bit.and_then(|v| defs.get(&v)).is_some_and(|&(at, i)| {
                        let d = &p.blocks[at].insts[i];
                        let Fields::Sop { src, .. } = &d.fields else {
                            return false;
                        };
                        let mask = match src.first() {
                            Some(&Operand::Scalar { code, .. }) => d.read(Reg::S(code)),
                            _ => None,
                        };
                        // The block that writes the mask, where its number
                        // is taken.
                        let written = |m: Value| match defs.get(&m) {
                            Some(&(b, _)) => Some(b),
                            None => (0..p.blocks.len())
                                .find(|&b| p.blocks[b].phis.iter().any(|phi| phi.value == m))
                                .or(p.inputs.iter().any(|i| i.1 == m).then_some(0)),
                        };
                        // (lanes that left a loop in different passes hold
                        // different numbers from it)
                        d.name() == "s_ff1_i32_b64"
                            && (!cycle[at] || at == b)
                            && mask.is_some_and(|m| {
                                st.masks.contains_key(&m)
                                    && written(m).is_some_and(|w| !cycle[w] || w == at)
                            })
                    });
                }
                _ => {}
            }
        }
        let (true, Some(guards)) = (first, guards) else {
            continue;
        };
        // (a lane the `v_mbcnt` left out holds another number)
        one.extend(inst.writes.iter().filter_map(|w| {
            let &m = st.masks.get(&w.1)?;
            guards
                .iter()
                .flatten()
                .all(|&g| !st.bools.meet(&[(m, true), (g, false)]))
                .then_some(m)
        }));
    }
    one
}

/// The blocks at most one lane of a console wave runs. Every lane that
/// enters has passed a test of its thread id against zero in each dimension
/// the group has more than one thread in (`v_cmp_eq_u32 0, v0 | v1`, the
/// console compiler's `all(GroupThreadID == 0)`): one thread of the group.
/// Or one of its lane number ([`first_lane_tests`]): one of each wave.
pub(super) fn solo_blocks(x: &Parts, threads: [u32; 3], id_vgprs: u8) -> Vec<bool> {
    let p = x.p;
    let need = (0..3)
        .filter(|&k| threads[k] > 1)
        .fold(0u8, |m, k| m | 1 << k);
    let defs: Map<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block
                .insts
                .iter()
                .enumerate()
                .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
        })
        .collect();
    // The thread id components `v` holds a zero of only where they are zero.
    fn pins(
        x: &Parts,
        defs: &Map<Value, (usize, usize)>,
        id_vgprs: u8,
        v: Value,
        depth: u32,
    ) -> u8 {
        if let Some(&(Reg::V(k), _)) = x.p.inputs.iter().find(|i| i.1 == v) {
            return if k < id_vgprs { 1 << k } else { 0 };
        }
        let Some(&(b, i)) = defs.get(&v) else {
            return 0;
        };
        let inst = &x.p.blocks[b].insts[i];
        let Fields::Valu(valu) = &inst.fields else {
            return 0;
        };
        // (a guarded write leaves the old value in the lanes it skips)
        if depth == 0 || valu.sdwa.is_some() || valu.dpp.is_some() || x.st.guards[b][i].is_some() {
            return 0;
        }
        // The flat id of two components, `a << k | b` made as a sum: zero
        // only where both are (thread ids are below 2^16, and so is `k`).
        if inst.name() == "v_lshl_add_u32" {
            let id = |op: Operand| match op {
                Operand::Vector { reg, .. } => inst
                    .read(Reg::V(reg))
                    .and_then(|a| x.p.inputs.iter().find(|i| i.1 == a))
                    .and_then(|i| match i.0 {
                        Reg::V(k) if k < id_vgprs => Some(1 << k),
                        _ => None,
                    }),
                _ => None,
            };
            return match valu.src[..] {
                [a, k, b] if scalar_constant(k).is_some_and(|k| k <= 16) => {
                    id(a).zip(id(b)).map_or(0, |(a, b)| a | b)
                }
                _ => 0,
            };
        }
        if inst.name() != "v_or_b32" {
            return 0;
        }
        valu.src.iter().fold(0, |m, op| match *op {
            Operand::Vector { reg, .. } => match inst.read(Reg::V(reg)) {
                Some(a) => m | pins(x, defs, id_vgprs, a, depth - 1),
                None => m,
            },
            _ => m,
        })
    }
    let mut one: Vec<Bool> = Vec::new();
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        let Fields::Valu(valu) = &inst.fields else {
            continue;
        };
        if inst.name() != "v_cmp_eq_u32" || valu.sdwa.is_some() || valu.dpp.is_some() {
            continue;
        }
        let (mut zero, mut pinned) = (false, 0);
        for op in &valu.src {
            match *op {
                Operand::Vector { reg, .. } => {
                    if let Some(a) = inst.read(Reg::V(reg)) {
                        pinned |= pins(x, &defs, id_vgprs, a, 4);
                    }
                }
                op => zero |= scalar_constant(op) == Some(0),
            }
        }
        if !zero || pinned & need != need {
            continue;
        }
        one.extend(inst.writes.iter().filter_map(|w| x.st.masks.get(&w.1)));
    }
    // Or a test of its lane number: the one lane of each console wave.
    if x.native.wave64 {
        one.extend(first_lane_tests(p, x.st));
    }
    (0..p.blocks.len())
        .map(|b| {
            one.iter()
                .any(|&m| !x.st.bools.meet(&[(x.st.reach[b], true), (m, false)]))
        })
        .collect()
}

/// A console wave64 compute program made for host waves of 32 lanes
/// ([`Options::narrow_waves`](super::Options)): each host wave holds half a
/// console wave (lanes 0-31 or 32-63), told by the thread ids it holds.
/// Wave operations within a half are the host's; what the console takes
/// over the whole wave is taken per half and passed between the halves
/// through group-shared memory at a group barrier, where all threads of
/// the group are (`full`; a mask and the first lane in EXEC also where
/// only those that have not ended are, `meet`). Elsewhere such an
/// operation is the console's only where one lane of a wave runs alone or
/// the value is the same in every lane; any other refuses the form
/// ([`Lower::refuse`]).
pub(super) struct Halves {
    /// The group is one host wave (32 threads or fewer).
    pub(super) whole: bool,
    /// Halves in a group.
    pub(super) count: u32,
    /// Where the halves meet with every lane there: [`wave_blocks`] of a
    /// group that is one console wave, [`Lower::group_blocks`] of a larger
    /// one.
    pub(super) full: Vec<bool>,
    /// Where the halves meet: `full`, and the [`Lower::group_blocks`] every
    /// thread that has not ended runs.
    pub(super) meet: Vec<bool>,
    /// The [`Lower::group_blocks`] outside every loop: an exchange there
    /// is not made again.
    pub(super) once: Vec<bool>,
    /// [`solo_blocks`].
    pub(super) solo: Vec<bool>,
    /// The relabelled flat thread id (made in the entry block).
    pub(super) flat: Option<V>,
    /// Exchanges made: each has its own shared dwords.
    pub(super) sites: u32,
}

/// Console wave64 programs kept at `WaveSize(64)` though narrow host waves
/// were asked for: some wave operation had no exact form in two halves.
pub(super) static KEPT_WIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Why the last of them was.
pub(super) static KEPT_WIDE_WHY: std::sync::Mutex<Option<Why>> = std::sync::Mutex::new(None);

thread_local! {
    /// Why the program last lowered on this thread kept the console's wave
    /// size though narrow host waves were asked for, if it did.
    pub(super) static WIDE: std::cell::Cell<Option<Why>> = const { std::cell::Cell::new(None) };
    /// The IR blocks where the halves form of the program last lowered on
    /// this thread found the halves of a wave apart ([`take_needs`]).
    pub(super) static NEEDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Takes [`NEEDS`]: the blocks where the halves of a console wave had to
/// meet (an exchange, an LDS read to sync) and could not, sorted.
pub(crate) fn take_needs() -> Vec<usize> {
    let mut n = NEEDS.take();
    n.sort_unstable();
    n.dedup();
    n
}

/// The per-lane branches (blocks closed by one, not yet in
/// [`Structured::wide`]) that decide whether or how often a lane runs one
/// of the blocks `needs`, directly or through the branches around them: the
/// branches to take wave-wide ([`crate::structure::structure_wide`]) so
/// that the halves of a console wave are together there. `ending`: also the
/// branches whose other way is the program's end (`if (...) return;`),
/// which the halves meet past without ([`wave_blocks`]), but not with every
/// lane.
pub(crate) fn branches_around(
    p: &Program,
    st: &Structured,
    needs: &[usize],
    ending: bool,
) -> Vec<usize> {
    let n = p.blocks.len();
    let succ: Vec<Vec<usize>> = p.blocks.iter().map(|b| b.exit.successors()).collect();
    let pdom = postdominators(p);
    let depends = |b: usize, q: usize| {
        succ[q].len() > 1 && succ[q].iter().any(|&s| pdom[s][b]) && (b == q || !pdom[q][b])
    };
    // Any branch on a lane mask: one that no lane takes as the per-lane form
    // has it (a skip around code inside a per-lane branch, `when` false in
    // every lane there) is taken by the wave once the wave goes on where
    // some lane does.
    let per_lane = |q: usize| match st.exits[q] {
        LaneExit::Branch {
            taken, not_taken, ..
        } => {
            taken != not_taken
                && !st.wide.contains(&q)
                && st.wave_any[q].is_some()
                && (ending || !succ[q].iter().any(|&s| ends(p, s)))
        }
        _ => false,
    };
    let mut seen = vec![false; n];
    let mut work: Vec<usize> = needs.iter().copied().filter(|&b| b < n).collect();
    let mut found = Vec::new();
    while let Some(b) = work.pop() {
        if std::mem::replace(&mut seen[b], true) {
            continue;
        }
        for q in 0..n {
            if depends(b, q) {
                if per_lane(q) && !found.contains(&q) {
                    found.push(q);
                }
                work.push(q);
            }
        }
    }
    found.sort_unstable();
    found
}

/// Per block, the blocks that postdominate it (itself among them).
pub(super) fn postdominators(p: &Program) -> Vec<Vec<bool>> {
    let n = p.blocks.len();
    let succ: Vec<Vec<usize>> = p.blocks.iter().map(|b| b.exit.successors()).collect();
    let mut pdom = vec![vec![true; n]; n];
    for b in 0..n {
        if succ[b].is_empty() {
            pdom[b] = (0..n).map(|q| q == b).collect();
        }
    }
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let new: Vec<bool> = (0..n)
                .map(|q| q == b || succ[b].iter().all(|&s| pdom[s][q]))
                .collect();
            changed |= new != pdom[b];
            pdom[b] = new;
        }
        if !changed {
            return pdom;
        }
    }
}

/// Takes [`WIDE`].
pub(crate) fn take_wide() -> Option<Why> {
    WIDE.take()
}

/// Counts a translation left at the console's wave size.
pub(crate) fn count_wide(why: Why) {
    KEPT_WIDE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    *KEPT_WIDE_WHY.lock().unwrap_or_else(|e| e.into_inner()) = Some(why);
}

/// How many translations [`Options::narrow_waves`](super::Options) left
/// at the console's wave size (a host of 32-lane waves does not run them),
/// and why the last one.
pub fn kept_wide() -> (u64, Option<&'static str>) {
    (
        KEPT_WIDE.load(std::sync::atomic::Ordering::Relaxed),
        *KEPT_WIDE_WHY.lock().unwrap_or_else(|e| e.into_inner()),
    )
}

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barrier between one lane's LDS write and another's read. Here a
/// read (or atomic) of LDS that a write may have changed since the last
/// barrier gets one before it. Barriers need every lane of the group, so
/// they go only in blocks each lane runs exactly once (no cycle through
/// them, on every path from the entry to an end); such a block still dirty
/// at its end syncs there when a successor is not one, so divergent code
/// after it reads clean LDS. `None`: a read in divergent code may see a
/// write made in divergent code since the last such block.
///
/// An atomic that gives nothing back needs no barrier after others of its
/// kind (sums, ORs, minima: any order leaves the same dword), so lanes may
/// accumulate in divergent code and loops; the next read syncs.
#[cfg(test)]
pub(crate) fn split_wave_barriers(p: &Program) -> Option<Set<(usize, usize)>> {
    split_wave_barriers_in(p, &once_blocks(p)).ok()
}

/// [`split_wave_barriers`] with barriers in the blocks `uniform` marks
/// (one console wave in a group: [`wave_blocks`]); `Err` with the blocks
/// whose reads would need one where none can be placed.
pub(super) fn split_wave_barriers_in(
    p: &Program,
    uniform: &[bool],
) -> Result<Set<(usize, usize)>, Vec<usize>> {
    // LDS since the last sync: untouched, changed only by atomics of one
    // commuting kind (1..), or written.
    const CLEAN: u8 = 0;
    const DIRTY: u8 = u8::MAX;
    const COMMUTING: [&str; 8] = [
        "ds_add_u32",
        "ds_or_b32",
        "ds_and_b32",
        "ds_xor_b32",
        "ds_min_u32",
        "ds_max_u32",
        "ds_min_i32",
        "ds_max_i32",
    ];
    let join = |a: u8, b: u8| match (a, b) {
        (CLEAN, x) | (x, CLEAN) => x,
        (a, b) if a == b => a,
        _ => DIRTY,
    };
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut dirty_out = vec![CLEAN; n];
    let mut apart = Vec::new();
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut dirty = p.blocks[b]
                .preds
                .iter()
                .fold(CLEAN, |d, &q| join(d, dirty_out[q]));
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                let name = inst.name();
                if name == "s_barrier" {
                    dirty = CLEAN;
                }
                if !matches!(inst.fields, Fields::Ds { .. })
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute")
                {
                    continue;
                }
                let kind = COMMUTING
                    .iter()
                    .position(|&k| k == name)
                    .map(|k| k as u8 + 1);
                let commutes = kind.is_some_and(|k| dirty == k);
                if dirty != CLEAN && !name.starts_with("ds_write") && !commutes {
                    if uniform[b] {
                        at.insert((b, i));
                    } else if !apart.contains(&b) {
                        apart.push(b);
                    }
                }
                dirty = match kind {
                    Some(k) => k,
                    None if name.starts_with("ds_read") => CLEAN,
                    None => DIRTY,
                };
            }
            if dirty != CLEAN && uniform[b] && succ(b).iter().any(|&s| !uniform[s]) {
                at.insert((b, p.blocks[b].insts.len()));
                dirty = CLEAN;
            }
            changed |= std::mem::replace(&mut dirty_out[b], dirty) != dirty;
        }
        if !changed {
            return if apart.is_empty() { Ok(at) } else { Err(apart) };
        }
    }
}
