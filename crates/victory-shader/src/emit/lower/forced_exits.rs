//! Loop limits kept at an existing exit branch ([`super::super::limit_exits`]).
//! After SlaktarnVI / avatarmanz e823263 (`find_forced_exits`).

use std::collections::HashSet;

use super::Lower;
use crate::structure::{Atom, LaneExit};

impl Lower<'_> {
    /// The loops jumped back to without a test (their single latch ends in
    /// a jump to the header) whose limit can be kept at an exit branch
    /// instead of a return: the latest branch on the dominator path from the
    /// latch up to the header that leaves the loop one way and stays in it
    /// the other. Every iteration passes it, so a lane past the limit leaves
    /// there, at most one iteration later, through an exit the loop has.
    ///
    /// Looked for only when the switch is on or a translation is asked what
    /// it meets; applied (`forced_exits`, `counted_only`) only with the
    /// switch on, so with it off the translation is unchanged.
    pub(super) fn find_forced_exits(&mut self, order: &[usize]) {
        use super::super::limit_exits as le;
        if self.counters.is_empty() || !le::looking() {
            return;
        }
        let p = self.x.p;
        let st = self.x.st;
        let idom = crate::ir::idoms(&p.blocks, order, &self.rpo);
        let reached = |s: &Self, q: usize| s.block[q].is_some() && s.rpo[q] != usize::MAX;
        let back = |s: &Self, from: usize, to: usize| {
            s.counters.contains_key(&to) && s.rpo[to] <= s.rpo[from]
        };
        let mut headers: Vec<usize> = self.counters.keys().copied().collect();
        headers.sort_by_key(|&h| self.rpo[h]);
        let mut found = Vec::new();
        let mut taken_blocks = HashSet::new();
        for h in headers {
            let latches: Vec<usize> = p.blocks[h]
                .preds
                .iter()
                .copied()
                .filter(|&q| reached(self, q) && self.rpo[h] <= self.rpo[q])
                .collect();
            let [latch] = latches[..] else { continue };
            if !matches!(st.exits[latch], LaneExit::Jump(t) if t == h) || self.sunk.contains_key(&latch) {
                continue;
            }
            // The natural loop of the back edge.
            let mut body = HashSet::from([h]);
            let mut stack = vec![latch];
            while let Some(x) = stack.pop() {
                if reached(self, x) && body.insert(x) {
                    stack.extend(p.blocks[x].preds.iter().copied());
                }
            }
            let mut d = latch;
            loop {
                if let LaneExit::Branch {
                    when,
                    taken,
                    not_taken,
                } = st.exits[d]
                    && body.contains(&taken) != body.contains(&not_taken)
                    && !back(self, d, taken)
                    && !back(self, d, not_taken)
                    // Not a wave-wide decision, nor a branch sunk below the
                    // code it skips: both are taken by whole waves.
                    && !matches!(st.bools.node(when), Some((Atom::Wave(_), _, _)))
                    && !self.sunk.contains_key(&d)
                    && !taken_blocks.contains(&d)
                {
                    let out = if body.contains(&taken) { not_taken } else { taken };
                    taken_blocks.insert(d);
                    found.push((d, h, out));
                    break;
                }
                if d == h || idom[d] == usize::MAX || idom[d] == d {
                    break;
                }
                d = idom[d];
            }
        }
        if found.is_empty() {
            return;
        }
        le::met();
        if le::limit_exits() {
            for (d, h, out) in found {
                self.forced_exits.insert(d, (h, out));
                self.counted_only.insert(h);
            }
        }
    }

    /// At the exit branch of block `b` that keeps loop `header`'s limit
    /// (`out` the successor that leaves): `cond` (taken when it holds), also
    /// taken past the limit if `out` is `taken`, or also not taken past it
    /// otherwise.
    pub(super) fn forced_exit_cond(
        &mut self,
        cond: super::V,
        header: usize,
        out: usize,
        taken: usize,
    ) -> super::V {
        use super::{Bin, pred};
        let (count, _) = self.counters[&header];
        let limit = self.iteration_limit();
        let limit = self.s.m.i32(limit);
        if out == taken {
            let past = self.s.m.cmp(self.cur, pred::IUGE, count, limit);
            self.s.m.bin(self.cur, Bin::Or, cond, past)
        } else {
            let within = self.s.m.cmp(self.cur, pred::IULT, count, limit);
            self.s.m.bin(self.cur, Bin::And, cond, within)
        }
    }

    /// A jump back to `header` whose limit is kept at an exit: the count
    /// goes up, and the jump is taken whatever it is.
    pub(super) fn count_up(&mut self, header: usize) {
        let c = self.cur;
        let (phi, _) = self.counters[&header];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, super::Bin::Add, phi, one);
        self.counters
            .get_mut(&header)
            .expect("header")
            .1
            .push((next, c));
    }
}
