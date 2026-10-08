"""Average-linkage merge hierarchy over the superpoint graph (Sec. 4.4)."""
import heapq
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hierarchy:
    n_leaves: int
    merges: list                       # (a, b, new_id, linkage) in merge order
    members: dict = field(default_factory=dict)   # node id -> np.array of superpoint ids
    children: dict = field(default_factory=dict)  # node id -> (a, b)
    linkage: dict = field(default_factory=dict)   # node id -> linkage at creation (leaves: inf)

    @property
    def n_nodes(self):
        return self.n_leaves + len(self.merges)

    @property
    def root(self):
        return self.n_nodes - 1

    def parents(self):
        par = {}
        for a, b, new, _ in self.merges:
            par[a] = new
            par[b] = new
        return par

    def flat(self, n_clusters=None, theta=None, sizes=None, min_frac=0.0):
        """Replay merges until n_clusters remain, or until linkage drops below theta.

        sizes/min_frac: clusters holding less than min_frac of the points do not count
        toward n_clusters and are absorbed afterwards into the largest significant
        cluster under their nearest ancestor (so stray fragments cannot use up K).
        Returns superpoint -> cluster labels (compact) and the list of node ids.
        """
        w = np.ones(self.n_leaves) if sizes is None else np.asarray(sizes, float)
        total = w.sum()
        big = lambda nid: w[self.members[nid]].sum() >= min_frac * total
        # number of significant clusters after each merge; stop at the last merge that
        # keeps at least n_clusters of them
        n_big = [sum(big(i) for i in range(self.n_leaves))]
        for a, b, new, _ in self.merges:
            n_big.append(n_big[-1] + big(new) - big(a) - big(b))
        stop = len(self.merges)
        if theta is not None:
            for k, (_, _, _, link) in enumerate(self.merges):
                if link < theta:
                    stop = k
                    break
        if n_clusters is not None:
            ok = [i for i in range(stop + 1) if n_big[i] >= n_clusters]
            stop = ok[-1] if ok else stop
        alive = set(range(self.n_leaves))
        for a, b, new, _ in self.merges[:stop]:
            alive -= {a, b}
            alive.add(new)
        if min_frac > 0:
            alive = self._absorb_small(alive, big)
        nodes = sorted(alive)
        lab = np.empty(self.n_leaves, dtype=int)
        for k, nid in enumerate(nodes):
            lab[self.members[nid]] = k
        return lab, nodes

    def _absorb_small(self, alive, big):
        par = self.parents()
        bigs = [x for x in alive if big(x)]
        if not bigs:
            return alive
        owner = {}
        for x in bigs:
            for m in self.members[x]:
                owner[m] = x
        keep = set(bigs)
        merged = {x: [x] for x in bigs}
        for x in alive:
            if x in keep:
                continue
            node, target = x, None
            while node in par and target is None:
                node = par[node]
                cand = {owner[m] for m in self.members[node] if m in owner}
                if cand:
                    target = max(cand, key=lambda c: len(self.members[c]))
            if target is None:
                target = max(bigs, key=lambda c: len(self.members[c]))
            merged[target].append(x)
        # materialize merged groups as pseudo-nodes in members
        out = set()
        for tgt, group in merged.items():
            if len(group) == 1:
                out.add(tgt)
            else:
                nid = -(len(self.members) + 1)
                self.members[nid] = np.concatenate([self.members[g] for g in group])
                out.add(nid)
        return out


def agglomerate(n, pairs, weights, veto=None, stop_below=None):
    """Greedy average-linkage merging on a sparse graph.

    n        : number of leaves (superpoints)
    pairs    : (E, 2) leaf pairs with an edge
    weights  : (E,) affinity per edge in [0, 1] (higher = more similar)
    veto     : optional f(members_a, members_b) -> True to forbid that merge
    stop_below: stop once the best available linkage is below this value

    Linkage between clusters = mean edge affinity over the edges joining them.
    Components with no remaining edges are joined at linkage 0 so the result is
    always a single tree (unless stopped or vetoed).
    """
    nbrs = [dict() for _ in range(n)]
    for (i, j), w in zip(pairs, weights):
        if i == j:
            continue
        s = nbrs[i].get(j, [0.0, 0])
        s[0] += w
        s[1] += 1
        nbrs[i][j] = s
        nbrs[j][i] = s
    members = {i: np.array([i]) for i in range(n)}
    linkage = {i: np.inf for i in range(n)}
    children = {}
    heap = [(-s[0] / s[1], i, j) for i in range(n) for j, s in nbrs[i].items() if i < j]
    heapq.heapify(heap)
    alive = set(range(n))
    merges = []
    nxt = n
    vetoed = set()

    def do_merge(a, b, link):
        nonlocal nxt
        new = nxt
        nxt += 1
        merged = {}
        for src in (a, b):
            for k, s in nbrs[src].items():
                if k in (a, b):
                    continue
                t = merged.setdefault(k, [0.0, 0])
                t[0] += s[0]
                t[1] += s[1]
        nbrs.append(merged)
        for k, s in merged.items():
            nbrs[k].pop(a, None)
            nbrs[k].pop(b, None)
            nbrs[k][new] = s
            heapq.heappush(heap, (-s[0] / s[1], min(k, new), max(k, new)))
        nbrs[a] = {}
        nbrs[b] = {}
        alive.difference_update((a, b))
        alive.add(new)
        members[new] = np.concatenate([members[a], members[b]])
        children[new] = (a, b)
        linkage[new] = link
        merges.append((a, b, new, link))

    while heap:
        negl, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive or b not in nbrs[a]:
            continue
        s = nbrs[a][b]
        if abs(-negl - s[0] / s[1]) > 1e-12:
            continue  # stale entry
        if stop_below is not None and -negl < stop_below:
            break
        if veto is not None and (a, b) not in vetoed and veto(members[a], members[b]):
            vetoed.add((a, b))
            continue
        do_merge(a, b, -negl)

    if stop_below is None and veto is None:
        # join disconnected components at linkage 0
        rest = sorted(alive)
        while len(rest) > 1:
            a, b = rest[0], rest[1]
            do_merge(a, b, 0.0)
            rest = sorted(alive)
    return Hierarchy(n_leaves=n, merges=merges, members=members, children=children, linkage=linkage)
