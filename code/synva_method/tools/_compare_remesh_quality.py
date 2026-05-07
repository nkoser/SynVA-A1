#!/usr/bin/env python
"""Compare bridge quality between two stitched test runs (NEW vs REMESH).

For each case x tag, compute mesh quality stats from the *_attached.obj:
- max edge length, p99 edge length
- max aspect ratio, p99 aspect ratio
- # boundary edges (should be 0 for watertight)

Then aggregate per tag and overall.
"""
import os, sys, json, argparse
import numpy as np
import trimesh
from collections import Counter

TAGS = ["A", "C", "D", "E", "baseline", "gt"]

def quality(mesh_path):
    m = trimesh.load(mesh_path, process=False)
    v = np.asarray(m.vertices); f = np.asarray(m.faces, dtype=np.int64)
    e0 = np.linalg.norm(v[f[:,1]]-v[f[:,0]],axis=1)
    e1 = np.linalg.norm(v[f[:,2]]-v[f[:,1]],axis=1)
    e2 = np.linalg.norm(v[f[:,0]]-v[f[:,2]],axis=1)
    e = np.stack([e0,e1,e2], axis=1)
    asp = e.max(1) / (e.min(1) + 1e-12)
    edges = []
    for tri in f:
        a,b,c = sorted(int(x) for x in tri)
        edges += [(a,b),(b,c),(a,c)]
    cnt = Counter(edges)
    boundary = sum(1 for k,vv in cnt.items() if vv==1)
    return dict(
        max_edge=float(e.max()),
        p99_edge=float(np.quantile(e,.99)),
        max_aspect=float(asp.max()),
        p99_aspect=float(np.quantile(asp,.99)),
        boundary_edges=int(boundary),
        n_v=int(len(v)),
        n_f=int(len(f)),
    )

def collect(root, tags):
    rows = {}
    for case in sorted(os.listdir(root)):
        cdir = os.path.join(root, case)
        if not os.path.isdir(cdir) or case.startswith("_"):
            continue
        for t in tags:
            p = os.path.join(cdir, f"{t}_attached.obj")
            if not os.path.isfile(p):
                continue
            try:
                rows[(case,t)] = quality(p)
            except Exception as ex:
                print(f"  fail {case}/{t}: {ex}")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="NEW (greedy+rot-align, no remesh) run dir")
    ap.add_argument("--new", required=True, help="REMESH run dir")
    ap.add_argument("--tags", nargs="+", default=TAGS)
    args = ap.parse_args()

    print(f"[old] {args.old}")
    a = collect(args.old, args.tags)
    print(f"[new] {args.new}")
    b = collect(args.new, args.tags)
    common = sorted(set(a.keys()) & set(b.keys()))
    print(f"\nCommon (case,tag) pairs: {len(common)}")

    # per-tag aggregation
    print(f"\n{'tag':<10s} {'n':>4s}  {'max_edge old→new':>22s}  {'max_aspect old→new':>22s}  {'p99_edge old→new':>22s}")
    by_tag = {t: {"old":[], "new":[]} for t in args.tags}
    for k in common:
        c,t = k
        by_tag[t]["old"].append(a[k])
        by_tag[t]["new"].append(b[k])
    for t in args.tags:
        oo = by_tag[t]["old"]; nn = by_tag[t]["new"]
        if not oo: continue
        me_o = np.mean([x["max_edge"] for x in oo]); me_n = np.mean([x["max_edge"] for x in nn])
        ma_o = np.mean([x["max_aspect"] for x in oo]); ma_n = np.mean([x["max_aspect"] for x in nn])
        pe_o = np.mean([x["p99_edge"] for x in oo]); pe_n = np.mean([x["p99_edge"] for x in nn])
        print(f"{t:<10s} {len(oo):>4d}  {me_o:>9.4f} → {me_n:>9.4f}  {ma_o:>9.1f} → {ma_n:>9.1f}  {pe_o:>9.4f} → {pe_n:>9.4f}")

    # overall
    all_o_me = [a[k]["max_edge"] for k in common]
    all_n_me = [b[k]["max_edge"] for k in common]
    all_o_ma = [a[k]["max_aspect"] for k in common]
    all_n_ma = [b[k]["max_aspect"] for k in common]
    print(f"\nOverall mean max_edge:    old={np.mean(all_o_me):.4f}  new={np.mean(all_n_me):.4f}  Δ={np.mean(all_n_me)-np.mean(all_o_me):+.4f}")
    print(f"Overall mean max_aspect:  old={np.mean(all_o_ma):.1f}    new={np.mean(all_n_ma):.1f}    Δ={np.mean(all_n_ma)-np.mean(all_o_ma):+.1f}")
    # also paired wins
    wins = sum(1 for k in common if b[k]["max_edge"] < a[k]["max_edge"])
    print(f"Per-pair wins (new better max_edge): {wins}/{len(common)}")
    wins_a = sum(1 for k in common if b[k]["max_aspect"] < a[k]["max_aspect"])
    print(f"Per-pair wins (new better max_aspect): {wins_a}/{len(common)}")

if __name__ == "__main__":
    main()
