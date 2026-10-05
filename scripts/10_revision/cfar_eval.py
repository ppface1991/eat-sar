#!/usr/bin/env python3
"""Step H — clutter-invariant (CFAR-like) operation of a deep detector.

Question: at the SAME overall false-alarm budget, does conditioning the
score gate on clutter make the false-alarm rate (and precision) uniform
across TRUE clutter strata, where a global threshold and a random-binning
placebo cannot?

Protocol (fit on calibration split, evaluate on test split):
  * clutter strata = tertiles of e on the calibration split (as in Table I)
  * targets are PRE-SPECIFIED, never chosen on test:
      FA/img  : {0.5x, 1x, 2x} the calibration-split FA/img of the
                production default T=0.40  (same budget, redistributed)
      precision: {0.85, 0.90, 0.95}
  * policies per target
      G040     global T=0.40 (reference, not target-matched)
      GLOBAL   one global T fitted on calibration to meet the target overall
      COND     one T per true clutter stratum fitted to meet the target
      PLACEBO  one T per RANDOM stratum (same sizes), n_perm permutations
    Fitting rule: the lowest T on the scan grid that meets the target
    (i.e. maximum recall subject to the FA/precision constraint).
  * metrics on test, per TRUE stratum: FA/img, P, R, F1;
    disparity = max-min across strata and mean |dev| from target;
    bootstrap CI (images resampled within strata) of
    disparity(GLOBAL) - disparity(COND); placebo z-score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from revkit import (BIN_LABELS, assign_bins, filter_gt, load_all,
                    read_manifest, tertile_edges)

SCAN = np.round(np.arange(0.05, 0.9001, 0.005), 3)


# --------------------------------------------------------------------------- #
# fast per-image counting
# --------------------------------------------------------------------------- #
class Pool:
    """Per-image sorted scores + cumulative TP, for O(1) counting at any T."""

    def __init__(self, records: dict, stems):
        self.stems = np.asarray(stems)
        self.neg = []   # -scores ascending  (== scores descending)
        self.ctp = []   # cumulative tp, with leading 0
        self.ngt = np.zeros(len(self.stems), dtype=np.int64)
        for i, st in enumerate(self.stems):
            r = records[st]
            s = np.asarray(r["scores"], dtype=np.float64)
            t = np.asarray(r["tp"], dtype=np.int64)
            o = np.argsort(-s, kind="stable")
            s, t = s[o], t[o]
            self.neg.append(-s)
            self.ctp.append(np.concatenate([[0], np.cumsum(t)]))
            self.ngt[i] = r["n_gt"]

    def counts(self, T_vec):
        """Per-image (n_kept, n_tp) for a per-image threshold vector."""
        n = len(self.stems)
        kept = np.empty(n, dtype=np.int64)
        tp = np.empty(n, dtype=np.int64)
        for i in range(n):
            k = int(np.searchsorted(self.neg[i], -T_vec[i], side="right"))
            kept[i] = k
            tp[i] = self.ctp[i][k]
        return kept, tp

    def curve(self, idx):
        """For image subset idx: arrays over SCAN of (n_kept, n_tp), plus n_gt, n_img."""
        if len(idx) == 0:
            z = np.zeros(len(SCAN), dtype=np.int64)
            return z, z, 0, 0
        s = np.concatenate([-self.neg[i] for i in idx]) if len(idx) else np.array([])
        t = np.concatenate([np.diff(self.ctp[i]) for i in idx]) if len(idx) else np.array([])
        o = np.argsort(-s, kind="stable")
        s, t = s[o], t[o]
        ctp = np.concatenate([[0], np.cumsum(t)])
        k = np.searchsorted(-s, -SCAN, side="right")
        return k, ctp[k], int(self.ngt[idx].sum()), len(idx)


def metrics(kept, tp, ngt, nimg):
    fp = kept - tp
    P = tp / np.maximum(kept, 1)
    R = tp / max(ngt, 1)
    F1 = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    return dict(FAPI=fp / max(nimg, 1), P=P, R=R, F1=F1)


def fit_T(pool, idx, kind, target):
    """Lowest T in SCAN meeting the target on subset idx; flags floor/ceiling."""
    k, tp, ngt, nimg = pool.curve(idx)
    m = metrics(k, tp, ngt, nimg)
    ok = (m["FAPI"] <= target) if kind == "fa" else (m["P"] >= target)
    if ok.any():
        j = int(np.argmax(ok))
        return float(SCAN[j]), ("floor" if j == 0 else "ok")
    # infeasible: best effort (lowest FA/img or highest precision) with >=1 kept box
    has = k > 0
    score = np.where(has, -m["FAPI"] if kind == "fa" else m["P"], -np.inf)
    return float(SCAN[int(np.argmax(score))]), "infeasible"


def stratum_eval(pool, T_vec, groups):
    kept, tp = pool.counts(T_vec)
    out = {}
    for name, idx in groups.items():
        K, TP, G, N = kept[idx].sum(), tp[idx].sum(), pool.ngt[idx].sum(), len(idx)
        FP = K - TP
        P = TP / max(K, 1); R = TP / max(G, 1)
        out[name] = dict(n_img=int(N), FAPI=float(FP / max(N, 1)), P=float(P), R=float(R),
                         F1=float(2 * P * R / (P + R)) if P + R > 0 else 0.0)
    return out


def disparity_from_values(v, key, target):
    v = np.asarray(v, dtype=float)
    if key == "FAPI":   # worst = heaviest false-alarm load relative to budget
        return dict(spread=float(v.max() - v.min()),
                    worst=float(v.max() / max(target, 1e-12)),
                    mean_rel_dev=float(np.mean(np.abs(v - target)) / max(target, 1e-12)))
    # precision: worst = largest shortfall below target (pp, 0 if all met)
    return dict(spread=float(100 * (v.max() - v.min())),
                worst=float(100 * max(target - v.min(), 0.0)),
                mean_rel_dev=float(100 * np.mean(np.abs(v - target))))


def disparity(per_bin, key, target):
    return disparity_from_values([per_bin[b][key] for b in BIN_LABELS], key, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-labels", required=True)
    ap.add_argument("--target-class", type=int, default=None)
    ap.add_argument("--e-col", default="e_pred")
    ap.add_argument("--baseline-T", type=float, default=0.40)
    ap.add_argument("--precision-targets", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    ap.add_argument("--fa-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-gt", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    val = read_manifest(args.val_manifest).dropna(subset=[args.e_col])
    test = read_manifest(args.test_manifest).dropna(subset=[args.e_col])
    rv = load_all(val, Path(args.val_labels), args.target_class, desc="load val")
    rt = load_all(test, Path(args.test_labels), args.target_class, desc="load test")
    sv, ev = filter_gt(rv, val["stem"].values, val[args.e_col].values, args.require_gt)
    st, et = filter_gt(rt, test["stem"].values, test[args.e_col].values, args.require_gt)

    edges = tertile_edges(ev)
    bv = assign_bins(ev, edges); bt = assign_bins(et, edges)
    print(f"[cfar] edges {edges[1]:.3f}/{edges[2]:.3f}  val {len(sv)}  test {len(st)}")

    PV, PT = Pool(rv, sv), Pool(rt, st)
    gv = {b: np.where(bv == b)[0] for b in BIN_LABELS}
    gt = {"ALL": np.arange(len(st)), **{b: np.where(bt == b)[0] for b in BIN_LABELS}}

    # production-default FA budget on the CALIBRATION split
    k, tp, ngt, n = PV.curve(np.arange(len(sv)))
    j40 = int(np.argmin(np.abs(SCAN - args.baseline_T)))
    fa40 = float((k[j40] - tp[j40]) / n)
    targets = ([("fa", round(m * fa40, 5), f"FA/img={m:g}x T0.40-budget") for m in args.fa_multipliers]
               + [("prec", p, f"P>={p:g}") for p in args.precision_targets])
    print(f"[cfar] calibration FA/img at T={args.baseline_T}: {fa40:.4f}")

    res = dict(edges=edges[1:3], n_val=int(len(sv)), n_test=int(len(st)),
               fa_per_img_T040_val=fa40, scan_step=0.005, targets=[])
    T040 = np.full(len(st), args.baseline_T)
    base = stratum_eval(PT, T040, gt)
    res["G040"] = base

    for kind, tgt, label in targets:
        key = "FAPI" if kind == "fa" else "P"
        entry = dict(kind=kind, target=tgt, label=label)

        Tg, fg = fit_T(PV, np.arange(len(sv)), kind, tgt)
        g = stratum_eval(PT, np.full(len(st), Tg), gt)

        Tb, flags = {}, {}
        for b in BIN_LABELS:
            Tb[b], flags[b] = fit_T(PV, gv[b], kind, tgt)
        Tvec_c = np.array([Tb[b] for b in bt])
        c = stratum_eval(PT, Tvec_c, gt)

        # placebo: random strata of the same sizes on calibration; the T fitted
        # for each random stratum is applied to a random test stratum of the
        # same size; disparity is ALWAYS measured on the TRUE clutter strata.
        sizes_v = [len(gv[b]) for b in BIN_LABELS]
        sizes_t = [len(gt[b]) for b in BIN_LABELS]
        plc = []
        for _ in range(args.n_perm):
            pv = rng.permutation(len(sv)); cut = np.cumsum(sizes_v)[:-1]
            Tr = [fit_T(PV, part, kind, tgt)[0] for part in np.split(pv, cut)]
            pt = rng.permutation(len(st)); cutt = np.cumsum(sizes_t)[:-1]
            Tvec = np.empty(len(st))
            for part, T in zip(np.split(pt, cutt), Tr):
                Tvec[part] = T
            r = stratum_eval(PT, Tvec, gt)
            plc.append(disparity(r, key, tgt))
        dg, dc, d0 = disparity(g, key, tgt), disparity(c, key, tgt), disparity(base, key, tgt)
        MET = ("spread", "worst")
        plc_stats = {}
        for mk in MET:
            arr = np.array([d[mk] for d in plc])
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            plc_stats[mk] = dict(mean=float(arr.mean()), std=sd,
                                 z_COND=float((dc[mk] - arr.mean()) / sd) if sd > 0 else float("nan"))

        # bootstrap: resample test images within each true stratum
        boots = {mk: [] for mk in MET}
        kg, tg_ = PT.counts(np.full(len(st), Tg)); kc, tc = PT.counts(Tvec_c)
        for _ in range(args.n_boot):
            vals = {"g": [], "c": []}
            for b in BIN_LABELS:
                idx = gt[b]; smp = rng.choice(idx, size=len(idx), replace=True)
                for nm, (K, TP) in (("g", (kg, tg_)), ("c", (kc, tc))):
                    Ks, TPs = K[smp].sum(), TP[smp].sum()
                    vals[nm].append((Ks - TPs) / len(smp) if key == "FAPI" else TPs / max(Ks, 1))
            Dg = disparity_from_values(vals["g"], key, tgt)
            Dc = disparity_from_values(vals["c"], key, tgt)
            for mk in MET:
                boots[mk].append(Dg[mk] - Dc[mk])
        boot_out = {}
        for mk in MET:
            a = np.array(boots[mk])
            boot_out[mk] = dict(mean=float(a.mean()),
                                ci95=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                                frac_positive=float((a > 0).mean()))

        entry.update(
            GLOBAL=dict(T=Tg, flag=fg, per_bin=g, disparity=dg),
            COND=dict(T=Tb, flags=flags, per_bin=c, disparity=dc),
            G040_disparity=d0,
            PLACEBO=dict(n_perm=args.n_perm, **plc_stats),
            boot_GLOBAL_minus_COND=boot_out,
        )
        res["targets"].append(entry)

        unit = "(FA/img)" if key == "FAPI" else "(pp)"
        print(f"\n== {label}  (target {tgt:g}) ==")
        print(f"  GLOBAL T={Tg:.3f}[{fg}]  COND T=({Tb['low']:.3f},{Tb['medium']:.3f},{Tb['high']:.3f}) "
              f"flags={flags}")
        for nm, r in (("G040", base), ("GLOBAL", g), ("COND", c)):
            row = "  ".join(f"{b[:3]}:{r[b][key]:.4f}" for b in BIN_LABELS)
            print(f"  {nm:7s} {key} per stratum  {row}   | ALL P={r['ALL']['P']:.4f} "
                  f"R={r['ALL']['R']:.4f} F1={r['ALL']['F1']:.4f} FA/img={r['ALL']['FAPI']:.4f}")
        for mk, desc in (("spread", f"spread{unit}"),
                         ("worst", "worst x-budget" if key == "FAPI" else "worst shortfall(pp)")):
            ps = plc_stats[mk]; bo = boot_out[mk]
            print(f"  {desc:22s} G040 {d0[mk]:.4f}  GLOBAL {dg[mk]:.4f}  COND {dc[mk]:.4f}  "
                  f"PLACEBO {ps['mean']:.4f}+-{ps['std']:.4f} (z_COND={ps['z_COND']:.2f})  "
                  f"| boot GLOBAL-COND {bo['mean']:.4f} CI95 [{bo['ci95'][0]:.4f},{bo['ci95'][1]:.4f}] "
                  f"P>0={bo['frac_positive']:.2f}")

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
