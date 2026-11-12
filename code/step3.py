#!/usr/bin/env python
"""第 3 步：由充分统计量构造分解矩阵。

读 step2.py 落盘的 stats_*.npz（第 2 步的 S^(f)、n^(f)、C），对每个因素算：

    3.1  组均值      abar[:, g] = S[:, g] / n[g]
    3.2  可解释分量  C_par(f)   = S diag(1/n) S^T
    3.3  残差分量    C_perp(f)  = C - C_par(f)
    3.4  零模型      C_null     = T p p^T,  p = diag(C)/T   （与因素无关）

不含第 4 步的 Louvain 打分，所以这里报的 trace 份额只是"对角线解释比例"，不是 ΔQ(f)。

    python code/step3.py runs_trial/Olmo-3-1025-7B/stats_code_L0.npz
    python code/step3.py runs_trial/Olmo-3-1025-7B/stats_*_L0.npz --save_matrices
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def c_parallel(S: np.ndarray, n: np.ndarray) -> np.ndarray:
    """C_par = S diag(1/n) S^T，写成 B B^T 以保证数值上半正定。

    B[:, g] = S[:, g] / sqrt(n_g) 就是 sqrt(n_g) * abar[:, g]，于是
    C_par = sum_g n_g abar_g abar_g^T —— 计划 3.2 的两种写法完全等价。
    """
    B = S / np.sqrt(np.maximum(n, 1))[None, :]
    return B @ B.T


def c_parallel_cf(SA, nA, SB, nB, n):
    """交叉拟合版 C_par：两折独立的组均值做外积，消掉"噪声自乘"带来的正偏差。

        C_cf = sum_g n_g * abar_g^A (abar_g^B)^T,   abar_g^X = S^X[:, g] / n^X_g

    朴素版用同一份 abar 平方，E[abar^2] = q^2 + q(1-q)/n_g，多出来的方差项累加成
    约 G_f * q(1-q) 的系统性正偏差——组数越多越严重。两折独立时 E[abar^A abar^B] = q^2，
    偏差在期望上为零（文档第 6 步对角线版本的推广）。

    代价：外积不再对称，这里取 (M + M^T)/2。两折均值都非负，所以结果仍然逐元素非负，
    Louvain 能用；但它不再是投影，半正定性不保证。
    返回 ``(C_cf, dropped)``，dropped 是某一折为空、只能整组丢弃的组数。
    """
    ok = (nA > 0) & (nB > 0)
    w = np.zeros_like(n, dtype=np.float64)
    w[ok] = n[ok] / (nA[ok] * nB[ok])
    M = (SA * w[None, :]) @ SB.T
    return 0.5 * (M + M.T), int((~ok).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="第 2 步的 stats_*.npz")
    ap.add_argument("--probes", type=int, default=8, help="半正定性随机探针数（C_perp 的 v^T C_perp v）")
    ap.add_argument("--save_matrices", action="store_true", help="额外存 C_par / C_perp（每个 N*N float32）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for path in args.stats:
        path = Path(path)
        z = np.load(path)
        T = int(z["T"])
        C = z["C"].astype(np.float64)
        p = np.diag(C) / T
        tr_C = float(np.trace(C))
        rng = np.random.default_rng(args.seed)
        V = rng.standard_normal((C.shape[0], args.probes))
        V /= np.linalg.norm(V, axis=0, keepdims=True)

        # 3.4 零模型：列独立打散后 S_null[:, g] = n_g * p，代回 3.2 得 T p p^T，与 f 无关。
        C_null = T * np.outer(p, p)
        report = {
            "file": path.name,
            "T": T,
            "N": int(C.shape[0]),
            "k": int(z["k"]),
            "trace_C": tr_C,
            "trace_C_null": float(np.trace(C_null)),
            "factors": {},
        }
        print(f"\n=== {path.name}  T={T}  N={C.shape[0]}  k={int(z['k'])}  tr(C)={tr_C:.4g}")

        par = {}
        for key in [f for f in z.files if f.startswith("S_")]:
            name = key[2:]
            S, n = z[key].astype(np.float64), z[f"n_{name}"].astype(np.float64)
            abar = S / np.maximum(n, 1)[None, :]  # 3.1 组均值
            C_par = c_parallel(S, n)  # 3.2
            C_perp = C - C_par  # 3.3
            par[name] = C_par

            # C_perp = A^T (I - P) A 在数学上半正定；这里用随机探针和对角线做数值体检。
            quad = np.einsum("ik,ij,jk->k", V, C_perp, V)
            entry = {
                "G": int(n.size),
                "n_min": int(n.min()),
                "n_max": int(n.max()),
                "trace_C_par": float(np.trace(C_par)),
                "trace_C_perp": float(np.trace(C_perp)),
                "diag_share": float(np.trace(C_par) / tr_C),
                "abar_max": float(abar.max()),
                "min_diag_C_perp": float(np.diag(C_perp).min()),
                "min_probe_C_perp": float(quad.min()),
                # 矩阵层面的 Frobenius 内积：不为 0，正是"因素之间不是两两正交"的直接证据。
                "frob_inner_par_perp": float(np.vdot(C_par, C_perp)),
            }
            report["factors"][name] = entry
            print(
                f"  {name:10s} G={entry['G']:5d} n_min={entry['n_min']:6d} "
                f"tr(C_par)={entry['trace_C_par']:12.4g} 对角份额={entry['diag_share']:.4f} "
                f"min diag(C_perp)={entry['min_diag_C_perp']:.3g} min probe={entry['min_probe_C_perp']:.3g}"
            )

        # 恒等式：常数因素的 C_par 就是零模型 T p p^T（因为 p = mu0），故 ΔQ(常数) 必然为 0。
        if "const" in par:
            report["max_abs_C_par_const_minus_C_null"] = float(np.abs(par["const"] - C_null).max())
            print(f"  C_par(常数) vs 零模型 T p p^T: 最大逐元素偏差 = {report['max_abs_C_par_const_minus_C_null']:.3g}")

        # 嵌套（refinement）序：token-id / 位置桶 都细分常数组，故 C_par 之差应半正定。
        for name in ("token_id", "pos_bucket"):
            if name in par and "const" in par:
                D = par[name] - par["const"]
                quad = np.einsum("ik,ij,jk->k", V, D, V)
                report["factors"][name]["refines_const_min_probe"] = float(quad.min())
                print(f"  C_par({name}) - C_par(常数): min probe = {quad.min():.3g}  (细分 => 应 >= 0)")

        # 各因素独立算出来的份额没有任何理由加到 1：每个 C_par(f) 都含了常数分量（见上面的嵌套探针），
        # 彼此重叠，和值可以 <1 也可以 >1。要得到和为 1 的分解必须走第 5 步的序贯联合因素。
        naive = sum(report["factors"][f]["trace_C_par"] for f in report["factors"])
        report["naive_sum_diag_share"] = float(naive / tr_C)
        print(f"  各因素对角份额朴素相加 = {naive / tr_C:.4f}  (无约束，不是 1)")

        out = path.with_name(path.stem + "_step3.json")
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if args.save_matrices:
            np.savez_compressed(
                path.with_name(path.stem + "_step3.npz"),
                C_null=C_null.astype(np.float32),
                **{f"C_par_{k}": v.astype(np.float32) for k, v in par.items()},
            )
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
