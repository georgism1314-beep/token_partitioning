#!/usr/bin/env python
"""第 4 步：打分子程序 SCORE_kappa。

读 step2.py 的 stats_*.npz，对第 3 步的矩阵族（零模型、C、各因素的 C_par）打分：

    4.2  归一化管线 Pi_kappa：去自环 -> 缩放 -> log1p 压长尾 -> 阈值保留 kappa 比例的边
    4.3  Q = Louvain 模块度
    4.4  地板 Q_floor = Q(Pi_kappa(T p p^T))
    4.5  SCORE_kappa(W) = Q(Pi_kappa(W)) - Q_floor
    4.6  Q_full = SCORE_kappa(C),  Delta_Q(f) = SCORE_kappa(C_par(f))

与文档的偏离，都是为了让不同因素 / 检查点之间的分数可比：

* 边权二值化（默认）。阈值之后把留下的边统一置 1，只保留"是否显著共激活"。
  这让管线第 2、3 步在数学上成为空操作：缩放是乘正常数、log1p 单调递增，都不改变
  元素的大小排序，因此按分位数选出的边集完全相同，置 1 后权重也相同。
  --weighted 保留 log1p 权重走文档原样，可作对照。
* tau_kappa 取分位数而非绝对值。C 与各 C_par(f) 量级差一个数量级以上，固定绝对阈值
  会让图密度天差地别，Q 就不可比。这里统一保留 kappa 比例的点对。

另外提供两个可选修正（默认关闭，保持与文档一致）：

* --floor sampled：地板用"列独立打散"的一次真实抽样 C_ij ~ Binomial(T, p_i p_j)，
  而不是它的期望 T p p^T。期望矩阵严格秩一、完全无噪，阈值化后是一张完美嵌套图，
  Louvain 会把它读成很强的核心-边缘结构，地板因此偏高。
* --edge_score zscore：按 Poisson 偏差 (W - E)/sqrt(E) 而不是原始计数排序选边，
  E = T p p^T。原始计数被激活率异质性主导（实测 C 偏离 E 仅约 39%），选出来的边
  大体只反映 p_i p_j；换成 z 分数后留下的是真正的超额共激活。此模式下零模型的
  z 分数恒为 0，故地板自动改用抽样零模型。

    python code/step4.py runs_trial/Olmo-3-1025-7B/stats_code_L0.npz
    python code/step4.py runs_trial/.../stats_*_L0.npz --edge_score zscore --floor sampled --seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

from step3 import c_parallel, c_parallel_cf


def sampled_null(T: int, p: np.ndarray, rng, chunk: int = 1024) -> np.ndarray:
    """列独立打散的一次抽样：C_ij ~ Binomial(T, p_i p_j)，期望即 T p p^T。"""
    N = p.size
    S = np.zeros((N, N))
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        S[a:b] = rng.binomial(T, np.clip(np.outer(p[a:b], p), 0.0, 1.0))
    S = np.triu(S, 1)  # 只留上三角再对称化，保证 S == S^T
    return S + S.T


def pi_kappa(W: np.ndarray, kappa: float, binarize: bool) -> tuple[sp.csr_matrix, dict]:
    """归一化建图管线。W 会被就地修改，调用方负责传副本。"""
    N = W.shape[0]
    np.fill_diagonal(W, 0.0)  # 4.2.1 去自环

    if not binarize:
        W *= N * (N - 1) / max(W.sum(), 1e-30)  # 4.2.2 缩放（已去自环，W.sum() 即非对角和）
        np.log1p(W, out=W)  # 4.2.3 压长尾

    # 4.2.4 阈值：保留 kappa 比例的点对。全矩阵里每个点对出现两次，故取 top-2m 个元素。
    m_pairs = max(1, int(round(kappa * N * (N - 1) / 2)))
    tau = float(np.partition(W.ravel(), -2 * m_pairs)[-2 * m_pairs])
    mask = W >= tau
    rows, cols = np.nonzero(mask)
    data = np.ones(rows.size) if binarize else W[rows, cols].astype(np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # 计数矩阵在 tau 上会大量并列，实际保留的边可能多于目标；把膨胀倍数报出来。
    kept = rows.size // 2
    deg = np.asarray(A.sum(1)).ravel()
    return A, {
        "tau": tau,
        "pairs_target": m_pairs,
        "pairs_kept": int(kept),
        "tie_inflation": round(kept / m_pairs, 3),
        "mean_degree": round(2 * kept / N, 1),
        "isolated_nodes": int((deg == 0).sum()),
    }


def louvain_q(A: sp.csr_matrix, seeds: int, resolution: float) -> dict:
    """多 seed 跑 Louvain，报中位数 + IQR（文档 5.11.5 的误差棒要求）。"""
    G = nx.from_scipy_sparse_array(A)
    qs, sizes = [], []
    for seed in range(seeds):
        part = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        qs.append(nx.community.modularity(G, part, weight="weight", resolution=resolution))
        sizes.append(sum(1 for c in part if len(c) > 1))  # 只数非单点社群
    q = np.array(qs)
    return {
        "Q": float(np.median(q)),
        "Q_iqr": float(np.percentile(q, 75) - np.percentile(q, 25)),
        "Q_min": float(q.min()),
        "Q_max": float(q.max()),
        "communities": float(np.median(sizes)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="第 2 步的 stats_*.npz")
    ap.add_argument("--kappa", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02],
                    help="保留边比例；文档要求四个档位方向一致才算过闸门")
    ap.add_argument("--seeds", type=int, default=5, help="Louvain 种子数；文档建议 >= 10")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--weighted", action="store_true", help="保留 log1p 边权走文档原样（默认二值化）")
    ap.add_argument("--floor", default="expected", choices=["expected", "sampled"],
                    help="地板用零模型的期望 T p p^T，还是一次真实抽样")
    ap.add_argument("--edge_score", default="raw", choices=["raw", "zscore"],
                    help="选边依据：原始计数，或 Poisson 偏差 (W - E)/sqrt(E)")
    ap.add_argument("--cross_fit", action="store_true", default=True,
                    help="同时给交叉拟合版 C_par^cf 打分作对照（需要 step2 存了 SA_/SB_）")
    ap.add_argument("--no_cross_fit", dest="cross_fit", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    binarize = not args.weighted
    rng = np.random.default_rng(args.seed)

    for path in args.stats:
        path = Path(path)
        z = np.load(path)
        T, N = int(z["T"]), int(z["N"])
        C = z["C"].astype(np.float64)
        p = np.diag(C) / T
        E = T * np.outer(p, p)  # 独立零模型的期望共现
        print(f"\n=== {path.name}  T={T}  N={N}  k={int(z['k'])}  "
              f"边权={'二值' if binarize else 'log1p'}  选边={args.edge_score}  地板={args.floor}")

        # 信噪诊断：C 偏离独立期望的幅度，和同 T 下纯抽样噪声的幅度相比。
        # 比值接近 1 说明这个 T 下根本没有可分辨的协同结构，SCORE 只是在比两个噪声。
        need_sample = args.floor == "sampled" or args.edge_score == "zscore"
        S = sampled_null(T, p, rng) if need_sample else None
        S_diag = S if S is not None else sampled_null(T, p, rng)
        C_off, E_off = C - np.diag(np.diag(C)), E - np.diag(np.diag(E))
        sig = float(np.linalg.norm(C_off - E_off))
        noi = float(np.linalg.norm(S_diag - E_off))
        diag = {"deviation_from_null": sig, "sampling_noise": noi, "signal_to_noise": sig / noi,
                "relative_deviation": sig / float(np.linalg.norm(E_off))}
        print(f"  诊断: ||C-E||={sig:.3g}  纯噪声={noi:.3g}  信噪比={sig / noi:.2f}  "
              f"相对偏离={diag['relative_deviation'] * 100:.1f}%"
              + ("   <- 信噪比 < 3，结论不可信" if sig / noi < 3 else ""))

        # 待打分的矩阵族：零模型（地板）、C（全局）、各因素的 C_par。
        null_W = S if args.floor == "sampled" else E
        named = [("null", null_W), ("C", C)]
        for key in [f for f in z.files if f.startswith("S_")]:
            name = key[2:]
            n = z[f"n_{name}"].astype(np.float64)
            named.append((name, c_parallel(z[key].astype(np.float64), n)))
            if args.cross_fit and f"SA_{name}" in z.files:
                cf, dropped = c_parallel_cf(
                    z[f"SA_{name}"].astype(np.float64), z[f"nA_{name}"].astype(np.float64),
                    z[f"SB_{name}"].astype(np.float64), z[f"nB_{name}"].astype(np.float64), n,
                )
                named.append((f"{name}_cf", cf))
                if dropped:
                    print(f"  [{name}_cf] {dropped}/{n.size} 组在某一折为空，已丢弃", flush=True)

        if args.edge_score == "zscore":
            # E 自己的 z 分数恒为 0，无法建图，故地板必须改用抽样零模型。
            sd = np.sqrt(np.maximum(E, 1.0))
            named = [(n, (W - E) / sd) for n, W in named]
            named[0] = ("null", (S - E) / sd)

        report = {"file": path.name, "T": T, "N": N, "k": int(z["k"]), "binarize": binarize,
                  "edge_score": args.edge_score, "floor": args.floor, "seeds": args.seeds,
                  "diagnostics": diag, "kappa": {}}

        for kappa in args.kappa:
            rows = {}
            for name, W in named:
                t0 = time.time()
                # z 分数模式下常数因素的残差恒为 0（C_par(const) == E），没有可建图的结构。
                if args.edge_score == "zscore" and np.abs(W).max() < 1e-3:
                    rows[name] = {"degenerate": True, "Q": None, "Q_iqr": None}
                    continue
                A, info = pi_kappa(W.copy(), kappa, binarize)
                rows[name] = {**info, **louvain_q(A, args.seeds, args.resolution), "degenerate": False}
                rows[name]["seconds"] = round(time.time() - t0, 1)

            q_floor = rows["null"]["Q"]  # 4.4
            for r in rows.values():  # 4.5
                r["SCORE"] = 0.0 if r["degenerate"] else r["Q"] - q_floor
            report["kappa"][str(kappa)] = {"Q_floor": q_floor, "matrices": rows}

            print(f"  kappa={kappa}  Q_floor={q_floor:.4f}  边数={rows['C']['pairs_kept']}  "
                  f"平均度={rows['C']['mean_degree']}  并列膨胀={rows['C']['tie_inflation']}x  "
                  f"孤立点={rows['C']['isolated_nodes']}")
            for name, r in rows.items():
                tag = {"C": "Q_full", "null": "地板自身"}.get(name, f"ΔQ({name})")
                if r["degenerate"]:
                    print(f"    {name:12s} 残差恒为 0，无结构可建图   SCORE=+0.0000  {tag}")
                else:
                    print(f"    {name:12s} Q={r['Q']:.4f}±{r['Q_iqr']:.4f}  SCORE={r['SCORE']:+.4f}  "
                          f"社群={r['communities']:.0f}  {tag}")
            if rows["C"]["SCORE"] <= 0:
                print(f"    [警告] Q_full={rows['C']['SCORE']:+.4f} <= 0，第 5 步的份额 "
                      f"rho_q = ΔQ_q / Q_full 会整体翻号，此档不可用")

        # 文件名带上选边模式，免得 raw 和 zscore 的结果互相覆盖。
        out = path.with_name(f"{path.stem}_step4_{args.edge_score}.json")
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
