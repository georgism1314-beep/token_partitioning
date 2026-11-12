"""读 cache_acts.py 的原始 FFN 激活和 cache_attn_topk.py 的同层 attention top-k，看"注意力源 token"能解释多少 FFN 方差，
并与当前 token 比较。不需要前向。

第 l 层 FFN 的输入 h_l = x_l + RMSNorm(Attn_l(x_l))，所以用同一层 l 的 attention（头平均权重）。
数据里没有 BOS，模型会把序列开头附近的某个 token（第一个 "."、" the"、"," 等）当 attention sink：
同一条序列里大量 query 的 top1 都指向它。直接用 top1 的 token 等于给序列打标签，所以另外定义去掉 sink 的版本：
    sink    某条序列某层里，被 >= sink_frac 比例的 query 当作 top1 的位置
    attn1   top1 位置上的 token-id（原始定义，含自己和 sink）
    attnc   top-k 里第一个既不是自己、也不是 sink 的位置上的 token-id（内容源）；top-k 里找不到则单独成"无"组
token-id 统一用 tok 的高频词表：两折都出现过的最高频 n_groups 个 token 各一组，其余为 OTHER。

分组：
    tok      当前 token
    attn1    原始 top1 源 token（+ OTHER）
    attnc    内容源 token（+ OTHER + 无）
    tokattn  (当前 token, 内容源 token) 格子：两折都出现过的最高频 n_pairs 个格子各一组，其余退回 tok 组（tok 的细分）
每个神经元（cf = cross-fit η²）：
    Δ(attnc | tok) = η²(tokattn) − η²(tok)
    Δ(tok | attnc) = η²(tokattn) − η²(attnc)（退回组混合了源 token，只是近似）
    attnc 先于 tok 被贪心选中的神经元比例 = mean(η²(attnc) > η²(tok))

    python attn_decomp.py --names step10000 --layers 4 31
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--acts_root", default=str(root / "acts"))
parser.add_argument("--attn_root", default=str(root / "attn_topk"))
parser.add_argument("--names", nargs="+", default=[f"step{s}" for s in range(0, 10001, 1000)] + ["Olmo-3-1025-7B"])
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--n_pairs", type=int, default=20000)
parser.add_argument("--sink_frac", type=float, default=0.05)
parser.add_argument("--chunk_tokens", type=int, default=16384)
parser.add_argument("--out_dir", default=str(root / "runs_decomp_attn"))
args = parser.parse_args()
device = "cuda"
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- 数据：与 cache_acts.py 相同的冻结子集前缀 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
fold = np.repeat(frozen["doc_fold"][:n_seqs], seq_lens)  # 每个 token 所在文档的折：0 = A，1 = B
seq_of = np.repeat(np.arange(n_seqs), seq_lens)
q = np.arange(T) - starts[seq_of]  # 序列内位置
max_len = int(seq_lens.max())
V = int(token_ids.max()) + 1

# ---- tok：两折都出现过的最高频 n_groups 个 token 各一组，其余为 OTHER ----
counts = np.bincount(token_ids, minlength=V)
counts_A = np.bincount(token_ids[fold == 0], minlength=V)
in_both_folds = (counts_A > 0) & (counts - counts_A > 0)
top = np.argsort(-np.where(in_both_folds, counts, 0))[: args.n_groups]
top = top[in_both_folds[top]]
K = len(top)
group_of_token = np.full(V, K)
group_of_token[top] = np.arange(K)
tok = group_of_token[token_ids]
G_tok = K + 1
NONE = K + 1  # attnc 在 top-k 里找不到合格位置
print(f"T={T:,}  tok 组={G_tok}（OTHER 占比 {np.mean(tok == K):.3f}）", flush=True)

in_A = torch.from_numpy(fold == 0).to(device)
tok_t = torch.from_numpy(tok).to(device)

for name in args.names:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    cache, attn_dir = Path(args.acts_root) / name, Path(args.attn_root) / name
    while not ((cache / "done").exists() and (attn_dir / "done").exists()):
        time.sleep(60)
    assert np.array_equal(np.load(cache / "meta.npz")["token_ids"], token_ids), f"{cache} 的 token 顺序不一致"
    assert np.array_equal(np.load(attn_dir / "meta.npz")["token_ids"], token_ids), f"{attn_dir} 的 token 顺序不一致"
    t0 = time.time()
    result = {}

    for l in args.layers:
        layer_path = out_dir / f"{name}_L{l}.npz"  # 每层单独落盘，进程被杀后重跑只补缺的层
        if layer_path.exists():
            result.update(dict(np.load(layer_path)))
            print(f"[{name}] L{l:2d} 已有结果，跳过", flush=True)
            continue

        # ---- 本层的注意力标签 ----
        src_pos = np.load(attn_dir / f"L{l}_pos.npy")  # [T, k]，序列内位置，-1 = 无
        src_tok = np.load(attn_dir / f"L{l}_tok.npy")
        top1 = src_pos[:, 0]
        # sink：被同一序列里 >= sink_frac 的 query 选为 top1 的位置
        hits = np.bincount(seq_of * max_len + top1, minlength=n_seqs * max_len)
        is_sink = hits >= args.sink_frac * np.repeat(seq_lens, max_len)
        key = seq_of[:, None] * max_len + np.maximum(src_pos, 0)
        ok = (src_pos >= 0) & (src_pos != q[:, None]) & ~is_sink[key]
        has = ok.any(1)
        first = ok.argmax(1)
        ctx_pos = np.where(has, src_pos[np.arange(T), first], -1)
        ctx_tok = np.where(has, src_tok[np.arange(T), first], -1)
        attn1 = group_of_token[src_tok[:, 0]]
        attnc = np.where(has, group_of_token[np.maximum(ctx_tok, 0)], NONE)
        del src_pos, src_tok, key, ok

        # ---- tokattn：(当前 token, 内容源 token) 格子，退回 tok 组 ----
        eligible = (tok < K) & (attnc < K)
        pairs, pair_inv, pair_counts = np.unique(tok[eligible].astype(np.int64) * (K + 2) + attnc[eligible],
                                                 return_inverse=True, return_counts=True)
        pair_counts_A = np.bincount(pair_inv[fold[eligible] == 0], minlength=len(pairs))
        pair_in_both = (pair_counts_A > 0) & (pair_counts - pair_counts_A > 0)
        top_pairs = np.argsort(-np.where(pair_in_both, pair_counts, 0))[: args.n_pairs]
        top_pairs = top_pairs[pair_in_both[top_pairs]]
        P = len(top_pairs)
        group_of_pair = np.full(len(pairs), -1)
        group_of_pair[top_pairs] = np.arange(P)
        tokattn = P + tok
        pair_group = group_of_pair[pair_inv]
        tokattn[np.where(eligible)[0][pair_group >= 0]] = pair_group[pair_group >= 0]
        cell_tok = np.concatenate([pairs[top_pairs] // (K + 2), np.arange(G_tok)])
        assert (cell_tok[tokattn] == tok).all(), "tokattn 必须是 tok 的细分"

        m = q >= 1
        result[f"L{l}_top1_self"] = np.mean(top1[m] == q[m])
        result[f"L{l}_top1_prev"] = np.mean(top1[m] == q[m] - 1)
        result[f"L{l}_top1_sink"] = np.mean(is_sink[seq_of * max_len + top1][m])
        result[f"L{l}_top1_same_tok"] = np.mean(token_ids[starts[seq_of] + top1][m] == token_ids[m])
        result[f"L{l}_ctx_found"] = np.mean(has[m])
        result[f"L{l}_ctx_prev"] = np.mean((ctx_pos == q - 1)[m])
        result[f"L{l}_ctx_same_tok"] = np.mean((ctx_tok == token_ids)[m & has])
        result[f"L{l}_ctx_dist_median"] = float(np.median((q - ctx_pos)[m & has]))
        result[f"L{l}_tokattn_cover"] = np.mean(tokattn < P)

        labels = {"tokattn": torch.from_numpy(tokattn).to(device),
                  "attn1": torch.from_numpy(attn1).to(device),
                  "attnc": torch.from_numpy(attnc).to(device)}
        sizes = {"tokattn": P + G_tok, "attn1": G_tok, "attnc": G_tok + 1}
        factors = {"tok": ("tokattn", torch.from_numpy(cell_tok).to(device)),
                   "tokattn": ("tokattn", None), "attn1": ("attn1", None), "attnc": ("attnc", None)}
        n_of = {p: torch.bincount(g, minlength=sizes[p]).double() for p, g in labels.items()}
        nA_of = {p: torch.bincount(g[in_A], minlength=sizes[p]).double() for p, g in labels.items()}

        # ---- 充分统计量（float64）：Σx、Σx²，以及每种分组下按组的 Σx（全体 S、A 折 SA） ----
        acts = np.load(cache / f"L{l}.npy", mmap_mode="r")
        N = acts.shape[1]
        s1 = torch.zeros(N, dtype=torch.float64, device=device)
        s2 = torch.zeros(N, dtype=torch.float64, device=device)
        S_of = {p: torch.zeros(sizes[p], N, dtype=torch.float64, device=device) for p in labels}
        SA_of = {p: torch.zeros(sizes[p], N, dtype=torch.float64, device=device) for p in labels}
        for lo in range(0, T, args.chunk_tokens):
            hi = min(lo + args.chunk_tokens, T)
            h = torch.from_numpy(np.ascontiguousarray(acts[lo:hi])).to(device).view(torch.bfloat16).double()
            a = in_A[lo:hi]
            s1 += h.sum(0)
            s2 += (h * h).sum(0)
            for p, g in labels.items():
                S_of[p].index_add_(0, g[lo:hi], h)
                SA_of[p].index_add_(0, g[lo:hi][a], h[a])
        del acts, h

        mu = s1 / T
        var = s2 / T - mu**2
        cf = {}
        for f, (p, parent) in factors.items():
            S, SA, n, nA = S_of[p], SA_of[p], n_of[p], nA_of[p]
            if parent is not None:
                G = int(parent.max().item()) + 1
                S = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, S)
                SA = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, SA)
                n = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, n)
                nA = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, nA)
            between_cf = torch.zeros(N, dtype=torch.float64, device=device)
            for r in range(0, len(n), 4096):
                Sr, SAr, nr, nAr = S[r : r + 4096], SA[r : r + 4096], n[r : r + 4096], nA[r : r + 4096]
                nBr = nr - nAr
                both = (nAr > 0) & (nBr > 0)
                muA = SAr[both] / nAr[both, None]
                muB = (Sr - SAr)[both] / nBr[both, None]
                between_cf += (nr[both, None] / T * (muA - mu) * (muB - mu)).sum(0)
            cf[f] = (between_cf / var).cpu().numpy()
            result[f"L{l}_{f}_cf"] = cf[f].astype(np.float32)
            result[f"L{l}_{f}_cf_weighted"] = (between_cf.sum() / var.sum()).item()
            del S, SA, Sr, SAr, muA, muB
        del S_of, SA_of
        torch.cuda.empty_cache()

        result[f"L{l}_attnc_given_tok"] = (cf["tokattn"] - cf["tok"]).astype(np.float32)
        result[f"L{l}_tok_given_attnc"] = (cf["tokattn"] - cf["attnc"]).astype(np.float32)
        result[f"L{l}_attnc_first"] = cf["attnc"] > cf["tok"]
        result[f"L{l}_attn1_first"] = cf["attn1"] > cf["tok"]
        print(f"[{name}] L{l:2d}  top1: 自己 {result[f'L{l}_top1_self']:.3f} 前一个 {result[f'L{l}_top1_prev']:.3f}"
              f" sink {result[f'L{l}_top1_sink']:.3f} 同tok {result[f'L{l}_top1_same_tok']:.3f}"
              f" | 内容源: 前一个 {result[f'L{l}_ctx_prev']:.3f} 同tok {result[f'L{l}_ctx_same_tok']:.3f}"
              f" 距离中位 {result[f'L{l}_ctx_dist_median']:.0f}"
              f" | η² 中位 tok {np.nanmedian(cf['tok']):.4f} attn1 {np.nanmedian(cf['attn1']):.4f}"
              f" attnc {np.nanmedian(cf['attnc']):.4f} tokattn {np.nanmedian(cf['tokattn']):.4f}"
              f"  +attnc|tok {np.nanmedian(result[f'L{l}_attnc_given_tok']):.4f}"
              f"  attnc 先选 {result[f'L{l}_attnc_first'].mean():.3f} attn1 先选 {result[f'L{l}_attn1_first'].mean():.3f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        np.savez(layer_path, **{k: v for k, v in result.items() if k.startswith(f"L{l}_")})

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), G_tok=G_tok, sink_frac=args.sink_frac, **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
