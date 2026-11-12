"""实验 B：对 32 个 attention head 分别定义"内容源 token"因素，看是否存在 neuron-specific 的 head。
读 cache_acts.py 的原始 FFN 激活和 cache_attn_heads.py 的逐头 top-k，不需要前向。

与 attn_decomp.py（实验 A，头平均）完全相同的定义，只是换成单个 head h 的权重：
    sink_h      某条序列里，被 head h 的 >= sink_frac 比例的 query 当作 top1 的位置
    attnc_h     head h 的 top-k 里第一个既不是自己、也不是 sink_h 的位置上的 token-id；找不到则单独成"无"组
    tokattn_h   (当前 token, attnc_h) 格子：两折都出现过的最高频 n_pairs 个格子各一组，其余退回 tok 组（tok 的细分）
token-id 用 tok 的高频词表（两折都出现过的最高频 n_groups 个 token + OTHER），与实验 A 一致。
每个神经元、每个 head（cf = cross-fit η²）：
    attnc_cf[h]        η²(attnc_h)
    given_tok[h]       Δ(attnc_h | tok) = η²(tokattn_h) − η²(tok)
汇总时看每个神经元最好的 head（max_h），以及最好的 head 在神经元之间分布得集中还是分散。
注意 max_h 是 32 个估计里取最大，会有少量正向选择偏差。

逐头的分组统计量很大（每个 head 约 6 GB，float64），所以每次只处理 heads_per_pass 个 head、重读一遍激活；
每个 (层, head 块) 单独落盘，进程被杀后重跑只补缺的部分。

    python attn_head_decomp.py --names step10000 --layers 4 31
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--acts_root", default=str(root / "acts"))
parser.add_argument("--heads_root", default=str(root / "attn_heads"))
parser.add_argument("--names", nargs="+", default=[f"step{s}" for s in range(0, 10001, 1000)] + ["Olmo-3-1025-7B"])
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--n_pairs", type=int, default=20000)
parser.add_argument("--sink_frac", type=float, default=0.05)
parser.add_argument("--heads_per_pass", type=int, default=4)
parser.add_argument("--chunk_tokens", type=int, default=16384)
parser.add_argument("--out_dir", default=str(root / "runs_decomp_attn_heads"))
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
token_ids = frozen["token_ids"][:T].astype(np.int64)
fold = np.repeat(frozen["doc_fold"][:n_seqs], seq_lens)  # 每个 token 所在文档的折：0 = A，1 = B
seq_of = np.repeat(np.arange(n_seqs), seq_lens)
q = np.arange(T) - starts[seq_of]  # 序列内位置
max_len = int(seq_lens.max())
V = int(token_ids.max()) + 1

# ---- tok：两折都出现过的最高频 n_groups 个 token 各一组，其余为 OTHER（与实验 A 相同） ----
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
NONE = K + 1
print(f"T={T:,}  tok 组={G_tok}（OTHER 占比 {np.mean(tok == K):.3f}）", flush=True)

in_A = torch.from_numpy(fold == 0).to(device)

for name in args.names:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    cache, heads_dir = Path(args.acts_root) / name, Path(args.heads_root) / name
    while not ((cache / "done").exists() and (heads_dir / "done").exists()):
        time.sleep(60)
    assert np.array_equal(np.load(cache / "meta.npz")["token_ids"][:T], token_ids), f"{cache} 的 token 顺序不一致"
    assert np.array_equal(np.load(heads_dir / "meta.npz")["token_ids"][:T], token_ids), f"{heads_dir} 的 token 顺序不一致"
    t0 = time.time()
    result = {}

    for l in args.layers:
        layer_path = out_dir / f"{name}_L{l}.npz"
        if layer_path.exists():
            result.update(dict(np.load(layer_path)))
            print(f"[{name}] L{l:2d} 已有结果，跳过", flush=True)
            continue

        # ---- 本层每个 head 的标签：sink_h、attnc_h、tokattn_h ----
        pos_all = np.load(heads_dir / f"L{l}_pos.npy")  # [T, H, k] int16，-1 = 无
        H = pos_all.shape[1]
        attnc_all = np.zeros((H, T), dtype=np.int64)
        tokattn_all = np.zeros((H, T), dtype=np.int64)
        cell_tok_all, P_all = [], []
        diag = {k: np.zeros(H) for k in ("top1_self", "top1_prev", "top1_sink", "ctx_found", "ctx_prev",
                                          "ctx_same_tok", "ctx_dist_median", "tokattn_cover")}
        m = q >= 1
        for h in range(H):
            src_pos = pos_all[:, h, :].astype(np.int64)
            top1 = src_pos[:, 0]
            hits = np.bincount(seq_of * max_len + top1, minlength=n_seqs * max_len)
            is_sink = hits >= args.sink_frac * np.repeat(seq_lens, max_len)
            key = seq_of[:, None] * max_len + np.maximum(src_pos, 0)
            ok = (src_pos >= 0) & (src_pos != q[:, None]) & ~is_sink[key]
            has = ok.any(1)
            ctx_pos = np.where(has, src_pos[np.arange(T), ok.argmax(1)], -1)
            ctx_tok = np.where(has, token_ids[starts[seq_of] + np.maximum(ctx_pos, 0)], -1)
            attnc = np.where(has, group_of_token[np.maximum(ctx_tok, 0)], NONE)

            eligible = (tok < K) & (attnc < K)
            pairs, pair_inv, pair_counts = np.unique(tok[eligible] * (K + 2) + attnc[eligible],
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

            attnc_all[h], tokattn_all[h] = attnc, tokattn
            cell_tok_all.append(cell_tok)
            P_all.append(P)
            diag["top1_self"][h] = np.mean(top1[m] == q[m])
            diag["top1_prev"][h] = np.mean(top1[m] == q[m] - 1)
            diag["top1_sink"][h] = np.mean(is_sink[seq_of * max_len + top1][m])
            diag["ctx_found"][h] = np.mean(has[m])
            diag["ctx_prev"][h] = np.mean((ctx_pos == q - 1)[m])
            diag["ctx_same_tok"][h] = np.mean((ctx_tok == token_ids)[m & has])
            diag["ctx_dist_median"][h] = float(np.median((q - ctx_pos)[m & has])) if (m & has).any() else np.nan
            diag["tokattn_cover"][h] = np.mean(tokattn < P)
        del pos_all, src_pos, key, ok
        print(f"[{name}] L{l:2d} 标签完成  ({time.time() - t0:.0f}s)", flush=True)

        # ---- 每次处理一块 head：按组累计 Σx（全体 S、A 折 SA），再算交叉拟合 η² ----
        acts = np.load(cache / f"L{l}.npy", mmap_mode="r")
        N = acts.shape[1]
        attnc_cf = np.zeros((H, N), dtype=np.float32)
        tokattn_cf = np.zeros((H, N), dtype=np.float32)
        tok_cf = None
        for h0 in range(0, H, args.heads_per_pass):
            heads = list(range(h0, min(h0 + args.heads_per_pass, H)))
            block_path = out_dir / f"{name}_L{l}_h{h0:02d}.npz"
            if block_path.exists():
                blk = np.load(block_path)
                attnc_cf[heads], tokattn_cf[heads], tok_cf = blk["attnc_cf"], blk["tokattn_cf"], blk["tok_cf"]
                continue
            labels, sizes = {}, {}
            for h in heads:
                labels[("attnc", h)] = torch.from_numpy(attnc_all[h]).to(device)
                labels[("tokattn", h)] = torch.from_numpy(tokattn_all[h]).to(device)
                sizes[("attnc", h)] = G_tok + 1
                sizes[("tokattn", h)] = P_all[h] + G_tok
            n_of = {p: torch.bincount(g, minlength=sizes[p]).double() for p, g in labels.items()}
            nA_of = {p: torch.bincount(g[in_A], minlength=sizes[p]).double() for p, g in labels.items()}
            s1 = torch.zeros(N, dtype=torch.float64, device=device)
            s2 = torch.zeros(N, dtype=torch.float64, device=device)
            S_of = {p: torch.zeros(sizes[p], N, dtype=torch.float64, device=device) for p in labels}
            SA_of = {p: torch.zeros(sizes[p], N, dtype=torch.float64, device=device) for p in labels}
            for lo in range(0, T, args.chunk_tokens):
                hi = min(lo + args.chunk_tokens, T)
                x = torch.from_numpy(np.ascontiguousarray(acts[lo:hi])).to(device).view(torch.bfloat16).double()
                a = in_A[lo:hi]
                s1 += x.sum(0)
                s2 += (x * x).sum(0)
                for p, g in labels.items():
                    S_of[p].index_add_(0, g[lo:hi], x)
                    SA_of[p].index_add_(0, g[lo:hi][a], x[a])
            del x
            mu = s1 / T
            var = s2 / T - mu**2

            # tok 由第一个 head 的 tokattn 合并回去得到（每块都算一遍，顺便核对各块一致）
            parent = torch.from_numpy(cell_tok_all[heads[0]]).to(device)
            p0 = ("tokattn", heads[0])
            jobs = [("tok", None,
                     torch.zeros(G_tok, N, dtype=torch.float64, device=device).index_add_(0, parent, S_of[p0]),
                     torch.zeros(G_tok, N, dtype=torch.float64, device=device).index_add_(0, parent, SA_of[p0]),
                     torch.zeros(G_tok, dtype=torch.float64, device=device).index_add_(0, parent, n_of[p0]),
                     torch.zeros(G_tok, dtype=torch.float64, device=device).index_add_(0, parent, nA_of[p0]))]
            jobs += [(p[0], p[1], S_of[p], SA_of[p], n_of[p], nA_of[p]) for p in labels]
            for f, h, S, SA, n, nA in jobs:
                between_cf = torch.zeros(N, dtype=torch.float64, device=device)
                for r in range(0, len(n), 4096):
                    Sr, SAr, nr, nAr = S[r : r + 4096], SA[r : r + 4096], n[r : r + 4096], nA[r : r + 4096]
                    nBr = nr - nAr
                    both = (nAr > 0) & (nBr > 0)
                    muA = SAr[both] / nAr[both, None]
                    muB = (Sr - SAr)[both] / nBr[both, None]
                    between_cf += (nr[both, None] / T * (muA - mu) * (muB - mu)).sum(0)
                cf = (between_cf / var).cpu().numpy().astype(np.float32)
                if f == "tok":
                    assert tok_cf is None or np.allclose(tok_cf, cf, atol=1e-5, equal_nan=True), "各块的 tok η² 不一致"
                    tok_cf = cf
                elif f == "attnc":
                    attnc_cf[h] = cf
                else:
                    tokattn_cf[h] = cf
            np.savez(block_path, attnc_cf=attnc_cf[heads], tokattn_cf=tokattn_cf[heads], tok_cf=tok_cf)
            del S_of, SA_of, jobs, S, SA, labels
            torch.cuda.empty_cache()
            print(f"[{name}] L{l:2d} head {heads[0]}-{heads[-1]} 完成  ({time.time() - t0:.0f}s)", flush=True)
        del acts

        given_tok = tokattn_cf - tok_cf[None, :]
        best = np.nanargmax(np.nan_to_num(given_tok, nan=-np.inf), axis=0)  # 每个神经元 Δ(attnc_h|tok) 最大的 head
        best_share = np.bincount(best, minlength=H) / N
        layer_result = {f"L{l}_tok_cf": tok_cf, f"L{l}_attnc_cf": attnc_cf, f"L{l}_given_tok": given_tok.astype(np.float32),
                        f"L{l}_best_head": best.astype(np.int16), f"L{l}_best_head_share": best_share,
                        **{f"L{l}_head_{k}": v for k, v in diag.items()}}
        np.savez(layer_path, **layer_result)
        result.update(layer_result)
        for h0 in range(0, H, args.heads_per_pass):
            (out_dir / f"{name}_L{l}_h{h0:02d}.npz").unlink(missing_ok=True)

        max_attnc = np.nanmax(attnc_cf, 0)
        max_given = np.nanmax(given_tok, 0)
        order = np.argsort(-best_share)[:3]
        print(f"[{name}] L{l:2d}  η² 中位 tok {np.nanmedian(tok_cf):.4f}"
              f" | attnc_h: 各头中位数的最大 {np.nanmax(np.nanmedian(attnc_cf, 1)):.4f} 逐神经元 max_h 的中位 {np.nanmedian(max_attnc):.4f}"
              f" | Δ(attnc_h|tok): 各头中位数的最大 {np.nanmax(np.nanmedian(given_tok, 1)):.4f}"
              f" 逐神经元 max_h 的中位 {np.nanmedian(max_given):.4f}"
              f" | attnc_h 先于 tok 的神经元 {np.mean(max_attnc > tok_cf):.3f}"
              f" | 最佳 head 占比前三 " + " ".join(f"h{h}:{best_share[h]:.2f}" for h in order) +
              f"  ({time.time() - t0:.0f}s)", flush=True)

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), G_tok=G_tok, sink_frac=args.sink_frac, **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
