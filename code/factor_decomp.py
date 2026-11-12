"""读 cache_acts.py 缓存的原始 FFN 激活，做"当前 token / 前一个 token / 前两个 token"的嵌套方差分解，不需要前向。

A_i 为神经元 i 的原始激活 silu(gate_proj(x)) * up_proj(x)。所有分组都由同一个最细的 trigram 划分合并得到，
保证嵌套关系严格成立：
    trigram 当前位置属于高频 bigram 格子、且 (前两个, 前一个, 当前) 两折都出现过的最高频 n_trigrams 个 trigram 各一组；
            其余 token 退回自己的 bigram 组
    bigram  当前 token 是高频 token、且 (前一个, 当前) 两折都出现过的最高频 n_bigrams 个 bigram 各一组；
            其余 token 退回自己的当前 token 组（= trigram 按 bigram 合并）
    tok     当前 token：两折都出现过的最高频 n_groups 个 token 各一组，其余为 OTHER（= bigram 按当前 token 合并）
    prev    前一个 token：bigram 格子按前一个 token 合并；退回组里前一个 token 不可知，合成一个"稀有"组
tok ⊂ bigram ⊂ trigram 逐级细分，prev 也是 bigram 的粗化，所以（总体上）η² 沿这条链单调不减。

每个神经元的分解（cf = cross-fit η²，按文档分两折，无偏）：
    链式   η²(tok) + Δ(prev | tok) + Δ(prev2 | bigram) + 剩余 = 1，
           Δ(prev | tok) = η²(bigram) − η²(tok)，Δ(prev2 | bigram) = η²(trigram) − η²(bigram)，剩余 = 1 − η²(trigram)
    贪心   第一步选 η²(tok) 与 η²(prev) 中较大者，第二步增量 = η²(bigram) − 第一步
    独有   Δ(tok | prev) = η²(bigram) − η²(prev)
    共享   η²(tok) + η²(prev) − η²(bigram)（为负表示两者有协同）

    python code/factor_decomp.py --names step0 step10000 --layers 0 31
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]  # repo root
parser = argparse.ArgumentParser()
parser.add_argument("--acts_root", default=str(root / "acts"))
parser.add_argument("--names", nargs="+", default=[f"step{s}" for s in range(0, 10001, 1000)] + ["Olmo-3-1025-7B"])
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--n_bigrams", type=int, default=20000)
parser.add_argument("--n_trigrams", type=int, default=20000)
parser.add_argument("--chunk_tokens", type=int, default=16384)
parser.add_argument("--out_dir", default=str(root / "runs_decomp_trigram"))
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
prev_ids = np.concatenate([[-1], token_ids[:-1]])
prev_ids[starts] = -1  # 序列第一个 token 没有前文
prev2_ids = np.concatenate([[-1, -1], token_ids[:-2]])
prev2_ids[starts] = -1
prev2_ids[starts + 1] = -1  # 序列前两个 token 没有"前两个"
V = int(token_ids.max()) + 1

# ---- tok：两折都出现过的最高频 n_groups 个当前 token 各一组，其余为 OTHER ----
# 只出现在一折的组无法 cross-fit（会被当成 0 贡献），所以只让两折都出现过的 token / bigram 单独成组
counts = np.bincount(token_ids, minlength=V)
counts_A = np.bincount(token_ids[fold == 0], minlength=V)
in_both_folds = (counts_A > 0) & (counts - counts_A > 0)
top = np.argsort(-np.where(in_both_folds, counts, 0))[: args.n_groups]
top = top[in_both_folds[top]]
group_of_token = np.full(V, len(top))
group_of_token[top] = np.arange(len(top))
tok = group_of_token[token_ids]
G_tok = len(top) + 1

# ---- bigram：组号 0..B-1 为高频 bigram，B + (tok 组号) 为退回的当前 token 组 ----
eligible = (prev_ids >= 0) & (tok != G_tok - 1)
pairs, pair_inv, pair_counts = np.unique(token_ids[eligible] * V + prev_ids[eligible], return_inverse=True, return_counts=True)
pair_counts_A = np.bincount(pair_inv[fold[eligible] == 0], minlength=len(pairs))
pair_in_both = (pair_counts_A > 0) & (pair_counts - pair_counts_A > 0)
top_pairs = np.argsort(-np.where(pair_in_both, pair_counts, 0))[: args.n_bigrams]
top_pairs = top_pairs[pair_in_both[top_pairs]]
B = len(top_pairs)
group_of_pair = np.full(len(pairs), -1)
group_of_pair[top_pairs] = np.arange(B)
bigram = B + tok
pair_group = group_of_pair[pair_inv]
bigram[np.where(eligible)[0][pair_group >= 0]] = pair_group[pair_group >= 0]
G_big = B + G_tok

# ---- 每个 bigram 格子对应的 tok 组与 prev 组（粗化映射） ----
cell_tok = np.concatenate([group_of_token[pairs[top_pairs] // V], np.arange(G_tok)])
cell_prev_id = np.concatenate([pairs[top_pairs] % V, np.full(G_tok, -1)])
prev_vocab, cell_prev = np.unique(cell_prev_id, return_inverse=True)  # -1（"稀有"组）排在第 0 位
G_prev = len(prev_vocab)
assert (cell_tok[bigram] == tok).all(), "bigram 必须是 tok 的细分"

# ---- trigram：组号 0..R-1 为高频 trigram，R + (bigram 组号) 为退回的 bigram 组 ----
eligible3 = (bigram < B) & (prev2_ids >= 0)
triples, tri_inv, tri_counts = np.unique(bigram[eligible3] * V + prev2_ids[eligible3], return_inverse=True, return_counts=True)
tri_counts_A = np.bincount(tri_inv[fold[eligible3] == 0], minlength=len(triples))
tri_in_both = (tri_counts_A > 0) & (tri_counts - tri_counts_A > 0)
top_triples = np.argsort(-np.where(tri_in_both, tri_counts, 0))[: args.n_trigrams]
top_triples = top_triples[tri_in_both[top_triples]]
R = len(top_triples)
group_of_triple = np.full(len(triples), -1)
group_of_triple[top_triples] = np.arange(R)
trigram = R + bigram
tri_group = group_of_triple[tri_inv]
trigram[np.where(eligible3)[0][tri_group >= 0]] = tri_group[tri_group >= 0]
G_tri = R + G_big
cell_big = np.concatenate([triples[top_triples] // V, np.arange(G_big)])
assert (cell_big[trigram] == bigram).all(), "trigram 必须是 bigram 的细分"

print(f"T={T:,}  tok 组={G_tok}（OTHER 占比 {np.mean(tok == G_tok - 1):.3f}）  bigram 单独成组={B}"
      f"（覆盖 {np.mean(bigram < B):.3f} 的 token）  trigram 单独成组={R}（覆盖 {np.mean(trigram < R):.3f} 的 token）"
      f"  prev 组={G_prev}（\"稀有\"组占比 {np.mean(cell_prev[bigram] == 0):.3f}）", flush=True)

fine_t = torch.from_numpy(trigram).to(device)
in_A = torch.from_numpy(fold == 0).to(device)
coarsen = {"trigram": torch.arange(G_tri, device=device),
           "bigram": torch.from_numpy(cell_big).to(device),
           "tok": torch.from_numpy(cell_tok[cell_big]).to(device),
           "prev": torch.from_numpy(cell_prev[cell_big]).to(device)}
n_fine = torch.bincount(fine_t, minlength=G_tri).double()
nA_fine = torch.bincount(fine_t[in_A], minlength=G_tri).double()

for name in args.names:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    cache = Path(args.acts_root) / name
    while not (cache / "done").exists():
        time.sleep(60)  # 等 cache_acts.py 写完
    assert np.array_equal(np.load(cache / "meta.npz")["token_ids"], token_ids), f"{cache} 的 token 顺序与冻结子集不一致"
    t0 = time.time()
    result = {}

    for l in args.layers:
        acts = np.load(cache / f"L{l}.npy", mmap_mode="r")
        N = acts.shape[1]

        # 充分统计量（float64）：Σx、Σx²，以及按 trigram 格子的 Σx（全体 S、A 折 SA）；其余因素的由它合并得到
        s1 = torch.zeros(N, dtype=torch.float64, device=device)
        s2 = torch.zeros(N, dtype=torch.float64, device=device)
        S_fine = torch.zeros(G_tri, N, dtype=torch.float64, device=device)
        SA_fine = torch.zeros(G_tri, N, dtype=torch.float64, device=device)
        for lo in range(0, T, args.chunk_tokens):
            hi = min(lo + args.chunk_tokens, T)
            h = torch.from_numpy(np.ascontiguousarray(acts[lo:hi])).to(device).view(torch.bfloat16).double()
            g, a = fine_t[lo:hi], in_A[lo:hi]
            s1 += h.sum(0)
            s2 += (h * h).sum(0)
            S_fine.index_add_(0, g, h)
            SA_fine.index_add_(0, g[a], h[a])
        del acts, h

        mu = s1 / T
        var = s2 / T - mu**2
        result[f"L{l}_mean"] = mu.float().cpu().numpy()
        result[f"L{l}_var"] = var.float().cpu().numpy()
        cf = {}
        for f, parent in coarsen.items():
            if f == "trigram":
                S, SA, n, nA = S_fine, SA_fine, n_fine, nA_fine
            else:
                G = int(parent.max().item()) + 1
                S = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, S_fine)
                SA = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, SA_fine)
                n = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, n_fine)
                nA = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, nA_fine)
            G_eff = int((n > 0).sum().item())

            # 按组分块累加，避免 [G, N] 大小的临时张量（trigram 每个 4.2 GB）同时存在
            between = torch.zeros(N, dtype=torch.float64, device=device)
            between_cf = torch.zeros(N, dtype=torch.float64, device=device)
            for r in range(0, len(n), 4096):
                Sr, SAr, nr, nAr = S[r : r + 4096], SA[r : r + 4096], n[r : r + 4096], nA[r : r + 4096]
                between += (Sr**2 / nr.clamp(min=1)[:, None]).sum(0)
                nBr = nr - nAr
                both = (nAr > 0) & (nBr > 0)
                muA = SAr[both] / nAr[both, None]
                muB = (Sr - SAr)[both] / nBr[both, None]
                between_cf += (nr[both, None] / T * (muA - mu) * (muB - mu)).sum(0)
            between = between / T - mu**2
            eps2 = between - (G_eff - 1) / (T - G_eff) * (var - between)
            cf[f] = (between_cf / var).cpu().numpy()
            result[f"L{l}_{f}_naive"] = (between / var).float().cpu().numpy()
            result[f"L{l}_{f}_eps2"] = (eps2 / var).float().cpu().numpy()
            result[f"L{l}_{f}_cf"] = cf[f].astype(np.float32)
            result[f"L{l}_{f}_cf_weighted"] = (between_cf.sum() / var.sum()).item()
            del S, SA, Sr, SAr, muA, muB
        del S_fine, SA_fine
        torch.cuda.empty_cache()

        # 每个神经元的分解
        tok_first = cf["tok"] >= cf["prev"]
        first = np.maximum(cf["tok"], cf["prev"])
        result[f"L{l}_tok_first"] = tok_first
        result[f"L{l}_greedy_step2"] = (cf["bigram"] - first).astype(np.float32)
        result[f"L{l}_uniq_prev"] = (cf["bigram"] - cf["tok"]).astype(np.float32)
        result[f"L{l}_uniq_tok"] = (cf["bigram"] - cf["prev"]).astype(np.float32)
        result[f"L{l}_shared"] = (cf["tok"] + cf["prev"] - cf["bigram"]).astype(np.float32)
        result[f"L{l}_uniq_prev2"] = (cf["trigram"] - cf["bigram"]).astype(np.float32)
        result[f"L{l}_resid_bigram"] = (1 - cf["bigram"]).astype(np.float32)
        result[f"L{l}_resid"] = (1 - cf["trigram"]).astype(np.float32)
        print(f"[{name}] L{l:2d}  中位数  tok {np.nanmedian(cf['tok']):.4f}  bigram {np.nanmedian(cf['bigram']):.4f}"
              f"  trigram {np.nanmedian(cf['trigram']):.4f}  | 链式 +prev|tok {np.nanmedian(result[f'L{l}_uniq_prev']):.4f}"
              f"  +prev2|bigram {np.nanmedian(result[f'L{l}_uniq_prev2']):.4f}  剩余 {np.nanmedian(result[f'L{l}_resid']):.4f}"
              f"  | prev {np.nanmedian(cf['prev']):.4f}  先选 tok {tok_first.mean():.3f}"
              f"  共享 {np.nanmedian(result[f'L{l}_shared']):+.4f}", flush=True)

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), G_tok=G_tok, G_prev=G_prev, G_bigram=G_big,
             G_trigram=G_tri, **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
