"""读 cache_acts.py 缓存的原始 FFN 激活和 pos_tags.py 的标签，做"字符类 / token-id / POS"的方差分解，不需要前向。

A_i 为神经元 i 的原始激活 silu(gate_proj(x)) * up_proj(x)。分组：
    charclass  字符类（只看 token 字符串，约 18 类）
    tok        当前 token：两折都出现过的最高频 n_groups 个 token 各一组，其余 token 按字符类分成若干 OTHER 组
               （这样 tok 严格是 charclass 的细分）
    tokpos     (当前 token, 上下文 POS) 格子：tok 不是 OTHER、两折都出现过且出现 >= min_count 次的格子各一组；
               其余 token 退回自己的 tok 组（所以 tokpos 严格是 tok 的细分）
    pos        上下文 POS 直接分组（spaCy UPOS + SPACE / NEWLINE）
    pos_multi  同上，但跨多个词且 POS 冲突的 token 单独成 MULTI 组（敏感性分析）

每个神经元的分解（cf = cross-fit η²，按文档分两折，无偏）：
    链式   η²(charclass) + Δ(tok | charclass) + Δ(POS | tok) + 剩余 = 1，
           Δ(tok | charclass) = η²(tok) − η²(charclass)，Δ(POS | tok) = η²(tokpos) − η²(tok)，剩余 = 1 − η²(tokpos)
    反序   Δ(tok | POS) = η²(tokpos) − η²(pos)（退回组混合了 POS，所以只是近似的细分）
    共享   η²(tok) + η²(pos) − η²(tokpos)

    python code/pos_decomp.py --names step0 step10000 --layers 0 31
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
parser.add_argument("--pos", default=str(root / "frozen_5e6" / "pos_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--min_count", type=int, default=10)
parser.add_argument("--chunk_tokens", type=int, default=16384)
parser.add_argument("--out_dir", default=str(root / "runs_decomp_pos"))
args = parser.parse_args()
device = "cuda"
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- 数据：与 cache_acts.py 相同的冻结子集前缀 + pos_tags.py 的标签 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
fold = np.repeat(frozen["doc_fold"][:n_seqs], seq_lens)  # 每个 token 所在文档的折：0 = A，1 = B
V = int(token_ids.max()) + 1
tags = np.load(args.pos)
assert np.array_equal(tags["token_ids"], token_ids), "pos 标签与冻结子集的 token 顺序不一致"
charclass = tags["charclass"].astype(np.int64)
pos = tags["upos"].astype(np.int64)
pos_multi = tags["upos_multi"].astype(np.int64)
n_cc, n_pos = len(tags["charclass_names"]), len(tags["upos_names"])

# ---- tok：两折都出现过的最高频 n_groups 个 token 各一组，其余按字符类分 OTHER 组 ----
counts = np.bincount(token_ids, minlength=V)
counts_A = np.bincount(token_ids[fold == 0], minlength=V)
in_both_folds = (counts_A > 0) & (counts - counts_A > 0)
top = np.argsort(-np.where(in_both_folds, counts, 0))[: args.n_groups]
top = top[in_both_folds[top]]
K = len(top)
is_top = np.zeros(V, dtype=bool)
is_top[top] = True
group_of_token = np.full(V, -1)
group_of_token[top] = np.arange(K)
tok = np.where(is_top[token_ids], group_of_token[token_ids], K + charclass)
G_tok = K + n_cc
tok_charclass = np.zeros(G_tok, dtype=np.int64)
tok_charclass[tok] = charclass  # 每个 tok 组对应的字符类（粗化映射）

# ---- tokpos：组号 0..C-1 为高频 (tok, POS) 格子，C + (tok 组号) 为退回的 tok 组 ----
eligible = tok < K
cells, cell_inv, cell_counts = np.unique(tok[eligible] * n_pos + pos[eligible], return_inverse=True, return_counts=True)
cell_counts_A = np.bincount(cell_inv[fold[eligible] == 0], minlength=len(cells))
keep = (cell_counts_A > 0) & (cell_counts - cell_counts_A > 0) & (cell_counts >= args.min_count)
C = int(keep.sum())
group_of_cell = np.full(len(cells), -1)
group_of_cell[keep] = np.arange(C)
tokpos = C + tok
cell_group = group_of_cell[cell_inv]
tokpos[np.where(eligible)[0][cell_group >= 0]] = cell_group[cell_group >= 0]
G_tp = C + G_tok
cell_tok = np.concatenate([cells[keep] // n_pos, np.arange(G_tok)])
assert (cell_tok[tokpos] == tok).all(), "tokpos 必须是 tok 的细分"
assert (tok_charclass[tok] == charclass).all(), "tok 必须是 charclass 的细分"

print(f"T={T:,}  charclass 组={n_cc}  tok 组={G_tok}（OTHER 占比 {np.mean(tok >= K):.3f}）"
      f"  tokpos 单独成组={C}（覆盖 {np.mean(tokpos < C):.3f} 的 token，有 >=2 个 POS 格子的 token 覆盖"
      f" {np.mean(np.bincount(cells[keep] // n_pos, minlength=G_tok)[tok] >= 2):.3f}）  pos 组={n_pos}", flush=True)

in_A = torch.from_numpy(fold == 0).to(device)
labels = {"tokpos": torch.from_numpy(tokpos).to(device),
          "charclass": torch.from_numpy(charclass).to(device),
          "pos": torch.from_numpy(pos).to(device),
          "pos_multi": torch.from_numpy(pos_multi).to(device)}
sizes = {"tokpos": G_tp, "charclass": n_cc, "pos": n_pos, "pos_multi": n_pos}
# 因素 -> (累加用的分组, 合并到该因素的映射；None 表示直接用)
factors = {"charclass": ("charclass", None),
           "tok": ("tokpos", torch.from_numpy(cell_tok).to(device)),
           "tokpos": ("tokpos", None),
           "pos": ("pos", None),
           "pos_multi": ("pos_multi", None)}
n_of = {p: torch.bincount(g, minlength=sizes[p]).double() for p, g in labels.items()}
nA_of = {p: torch.bincount(g[in_A], minlength=sizes[p]).double() for p, g in labels.items()}

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

        # 充分统计量（float64）：Σx、Σx²，以及每种分组下按组的 Σx（全体 S、A 折 SA）
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
        result[f"L{l}_mean"] = mu.float().cpu().numpy()
        result[f"L{l}_var"] = var.float().cpu().numpy()
        cf = {}
        for f, (p, parent) in factors.items():
            S, SA, n, nA = S_of[p], SA_of[p], n_of[p], nA_of[p]
            if parent is not None:
                G = int(parent.max().item()) + 1
                S = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, S)
                SA = torch.zeros(G, N, dtype=torch.float64, device=device).index_add_(0, parent, SA)
                n = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, n)
                nA = torch.zeros(G, dtype=torch.float64, device=device).index_add_(0, parent, nA)
            G_eff = int((n > 0).sum().item())

            # 按组分块累加，避免 [G, N] 大小的临时张量同时存在
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
        del S_of, SA_of
        torch.cuda.empty_cache()

        # 每个神经元的分解
        result[f"L{l}_tok_given_cc"] = (cf["tok"] - cf["charclass"]).astype(np.float32)
        result[f"L{l}_pos_given_tok"] = (cf["tokpos"] - cf["tok"]).astype(np.float32)
        result[f"L{l}_tok_given_pos"] = (cf["tokpos"] - cf["pos"]).astype(np.float32)
        result[f"L{l}_shared_tok_pos"] = (cf["tok"] + cf["pos"] - cf["tokpos"]).astype(np.float32)
        result[f"L{l}_resid"] = (1 - cf["tokpos"]).astype(np.float32)
        print(f"[{name}] L{l:2d}  中位数  charclass {np.nanmedian(cf['charclass']):.4f}  tok {np.nanmedian(cf['tok']):.4f}"
              f"  tokpos {np.nanmedian(cf['tokpos']):.4f}  pos {np.nanmedian(cf['pos']):.4f}"
              f"  pos_multi {np.nanmedian(cf['pos_multi']):.4f}  | 链式 +tok|cc {np.nanmedian(result[f'L{l}_tok_given_cc']):.4f}"
              f"  +pos|tok {np.nanmedian(result[f'L{l}_pos_given_tok']):.4f}  剩余 {np.nanmedian(result[f'L{l}_resid']):.4f}"
              f"  | 反序 +tok|pos {np.nanmedian(result[f'L{l}_tok_given_pos']):.4f}"
              f"  共享 {np.nanmedian(result[f'L{l}_shared_tok_pos']):+.4f}", flush=True)

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), G_charclass=n_cc, G_tok=G_tok, G_tokpos=G_tp,
             G_pos=n_pos, **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
