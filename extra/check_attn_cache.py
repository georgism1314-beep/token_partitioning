"""逐条核对 attn_topk/ 里已经写好的缓存，确认缺失值约定和因果性都没问题。

对每个 checkpoint、每层、每个 token（序列内位置 t）检查：
    1. 有效项个数 == min(K, t+1)，且有效项在前、缺失项在后（不会夹在中间）
    2. 缺失项：tok == -1、pos == -1、w == 0（不会残留因果范围外的 token-id）
    3. 有效项：0 <= pos <= t；tok == 该序列 pos 处的真实 token-id；位置互不重复
    4. 权重按名次单调不增、非负，且 Σ top-K 权重 <= 1（+容差）
另外打印 Σ top-K 权重的均值（t >= K 的行），作为 top-K 截断程度的粗略参考。

    python extra/check_attn_cache.py
"""
import argparse
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]  # repo root
parser = argparse.ArgumentParser()
parser.add_argument("--attn_root", default=str(root / "attn_topk"))
parser.add_argument("--chunk", type=int, default=250_000)
args = parser.parse_args()

dirs = sorted(p for p in Path(args.attn_root).iterdir() if (p / "done").exists())
n_bad_total = 0
for d in dirs:
    meta = np.load(d / "meta.npz")
    token_ids, seq_lens, K = meta["token_ids"].astype(np.int64), meta["seq_lens"], int(meta["topk"])
    T = int(seq_lens.sum())
    starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
    row_start = np.repeat(starts, seq_lens)              # 每个 token 所在序列的起点
    q_all = np.arange(T) - row_start                     # 序列内位置 t
    for l in meta["layers"]:
        tok_f = np.load(d / f"L{l}_tok.npy", mmap_mode="r")
        pos_f = np.load(d / f"L{l}_pos.npy", mmap_mode="r")
        w_f = np.load(d / f"L{l}_w.npy", mmap_mode="r")
        assert tok_f.shape == pos_f.shape == w_f.shape == (T, K)
        bad = {k: 0 for k in ("n_valid", "contiguous", "missing", "causal", "tok_match", "dup", "monotone", "sum")}
        cover_sum, cover_n, max_sum = 0.0, 0, 0.0
        for c in range(0, T, args.chunk):
            tok = np.asarray(tok_f[c : c + args.chunk]).astype(np.int64)
            pos = np.asarray(pos_f[c : c + args.chunk]).astype(np.int64)
            w = np.asarray(w_f[c : c + args.chunk]).astype(np.float32)
            q = q_all[c : c + args.chunk]
            valid = pos >= 0

            bad["n_valid"] += int((valid.sum(1) != np.minimum(K, q + 1)).sum())
            bad["contiguous"] += int((valid[:, 1:] & ~valid[:, :-1]).any(1).sum())
            bad["missing"] += int(((tok[~valid] != -1) | (w[~valid] != 0)).sum())
            bad["causal"] += int((valid & (pos > q[:, None])).any(1).sum())
            src = np.where(valid, row_start[c : c + args.chunk, None] + pos, 0)
            bad["tok_match"] += int((valid & (tok != token_ids[src])).any(1).sum())
            sp = np.sort(np.where(valid, pos, -1 - np.arange(K)[None, :]), axis=1)  # 缺失项换成互不相同的负数
            bad["dup"] += int((np.diff(sp, axis=1) == 0).any(1).sum())
            bad["monotone"] += int(((w[:, 1:] > w[:, :-1]).any(1) | (w < 0).any(1)).sum())
            s = w.sum(1)
            bad["sum"] += int((s > 1.01).sum())
            max_sum = max(max_sum, float(s.max()))
            full = q >= K
            cover_sum += float(s[full].sum())
            cover_n += int(full.sum())
        n_bad = sum(bad.values())
        n_bad_total += n_bad
        flag = "OK " if n_bad == 0 else "BAD"
        detail = "" if n_bad == 0 else "  " + " ".join(f"{k}={v}" for k, v in bad.items() if v)
        print(f"{flag} {d.name:>18s} L{l:2d}  max Σw {max_sum:.4f}  mean Σ top{K} w (t≥{K}) {cover_sum / cover_n:.3f}{detail}",
              flush=True)
print(f"共 {len(dirs)} 个 checkpoint，违规行总数 {n_bad_total}", flush=True)
