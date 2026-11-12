"""各分组因素对单个 FFN 神经元的方差解释比例 η²，跨 checkpoint、跨层。

A_i 取神经元 i 的原始 FFN 激活 h = silu(gate_proj(x)) * up_proj(x)（即 down_proj 的输入），
不做二值化 / topk / 阈值 / log。对每个分组因素 G，按全方差公式
    Var(A_i) = Var(E[A_i | G]) + E[Var(A_i | G)]
    η²_i     = Var(E[A_i | G]) / Var(A_i)
因素（组都只取两折都出现过的，其余并入兜底组）：
    tok     当前 token-id：最高频 n_groups 个 token 各一组，其余为 OTHER
    plc     placebo：tok 组标签的随机置换，校正后应 ≈ 0
    prev    前一个 token-id：最高频 n_groups 个各一组，其余为 OTHER，序列第一个 token 单独一组
    bigram  (前一个 token, 当前 token)：当前 token 属于 tok 的非 OTHER 组时，最高频 n_bigrams 个 bigram 各一组；
            其余 token 退回自己的 tok 组。所以 bigram 是 tok 的细分，η²(bigram) − η²(tok) 就是
            "已知当前 token 后，前一个 token 还能多解释多少"

朴素 η² 有约 (G-1)/T 的正偏差（每组均值自带噪声），所以同时给出
    eps2  ε² 校正：between - (G-1)/(T-G) * within
    cf    cross-fit：按文档分 A/B 两折，between = Σ_g n_g/T · (μ^A_g - μ)(μ^B_g - μ)，两折噪声独立故无偏

    python token_variance.py --steps 0 1000 2000 --layers 0 4 8 16 24 31
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_root", default=str(root / "ckpts_deep"))
parser.add_argument("--steps", type=int, nargs="*", default=list(range(0, 10001, 1000)))
parser.add_argument("--extra_ckpts", nargs="*", default=[], help="额外的完整模型目录（如最终基座），结果以目录名命名")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--factors", nargs="+", default=["tok", "plc", "prev", "bigram"])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--n_bigrams", type=int, default=20000)
parser.add_argument("--batch_seqs", type=int, default=8)
parser.add_argument("--layers_per_pass", type=int, default=1)
parser.add_argument("--out_dir", default=str(root / "runs_factors"))
args = parser.parse_args()
device = "cuda"
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- 数据：冻结子集（Nemotron SFT 已 tokenize）的前缀，按整条序列截取；每条序列就是一个文档 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
fold = np.repeat(frozen["doc_fold"][:n_seqs], seq_lens)  # 每个 token 所在文档的折：0 = A，1 = B
prev_ids = np.concatenate([[-1], token_ids[:-1]])
prev_ids[starts] = -1  # 序列第一个 token 没有前文
V = int(token_ids.max()) + 1

# ---- 分组。只出现在一折的组无法 cross-fit（会被当成 0 贡献），所以只让两折都出现过的 token / bigram 单独成组 ----
groups = {}

# tok：当前 token-id
counts = np.bincount(token_ids, minlength=V)
counts_A = np.bincount(token_ids[fold == 0], minlength=V)
in_both_folds = (counts_A > 0) & (counts - counts_A > 0)
top = np.argsort(-np.where(in_both_folds, counts, 0))[: args.n_groups]
top = top[in_both_folds[top]]
group_of_token = np.full(V, len(top))
group_of_token[top] = np.arange(len(top))
groups["tok"] = group_of_token[token_ids]
tok_other = len(top)

# plc：tok 标签的随机置换
groups["plc"] = np.random.default_rng(0).permutation(groups["tok"])

# prev：前一个 token-id；组号 0..K-1 为高频 token，K 为 OTHER，K+1 为序列开头
has_prev = prev_ids >= 0
prev_counts = np.bincount(prev_ids[has_prev], minlength=V)
prev_counts_A = np.bincount(prev_ids[has_prev & (fold == 0)], minlength=V)
prev_in_both = (prev_counts_A > 0) & (prev_counts - prev_counts_A > 0)
top_prev = np.argsort(-np.where(prev_in_both, prev_counts, 0))[: args.n_groups]
top_prev = top_prev[prev_in_both[top_prev]]
group_of_prev = np.full(V, len(top_prev))
group_of_prev[top_prev] = np.arange(len(top_prev))
groups["prev"] = np.where(has_prev, group_of_prev[np.maximum(prev_ids, 0)], len(top_prev) + 1)

# bigram：组号 0..B-1 为高频 bigram，B + (tok 组号) 为退回的 tok 组
eligible = has_prev & (groups["tok"] != tok_other)
pairs, pair_inv, pair_counts = np.unique(token_ids[eligible] * V + prev_ids[eligible], return_inverse=True, return_counts=True)
pair_counts_A = np.bincount(pair_inv[fold[eligible] == 0], minlength=len(pairs))
pair_in_both = (pair_counts_A > 0) & (pair_counts - pair_counts_A > 0)
top_pairs = np.argsort(-np.where(pair_in_both, pair_counts, 0))[: args.n_bigrams]
top_pairs = top_pairs[pair_in_both[top_pairs]]
group_of_pair = np.full(len(pairs), -1)
group_of_pair[top_pairs] = np.arange(len(top_pairs))
groups["bigram"] = len(top_pairs) + groups["tok"]
pair_group = group_of_pair[pair_inv]
groups["bigram"][np.where(eligible)[0][pair_group >= 0]] = pair_group[pair_group >= 0]
assert len(np.unique(groups["bigram"])) == len(np.unique(groups["bigram"] * (tok_other + 1) + groups["tok"])), "bigram 必须是 tok 的细分"

print(f"T={T:,}  序列数={n_seqs}  tok 组={tok_other + 1}（OTHER 占比 {np.mean(groups['tok'] == tok_other):.3f}）"
      f"  prev 组={len(top_prev) + 2}（OTHER 占比 {np.mean(groups['prev'] == len(top_prev)):.3f}）"
      f"  bigram 单独成组={len(top_pairs)}（覆盖 {np.mean(groups['bigram'] < len(top_pairs)):.3f} 的 token）", flush=True)

in_A = torch.from_numpy(fold == 0).to(device)
factors = {}
for f in args.factors:
    g = torch.from_numpy(groups[f]).to(device)
    G = int(g.max().item()) + 1
    n = torch.bincount(g, minlength=G).double()
    factors[f] = {"g": g, "G": G, "G_eff": int((n > 0).sum().item()), "n": n,
                  "nA": torch.bincount(g[in_A], minlength=G).double()}

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    while not any(ckpt.glob("model*.safetensors")):
        time.sleep(60)  # 等 fetch_ckpt.py 下完
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
    N = model.config.intermediate_size
    all_layers = model.model.layers
    result = {}

    # 显存不够同时放所有层的统计量（四个因素每层约 9 GB），所以每趟只看 layers_per_pass 层，前向也只算到这一趟最深的那层
    passes = [args.layers[i : i + args.layers_per_pass] for i in range(0, len(args.layers), args.layers_per_pass)]
    for pass_layers in passes:
        model.model.layers = all_layers[: max(pass_layers) + 1]

        # down_proj 的输入就是 silu(gate_proj(x)) * up_proj(x)
        acts = {}
        hooks = [all_layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs, l=l: acts.__setitem__(l, inputs[0])) for l in pass_layers]

        # 每层的充分统计量（float64）：Σx、Σx²，以及每个因素按组的 Σx（全体 S、A 折 SA）
        stats = {}
        for l in pass_layers:
            stats[l] = {"s1": torch.zeros(N, dtype=torch.float64, device=device),
                        "s2": torch.zeros(N, dtype=torch.float64, device=device)}
            for f, fac in factors.items():
                stats[l][f"S_{f}"] = torch.zeros(fac["G"], N, dtype=torch.float64, device=device)
                stats[l][f"SA_{f}"] = torch.zeros(fac["G"], N, dtype=torch.float64, device=device)

        with torch.inference_mode():
            for b in range(0, n_seqs, args.batch_seqs):
                lens = seq_lens[b : b + args.batch_seqs]
                ids = torch.zeros(len(lens), int(lens.max()), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                    ids[j, :m] = torch.from_numpy(token_ids[s : s + m])
                    mask[j, :m] = 1
                model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)

                # 右侧 padding + 因果注意力：有效 token 不受 padding 影响；按 mask 取出后顺序与 token_ids 一致
                valid = mask.bool().to(device)
                lo, hi = int(starts[b]), int(starts[b] + lens.sum())
                a = in_A[lo:hi]
                for l in pass_layers:
                    h = acts[l][valid].double()  # [t, N]
                    st = stats[l]
                    st["s1"] += h.sum(0)
                    st["s2"] += (h * h).sum(0)
                    for f, fac in factors.items():
                        g = fac["g"][lo:hi]
                        st[f"S_{f}"].index_add_(0, g, h)
                        st[f"SA_{f}"].index_add_(0, g[a], h[a])
        for hook in hooks:
            hook.remove()

        for l in pass_layers:
            st = stats[l]
            mu = st["s1"] / T
            var = st["s2"] / T - mu**2
            result[f"L{l}_mean"] = mu.float().cpu().numpy()
            result[f"L{l}_var"] = var.float().cpu().numpy()
            for f, fac in factors.items():
                S, SA, n, nA, G_eff = st[f"S_{f}"], st[f"SA_{f}"], fac["n"], fac["nA"], fac["G_eff"]
                between = (S**2 / n.clamp(min=1)[:, None]).sum(0) / T - mu**2
                within = var - between
                eps2 = between - (G_eff - 1) / (T - G_eff) * within
                nB = n - nA
                both = (nA > 0) & (nB > 0)
                muA = SA[both] / nA[both, None]
                muB = (S - SA)[both] / nB[both, None]
                between_cf = (n[both, None] / T * (muA - mu) * (muB - mu)).sum(0)
                result[f"L{l}_{f}_naive"] = (between / var).float().cpu().numpy()
                result[f"L{l}_{f}_eps2"] = (eps2 / var).float().cpu().numpy()
                result[f"L{l}_{f}_cf"] = (between_cf / var).float().cpu().numpy()
                result[f"L{l}_{f}_cf_weighted"] = (between_cf.sum() / var.sum()).item()
                del S, SA, between, within, eps2, muA, muB
                st[f"S_{f}"] = st[f"SA_{f}"] = None  # 逐个释放，避免大组数因素的临时张量叠加

            r = {k[len(f"L{l}_"):]: v for k, v in result.items() if k.startswith(f"L{l}_")}
            line = f"[{name}] L{l:2d}  cf η² 中位数"
            for f in factors:
                line += f"  {f} {np.nanmedian(r[f'{f}_cf']):+.4f}"
            if "tok" in factors and "bigram" in factors:
                line += f"  | bigram−tok 中位数 {np.nanmedian(r['bigram_cf'] - r['tok_cf']):.4f}"
            print(line, flush=True)
        del stats, acts
        torch.cuda.empty_cache()

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), factors=np.array(list(factors)),
             **{f"G_{f}": fac["G_eff"] for f, fac in factors.items()}, **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
    del model, all_layers
    torch.cuda.empty_cache()
