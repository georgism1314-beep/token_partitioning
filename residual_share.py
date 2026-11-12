"""在残差流层面看 token-id 解释了多少方差，以及每层 attention 把多少权重放在当前 token 自己身上。

Olmo-3 的 decoder layer 是后置归一化（attention / FFN 的输入都不做 norm，只对输出做 RMSNorm 再加回残差）：
    h_l     = x_l + a_l,   a_l = RMSNorm(Attn(x_l))      h_l 就是第 l 层 FFN 的输入
    x_{l+1} = h_l + m_l,   m_l = RMSNorm(MLP(h_l))
    x_0     = e_t（当前 token 的嵌入）
所以 h_l = e_t + Σ_{k<l}(a_k + m_k) + a_l。

对每个探测层、每个向量 v ∈ {h_l, a_l, m_l} 以及 e_t，记录（对 token、对 4096 维求和）：
    V     tr Var(v)                         总方差
    eta2  Σ_d Var(E[v_d | token]) / V       token-id 解释的方差占比（ε² 去偏，组 = 样本内最高频 n_groups 个 token + OTHER）
以及每一层 attention 的 self / prev（放在自己、前一个 token 上的权重）和 ent（熵，nats），对 head 与位置 ≥1 平均。

    python residual_share.py --steps 0 1000 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_root", default=str(root / "ckpts_deep"))
parser.add_argument("--steps", type=int, nargs="*", default=list(range(0, 10001, 1000)))
parser.add_argument("--extra_ckpts", nargs="*", default=[])
parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--n_seqs", type=int, default=128)
parser.add_argument("--batch_seqs", type=int, default=2)
parser.add_argument("--n_groups", type=int, default=2000)
parser.add_argument("--out_dir", default=str(root / "runs_residual"))
args = parser.parse_args()
device = "cuda"
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---- 数据：冻结子集里前 n_seqs 条满长（1024）的序列，免去 padding ----
frozen = np.load(args.frozen)
seq_lens = frozen["seq_lens"]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
full = np.where(seq_lens == seq_lens.max())[0][: args.n_seqs]
ids_all = torch.from_numpy(np.stack([frozen["token_ids"][starts[i] : starts[i] + seq_lens[i]] for i in full]))
n_all, L = ids_all.shape
T = n_all * L

# ---- 分组：样本内最高频 n_groups 个 token 各一组，其余为 OTHER ----
counts = np.bincount(ids_all.flatten().numpy())
top = np.argsort(-counts)[: args.n_groups]
top = top[counts[top] > 0]
group_of_token = np.full(len(counts), len(top))
group_of_token[top] = np.arange(len(top))
G = len(top) + 1
group_all = torch.from_numpy(group_of_token[ids_all.numpy()]).to(device)  # [n_all, L]
n = torch.bincount(group_all.flatten(), minlength=G).double()
print(f"T={T:,}  G={G}  OTHER 占比={(n[-1] / T).item():.3f}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    layers = model.model.layers
    n_layers, d = len(layers), model.config.hidden_size

    cap = {}
    hooks = []
    for l, layer in enumerate(layers):
        hooks.append(layer.self_attn.register_forward_hook(lambda m, inp, out, l=l: cap.__setitem__(("w", l), out[1])))
    for l in args.layers:
        hooks.append(layers[l].mlp.register_forward_pre_hook(lambda m, inp, l=l: cap.__setitem__(("h", l), inp[0])))
        hooks.append(layers[l].post_attention_layernorm.register_forward_hook(lambda m, inp, out, l=l: cap.__setitem__(("a", l), out)))
        hooks.append(layers[l].post_feedforward_layernorm.register_forward_hook(lambda m, inp, out, l=l: cap.__setitem__(("m", l), out)))

    # 每个向量的充分统计量（float64）：Σv、Σv²（逐维），以及按 token 组的 Σv
    keys = [("e", 0)] + [(kind, l) for l in args.layers for kind in ("h", "a", "m")]
    stats = {k: {"s1": torch.zeros(d, dtype=torch.float64, device=device),
                 "s2": torch.zeros(d, dtype=torch.float64, device=device),
                 "S": torch.zeros(G, d, dtype=torch.float64, device=device)} for k in keys}
    attn = {k: np.zeros(n_layers) for k in ("self", "prev", "ent")}
    sq_norm_h = {l: 0.0 for l in args.layers}

    with torch.inference_mode():
        for b in range(0, n_all, args.batch_seqs):
            ids = ids_all[b : b + args.batch_seqs].to(device)
            model.model(input_ids=ids, use_cache=False)
            cap[("e", 0)] = model.model.embed_tokens(ids)
            g = group_all[b : b + args.batch_seqs].flatten()
            for k in keys:
                v = cap[k].reshape(-1, d).double()
                st = stats[k]
                st["s1"] += v.sum(0)
                st["s2"] += (v * v).sum(0)
                st["S"].index_add_(0, g, v)
                if k[0] == "h":
                    sq_norm_h[k[1]] += (v * v).sum().item()
            for l in range(n_layers):
                # w: [B, heads, L, L]；位置 0 只能看自己，不计入
                w = cap[("w", l)].float()
                attn["self"][l] += w.diagonal(dim1=-2, dim2=-1)[..., 1:].mean(1).sum().item()
                attn["prev"][l] += w.diagonal(offset=-1, dim1=-2, dim2=-1).mean(1).sum().item()
                attn["ent"][l] += (-(w * (w + 1e-12).log()).sum(-1))[..., 1:].mean(1).sum().item()
            cap.clear()
    for hook in hooks:
        hook.remove()

    result = {k: v / (n_all * (L - 1)) for k, v in attn.items()}
    for k in keys:
        st = stats[k]
        mu = st["s1"] / T
        var = st["s2"] / T - mu**2
        between = (st["S"] ** 2 / n.clamp(min=1)[:, None]).sum(0) / T - mu**2
        eps2 = between - (G - 1) / (T - G) * (var - between)
        result[f"{k[0]}{k[1]}_V"] = var.sum().item()
        result[f"{k[0]}{k[1]}_eta2"] = (eps2.sum() / var.sum()).item()
    for l in args.layers:
        result[f"h{l}_sqnorm"] = sq_norm_h[l] / T
    result["emb_rms"] = model.model.embed_tokens.weight.float().pow(2).mean().sqrt().item()
    result["gain_attn"] = np.array([layer.post_attention_layernorm.weight.float().abs().mean().item() for layer in layers])
    result["gain_ffn"] = np.array([layer.post_feedforward_layernorm.weight.float().abs().mean().item() for layer in layers])
    np.savez(out_dir / f"{name}.npz", **result)

    print(f"[{name}] 嵌入 RMS {result['emb_rms']:.4f}  tr Var(e) {result['e0_V']:.3g}", flush=True)
    for l in args.layers:
        print(f"[{name}] L{l:2d}  E||h||² {result[f'h{l}_sqnorm']:.3g}  trVar h {result[f'h{l}_V']:.3g} a {result[f'a{l}_V']:.3g}"
              f" m {result[f'm{l}_V']:.3g}  e/h {result['e0_V'] / result[f'h{l}_V']:.4f}"
              f"  | token η²  h {result[f'h{l}_eta2']:.3f}  a {result[f'a{l}_eta2']:.3f}  m {result[f'm{l}_eta2']:.3f}"
              f"  | 注意力 自己 {result['self'][l]:.3f} 前一个 {result['prev'][l]:.3f} 熵 {result['ent'][l]:.2f}", flush=True)
    del model, layers, stats
    torch.cuda.empty_cache()
