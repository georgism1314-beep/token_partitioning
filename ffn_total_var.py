"""前向一次，统计每一层 FFN 神经元原始激活 silu(gate_proj(x)) * up_proj(x)（down_proj 的输入）的均值和方差，
看总方差随训练步数、随层数怎么变。数据与 cache_acts.py 相同（冻结子集前缀，T≈2×10⁶），不写激活缓存。

每个 checkpoint 存 runs_ffn_var/{name}.npz：
    mean, var     [L, N]  每层每个神经元在 T 个 token 上的均值、方差（总体方差，除以 T）
    sqmean        [L, N]  E[x²]
总方差 = Σ_n var[l, n]（对神经元求和），也给出每个神经元方差的中位数。

    python ffn_total_var.py --steps 0 10000 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
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
parser.add_argument("--extra_ckpts", nargs="*", default=[])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--batch_seqs", type=int, default=2)
parser.add_argument("--out_dir", default=str(root / "runs_ffn_var"))
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
print(f"T={T:,}  序列数={n_seqs}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()  # 与 cache_acts.py 相同的实现
    n_layers, N = model.config.num_hidden_layers, model.config.intermediate_size

    # 每层逐神经元 Σx、Σx²（float64）；hook 里直接累计，不保留激活
    s1 = torch.zeros(n_layers, N, dtype=torch.float64, device=device)
    s2 = torch.zeros(n_layers, N, dtype=torch.float64, device=device)
    valid = {}
    hooks = []
    for l in range(n_layers):
        def hook(module, inputs, l=l):
            x = inputs[0][valid["mask"]].double()  # 只取有效 token
            s1[l] += x.sum(0)
            s2[l] += (x * x).sum(0)
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))

    with torch.inference_mode():
        for b in range(0, n_seqs, args.batch_seqs):
            lens = seq_lens[b : b + args.batch_seqs]
            ids = torch.zeros(len(lens), int(lens.max()), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                ids[j, :m] = torch.from_numpy(token_ids[s : s + m])
                mask[j, :m] = 1
            valid["mask"] = mask.bool().to(device)
            model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            if b % 256 == 0:
                print(f"[{name}] 序列 {b}/{n_seqs}  ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()

    mean = (s1 / T).cpu().numpy()
    sqmean = (s2 / T).cpu().numpy()
    var = sqmean - mean**2
    np.savez(out_path, name=name, T=T, mean=mean, var=var, sqmean=sqmean)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s  总方差（逐层）: "
          + " ".join(f"L{l}:{var[l].sum():.3g}" for l in range(n_layers)), flush=True)
    del model, s1, s2
    torch.cuda.empty_cache()
