"""前向一次，把指定层的原始 FFN 激活 silu(gate_proj(x)) * up_proj(x)（down_proj 的输入）原样写到磁盘，
之后的因素分析直接读缓存，不必再前向。

每个 checkpoint 一个目录 acts/{name}/：
    L{l}.npy   [T, 11008] int16，内容是 bf16 的位模式（numpy 没有 bf16）；读回：torch.from_numpy(x).view(torch.bfloat16)
    meta.npz   token_ids / seq_lens / doc_fold，T 个 token 的行顺序就是 token_ids 的顺序（冻结子集按整条序列截取的前缀）
    done       全部写完才创建，下游以它判断缓存完整
每层 T=2×10⁶ 时约 44 GB。

    python cached/cache_acts.py --steps 0 1000 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parents[1]  # repo root
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_root", default=str(root / "ckpts_deep"))
parser.add_argument("--steps", type=int, nargs="*", default=list(range(0, 10001, 1000)))
parser.add_argument("--extra_ckpts", nargs="*", default=[], help="额外的完整模型目录（如最终基座），缓存以目录名命名")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--batch_seqs", type=int, default=8)
parser.add_argument("--acts_root", default=str(root / "acts"))
args = parser.parse_args()
device = "cuda"

# ---- 数据：冻结子集（Nemotron SFT 已 tokenize）的前缀，按整条序列截取 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
print(f"T={T:,}  序列数={n_seqs}  层={args.layers}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out = Path(args.acts_root) / name
    if (out / "done").exists():
        print(f"[{name}] 已有缓存，跳过", flush=True)
        continue
    while not any(ckpt.glob("model*.safetensors")):
        time.sleep(60)  # 等 fetch_ckpt.py 下完
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
    N = model.config.intermediate_size
    model.model.layers = model.model.layers[: max(args.layers) + 1]

    # down_proj 的输入就是 silu(gate_proj(x)) * up_proj(x)
    acts = {}
    hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
        lambda module, inputs, l=l: acts.__setitem__(l, inputs[0])) for l in args.layers]
    files = {l: np.lib.format.open_memmap(out / f"L{l}.partial.npy", mode="w+", dtype=np.int16, shape=(T, N))
             for l in args.layers}

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
            for l in args.layers:
                files[l][lo:hi] = acts[l][valid].view(torch.int16).cpu().numpy()
    for hook in hooks:
        hook.remove()

    for l in args.layers:
        files[l].flush()
        del files[l]
        (out / f"L{l}.partial.npy").replace(out / f"L{l}.npy")
    np.savez(out / "meta.npz", token_ids=token_ids, seq_lens=seq_lens, doc_fold=frozen["doc_fold"][:n_seqs],
             layers=np.array(args.layers))
    (out / "done").touch()
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out}", flush=True)
    del model, acts
    torch.cuda.empty_cache()
