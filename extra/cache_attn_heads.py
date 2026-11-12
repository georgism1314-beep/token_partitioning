"""前向一次，对指定层的每个 attention head 分别取 top-k 前文位置（不做头平均），供 attn_head_decomp.py 用。
与 cache_attn_topk.py 的区别只在于不对 head 取平均；因果掩码、缺失值约定都相同。

每个 checkpoint 一个目录 attn_heads/{name}/：
    L{l}_pos.npy   [T, H, K] int16    第 h 个 head 得分第 1..K 高的前文位置（序列内 0..len-1）；不足 K 的填 -1
    L{l}_w.npy     [T, H, K] float16  对应的注意力权重；不足 K 的填 0
    meta.npz       token_ids / seq_lens / doc_fold / layers / topk
    done           全部写完才创建
token-id 不单独存：pos 加上序列起点就能从 token_ids 取到。K=10、H=32、6 层时每 ckpt 约 15 GB。

    python extra/cache_attn_heads.py --steps 0 10000 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
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
parser.add_argument("--extra_ckpts", nargs="*", default=[], help="额外的完整模型目录（如最终基座）")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--batch_seqs", type=int, default=2)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--out_root", default=str(root / "attn_heads"))
args = parser.parse_args()
device = "cuda"
K = args.topk

# ---- 数据：与 cache_acts.py 相同的冻结子集前缀 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
assert seq_lens.max() < 2**15, "位置用 int16 存"
print(f"T={T:,}  序列数={n_seqs}  层={args.layers}  每头 topk={K}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out = Path(args.out_root) / name
    if (out / "done").exists():
        print(f"[{name}] 已有缓存，跳过", flush=True)
        continue
    while not any(ckpt.glob("model*.safetensors")):
        time.sleep(60)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    # eager：self_attn 的 forward 返回 (attn_output, attn_weights)，weights 形状 [B, H, S, S]
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    model.model.layers = model.model.layers[: max(args.layers) + 1]
    H = model.config.num_attention_heads

    weights = {}
    hooks = [model.model.layers[l].self_attn.register_forward_hook(
        lambda module, inputs, output, l=l: weights.__setitem__(l, output[1])) for l in args.layers]
    files_pos = {l: np.lib.format.open_memmap(out / f"L{l}_pos.partial.npy", mode="w+", dtype=np.int16, shape=(T, H, K))
                 for l in args.layers}
    files_w = {l: np.lib.format.open_memmap(out / f"L{l}_w.partial.npy", mode="w+", dtype=np.float16, shape=(T, H, K))
               for l in args.layers}

    with torch.inference_mode():
        for b in range(0, n_seqs, args.batch_seqs):
            lens = seq_lens[b : b + args.batch_seqs]
            S = int(lens.max())
            ids = torch.zeros(len(lens), S, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                ids[j, :m] = torch.from_numpy(token_ids[s : s + m])
                mask[j, :m] = 1
            model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)

            for l in args.layers:
                a = weights[l]
                assert a is not None, f"层 {l} 没拿到 attention weights（需要 eager）"
                assert a.ndim == 4 and a.shape[0] == len(lens) and a.shape[1] == H and a.shape[2] == a.shape[3] == S, a.shape
                for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                    m = int(m)
                    assert int(mask[j].sum()) == m
                    # 有效 query × 有效 key：每个头每行有限、和为 1；右侧 padding 的 key 不分到权重
                    aj = a[j, :, :m].float()  # [H, m, S]
                    assert torch.isfinite(aj[..., :m]).all(), f"层 {l} 有效区域出现非有限权重"
                    row_sum = aj[..., :m].sum(dim=-1)
                    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-2), \
                        f"层 {l} 行和偏离 1：max |Σ-1| = {(row_sum - 1).abs().max().item():.3g}"
                    if m < S:
                        assert aj[..., m:].abs().max().item() <= 1e-6, f"层 {l} padding key 上有权重"

                    # 未来位置打成 -inf，前排 query 可见位置不足 K 时 topk 的尾部是 -inf，统一替换成缺失值
                    wj = aj[..., :m].masked_fill(
                        torch.triu(torch.ones(m, m, dtype=torch.bool, device=device), diagonal=1), float("-inf"))
                    k = min(K, m)
                    vals, idx = wj.topk(k, dim=-1)  # [H, m, k]
                    valid = torch.isfinite(vals)
                    q = torch.arange(m, device=device)
                    assert torch.equal(valid.sum(-1), torch.clamp(q + 1, max=k).expand(H, m))
                    buf_pos = torch.full((m, H, K), -1, dtype=torch.int16, device=device)
                    buf_wt = torch.zeros((m, H, K), dtype=torch.float16, device=device)
                    buf_pos[:, :, :k] = torch.where(valid, idx, -1).permute(1, 0, 2).short()
                    buf_wt[:, :, :k] = torch.where(valid, vals, 0.0).permute(1, 0, 2).half()
                    assert (buf_pos.long() <= q[:, None, None]).all()
                    files_pos[l][s : s + m] = buf_pos.cpu().numpy()
                    files_w[l][s : s + m] = buf_wt.cpu().numpy()
            weights.clear()
            if b % 128 == 0:
                print(f"[{name}] 序列 {b}/{n_seqs}  ({time.time() - t0:.0f}s)", flush=True)

    for hook in hooks:
        hook.remove()
    for l in args.layers:
        files_pos[l].flush(); files_w[l].flush()
        del files_pos[l], files_w[l]
        (out / f"L{l}_pos.partial.npy").replace(out / f"L{l}_pos.npy")
        (out / f"L{l}_w.partial.npy").replace(out / f"L{l}_w.npy")
    np.savez(out / "meta.npz", token_ids=token_ids, seq_lens=seq_lens, doc_fold=frozen["doc_fold"][:n_seqs],
             layers=np.array(args.layers), topk=K)
    (out / "done").touch()
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out}", flush=True)
    del model, weights
    torch.cuda.empty_cache()
