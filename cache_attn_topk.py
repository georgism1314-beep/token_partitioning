"""前向一次，用 hook 抓住指定层 attention 的权重，对每个 query 位置取头平均后的 top-k 前文位置，记下：
    被 attend 到的 token-id、序列内位置、平均注意力权重。
下游可拿 top1 当「注意力源 token」因素，或用 top-k 集合做敏感性分析；不存完整 T×T 矩阵。

每个 checkpoint 一个目录 attn_topk/{name}/：
    L{l}_tok.npy   [T, K] int32   得分第 1..K 高的前文位置上的 token-id；不足 K 的填 -1
    L{l}_pos.npy   [T, K] int32   对应的序列内位置（0..len-1）；不足 K 的填 -1
    L{l}_w.npy     [T, K] float16 头平均后的注意力权重
    meta.npz       token_ids / seq_lens / doc_fold / layers / topk
    done           全部写完才创建
K=50、T=2×10⁶、6 层时每 ckpt 约 2.4 GB（tok+pos）+ 1.2 GB（w）。

权重定义：对该层所有 head 的 attention weights 取平均，再在 key 维上 top-k（含自己）。
必须用 attn_implementation="eager"，否则 hook 拿不到权重矩阵。

    python cache_attn_topk.py --steps 0 10000 --topk 50 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
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
parser.add_argument("--extra_ckpts", nargs="*", default=[], help="额外的完整模型目录（如最终基座）")
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--batch_seqs", type=int, default=2)
parser.add_argument("--topk", type=int, default=50)
parser.add_argument("--attn_root", default=str(root / "attn_topk"))
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
print(f"T={T:,}  序列数={n_seqs}  层={args.layers}  topk={K}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out = Path(args.attn_root) / name
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

    weights = {}
    hooks = [model.model.layers[l].self_attn.register_forward_hook(
        lambda module, inputs, output, l=l: weights.__setitem__(l, output[1])) for l in args.layers]
    files_tok = {l: np.lib.format.open_memmap(out / f"L{l}_tok.partial.npy", mode="w+", dtype=np.int32, shape=(T, K))
                 for l in args.layers}
    files_pos = {l: np.lib.format.open_memmap(out / f"L{l}_pos.partial.npy", mode="w+", dtype=np.int32, shape=(T, K))
                 for l in args.layers}
    files_w = {l: np.lib.format.open_memmap(out / f"L{l}_w.partial.npy", mode="w+", dtype=np.float16, shape=(T, K))
               for l in args.layers}
    # 诊断：top1 是不是自己、top1 token 是不是当前 token
    n_self = {l: 0 for l in args.layers}
    n_same_tok = {l: 0 for l in args.layers}
    n_pos = 0

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

            # 按序列切掉 padding，再对 key 维 top-k；不足 K 的右侧填 -1 / 0
            for l in args.layers:
                a = weights[l]
                assert a is not None, f"层 {l} 没拿到 attention weights（需要 eager）"
                assert a.ndim == 4 and a.shape[0] == len(lens), a.shape
                assert a.shape[1] == model.config.num_attention_heads, a.shape
                assert a.shape[2] == a.shape[3] == S, a.shape
                w = a.float().mean(1)  # [B, S, S]，头平均
                for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                    m = int(m)
                    # m 是 seq_lens 给的真实长度，padding 在右侧，attention_mask 恰好有 m 个 1
                    assert int(mask[j].sum()) == m and int(mask[j, :m].sum()) == m
                    # 只在有效 query × 有效 key 上检查：每个头每行有限、和为 1（即没有整行被 mask），
                    # 右侧 padding 的 key 不分到权重
                    aj = a[j, :, :m].float()  # [H, m, S]
                    assert torch.isfinite(aj[..., :m]).all(), f"层 {l} 有效区域出现非有限权重"
                    row_sum = aj[..., :m].sum(dim=-1)
                    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-2), \
                        f"层 {l} 行和偏离 1：max |Σ-1| = {(row_sum - 1).abs().max().item():.3g}"
                    if m < S:
                        assert aj[..., m:].abs().max().item() <= 1e-6, f"层 {l} padding key 上有权重"
                    # 把未来位置打成 -inf，避免前 t+1 < K 个可见位置时 topk 选到因果掩码外的 0
                    wj = w[j, :m, :m].clone()
                    wj.masked_fill_(torch.triu(torch.ones(m, m, dtype=torch.bool, device=device), diagonal=1),
                                    float("-inf"))
                    k = min(K, m)
                    vals, idx = wj.topk(k, dim=-1)  # [m, k]；前排 query 后几名可能是 -inf
                    valid = torch.isfinite(vals)
                    ids_j = ids[j, :m].to(device)
                    tok = ids_j[idx]
                    buf_tok = torch.full((m, K), -1, dtype=torch.int32, device=device)
                    buf_pos = torch.full((m, K), -1, dtype=torch.int32, device=device)
                    buf_wt = torch.zeros((m, K), dtype=torch.float16, device=device)
                    buf_tok[:, :k] = torch.where(valid, tok, torch.tensor(-1, device=device)).int()
                    buf_pos[:, :k] = torch.where(valid, idx, torch.tensor(-1, device=device)).int()
                    buf_wt[:, :k] = torch.where(valid, vals, torch.zeros_like(vals)).half()
                    # idx 来自 [m, m] 矩阵，总在 [0, m) 内，ids_j[idx] 不会越界；无效项指向的是未来位置，
                    # 这里确认它们最终都被替换成缺失值，且第 t 行恰有 min(K, t+1) 个有效项、全部在 0..t 内
                    q = torch.arange(m, device=device)
                    assert torch.equal(valid.sum(1), torch.clamp(q + 1, max=k))
                    assert (idx[valid] <= q[:, None].expand_as(idx)[valid]).all()
                    assert (buf_tok[:, :k][~valid] == -1).all() and (buf_pos[:, :k][~valid] == -1).all()
                    assert (buf_wt[:, :k][~valid] == 0).all()
                    assert (buf_tok[:, k:] == -1).all() and (buf_pos[:, k:] == -1).all()
                    files_tok[l][s : s + m] = buf_tok.cpu().numpy()
                    files_pos[l][s : s + m] = buf_pos.cpu().numpy()
                    files_w[l][s : s + m] = buf_wt.cpu().numpy()
                    # top1 == 自己？ top1 token == 当前 token？
                    n_self[l] += int((idx[:, 0] == torch.arange(m, device=device)).sum().item())
                    n_same_tok[l] += int((tok[:, 0] == ids_j).sum().item())
                    if l == args.layers[0]:
                        n_pos += m
            weights.clear()
            if b % 64 == 0:
                print(f"[{name}] 序列 {b}/{n_seqs}  ({time.time() - t0:.0f}s)", flush=True)

    for hook in hooks:
        hook.remove()
    for l in args.layers:
        files_tok[l].flush(); files_pos[l].flush(); files_w[l].flush()
        del files_tok[l], files_pos[l], files_w[l]
        (out / f"L{l}_tok.partial.npy").replace(out / f"L{l}_tok.npy")
        (out / f"L{l}_pos.partial.npy").replace(out / f"L{l}_pos.npy")
        (out / f"L{l}_w.partial.npy").replace(out / f"L{l}_w.npy")
        print(f"[{name}] L{l:2d}  top1=自己 {n_self[l]/n_pos:.3f}  top1_tok=当前tok {n_same_tok[l]/n_pos:.3f}",
              flush=True)

    np.savez(out / "meta.npz", token_ids=token_ids, seq_lens=seq_lens, doc_fold=frozen["doc_fold"][:n_seqs],
             layers=np.array(args.layers), topk=K)
    (out / "done").touch()
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out}", flush=True)
    del model, weights
    torch.cuda.empty_cache()
