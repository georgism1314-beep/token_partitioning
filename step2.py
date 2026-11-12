#!/usr/bin/env python
"""Demo for research_plan.md 第二部分 (steps 0-2) on OLMo-3 + Llama-Nemotron SFT.

What it does, per 数据集 (one pure ``SFT/code`` set and one pure ``SFT/math`` set):

  第 0 步  freeze an analysis subset: tokenize the Nemotron examples once and keep
           w_1..w_T, doc(t), pos(t) and a by-document cross-fit split H_A/H_B.
  第 1 步  one forward of the frozen tokens through checkpoint theta^(s); at layer
           ell rebuild the SwiGLU neuron activations h_t = SiLU(W_g x_t) * (W_u x_t)
           and binarize with a_t = 1[i in TopK_k(h_t)].
  第 2 步  stream the sufficient statistics S^(f), n^(f) (and C = A^T A) for the
           three factors implemented here:
             f_0   常数      G=1, mu_0 = (1/T) sum_t a_t
             f_tok token-id  long-tail truncated (2.2) so every kept group has
                             n_g >= --min_token_count; rare types fall into OTHER,
                             which is what keeps group means off 1/1
             f_pos 位置桶    bucket of the token's position inside its sequence,
                             precomputed in 第 0 步 from the dataset alone

Steps 3+ (C_parallel, Louvain scoring, sequential decomposition) are deliberately
not implemented: this script only produces the frozen subset and the statistics
they consume.

Typical run (needs the HF cache / network for allenai/Olmo-3-1025-7B):

    export HF_HOME=/home/test/test06/.cache/huggingface
    python step2.py --revision stage1-step1000 --layers 0 --max_tokens 200000

Offline math check of the accumulators (tiny random model, no downloads):

    python step2.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.activations import ACT2FN


# --------------------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model_name", default="allenai/Olmo-3-1025-7B", help="Hub id or local checkpoint dir.")
    p.add_argument("--revision", default="stage1-step10000", help="Early-training checkpoint branch; ignored for local dirs.")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--layers", type=int, nargs="+", default=[0], help="Layer indices ell to instrument.")

    p.add_argument("--dataset_name", default="nvidia/Llama-Nemotron-Post-Training-Dataset")
    p.add_argument("--dataset_config", default="SFT")
    p.add_argument("--sets", nargs="+", default=["code", "math"], help="Nemotron SFT splits; one independent set each.")
    p.add_argument(
        "--local_data",
        nargs="+",
        default=[],
        metavar="SET=PATH",
        help="Read a set from a local jsonl/parquet file or directory instead of the Hub, e.g. "
        "code=/home/test/testdata/hf_data/nvidia/Llama-Nemotron-Post-Training-Dataset/SFT/code",
    )
    p.add_argument("--streaming", action="store_true", default=True, help="Stream the dataset (it is ~4M rows).")
    p.add_argument("--no_streaming", dest="streaming", action="store_false")
    p.add_argument("--max_docs", type=int, default=512, help="Documents read per set before tokenization filtering.")
    p.add_argument("--max_tokens", type=int, default=200_000, help="T, the frozen position count per set.")
    p.add_argument("--seq_len", type=int, default=1024, help="Tokenization truncation length.")
    p.add_argument("--min_seq_len", type=int, default=64, help="Drop shorter documents; they distort the position factor.")

    # 多个 k 共享同一次前向：隐状态与 k 无关，二值化发生在之后，所以扫 k 不用重复跑模型。
    p.add_argument("--topk_frac", type=float, nargs="+", default=[0.05],
                   help="k = round(topk_frac * N) when --topk is unset; 可给多个值做 k 扫描.")
    p.add_argument("--topk", type=int, nargs="+", default=None, help="Explicit k for TopK_k(h_t); 可给多个值.")
    p.add_argument("--topk_on", default="raw", choices=["raw", "abs"], help="'raw' is the plan's TopK_k(h_t).")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Tensor cores for the fp32 matmuls.")
    p.add_argument("--frozen_cache", default=None,
                   help="Dir holding frozen_{set}.npz from an earlier run; reuse it so every checkpoint sees identical tokens.")

    p.add_argument("--g_star", type=int, default=8000, help="G*, cap on the number of kept token-id groups (2.2).")
    p.add_argument(
        "--min_token_count",
        type=int,
        default=30,
        help="A vocabulary type needs this many frozen positions to get its own group; the plan's n_g floor.",
    )
    p.add_argument(
        "--pos_bucket_edges",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Left edges of the 位置桶 buckets; the last one is open-ended.",
    )

    p.add_argument("--batch_size", type=int, default=8, help="Sequences per forward.")
    p.add_argument("--row_chunk", type=int, default=2048, help="Token rows per activation/accumulation chunk.")
    p.add_argument("--skip_cooccurrence", action="store_true", help="Do not accumulate C = A^T A (N*N floats).")
    p.add_argument("--out_dir", default="runs", help="Output root, relative to this file unless absolute.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--placebo", action="store_true", default=True,
                   help="额外累加一个与 token-id 同组数、同组大小但随机分组的安慰剂因素作对照")
    p.add_argument("--no_placebo", dest="placebo", action="store_false")
    p.add_argument("--selftest", action="store_true", help="Run the offline accumulator check instead of a real model.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# 第 0 步: frozen analysis subset
# --------------------------------------------------------------------------------------
_SPECIAL = re.compile(r"<\|[^|>]*\|>")


def clean_text(example) -> str:
    """Flatten one Nemotron SFT row to prose.

    ``input`` comes either as a Llama chat-template string or as a ``[{role, content}]`` list
    depending on the dataset revision; ``output`` is always a string and still carries the
    ``<think>`` block, which is exactly the text we want the model to see.
    """
    parts = []
    for key in ("input", "output"):
        val = example.get(key)
        if isinstance(val, str):
            chunks = [val]
        elif isinstance(val, list):
            chunks = [m.get("content", "") for m in val if isinstance(m, dict)]
        else:
            continue
        for chunk in chunks:
            if isinstance(chunk, str) and chunk.strip():
                parts.append(_SPECIAL.sub("", chunk).strip())
    return "\n\n".join(parts)


def _local_shards(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    for pattern in ("*.parquet", "*.jsonl", "*.json"):
        files = sorted(path.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(f"no parquet/jsonl shard under {path}")


def iter_local_rows(path: Path, max_docs: int):
    """Read local shards row by row.

    Deliberately bypasses ``datasets`` here: the Nemotron SFT shards are tens of GB of jsonl and
    the builder would want a writable cache plus a full scan just to hand back the first few
    hundred rows.
    """
    shards = _local_shards(path)
    print(f"[data] {len(shards)} local shard(s), first={shards[0].name}", flush=True)
    seen = 0
    for shard in shards:
        if shard.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    yield row
                    seen += 1
                    if seen >= max_docs:
                        return
        else:
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
                        seen += 1
                        if seen >= max_docs:
                            return


def iter_texts(args, split):
    local = dict(item.split("=", 1) for item in args.local_data)
    if split in local:
        rows = iter_local_rows(Path(local[split]), args.max_docs)
    else:
        from datasets import load_dataset

        ds = load_dataset(args.dataset_name, args.dataset_config, split=split, streaming=args.streaming)
        rows = ds.take(args.max_docs) if args.streaming else ds.select(range(min(args.max_docs, len(ds))))
    for example in rows:
        text = clean_text(example)
        if text:
            yield text


@dataclass
class FrozenSubset:
    """Everything 第 0 步 must freeze, plus the per-sequence view the forward pass needs.

    ``seqs`` and the flat arrays are in the same order, so batch row ``r`` of the forward
    pass is frozen position ``t0 + r`` -- that is what lets 第 2 步 slice the factor arrays
    instead of recomputing them.
    """

    name: str
    seqs: list[np.ndarray]
    token_ids: np.ndarray  # w_t, [T]
    doc_id: np.ndarray  # doc(t), [T]
    pos: np.ndarray  # position inside the sequence, [T]
    doc_fold: np.ndarray  # H_A / H_B assignment per document, 0/1
    vocab_size: int

    @property
    def T(self) -> int:
        return int(self.token_ids.size)


def build_frozen_subset(name, seqs, vocab_size, seed) -> FrozenSubset:
    token_ids = np.concatenate(seqs).astype(np.int64)
    doc_id = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(seqs)])
    pos = np.concatenate([np.arange(len(s), dtype=np.int64) for s in seqs])
    fold = np.random.default_rng(seed).integers(0, 2, size=len(seqs)).astype(np.int8)
    return FrozenSubset(name, seqs, token_ids, doc_id, pos, fold, vocab_size)


def load_frozen_subset(path: Path, name: str, vocab_size: int) -> FrozenSubset:
    """复用已落盘的冻结子集。

    扫 checkpoint 阶梯时，每个 checkpoint 看的必须是**同一批** token（否则分数的变化里混进了
    数据的变化），而数据和 tokenizer 都不随 checkpoint 变，所以分词只该做一次。
    """
    z = np.load(path)
    lens = z["seq_lens"].astype(np.int64)
    seqs = np.split(z["token_ids"].astype(np.int64), np.cumsum(lens)[:-1])
    sub = FrozenSubset(name, seqs, z["token_ids"].astype(np.int64), z["doc_id"].astype(np.int64),
                       z["pos"].astype(np.int64), z["doc_fold"], vocab_size)
    assert sub.T == int(lens.sum()), (sub.T, lens.sum())
    return sub


def tokenize_subset(name, tokenizer, args) -> FrozenSubset:
    seqs, total = [], 0
    # Nemotron 的文档动辄两三万字符，但每篇只取前 seq_len 个 token。先按字符粗切，
    # 省掉把整篇 tokenize 再丢掉 85% 的开销；留足倍数以免切短于 seq_len。
    char_cap = args.seq_len * 16
    for text in iter_texts(args, name):
        ids = tokenizer(text[:char_cap], truncation=True, max_length=args.seq_len, add_special_tokens=True)["input_ids"]
        if len(ids) < args.min_seq_len:
            continue
        if total + len(ids) > args.max_tokens:
            ids = ids[: args.max_tokens - total]
            if len(ids) < args.min_seq_len:
                break
        seqs.append(np.asarray(ids, dtype=np.int64))
        total += len(ids)
        if total >= args.max_tokens:
            break
    if not seqs:
        raise RuntimeError(f"set '{name}': no sequence survived filtering; lower --min_seq_len or raise --max_docs")
    return build_frozen_subset(name, seqs, len(tokenizer), args.seed)


# --------------------------------------------------------------------------------------
# 第 2.1/2.2 步: factors
# --------------------------------------------------------------------------------------
@dataclass
class Factor:
    name: str
    n_groups: int
    group_of_t: np.ndarray  # f(t), [T]
    labels: np.ndarray | None  # group -> vocabulary id / bucket edge, for later interpretation


def factor_const(subset) -> Factor:
    return Factor("const", 1, np.zeros(subset.T, dtype=np.int64), None)


def factor_token_id(subset, g_star, min_count) -> Factor:
    """f_tok with the 2.2 long-tail truncation.

    Keeping a type only when it occurs >= ``min_count`` times is the whole point: a type seen
    once would give a group mean of a single binary vector (the 1/1 degeneracy), which then
    looks maximally "explained" by token identity for free.
    """
    counts = np.bincount(subset.token_ids, minlength=subset.vocab_size)
    eligible = np.flatnonzero(counts >= min_count)
    eligible = eligible[np.argsort(-counts[eligible], kind="stable")][:g_star]
    lut = np.full(subset.vocab_size, len(eligible), dtype=np.int64)  # default group = OTHER
    lut[eligible] = np.arange(len(eligible))
    labels = np.concatenate([eligible, np.array([-1])])  # -1 marks OTHER
    return Factor("token_id", len(eligible) + 1, lut[subset.token_ids], labels)


def factor_pos_bucket(subset, edges) -> Factor:
    """f_pos, computed from the frozen positions only -- no activation ever enters it.

    Buckets past the longest sequence get dropped here rather than later: an empty group has
    n_g = 0 and would make 第 3 步's 1/n^(f)_g rescaling divide by zero.
    """
    edges = np.asarray(sorted(set(int(e) for e in edges)), dtype=np.int64)
    raw = np.searchsorted(edges, subset.pos, side="right") - 1
    used = np.flatnonzero(np.bincount(raw, minlength=len(edges)) > 0)
    lut = np.full(len(edges), -1, dtype=np.int64)
    lut[used] = np.arange(len(used))
    return Factor("pos_bucket", len(used), lut[raw], edges[used])


def factor_placebo(like: Factor, seed: int) -> Factor:
    """安慰剂对照：组数与组大小分布都复制 ``like``，但分组与 token 内容完全无关。

    因素的组数 G_f 越大，组均值就有越多自由度去拟合噪声，C_par(f) 的 Delta_Q 会被
    这种过拟合抬高。于是不同 G_f 的因素之间直接比 Delta_Q 是不公平的。这个对照给出
    "纯靠组数能白拿多少分"的基线：它的 Delta_Q 应当接近 0，否则跨因素比较不成立。
    """
    perm = np.random.default_rng(seed).permutation(like.group_of_t)
    return Factor("placebo", like.n_groups, perm, None)


def build_factors(subset, args) -> list[Factor]:
    tok = factor_token_id(subset, args.g_star, args.min_token_count)
    factors = [factor_const(subset), tok, factor_pos_bucket(subset, args.pos_bucket_edges)]
    if args.placebo:
        factors.append(factor_placebo(tok, args.seed))
    return factors


# --------------------------------------------------------------------------------------
# 第 2.3 步: streaming sufficient statistics
# --------------------------------------------------------------------------------------
class FactorStats:
    """S^(f), n^(f) for every factor, plus C = A^T A, updated one token chunk at a time.

    除了整份 S^(f)，还按交叉拟合折 H_A / H_B 各累一份。朴素的 C_par 在外积两侧用了同一份
    组均值，估计噪声自己乘自己，产生系统性正偏差（量级约 (G_f/T)(1-q)/q，组数越多越严重）。
    有了 S^A 与 S^B，第 3 步就能用两折独立的组均值做外积把这个偏差消掉。
    """

    def __init__(self, n_neurons, factors, device, group_t, fold_t, with_cooccurrence=True):
        self.N = n_neurons
        self.factors = factors
        # S 必须用整型累加。float32 下 index_add_ 对重复下标走逐元素 atomicAdd，一旦某个
        # S[i, g] 超过 2^24，ulp 变成 2，再加 1.0 就被舍回原值，计数器彻底卡住——T=2e7 时
        # 常数因素（n_g = T）正好撞上这个坑。int32 到 2^31 都是精确的。
        z = lambda G: torch.zeros(n_neurons, G, dtype=torch.int32, device=device)
        self.S = [z(f.n_groups) for f in factors]
        self.SA = [z(f.n_groups) for f in factors]
        self.n = [torch.zeros(f.n_groups, dtype=torch.long, device=device) for f in factors]
        self.nA = [torch.zeros(f.n_groups, dtype=torch.long, device=device) for f in factors]
        # f(t) 与折归属与 k 无关，由调用方建一次后在所有 (layer, k) 之间共享：T=2e7 时
        # 每份是 8 字节 x T x 因素数，重复持有会白占几个 GB。
        self.g = group_t
        self.fold = fold_t
        # C 同样用整型累加。float32 下 diag(C) 在 T=2e7 时达 1.9e7 > 2^24，逐 chunk 相加每次
        # 有 <=1 的舍入，累计约 30 个计数——小到不影响打分，但足以破坏 diag(C) == S^(const)
        # 这个恒等式，让常数因素的 z 残差从 0 变成约 0.013 而不再被判为退化。
        self.C = torch.zeros(n_neurons, n_neurons, dtype=torch.int32, device=device) if with_cooccurrence else None
        self.T = 0

    def update(self, A: torch.Tensor, t0: int) -> None:
        """``A``: [B, N] binary rows a_t for frozen positions ``t0 .. t0 + B - 1``."""
        B = A.shape[0]
        if self.C is not None:
            # 每个 chunk 的计数 <= row_chunk，远小于 2^24，所以 fp32(TF32) 矩阵乘的结果是精确整数，
            # 转 int32 无损；累加也就无损。
            self.C += (A.t() @ A).to(torch.int32)
        At = A.t().to(torch.int32).contiguous()  # [N, B], one column per token
        in_a = self.fold[t0 : t0 + B] == 0
        col_a = At[:, in_a]
        for i, g in enumerate(self.g):
            gb = g[t0 : t0 + B]
            ga = gb[in_a]
            # 只累全量与 H_A 折：H_B 折等于两者之差，省掉三分之一的 index_add。
            self.S[i].index_add_(1, gb, At)
            self.SA[i].index_add_(1, ga, col_a)
            G = self.n[i].numel()
            self.n[i] += torch.bincount(gb, minlength=G)
            self.nA[i] += torch.bincount(ga, minlength=G)
        self.T += B

    def to_numpy(self) -> dict:
        out = {"T": self.T}
        for i, f in enumerate(self.factors):
            triples = (("", self.S[i], self.n[i]), ("A", self.SA[i], self.nA[i]),
                       ("B", self.S[i] - self.SA[i], self.n[i] - self.nA[i]))
            for tag, S, n in triples:
                out[f"S{tag}_{f.name}"] = S.cpu().numpy()
                out[f"n{tag}_{f.name}"] = n.cpu().numpy()
            if f.labels is not None:
                out[f"labels_{f.name}"] = f.labels
        out["mu0"] = (self.S[0][:, 0].double() / max(self.T, 1)).cpu().numpy()
        if self.C is not None:
            C = self.C.cpu().numpy()
            out["C"] = C
            out["p"] = np.diag(C) / max(self.T, 1)
        return out


# --------------------------------------------------------------------------------------
# 第 1 步: model plumbing
# --------------------------------------------------------------------------------------
def locate_trunk(model):
    for attr in ("model", "gpt_neox", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "layers"):
            return sub
    raise RuntimeError("Cannot locate the decoder stack for this architecture.")


def truncate_layers(model, keep_upto):
    """Drop layers above ``keep_upto``: layer k's FFN input never depends on them."""
    trunk = locate_trunk(model)
    if keep_upto + 1 >= len(trunk.layers):
        return 0
    dropped = len(trunk.layers) - (keep_upto + 1)
    trunk.layers = torch.nn.ModuleList(list(trunk.layers)[: keep_upto + 1])
    for field in ("num_hidden_layers", "num_layers"):
        if isinstance(getattr(model.config, field, None), int):
            setattr(model.config, field, keep_upto + 1)
    return dropped


def pick_ffn_pair(layer):
    """Return ``(up_proj, gate_proj)``; both read the same input, so one hook is enough."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise RuntimeError("Decoder layer has no .mlp")
    up = next((getattr(mlp, n) for n in ("up_proj", "w3", "wi_1") if isinstance(getattr(mlp, n, None), torch.nn.Linear)), None)
    gate = next((getattr(mlp, n) for n in ("gate_proj", "w1", "wi_0") if isinstance(getattr(mlp, n, None), torch.nn.Linear)), None)
    if up is None or gate is None:
        raise RuntimeError("This demo expects a gated (SwiGLU) MLP with up_proj and gate_proj.")
    return up, gate


def topk_order(h, k_max, mode):
    """TopK_k 的下标，按 h 降序。小 k 的 TopK 是大 k 的前缀，所以整轮 k 只排一次序。"""
    score = h.abs() if mode == "abs" else h
    return score.topk(k_max, dim=1).indices


@torch.no_grad()
def run_forward(model, subset, layers, args, ks):
    """Stream 第 1 步 + 第 2 步 over the frozen subset, returning one FactorStats per layer."""
    device = torch.device(args.device)
    trunk = locate_trunk(model)
    pad_id = getattr(model.config, "pad_token_id", None) or 0

    captured, hooks, weights = {}, [], {}
    for ell in layers:
        up, gate = pick_ffn_pair(trunk.layers[ell])
        weights[ell] = (
            gate.weight.detach().to(device=device, dtype=torch.float32).t().contiguous(),
            up.weight.detach().to(device=device, dtype=torch.float32).t().contiguous(),
        )
        hooks.append(up.register_forward_hook(lambda _m, inp, _o, ell=ell: captured.__setitem__(ell, inp[0].detach())))

    N = weights[layers[0]][0].shape[1]
    act_fn = ACT2FN[getattr(model.config, "hidden_act", "silu")]
    factors = build_factors(subset, args)
    # 折是按文档切的，所以同一篇文章的 token 不会跨折泄漏。
    group_t = [torch.from_numpy(f.group_of_t).to(device) for f in factors]
    fold_t = torch.from_numpy(subset.doc_fold[subset.doc_id].astype(np.int64)).to(device)
    stats = {
        (ell, k): FactorStats(N, factors, device, group_t, fold_t, not args.skip_cooccurrence)
        for ell in layers
        for k in ks
    }

    t0 = 0
    for start in range(0, len(subset.seqs), args.batch_size):
        batch = subset.seqs[start : start + args.batch_size]
        width = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for r, seq in enumerate(batch):
            input_ids[r, : len(seq)] = torch.from_numpy(seq)
            mask[r, : len(seq)] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        captured.clear()
        trunk(input_ids=input_ids, attention_mask=mask, use_cache=False)
        keep = mask.bool()

        for ell in layers:
            # Row-major mask select keeps the sequence-major order of the frozen arrays.
            x = captured[ell][keep].to(torch.float32)
            wg, wu = weights[ell]
            for off in range(0, x.shape[0], args.row_chunk):
                chunk = x[off : off + args.row_chunk]
                h = act_fn(chunk @ wg) * (chunk @ wu)
                idx = topk_order(h, max(ks), args.topk_on)
                A = torch.empty_like(h, dtype=torch.float32)
                for k in ks:
                    # a_t = 1[i in TopK_k(h_t)]；缓冲区复用是安全的，同一条 CUDA 流上
                    # update() 的读取一定排在下一轮 zero_() 之前。
                    A.zero_().scatter_(1, idx[:, :k], 1.0)
                    stats[(ell, k)].update(A, t0 + off)
        t0 += int(keep.sum())
        captured.clear()

    for h in hooks:
        h.remove()
    return stats, factors, N


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {} if os.path.isdir(args.model_name) else {"revision": args.revision}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=getattr(torch, args.dtype), **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    truncate_layers(model, max(args.layers))
    return tokenizer, model.to(args.device).eval()


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def diagnose(subset, factors, stats, k) -> dict:
    """The 第 0 / 第 2 步 sanity numbers: group occupancy and the mu_0 == p identity."""
    d = {"set": subset.name, "T": subset.T, "num_sequences": len(subset.seqs), "k": k}
    for f, n in zip(factors, [s.cpu().numpy() for s in stats.n]):
        entry = {"G": f.n_groups, "n_min": int(n.min()), "n_max": int(n.max()), "n_sum": int(n.sum())}
        if f.name == "token_id":
            kept = n[:-1]
            entry.update(
                kept_types=int(kept.size),
                other_positions=int(n[-1]),
                coverage=float(kept.sum() / max(n.sum(), 1)),
                n_min_kept=int(kept.min()) if kept.size else None,
            )
        d[f.name] = entry
    mu0 = (stats.S[0][:, 0].double() / stats.T).cpu().numpy()
    d["mu0_mean"] = float(mu0.mean())
    d["mu0_sum"] = float(mu0.sum())  # == k, since every row of A has exactly k ones
    if stats.C is not None:
        p = (torch.diagonal(stats.C).double() / stats.T).cpu().numpy()
        d["max_abs_mu0_minus_p"] = float(np.abs(mu0 - p).max())
    return d


def check_identities(report: dict, k: int) -> None:
    """Fail the real run, not just --selftest, when an accumulator loses counts.

    mu_0 = (1/T) sum_t a_t is read off S^(const), while p = diag(C)/T is read off the
    co-occurrence matmul; both count the same thing by two paths, so a mismatch means one
    accumulator dropped counts. Likewise sum_i mu_0,i must be exactly k, since every a_t
    has exactly k ones. These used to be reported without being enforced, which let a
    float32 overflow in S ship silently.
    """
    dev = report.get("max_abs_mu0_minus_p")
    if dev is not None and dev > 1e-6:
        raise AssertionError(f"mu0 != diag(C)/T (max dev {dev:.3g}): an accumulator lost counts")
    if abs(report["mu0_sum"] - k) > 1e-3:
        raise AssertionError(f"sum(mu0)={report['mu0_sum']:.6g} != k={k}: an accumulator lost counts")


def process_set(name, tokenizer, model, args, out_dir) -> list[dict]:
    cached = Path(args.frozen_cache) / f"frozen_{name}.npz" if args.frozen_cache else None
    if cached is not None and cached.exists():
        subset = load_frozen_subset(cached, name, len(tokenizer))
        print(f"[{name}] reused frozen subset from {cached}", flush=True)
    else:
        subset = tokenize_subset(name, tokenizer, args)
    print(f"[{subset.name}] frozen subset: T={subset.T} over {len(subset.seqs)} sequences", flush=True)
    np.savez_compressed(
        out_dir / f"frozen_{subset.name}.npz",
        token_ids=subset.token_ids,
        doc_id=subset.doc_id,
        pos=subset.pos,
        doc_fold=subset.doc_fold,
        seq_lens=np.array([len(s) for s in subset.seqs]),
    )

    n_intermediate = int(getattr(model.config, "intermediate_size"))
    ks = args.topk or [max(1, round(frac * n_intermediate)) for frac in args.topk_frac]
    stats, factors, N = run_forward(model, subset, args.layers, args, ks)

    reports = []
    for (ell, k), st in stats.items():
        payload = st.to_numpy()
        payload.update(N=N, k=k, layer=ell, revision=args.revision)
        np.savez_compressed(out_dir / f"stats_{subset.name}_L{ell}_k{k}.npz", **payload)
        report = diagnose(subset, factors, st, k)
        report["layer"] = ell
        reports.append(report)
        check_identities(report, k)
        print(f"[{subset.name}] layer {ell} k={k}: " + json.dumps(report, ensure_ascii=False), flush=True)
    return reports


# --------------------------------------------------------------------------------------
# offline self-test
# --------------------------------------------------------------------------------------
def selftest(args):
    """Validate the accumulators on a tiny random Olmo-3 and synthetic Zipf token streams."""
    from transformers import Olmo3Config, Olmo3ForCausalLM

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vocab, N = 128, 96
    config = Olmo3Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=N,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )
    model = Olmo3ForCausalLM(config).to(args.device).eval()

    # Zipf-ish ids so the long-tail truncation has both frequent and singleton types.
    weights = 1.0 / np.arange(1, vocab)
    seqs = [
        rng.choice(np.arange(1, vocab), size=int(length), p=weights / weights.sum())
        for length in rng.integers(args.min_seq_len, 200, size=24)
    ]
    subset = build_frozen_subset("selftest", seqs, vocab, args.seed)

    args.layers = [0, 1]
    ks = args.topk or [max(1, round(frac * N)) for frac in args.topk_frac]
    stats, factors, got_N = run_forward(model, subset, args.layers, args, ks)
    assert got_N == N, (got_N, N)

    for (ell, k), st in stats.items():
        assert st.T == subset.T, (st.T, subset.T)
        # 两折互补：各自的组计数相加必须等于整份。
        for n, nA in zip(st.n, st.nA):
            assert (n - nA).min() >= 0, "fold split lost positions"
        mu0 = (st.S[0][:, 0].double() / st.T).cpu().numpy()
        p = (torch.diagonal(st.C).double() / st.T).cpu().numpy()
        assert np.allclose(mu0, p, atol=1e-6), np.abs(mu0 - p).max()
        assert abs(mu0.sum() - k) < 1e-4, mu0.sum()  # every a_t has exactly k ones
        for f, S, n in zip(factors, st.S, st.n):
            n_np, S_np = n.cpu().numpy(), S.cpu().numpy()
            assert int(n_np.sum()) == subset.T, (f.name, n_np.sum())
            # int64 accumulation: k * n_g reaches ~1e10 at T=2e7, past int32 range.
            assert np.array_equal(S_np.sum(0, dtype=np.int64), k * n_np), f.name
            if f.name == "token_id":
                assert n_np[:-1].min() >= args.min_token_count, n_np[:-1].min()
                assert n_np[:-1].size < vocab, "long tail was not truncated at all"
        print(f"[selftest] layer {ell}: T={st.T} k={k} " + json.dumps(diagnose(subset, factors, st, k), ensure_ascii=False))
    print("[selftest] OK: mu0 == diag(C)/T, group counts sum to T, token-id tail truncated.")


# --------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    # C = A^T A 的输入恰好是 0/1，TF32 只截断输入尾数而累加仍在 fp32，所以共现计数逐位精确，
    # 却能用上张量核（实测该矩阵乘快 5.5 倍）。h 的两个投影会损失约 1e-3 相对精度，只可能
    # 让极少数卡在第 k 名边界上的神经元易位。
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if args.selftest:
        selftest(args)
        return 0

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    tag = Path(args.model_name).name
    out_dir = out_dir / (tag if os.path.isdir(args.model_name) else f"{tag}_{args.revision}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args)
    N = int(model.config.intermediate_size)
    note = ""
    if not args.skip_cooccurrence:
        note = f", C costs {N * N * 4 / 1024 ** 3:.2f} GiB per layer (--skip_cooccurrence to drop it)"
    print(f"loaded {args.model_name}@{args.revision}: N={N}, layers={args.layers}{note}", flush=True)

    reports = []
    for name in args.sets:
        reports.extend(process_set(name, tokenizer, model, args, out_dir))
    (out_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "reports": reports}, ensure_ascii=False, indent=2)
    )
    print(f"wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
