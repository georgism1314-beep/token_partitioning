#!/usr/bin/env python
"""按字节范围抓取 Olmo-3 某个 checkpoint 的前若干层，拼成一个可直接 from_pretrained 的目录。

第 ell 层的 FFN 激活只依赖 embed_tokens 和 model.layers.0..ell，所以只抓到 --max_layer 为止。
ell=0 时只要 1.14 GiB，而整个 checkpoint 是 14.6 GB；镜像带宽只有几 MiB/s，这个差别决定了
十个 checkpoint 是半小时还是半天。

安全性上只做了一件事：下来的张量必须和 safetensors 头里声明的 shape / dtype 完全对上，
否则直接报错，免得把截断的权重当成"早期 checkpoint 的特征"去解释。

    python fetch_ckpt.py --steps 1000 2000 --out_dir ckpts
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "allenai/Olmo-3-1025-7B"
UA = {"User-Agent": "python-requests/2.31"}
# bf16 以外的类型这里不会出现，但留着以免静默误读。
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def get(url: str, rng: tuple[int, int] | None = None, retries: int = 8) -> bytes:
    """镜像会对并发请求回 429，所以退避是必须的而不是可选的。"""
    headers = dict(UA)
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
                return r.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            code = getattr(exc, "code", None)
            time.sleep(min(60.0, 2.0**attempt) if code == 429 else 1.0 + attempt)
    raise AssertionError("unreachable")


def read_header(url: str) -> dict:
    """safetensors 布局：8 字节小端头长度，随后是该长度的 JSON 头。"""
    n = int.from_bytes(get(url, (0, 7)), "little")
    return json.loads(get(url, (8, 8 + n - 1))), 8 + n


def fetch_tensors(base: str, shard: str, names: list[str], workers: int) -> dict[str, torch.Tensor]:
    url = f"{base}/{shard}"
    header, data_start = read_header(url)
    missing = [n for n in names if n not in header]
    if missing:
        raise RuntimeError(f"{shard} 缺少张量: {missing}")

    def one(name: str) -> tuple[str, torch.Tensor]:
        spec = header[name]
        lo, hi = spec["data_offsets"]
        raw = get(url, (data_start + lo, data_start + hi - 1))
        if len(raw) != hi - lo:
            raise RuntimeError(f"{name}: 收到 {len(raw)} 字节，头里声明 {hi - lo}")
        dtype = DTYPES[spec["dtype"]]
        t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(spec["shape"])
        return name, t

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, names))


def fetch_step(step: int, out_dir: Path, base_model: Path, workers: int, max_layer: int) -> Path:
    dest = out_dir / f"stage1-step{step}"
    if (dest / "model.safetensors").exists():
        print(f"[step{step}] 已存在，跳过", flush=True)
        return dest
    base = f"https://hf-mirror.com/{REPO}/resolve/stage1-step{step}"
    weight_map = json.loads(get(f"{base}/model.safetensors.index.json"))["weight_map"]

    layer_prefixes = tuple(f"model.layers.{i}." for i in range(max_layer + 1))
    wanted = [k for k in weight_map if k.startswith("model.embed_tokens") or k.startswith(layer_prefixes)]
    wanted.append("model.norm.weight")  # 只有 4096 个数，但 from_pretrained 要它
    by_shard: dict[str, list[str]] = {}
    for k in wanted:
        by_shard.setdefault(weight_map[k], []).append(k)

    tensors: dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        tensors.update(fetch_tensors(base, shard, names, workers))
    got = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"[step{step}] 取到 {len(tensors)} 个张量，{got / 2**30:.2f} GiB", flush=True)

    # 架构在各 revision 之间不变，所以照抄本地那份已验证可用的 config，只截断层数。
    cfg = json.loads((base_model / "config.json").read_text())
    cfg["num_hidden_layers"] = max_layer + 1
    if isinstance(cfg.get("layer_types"), list):
        cfg["layer_types"] = cfg["layer_types"][: max_layer + 1]
    cfg["tie_word_embeddings"] = True  # 避免去下 820 MB 的 lm_head，它从不参与计算
    cfg["_fetched_revision"] = f"stage1-step{step}"

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        src = base_model / name
        if src.exists():
            shutil.copy2(src, dest / name)
    # 先写临时文件再改名：下游脚本以 model.safetensors 是否存在判断下载完成，不能让它读到半个文件。
    tmp = dest / "model.safetensors.tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    tmp.replace(dest / "model.safetensors")
    return dest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--out_dir", default="ckpts")
    p.add_argument("--base_model", default="/home/test/testdata/models/Olmo-3-1025-7B")
    p.add_argument("--workers", type=int, default=8, help="并发的 range 请求数，单流带宽很低")
    p.add_argument("--jobs", type=int, default=2, help="同时抓几个 checkpoint")
    p.add_argument("--max_layer", type=int, default=0, help="抓 model.layers.0..max_layer")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    base_model = Path(args.base_model)

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(fetch_step, s, out_dir, base_model, args.workers, args.max_layer): s for s in args.steps}
        for fut in cf.as_completed(futures):
            step = futures[fut]
            try:
                print(f"[step{step}] -> {fut.result()}", flush=True)
            except Exception as exc:
                print(f"[step{step}] 失败: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
