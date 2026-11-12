"""实验 C：用每个 head 的完整 attention 输出（而不是它看向哪个 token）去解释同层 FFN 神经元的激活。

第 l 层 FFN 的输入 h_l = x_l + RMSNorm(o_l)，o_l = Σ_h W_O^h z_h，z_h 是第 h 个 head 的输出（o_proj 的输入，128 维）。
W_O^h z_h 是 z_h 的线性变换，所以对线性回归来说"用 z_h"和"用 head h 实际写进残差的向量"等价；
32 个 head 一起回归（4096 维）等价于用整个 o_l。

z_h 是连续向量，不能分组算 η²，改用交叉拟合的岭回归（折按文档划分，A 折拟合、B 折评估，反之亦然）：
    r2_head[h]      R²(y ~ z_h)                         z_h 单独能解释的方差占比，对应实验 A/B 的 η²(attnc)
    r2_all          R²(y ~ z_1..z_32)
    dtok_head[h]    Δ(z_h | tok)：先减去 token 组均值（tok 固定效应），再用 z_h 回归残差，
                    解释掉的部分占总方差的比例，对应实验 A/B 的 Δ(attnc | tok)
    dtok_all        Δ(z_1..z_32 | tok)
    tok_cf          交叉拟合 η²(tok)，用来和实验 A 的 tok 基线核对
全部由充分统计量算出（每折的 ΣxxT、ΣxyT、Σy²，以及按 token 组的 Σx、Σy），一次前向即可。
线性回归只是下界：FFN 对输入是非线性的。max_h 在 32 个估计里取最大，会有少量正向选择偏差。

    python extra/attn_head_regress.py --steps 0 10000 --extra_ckpts /home/test/testdata/models/Olmo-3-1025-7B
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
parser.add_argument("--extra_ckpts", nargs="*", default=[])
parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 16, 24, 31])
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--batch_seqs", type=int, default=4)
parser.add_argument("--n_groups", type=int, default=8000)
parser.add_argument("--ridges", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    help="λ = ridge × tr(Σ_xx) / 维数；每个值都存，汇总按逐层中位 R² 选最好的")
parser.add_argument("--out_dir", default=str(root / "runs_attn_regress"))
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
token_ids = frozen["token_ids"][:T].astype(np.int64)
fold = np.repeat(frozen["doc_fold"][:n_seqs], seq_lens)

# ---- tok：两折都出现过的最高频 n_groups 个 token 各一组，其余为 OTHER（与实验 A 相同） ----
V = int(token_ids.max()) + 1
counts = np.bincount(token_ids, minlength=V)
counts_A = np.bincount(token_ids[fold == 0], minlength=V)
in_both_folds = (counts_A > 0) & (counts - counts_A > 0)
top = np.argsort(-np.where(in_both_folds, counts, 0))[: args.n_groups]
top = top[in_both_folds[top]]
group_of_token = np.full(V, len(top))
group_of_token[top] = np.arange(len(top))
G = len(top) + 1
tok_t = torch.from_numpy(group_of_token[token_ids]).to(device)
fold_t = torch.from_numpy(fold.astype(np.int64)).to(device)
print(f"T={T:,}  tok 组={G}", flush=True)

ckpts = [(f"step{s}", Path(args.ckpt_root) / f"stage1-step{s}") for s in args.steps]
ckpts += [(Path(p).name, Path(p)) for p in args.extra_ckpts]
for name, ckpt in ckpts:
    out_path = out_dir / f"{name}.npz"
    if out_path.exists():
        print(f"[{name}] 已有结果，跳过", flush=True)
        continue
    while not any(ckpt.glob("model*.safetensors")):
        time.sleep(60)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()  # 与 cache_acts.py 相同的实现
    model.model.layers = model.model.layers[: max(args.layers) + 1]
    H, N = model.config.num_attention_heads, model.config.intermediate_size
    Dh = model.model.layers[0].self_attn.head_dim
    D = H * Dh

    # z：o_proj 的输入（各 head 输出按头拼接）；y：down_proj 的输入 silu(gate)·up，即缓存里的原始 FFN 激活
    cap = {}
    hooks = []
    for l in args.layers:
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
            lambda m, i, l=l: cap.__setitem__(("z", l), i[0])))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: cap.__setitem__(("y", l), i[0])))

    # 每层每折的充分统计量（float64）
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=device)
    st = {(l, f): {"n": 0.0, "sx": zeros(D), "sy": zeros(N), "syy": zeros(N), "XtX": zeros(D, D), "XtY": zeros(D, N),
                   "gn": zeros(G), "gX": zeros(G, D), "gY": zeros(G, N)}
          for l in args.layers for f in (0, 1)}

    with torch.inference_mode():
        for b in range(0, n_seqs, args.batch_seqs):
            lens = seq_lens[b : b + args.batch_seqs]
            ids = torch.zeros(len(lens), int(lens.max()), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (s, m) in enumerate(zip(starts[b : b + args.batch_seqs], lens)):
                ids[j, :m] = torch.from_numpy(token_ids[s : s + m])
                mask[j, :m] = 1
            model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            valid = mask.bool().to(device)
            lo, hi = int(starts[b]), int(starts[b] + lens.sum())
            for l in args.layers:
                X = cap[("z", l)][valid].double()  # [n, D]
                Y = cap[("y", l)][valid].double()  # [n, N]
                for f in (0, 1):
                    sel = fold_t[lo:hi] == f
                    if not sel.any():
                        continue
                    Xf, Yf, gf = X[sel], Y[sel], tok_t[lo:hi][sel]
                    s = st[(l, f)]
                    s["n"] += float(sel.sum().item())
                    s["sx"] += Xf.sum(0)
                    s["sy"] += Yf.sum(0)
                    s["syy"] += (Yf * Yf).sum(0)
                    s["XtX"] += Xf.T @ Xf
                    s["XtY"] += Xf.T @ Yf
                    s["gn"] += torch.bincount(gf, minlength=G).double()
                    s["gX"].index_add_(0, gf, Xf)
                    s["gY"].index_add_(0, gf, Yf)
            cap.clear()
            if b % 256 == 0:
                print(f"[{name}] 序列 {b}/{n_seqs}  ({time.time() - t0:.0f}s)", flush=True)
    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()

    # ---- 汇总：每折的中心化矩阵（全局均值）和组内矩阵（减去 token 组均值），交叉拟合岭回归 ----
    result = {}
    for l in args.layers:
        mats = {}
        for f in (0, 1):
            s = st[(l, f)]
            n, mx, my = s["n"], s["sx"] / s["n"], s["sy"] / s["n"]
            occ = s["gn"] > 0
            gX, gY, gn = s["gX"][occ], s["gY"][occ], s["gn"][occ]
            mats[("marg", f)] = (s["XtX"] - n * torch.outer(mx, mx), s["XtY"] - n * torch.outer(mx, my), s["syy"] - n * my**2)
            mats[("tok", f)] = (s["XtX"] - (gX / gn[:, None]).T @ gX, s["XtY"] - (gX / gn[:, None]).T @ gY,
                                s["syy"] - (gY**2 / gn[:, None]).sum(0))
        Syy = mats[("marg", 0)][2] + mats[("marg", 1)][2]  # 两折各自的总平方和

        # 结果按 ridge 存成 [R, H, N] / [R, N]；另存逐层中位 R² 最好的那个 ridge 下的结果（不带后缀）
        idx = torch.arange(H, device=device)
        for kind in ("marg", "tok"):
            expl_head = zeros(len(args.ridges), H, N)
            expl_all = zeros(len(args.ridges), N)
            for fit, ev in ((0, 1), (1, 0)):
                Cxx, Cxy, _ = mats[(kind, fit)]
                Exx, Exy, Eyy = mats[(kind, ev)]
                Bxx = Cxx.view(H, Dh, H, Dh)[idx, :, idx, :]  # 每个 head 的对角块 [H, Dh, Dh]
                Exx_b = Exx.view(H, Dh, H, Dh)[idx, :, idx, :]
                for r, ridge in enumerate(args.ridges):
                    # 每个 head 单独
                    lam = ridge * Bxx.diagonal(dim1=1, dim2=2).mean(1)
                    beta = torch.linalg.solve(Bxx + lam[:, None, None] * torch.eye(Dh, dtype=torch.float64, device=device),
                                              Cxy.view(H, Dh, N))  # [H, Dh, N]
                    ss_res = Eyy - 2 * (beta * Exy.view(H, Dh, N)).sum(1) + (beta * torch.bmm(Exx_b, beta)).sum(1)
                    expl_head[r] += Eyy - ss_res
                    # 32 个 head 一起
                    lam = ridge * Cxx.diagonal().mean()
                    beta = torch.linalg.solve(Cxx + lam * torch.eye(D, dtype=torch.float64, device=device), Cxy)  # [D, N]
                    ss_res = Eyy - 2 * (beta * Exy).sum(0) + (beta * (Exx @ beta)).sum(0)
                    expl_all[r] += Eyy - ss_res
            key = "r2" if kind == "marg" else "dtok"
            head_r = (expl_head / Syy).cpu().numpy().astype(np.float32)
            all_r = (expl_all / Syy).cpu().numpy().astype(np.float32)
            result[f"L{l}_{key}_head_by_ridge"] = head_r
            result[f"L{l}_{key}_all_by_ridge"] = all_r
            best_r_head = int(np.argmax(np.nanmedian(np.nanmax(head_r, 1), 1)))
            best_r_all = int(np.argmax(np.nanmedian(all_r, 1)))
            result[f"L{l}_{key}_head"] = head_r[best_r_head]
            result[f"L{l}_{key}_all"] = all_r[best_r_all]
            result[f"L{l}_{key}_ridge_head"] = args.ridges[best_r_head]
            result[f"L{l}_{key}_ridge_all"] = args.ridges[best_r_all]

        # 交叉拟合 η²(tok)，与实验 A 的 tok 基线核对
        sA, sB = st[(l, 0)], st[(l, 1)]
        mu = (sA["sy"] + sB["sy"]) / T
        var = (sA["syy"] + sB["syy"]) / T - mu**2
        both = (sA["gn"] > 0) & (sB["gn"] > 0)
        muA = sA["gY"][both] / sA["gn"][both, None]
        muB = sB["gY"][both] / sB["gn"][both, None]
        nG = (sA["gn"] + sB["gn"])[both]
        result[f"L{l}_tok_cf"] = ((nG[:, None] / T * (muA - mu) * (muB - mu)).sum(0) / var).cpu().numpy().astype(np.float32)

        r2h, dth = result[f"L{l}_r2_head"], result[f"L{l}_dtok_head"]
        best = np.nanargmax(np.nan_to_num(dth, nan=-np.inf), axis=0)
        result[f"L{l}_best_head"] = best.astype(np.int16)
        result[f"L{l}_best_head_share"] = np.bincount(best, minlength=H) / N
        order = np.argsort(-result[f"L{l}_best_head_share"])[:3]
        print(f"[{name}] L{l:2d}  η²(tok) 中位 {np.nanmedian(result[f'L{l}_tok_cf']):.4f}"
              f" | R²: 单头逐神经元 max_h 的中位 {np.nanmedian(np.nanmax(r2h, 0)):.4f} 全部头 {np.nanmedian(result[f'L{l}_r2_all']):.4f}"
              f" | Δ(·|tok): 单头 max_h 中位 {np.nanmedian(np.nanmax(dth, 0)):.4f} 全部头 {np.nanmedian(result[f'L{l}_dtok_all']):.4f}"
              f" | 最佳 head 占比前三 " + " ".join(f"h{h}:{result[f'L{l}_best_head_share'][h]:.2f}" for h in order) +
              f" | ridge 单头/全部 {result[f'L{l}_r2_ridge_head']:g}/{result[f'L{l}_r2_ridge_all']:g}", flush=True)
    del st, mats
    torch.cuda.empty_cache()

    np.savez(out_path, name=name, T=T, layers=np.array(args.layers), G=G, ridges=np.array(args.ridges), **result)
    print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s -> {out_path}", flush=True)
