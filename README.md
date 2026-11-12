# Token Partitioning

在 LLM（主要是 OLMo-3）的 FFN 神经元上，把共激活结构按可解释因素（token-id、位置、前文 token、attention 源等）做分解，并跨训练检查点追踪这些因素能解释多少方差 / 模块度。

详细计算协议见 [`research_plan.md`](research_plan.md)。

## 目录

| 路径 | 内容 |
|------|------|
| `code/` | 主流程：冻结子集与前向统计（`step2`）、分解矩阵（`step3`）、Louvain 打分（`step4`），以及因素方差分解等分析脚本 |
| `cached/` | 缓存指定层原始 FFN 激活（`cache_acts.py`），供后续分析反复读取 |
| `extra/` | Attention 相关额外实验（top-k / 逐 head 缓存与分解）及报告导出 |

大数据目录（`acts/`、`ckpts*`、`runs_*` 等）默认留在仓库根目录，不入库；脚本里的路径也按仓库根解析。

## 环境

建议 Python ≥ 3.10，GPU 上跑前向。先装 CUDA 对应的 PyTorch，再装其余依赖：

```bash
# 按你的 CUDA 版本从 https://pytorch.org 选择命令，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

POS 相关脚本还需 spaCy 英文模型：

```bash
python -m spacy download en_core_web_sm
```

## 典型流程（概要）

1. 拉取 / 准备检查点：`python code/fetch_ckpt.py ...`
2. 冻结分析子集并跑充分统计：`python code/step2.py ...`
3. （可选）缓存连续 FFN 激活：`python cached/cache_acts.py ...`
4. 因素分解与打分：`python code/step3.py ...`、`python code/step4.py ...`，或 `factor_decomp.py` / `token_variance.py` 等
5. Attention 扩展实验见 `extra/`

各脚本顶部的 docstring 有具体参数示例。
