# 第二部分：完整计算协议

**输入：** 一组模型检查点 \(\{\theta^{(s)}\}_{s\in\mathcal{S}}\)，一个数据集 \(\mathcal{D}\)。

**输出：** 每个检查点、每一层上的一族指标

$$
\{\Delta Q(f),\ U(f),\ \lambda_i,\ \mathrm{spec}(c)\}
$$

本部分不跳步。每个符号在首次出现时给出维度与含义；所有下标的取值范围都明确标注。

---

## 记号总表

### 下标与维度

| 符号            | 类型 / 维度            | 含义                          |
| --------------- | ---------------------- | ----------------------------- |
| \(s\)           | 标量整数               | 训练步数，检查点编号          |
| \(\ell\)        | \(\in\{0,\dots,L-1\}\) | 层下标                        |
| \(N\)           | 标量整数               | 单层 FFN 中间维度             |
| \(V\)           | 标量整数               | 词表大小                      |
| \(\mathcal{D}\) | —                      | 采样得到的语料子集            |
| \(T\)           | 标量整数               | token 位置总数                |
| \(t\)           | \(\in\{1,\dots,T\}\)   | token 位置下标                |
| \(v\)           | \(\in\{1,\dots,V\}\)   | 词表下标                      |
| \(w_t\)         | \(\in\{1,\dots,V\}\)   | 位置 \(t\) 上 token 的词表 id |
| \(i,j\)         | \(\in\{1,\dots,N\}\)   | 神经元下标                    |
| \(g\)           | \(\in\{1,\dots,G_f\}\) | 分组下标                      |
| \(c\)           | \(\in\{1,\dots,K\}\)   | 社群下标                      |
| \(\kappa\)      | 百分比                 | 建图保留边比例                |

### 因素相关

| 符号              | 类型 / 维度                         | 含义                      |
| ----------------- | ----------------------------------- | ------------------------- |
| \(f\)             | \(\{1,\dots,T\}\to\{1,\dots,G_f\}\) | 因素                      |
| \(G_f\)           | 标量整数                            | 因素 \(f\) 的分组总数     |
| \(f_1,\dots,f_Q\) | —                                   | 序贯分解中的因素顺序      |
| \(q\)             | \(\in\{0,\dots,Q\}\)                | 链条级数下标              |
| \(f_{1:q}\)       | 因素                                | 前 \(q\) 个因素的联合取值 |

### 矩阵与标量指标

| 符号                      | 类型 / 维度                       | 含义                    |
| ------------------------- | --------------------------------- | ----------------------- |
| \(A\)                     | \(\{0,1\}^{T\times N}\)           | 二值激活矩阵            |
| \(S^{(f)}\)               | \(\mathbb{R}^{N\times G_f}\)      | 分组激活计数            |
| \(n^{(f)}\)               | \(\mathbb{Z}^{G_f}\)              | 每组位置数              |
| \(C\)                     | \(\mathbb{R}^{N\times N}\)        | \(A^\top A\)            |
| \(C_\parallel(f)\)        | \(\mathbb{R}^{N\times N}_{\ge0}\) | 被因素 \(f\) 解释的分量 |
| \(C_\perp(f)\)            | \(\mathbb{R}^{N\times N}\)        | 残差分量                |
| \(M_q\)                   | \(\mathbb{R}^{N\times N}\)        | 第 \(q\) 级矩阵增量     |
| \(\Pi_\kappa\)            | 矩阵 \(\to\) 矩阵                 | 归一化建图管线          |
| \(Q(\cdot)\)              | 矩阵 \(\to\) 标量                 | Louvain 模块度          |
| \(Q^{\mathrm{floor}}\)    | 标量                              | 管线地板                |
| \(\mathrm{SCORE}_\kappa\) | 矩阵 \(\to\) 标量                 | 超额模块度打分          |
| \(Q^{\mathrm{full}}\)     | 标量                              | 全局超额模块度          |
| \(\Delta Q(f)\)           | 标量                              | 因素 \(f\) 的边际贡献   |
| \(Q_q\)                   | 标量                              | 第 \(q\) 级累积分数     |
| \(\Delta Q_q\)            | 标量                              | 序贯增量                |
| \(\rho_q\)                | 标量                              | 主图份额                |
| \(\lambda_i\)             | \([0,1]\)                         | 神经元词法性            |

---

## 第 0 步：采样固定的分析子集

目的：所有检查点必须在完全相同的 token 上计算，否则跨 \(s\) 的比较没有意义。

从数据集中按领域分层抽样 \(M\) 篇文档，拼接得到 token 序列，总位置数 \(T\)。推荐 \(T\approx10^7\)。

产出并冻结：

- \(w_1,\dots,w_T\)；
- \(\mathrm{doc}(t)\)；
- \(\mathrm{dom}(m)\)；
- 交叉拟合划分 \(\mathcal{H}_A,\mathcal{H}_B\)，按文档切。

采样量依据：

|    \(T\) | 满足 \(n_g\ge30\) 的类型数 | 覆盖率 | 序列数（len=1024） |
| -------: | -------------------------: | -----: | -----------------: |
| \(10^5\) |                        266 |  49.3% |                 98 |
| \(10^6\) |                       2667 |  67.8% |                977 |
| \(10^7\) |                      26676 |  86.2% |               9766 |
| \(10^8\) |                     150000 | 100.0% |              97656 |

---

## 第 1 步：提取激活矩阵

对检查点 \(\theta^{(s)}\)、层 \(\ell\)，把冻结 token 序列喂进去做一次前向。

### 1a. 取 FFN 中间激活

设位置 \(t\) 的输入隐状态为 \(x_t\in\mathbb{R}^d\)。

对 SwiGLU：

$$
h_{t,i}
=
\underbrace{\mathrm{SiLU}(g_i^\top x_t)}_{\text{gate\_proj 第 }i\text{ 行}}
\cdot
\underbrace{(u_i^\top x_t)}_{\text{up\_proj 第 }i\text{ 行}}
$$

其中 \(g_i,u_i\in\mathbb{R}^d\)。

### 1b. 二值化

取每个位置上最大的 \(k\) 个神经元：

$$
a_{t,i}
=
\mathbb{1}[i\in\mathrm{TopK}_k(h_t)]
\in\{0,1\}
$$

记

$$
a_t=(a_{t,1},\dots,a_{t,N})^\top\in\{0,1\}^N
$$

把所有位置堆叠：

$$
A=
\begin{pmatrix}
a_1^\top\\
\vdots\\
a_T^\top
\end{pmatrix}
\in\{0,1\}^{T\times N}
$$

实现时不要真的存 \(A\)。

---

## 第 2 步：定义因素，并流式累加充分统计量

### 2.1 什么是“因素”

$$
f:\{1,\dots,T\}\to\{1,\dots,G_f\}
$$

建议因素：

| 因素                      | 定义                              |      \(G_f\) | 获取方式        |
| ------------------------- | --------------------------------- | -----------: | --------------- |
| \(f_0\) 常数              | \(f_0(t)=1\)                      |            1 | —               |
| \(f_{\mathrm{dom}}\)      | \(\mathrm{dom}(\mathrm{doc}(t))\) |    \(\sim8\) | ground-truth    |
| \(f_{\mathrm{pos}}\)      | 字符类/POS                        |   \(\sim48\) | 规则/POS tagger |
| \(f_{\mathrm{token-id}}\) | \(w_t\)                           | \(\sim8001\) | 免费            |
| $f_{位置}$                | position in input sequence        |              |                 |

### 2.2 长尾截断

取词频最高的 \(G^\star\) 个 token，其余归入 `OTHER`：

$$
f_{\mathrm{tok}}(t)
=
\begin{cases}
\mathrm{rank}(w_t), & \mathrm{rank}(w_t)\le G^\star\\
G^\star+1, & \text{否则}
\end{cases}
$$

### 2.3 流式累加

对每个因素维护：

$$
S^{(f)}\in\mathbb{R}^{N\times G_f},
\qquad
S^{(f)}_{i,g}
=
\sum_{t:f(t)=g}a_{t,i}
$$

$$
n^{(f)}\in\mathbb{Z}^{G_f},
\qquad
n^{(f)}_g
=
|\{t:f(t)=g\}|
$$

同时累加：

$$
C\in\mathbb{R}^{N\times N},
\qquad
C_{ij}
=
\sum_{t=1}^T a_{t,i}a_{t,j}
=
A^\top A
$$

伪代码：

```text
初始化 C ← 0_{N×N}
对每个因素 f:
    S^(f) ← 0_{N×G_f}
    n^(f) ← 0_{G_f}

for 每个 batch:
    前向 → 得到每个位置 t 的 top-k 神经元集合 I_t

    for t in batch:
        C[I_t, I_t] += 1

        for 每个因素 f:
            g ← f(t)
            S^(f)[I_t, g] += 1
            n^(f)[g] += 1
```

---

## 第 3 步：由充分统计量构造分解矩阵

### 3.1 组均值

$$
\bar a^{(f)}_g
=
\frac{1}{n^{(f)}_g}
S^{(f)}_{:,g}
\in\mathbb{R}^N
$$

### 3.2 可解释分量 \(C_\parallel\)

$$
\boxed{
C_\parallel(f)
=
\sum_{g=1}^{G_f}
n^{(f)}_g
\bar a^{(f)}_g
\big(\bar a^{(f)}_g\big)^\top
=
S^{(f)}
\mathrm{diag}\!\big(1/n^{(f)}\big)
\big(S^{(f)}\big)^\top
}
$$

定义 \(P^{(f)}\in\mathbb{R}^{T\times T}\)：

$$
P^{(f)}_{t,t'}
=
\begin{cases}
1/n^{(f)}_{f(t)}, & f(t)=f(t')\\
0, & \text{否则}
\end{cases}
$$

满足

$$
(P^{(f)})^2=P^{(f)},
\qquad
(P^{(f)})^\top=P^{(f)}
$$

因此是正交投影。

于是

$$
A
=
P^{(f)}A
+
(I-P^{(f)})A
$$

且

$$
(P^{(f)}A)^\top((I-P^{(f)})A)=0
$$

最终：

$$
C
=
C_\parallel(f)
+
C_\perp(f)
$$

### 3.3 残差分量

$$
C_\perp(f)=C-C_\parallel(f)
$$

注意：\(C_\perp\) 含负元素，不直接用于 Louvain。

### 3.4 零模型

列独立打散后：

$$
S^{(f),\mathrm{null}}_{i,g}
=
n^{(f)}_g\,p_i,
\qquad
p_i=C_{ii}/T
$$

于是

$$
C^{\mathrm{null}}_\parallel
=
Tpp^\top
$$

与因素无关。

---

## 第 4 步：打分子程序

### 4.1 输入输出

输入：非负对称矩阵 \(W\in\mathbb{R}_{\ge0}^{N\times N}\)、激活率 \(p\)、阈值 \(\kappa\)。

输出：

$$
\mathrm{SCORE}_\kappa(W)
$$

### 4.2 归一化管线 \(\Pi_\kappa\)

1. 去自环：

$$
W_{ii}\leftarrow0
$$

2. 缩放：

$$
\tilde W_{ij}
=
W_{ij}
\cdot
\frac{N(N-1)}
{\sum_{i'\ne j'}W_{i'j'}}
$$

3. 压长尾：

$$
\hat W_{ij}
=
\log(1+\tilde W_{ij})
$$

4. 阈值：

$$
W^\kappa_{ij}
=
\begin{cases}
\hat W_{ij}, & \hat W_{ij}\ge\tau_\kappa\\
0, & \text{否则}
\end{cases}
$$

### 4.3 模块度

$$
Q(W^\kappa,\pi)
=
\frac{1}{2m}
\sum_{i=1}^N
\sum_{j=1}^N
\left[
W^\kappa_{ij}
-
\frac{d_i d_j}{2m}
\right]
\delta(c(i),c(j))
$$

### 4.4 地板

$$
\boxed{
Q^{\mathrm{floor}}
=
Q\big(\Pi_\kappa(Tpp^\top)\big)
}
$$

### 4.5 打分函数

$$
\boxed{
\mathrm{SCORE}_\kappa(W)
=
Q\big(\Pi_\kappa(W)\big)
-
Q^{\mathrm{floor}}
}
$$

### 4.6 两个基本量

$$
Q^{\mathrm{full}}
=
\mathrm{SCORE}_\kappa(C)
$$

$$
\Delta Q(f)
=
\mathrm{SCORE}_\kappa(C_\parallel(f))
$$

---

## 第 5 步：序贯分解

### 5.1 为什么不能各算各的

因素之间相关，独立边际值会重复记账，因此主图必须用序贯增量。

### 5.2 联合因素

固定顺序 \(f_1,\dots,f_Q\)，定义：

$$
\boxed{
f_{1:q}(t)
=
\big(
f_1(t),
f_2(t),
\dots,
f_q(t)
\big)
}
$$

### 5.3 顺序约束

1. 被决定的因素必须在前；
2. 高基数因素太早会造成稀疏格子；
3. 交互项会被分给靠后因素。

### 5.4 主序与备序

主序：

$$
\boxed{
\text{常数}
\prec
\text{位置桶}
\prec
\text{字符类}
\prec
\times\text{token-id}
\prec
\times\text{induction}
\prec
\times\text{POS}
\prec
\times\text{领域}
\prec
\times\text{任务}
}
$$

备序：

$$
\text{常数}
\prec
\text{位置桶}
\prec
\text{领域}
\prec
\times\text{任务}
\prec
\times\text{POS}
\prec
\times\text{induction}
\prec
\times\text{字符类}
\prec
\times\text{token-id}
$$

只采信两个顺序下方向一致的结论。

### 5.5 算法

累积：

$$
C_\parallel(f_{1:q})
=
S^{(f_{1:q})}
\mathrm{diag}\big(1/n^{(f_{1:q})}\big)
\big(S^{(f_{1:q})}\big)^\top
$$

累积分数：

$$
Q_q
=
\mathrm{SCORE}_\kappa\big(C_\parallel(f_{1:q})\big)
$$

序贯增量：

$$
\boxed{
\Delta Q_q
=
Q_q-Q_{q-1}
}
$$

残差：

$$
\Delta Q_\perp
=
Q^{\mathrm{full}}-Q_Q
$$

份额：

$$
\rho_q
=
\frac{\Delta Q_q}{Q^{\mathrm{full}}},
\qquad
\rho_\perp
=
\frac{\Delta Q_\perp}{Q^{\mathrm{full}}}
$$

### 5.6 为什么等于“从上一级残差里再解释”

$$
M_q
\overset{\text{def}}{=}
C_\parallel(f_{1:q})
-
C_\parallel(f_{1:q-1})
$$

由嵌套投影关系可得，该量就是对上一级残差继续投影后新解释出来的部分。

### 5.7 为什么不对 \(M_q\) 直接算 \(Q\)

因为 \(M_q\) 虽然半正定，但含大量负元素，不满足 Louvain 的非负边权前提。

纪律：

$$
\boxed{
Q\text{ 只在累积量上算，减法只在标量上做。}
}
$$

### 5.8 排序依赖与 Shapley

顺序无关稳健性：

$$
\phi_q
=
\frac{1}{Q!}
\sum_{\sigma\in\mathfrak{S}_Q}
\left[
\mathrm{SCORE}(C_\parallel(f_{\sigma,\preceq q}))
-
\mathrm{SCORE}(C_\parallel(f_{\sigma,\prec q}))
\right]
$$

### 5.9 链条深度约束

$$
\boxed{
G_{1:Q}
\le
\frac{T}{n^{\min}},
\qquad
n^{\min}=8
\ \text{（推荐 30）}
}
$$

### 5.10 完整伪代码

```text
输入:
    冻结子集 (w_1..w_T, doc, dom)
    检查点 θ^(s)
    层 ℓ
    阈值档 κ
    因素顺序 f_1,...,f_Q

# ---- 采集 ----
C ← 0_{N×N}

for q in 0..Q:
    S[q] ← 0_{N×G_{1:q}}
    n[q] ← 0_{G_{1:q}}

for t in 1..T:
    I_t ← TopK_k(h_t)

    C[I_t, I_t] += 1

    for q in 0..Q:
        g ← f_{1:q}(t)
        S[q][I_t, g] += 1
        n[q][g] += 1

# ---- 打分 ----
p ← diag(C) / T
Q_floor ← Q(Π_κ(T · p pᵀ))
Q_full  ← Q(Π_κ(C)) − Q_floor

for q in 0..Q:
    C_par ← S[q] · diag(1/n[q]) · S[q]ᵀ
    Q_cum[q] ← Q(Π_κ(C_par)) − Q_floor

# ---- 分解 ----
for q in 1..Q:
    ΔQ[q] ← Q_cum[q] − Q_cum[q−1]

ΔQ_perp ← Q_full − Q_cum[Q]

ρ[q] ← ΔQ[q] / Q_full
ρ_perp ← ΔQ_perp / Q_full
```

### 5.11 跨检查点读数

#### 5.11.1 份额与绝对量

必须同时画：

- \(\rho_q(s)\)：谁主导；
- \(\Delta Q_q(s)\)：绝对解释力是否真的迁移。

#### 5.11.2 主导因素与迁移点

$$
\boxed{
q^\star(s)
=
\arg\max_q\rho_q(s)
}
$$

$$
\boxed{
s^\star_{a\to b}
=
\min
\Big\{
s:
\rho_b(s')>\rho_a(s')
\ \forall s'\ge s
\Big\}
}
$$

#### 5.11.3 平均阶数

$$
\bar o(s)
=
\frac{
\sum_q o_q\,\rho_q(s)
}{
\sum_q\rho_q(s)
}
$$

#### 5.11.4 层 × 训练步相图

以层 \(\ell\) 为纵轴、token 数为横轴，颜色取 \(q^\star\) 或 \(\bar o\)。

#### 5.11.5 误差棒

- Louvain：\(\ge10\) seed，取中位数 + IQR；
- 语料抽样：按文档 bootstrap；
- 趋势：Mann–Kendall 或 Spearman。

#### 5.11.6 四道闸门

1. 主序与备序方向一致；
2. 四个 \(\kappa\) 档方向一致；
3. layer 0 不迁移；
4. \(U(f)\) 给出一致的主导因素与迁移点。

#### 5.11.7 最小可行版本

只做 layer 0 与中间层的 \(\rho_{\mathrm{tok}}(s)\) 两条曲线，链条：

$$
\text{常数}
\prec
\text{字符类}
\prec
\times\text{token-id}
\prec
\times\text{领域}
$$

---

## 第 6 步：神经元级词法性 \(\lambda_i\)

$$
\boxed{
\lambda_i
=
\frac{
[C^{\mathrm{cf}}_\parallel(f_{\mathrm{tok}})]_{ii}
}{
C_{ii}
}
\in[0,1]
}
$$

交叉拟合版本：

$$
[C^{\mathrm{cf}}_\parallel(f)]_{ii}
=
\sum_{g=1}^{G_f}
n^{(f)}_g
\,
\bar a^{(f),A}_{g,i}
\cdot
\bar a^{(f),B}_{g,i}
$$

偏差对比：

| 每组样本数 \(n_g\) | 朴素对角偏差 | 交叉拟合偏差 |
| -----------------: | -----------: | -----------: |
|                  4 |       +31.5% |        -0.3% |
|                  8 |       +15.6% |        -0.1% |
|                 20 |        +6.3% |        -0.0% |
|                 50 |        +2.4% |        -0.1% |