"""给冻结子集的每个模型 token 打两种标签：字符类（只看 token 字符串）和上下文 POS（spaCy UPOS，对齐到模型 token）。

对齐方式：spaCy 只是中间工具，最终标签落在模型 token 上。
    1. 每条序列 decode 成文本；重新分词得到每个模型 token 的字符区间（id 对不上的少数序列用逐前缀解码的长度兜底）
    2. spaCy 在同一段文本上分词 + 词性标注，得到每个非空白字符所属的 spaCy 词
    3. 模型 token 去掉空白后的字符落在哪些 spaCy 词上：
           没有非空白字符          -> SPACE / NEWLINE              （whitespace）
           正好是一个完整的词      -> 该词的 POS                    （one_to_one）
           是一个词的一部分        -> 该词的 POS（子词继承）         （subword）
           跨多个词且 POS 相同     -> 该 POS                        （multi_same）
           跨多个词且 POS 不同     -> 字符重叠最多的词的 POS         （multi_conflict；upos_multi 里记为 MULTI，用于敏感性分析）
           字符区间为空            -> 继承前一个 token               （zero_width）
输出与冻结子集前缀等长的数组：upos、upos_multi、align、charclass，以及对齐统计和冲突样例。

    python pos_tags.py --n_process 32
"""
import argparse
import string
import time
from pathlib import Path

import numpy as np
import spacy
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/home/test/testdata/models/Olmo-3-1025-7B")
parser.add_argument("--frozen", default=str(root / "frozen_5e6" / "frozen_code.npz"))
parser.add_argument("--max_tokens", type=int, default=2_000_000)
parser.add_argument("--spacy_model", default="en_core_web_sm")
parser.add_argument("--n_process", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_examples", type=int, default=300)
parser.add_argument("--out", default=str(root / "frozen_5e6" / "pos_code.npz"))
args = parser.parse_args()
t0 = time.time()

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X", "SPACE", "NEWLINE", "MULTI"]
ALIGN = ["whitespace", "one_to_one", "subword", "multi_same", "multi_conflict", "zero_width", "unaligned"]
CHARCLASS = ["␣lower", "␣Cap", "␣UPPER", "lower", "Cap", "UPPER", "digit", "ident", "punct_end", "punct_sep",
             "bracket", "quote", "operator", "punct_mixed", "non_ascii", "space", "newline", "mixed"]
pos_index = {p: i for i, p in enumerate(UPOS)}
align_index = {a: i for i, a in enumerate(ALIGN)}

tk = AutoTokenizer.from_pretrained(args.model)
nlp = spacy.load(args.spacy_model, disable=["parser", "ner", "lemmatizer"])

# ---- 数据：与 factor_decomp.py 相同的冻结子集前缀 ----
frozen = np.load(args.frozen)
n_seqs = int(np.searchsorted(np.cumsum(frozen["seq_lens"]), args.max_tokens, side="right"))
seq_lens = frozen["seq_lens"][:n_seqs]
starts = np.concatenate([[0], np.cumsum(seq_lens)[:-1]])
T = int(seq_lens.sum())
token_ids = frozen["token_ids"][:T]
V = int(token_ids.max()) + 1

# ---- 字符类：只由 token 字符串决定（因此是 token-id 的粗化） ----
charclass_of_id = np.zeros(V, dtype=np.int8)
for i in np.unique(token_ids):
    s = tk.decode([int(i)])
    core = s.strip()
    if not core:
        c = "newline" if "\n" in s else "space"
    elif not core.isascii() or "\ufffd" in s:
        c = "non_ascii"
    elif core.isalpha():
        c = "lower" if core[0].islower() else ("UPPER" if core.isupper() and len(core) > 1 else "Cap")
        c = "␣" + c if s[0] == " " else c
    elif core.isdigit():
        c = "digit"
    elif all(ch.isalnum() or ch == "_" for ch in core):
        c = "ident"
    elif all(ch in string.punctuation for ch in core):
        if set(core) <= set(".!?"):
            c = "punct_end"
        elif set(core) <= set(".,;:!?"):
            c = "punct_sep"
        elif set(core) <= set("()[]{}"):
            c = "bracket"
        elif set(core) <= set("\"'`"):
            c = "quote"
        elif set(core) <= set("+-*/=<>^%|&~\\"):
            c = "operator"
        else:
            c = "punct_mixed"
    else:
        c = "mixed"
    charclass_of_id[i] = CHARCLASS.index(c)
charclass = charclass_of_id[token_ids]

# ---- 每个模型 token 在 decode 文本里的字符区间 ----
texts, spans = [], []
fallback_seq = np.zeros(n_seqs, dtype=bool)
for i in range(n_seqs):
    ids = token_ids[starts[i] : starts[i] + seq_lens[i]].tolist()
    txt = tk.decode(ids)
    enc = tk(txt, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == ids:
        span = np.array(enc["offset_mapping"])
    else:
        # 重新分词对不上（少数序列）：用逐前缀解码的字符长度作为每个 token 的结束位置
        ends = np.array([len(tk.decode(ids[: k + 1])) for k in range(len(ids))])
        ends = np.minimum(np.maximum.accumulate(ends), len(txt))
        span = np.stack([np.concatenate([[0], ends[:-1]]), ends], 1)
        fallback_seq[i] = True
    texts.append(txt)
    spans.append(span)
print(f"T={T:,}  序列={n_seqs}  重新分词 id 一致 {n_seqs - fallback_seq.sum()}，用前缀解码兜底 {fallback_seq.sum()}"
      f"  ({time.time() - t0:.0f}s)", flush=True)

# ---- spaCy 标注 + 对齐到模型 token ----
upos = np.zeros(T, dtype=np.int8)
align = np.zeros(T, dtype=np.int8)
examples = []
for i, doc in enumerate(nlp.pipe(texts, n_process=args.n_process, batch_size=args.batch_size)):
    txt, span = texts[i], spans[i]
    words = [w for w in doc if not w.is_space]
    word_pos = [pos_index.get(w.pos_, pos_index["X"]) for w in words]
    word_of_char = np.full(len(txt), -1)
    for j, w in enumerate(words):
        word_of_char[w.idx : w.idx + len(w.text)] = j

    for k, (s, e) in enumerate(span):
        t = starts[i] + k
        chars = [c for c in range(s, e) if not txt[c].isspace()]
        if e <= s:
            upos[t], align[t] = (upos[t - 1] if k > 0 else pos_index["X"]), align_index["zero_width"]
            continue
        if not chars:
            upos[t], align[t] = pos_index["NEWLINE" if "\n" in txt[s:e] else "SPACE"], align_index["whitespace"]
            continue
        ws = word_of_char[chars]
        ws = ws[ws >= 0]
        if len(ws) == 0:
            upos[t], align[t] = pos_index["X"], align_index["unaligned"]
            continue
        uniq, cnt = np.unique(ws, return_counts=True)
        best = uniq[np.argmax(cnt)]  # 字符重叠最多的词（并列取靠前的）
        upos[t] = word_pos[best]
        if len(uniq) == 1:
            whole = words[best].idx == chars[0] and words[best].idx + len(words[best].text) == chars[-1] + 1
            align[t] = align_index["one_to_one" if whole else "subword"]
        elif len({word_pos[w] for w in uniq}) == 1:
            align[t] = align_index["multi_same"]
        else:
            align[t] = align_index["multi_conflict"]
            if len(examples) < args.n_examples:
                examples.append(f"{txt[s:e]!r:24} -> {UPOS[upos[t]]:6} 词: "
                                + "  ".join(f"{words[w].text!r}/{UPOS[word_pos[w]]}" for w in uniq))
    if i % 200 == 0:
        print(f"  标注 {i}/{n_seqs}  ({time.time() - t0:.0f}s)", flush=True)

upos_multi = upos.copy()
upos_multi[align == align_index["multi_conflict"]] = pos_index["MULTI"]

# ---- 统计 ----
print("\n对齐类别（token 占比）：")
for a, name in enumerate(ALIGN):
    print(f"  {name:15} {np.mean(align == a):.4f}")
print(f"  兜底序列里的 token 占比 {np.repeat(fallback_seq, seq_lens).mean():.4f}")
print("\nPOS 分布：")
print("  " + "  ".join(f"{p} {np.mean(upos == j):.3f}" for j, p in enumerate(UPOS) if np.any(upos == j)))
print("字符类分布：")
print("  " + "  ".join(f"{c} {np.mean(charclass == j):.3f}" for j, c in enumerate(CHARCLASS) if np.any(charclass == j)))

# POS 的不确定性有多少是 token-id 解释不了的：H(POS) 与 H(POS | token-id)，单位 bit
joint, joint_counts = np.unique(token_ids.astype(np.int64) * len(UPOS) + upos, return_counts=True)
p_pos = np.bincount(upos, minlength=len(UPOS)) / T
H_pos = -np.sum(p_pos[p_pos > 0] * np.log2(p_pos[p_pos > 0]))
tok_counts = np.bincount(token_ids, minlength=V)
p_cond = joint_counts / tok_counts[joint // len(UPOS)]
H_pos_given_tok = -np.sum(joint_counts / T * np.log2(p_cond))
dominant = np.zeros(V)
np.maximum.at(dominant, joint // len(UPOS), p_cond)
print(f"\nH(POS) = {H_pos:.3f} bit   H(POS | token-id) = {H_pos_given_tok:.3f} bit"
      f"   主导 POS 占比 < 0.9 的 token 位置占比 {np.mean(dominant[token_ids] < 0.9):.3f}")

out = Path(args.out)
np.savez(out, token_ids=token_ids, upos=upos, upos_multi=upos_multi, align=align, charclass=charclass,
         fallback_seq=fallback_seq, upos_names=np.array(UPOS), align_names=np.array(ALIGN),
         charclass_names=np.array(CHARCLASS))
lines = [f"{tk.decode([int(token_ids[t])])!r:20} {CHARCLASS[charclass[t]]:12} {UPOS[upos[t]]:8} {ALIGN[align[t]]}"
         for t in range(200)]
(out.parent / "pos_code_examples.txt").write_text("# 第一条序列前 200 个 token：token / 字符类 / POS / 对齐类别\n"
                                                 + "\n".join(lines) + "\n\n# 跨词且 POS 冲突的 token\n"
                                                 + "\n".join(examples) + "\n")
print("\n冲突样例（前 15 个）：")
print("\n".join(examples[:15]))
print(f"\n完成，用时 {time.time() - t0:.0f}s -> {out}", flush=True)
