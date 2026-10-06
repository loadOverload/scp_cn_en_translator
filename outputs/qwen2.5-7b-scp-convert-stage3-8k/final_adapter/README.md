---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
language:
  - en
  - zh
license: cc-by-sa-3.0
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - transformers
  - trl
  - scp
  - wikidot
  - translation
---

# SCP 英中 Wikidot 转换 LoRA

把英文 SCP Wikidot 源码转换为中文 SCP Wikidot 源码的 QLoRA 适配器。

**基座**：[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · **权重**：LoRA r=16，77 MB

> **权重文件从 [v1.0 Release](https://github.com/loadOverload/scp_cn_en_translator/releases/tag/v1.0) 下载**（`scp-convert-7b-lora-v1.0.zip`）。本仓库只保留模型卡，二进制不进 git 历史。

> **它改进的是格式遵从，不是翻译质量。** 作为译者初稿工具使用，不要直接发布未经人工校对的产出。详见下方[已知问题](#已知问题)。

## 它做了什么

相对基座模型，它学会了不去翻译 Wikidot 结构：

```
基座 Qwen 会把标记也翻掉：            本适配器：
  [[*user JakdragonX]]                [[*user JakdragonX]]
  → [[*用户 JakdragonX]]

  [[[secure-facility-dossier-site-120|Site-120]]]
  → [[安全设施档案-120|120号站点]]     [[[secure-facility-dossier-site-120|Site-120]]]

  [[footnote]] → [[脚注]]              [[footnote]]

  **Object Class:** --Safe-- Keter     **项目等级：**--Safe-- Keter
  → **对象分类:** --安全-- 隐形

                                       :scp-wiki:X → :scp-wiki-cn:X
                                       scp-wiki.wikidot.com → scp-wiki-cn.wikidot.com
```

## 指标

18 条按长度分层的验证样本，贪心解码：

| | 基座 Qwen2.5-7B | 本适配器 |
|---|---|---|
| BLEU | 42.36 | **51.93** |
| chrF | 46.00 | **58.90** |
| ROUGE-L | 71.33 | **80.07** |
| Wikidot 标记保留率 | 0.8239 | **0.9547** |

标记保留分类（本适配器）：`tag` 0.8812 · `param` 0.9665 · `url` 0.9907 · `inline` 0.9796 · `code` 1.0 · `heading` 1.0

**零错误率仅 5.6%** —— 18 条中只有 1 条完全没有标记错误，平均每条仍有 2.39 处。

## 使用方法

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "path/to/final_adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
```

**提示词必须与训练时一致**（该模型只见过这一种格式，实测改动 prompt 后输出逐字不变）：

```python
SYSTEM = (
    "你是 SCP 维基的文本类型转换器。英文 SCP Wikidot 文本与中文 SCP Wikidot 文本 是同一篇"
    "文档的两种类型形式。你的任务是在这两种类型之间转换：Wikidot 标记、组件名、参数名、"
    "代码块、链接目标与 URL 一律逐字保持不变，只把正文转换为目标类型的语言。"
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"把下面的英文 SCP Wikidot 文本转换为中文 SCP Wikidot 文本：\n\n{source}"},
]
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids.to(model.device), max_new_tokens=4096, do_sample=False, repetition_penalty=1.05)
print(tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

**预算切分**：源码 + 输出 + 提示词开销必须 ≤ 上下文长度。以 8k 上下文为例：

```
3,978（源码） + 4,096（输出上限） + 110（实测提示词开销） = 8,184 ≤ 8,192
```

若把整个预算都给源码，输出会被截断，表面上看起来像"模型丢标记"。

## 训练

**语料**：SCP 英文维基 + SCP 中文分部 **7,467 篇**人工对照（英文 9,600 万字符 / 中文 4,330 万字符）。

```
按空行切段 → bge-m3 嵌入 → 单调 DP 对齐 → 383,201 个段落对齐单元（匹配率 93.4%）
           → 打包成 ≤8k token 的窗口 → 10,164 条训练样本

阶段 1  段落级  30,000 条  4k 上下文   3.2 小时
阶段 2  窗口级   9,977 条  8k 上下文  10.3 小时（单张 RTX 4090）
```

```
量化       4-bit NF4 + double quant，计算精度 bf16
LoRA       r=16, alpha=32, dropout=0.05
target     q/k/v/o_proj, gate/up/down_proj
可训练参数  40,370,176 / 4,393,342,464 (0.9189%)
```

## 已知问题

**① 中文表达受限于 7B 的容量，微调解决不了。**

```
原文     Well, that's what you get for assuming.
基座     这是你假设的结果。
本适配器  嗯，这就是你假设的结果。     ← 与基座几乎相同
人工     你早该想到的。
```

同一个习语，基座和微调都译错，错得几乎一样。这说明它是模型容量边界，不是训练方式的问题。

| 类型 | 例子 |
|---|---|
| 习语直译 | `that's what you get for assuming` → `这就是你假设的结果` |
| 词义选错 | `That the place?` → `那就是地方吗？`（应为"那儿"） |
| 词汇塌缩 | `crates, chests, barrels` → `箱子、箱子、桶` |
| 幻觉 | `Not really.` → `不是真的。`（源文无此意） |
| 语法崩坏 | `恰好瞥见那块牌子上写了什么？`（陈述动词接问号） |
| 指代错误 | `Their eyes. They were dark.` → `它们很黑` |
| 内容遗漏 | 丢掉 `located in a wide alleyway in the Sewer District` |

说明文与技术描述尚可（术语固定），**对话与叙事明显吃力**。

**② 训练语料本身的质量就是上限。** 语料是粉丝译文，带翻译腔（"的"密度 43.6/千字），也有误译：

```
原文     Starvation doesn't play favorites, Surratt.
人工     饥饿会影响工作的，Surratt。      ← 误译
本适配器  饥饿不会偏袒任何人，Surratt。    ← 生硬但正确
```

**③ 其他限制**

- 训练上下文 8k，更长的文档必须切窗；切窗无重叠，跨窗口语句可能被割裂；各窗口独立生成，长文档术语一致性无法保证
- 对提示词变化不敏感（只见过一种 system prompt）
- 47 篇文档（占语料 2.88%）因单个 Wikidot 标记块过长而未进入训练
- `inline_unbalanced`（`//` 不配对）未改善：基座 8 次 → 本适配器 11 次
- 本目录不含 `tokenizer.json`，请从基座加载 tokenizer（`AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")`）

## 适用场景

```
适用    译者初稿 —— 省去标记处理与格式整理，人工专注正文润色
不适用  无人校对直接发布；文学性强的对话段落
```

## 许可

语料版权归 SCP 基金会及 SCP 中文分部各原作者所有，遵循 **CC BY-SA 3.0**。本适配器为在既有译文上训练的衍生作品，同样以 **CC BY-SA 3.0** 发布。

*本项目与 SCP 基金会官方无关。使用者需自行确认产出符合所在社区的发布规范。*
