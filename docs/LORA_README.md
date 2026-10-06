# SCP 英中 Wikidot 转换 LoRA（Qwen2.5-7B）

英文 SCP Wikidot 源码 → 中文 SCP Wikidot 源码。基座 `Qwen/Qwen2.5-7B-Instruct`，发布物 `final_adapter/`（LoRA r=16，80.8 MB）。

**它改进的是格式遵从，不是翻译质量。** 作为初稿工具使用，不要直接发布未经人工校对的产出。

| | 基座 | 本模型 |
|---|---|---|
| BLEU | 42.36 | **51.93** |
| chrF | 46.00 | **58.90** |
| ROUGE-L | 71.33 | **80.07** |
| Wikidot 标记保留率 | 0.8239 | **0.9547** |
| 零错误样本比例 | 5.6% | 5.6% |

---

## 1. 语料

| 项 | 值 |
|---|---|
| 来源 | SCP 基金会英文维基 + SCP 中文分部 |
| 爬取时间 | 2026-10-03 |
| 文档数 | 7,467（抓取成功 7,465） |
| 英文 / 中文 | 96.0 M 字符 / 43.3 M 字符 |
| 系列分布 | S1–S10：998 / 998 / 997 / 854 / 807 / 728 / 630 / 541 / 533 / 381 |
| 存储 | SQLite `data/raw/crawl.db`，`items` 表 |

英中两侧都是**中文分部译者实际发布的版本**，因此包含真实的中文维基约定：

- 组件前缀：`:scp-wiki:component:X` → `:scp-wiki-cn:component:X`
- 域名：`scp-wiki.wikidot.com` → `scp-wiki-cn.wikidot.com`
- 图片名展开：`800px-SCP002-new.jpg` → `http://scp-wiki.wdfiles.com/local--files/scp-002/...`
- 专名不译：`Site-120`、`Keter`、角色名

**代价**：这些是粉丝译文，本身带翻译腔和误译（见 §6.2）。

## 2. 段落对齐

```
整篇文档 → split_paragraphs() 按空行切段 → bge-m3 嵌入（CLS + L2）
        → 单调 DP 对齐（非 Vecalign）→ 对齐单元
```

- 允许 6 种操作：`1:1` / `1:2` / `2:1` / `2:2` / `1:0`（删）/ `0:1`（插）
- 合并对取均值嵌入；低于阈值优先判为删/插而非强行配对
- 轻微位置先验防止长距离错配
- 参数：`min_similarity=0.70`、`gap_penalty=0.30`、`position_weight=0.10`

`min_similarity` 经标定：真实 1:1 约 0.98，真实合并约 0.94，**被无关段落污染的合并约 0.686** —— 0.70 恰好排除后者，0.55 会误收。

| 类型 | 数量 | 占比 | 平均相似度 |
|---|---|---|---|
| `1:1` | 331,465 | 86.5% | 0.8744 |
| `2:1` | 14,099 | 3.7% | 0.8233 |
| `1:2` | 5,229 | 1.4% | 0.8234 |
| `2:2` | 7,075 | 1.8% | 0.7979 |
| `1:0` | 19,144 | 5.0% | — 未匹配 |
| `0:1` | 6,189 | 1.6% | — 未匹配 |

合计 **383,201** 个单元，覆盖 7,461 / 7,467 篇，匹配率 **93.4%**，匹配上的平均相似度 **0.8701**（含未匹配的全部为 0.8126）。耗时 223.7 s（RTX 4090）。

## 3. 数据集

**短集**：从 `1:1` 单元生成，270,828 个候选中按文档轮询下采样到 **30,000 条**，覆盖 7,306 篇（每篇 1–5 条）。轮询是为了让文档分布均匀。

**长集**：连续对齐单元贪心打包成窗口。窗口边界落在单元之间，所以 `2:1` 合并对**永不被拆分**。

> **预算是源码 + 译文共享的**：`exact_cost = prompt_overhead + len(源码) + len(译文)`。这一点在推理时同样关键（见 §5.2）。

| | 16k 版 | **8k 版（发布用）** |
|---|---|---|
| 预算 | 16,384 | 8,185 |
| 样本数 | 8,184 | **10,164**（train 9,977 / val 187） |
| token 总量 | 49.2 M | 47.9 M |
| token 分布 | min 213 / p50 4,456 / max 16,333 | min 119 / p50 4,569 / max 8,172 |
| 超预算 | 0 | **0** |
| 整篇剔除 | 12 篇 | **47 篇** |

**为什么用 8,185 而非 8,192**：训练走 `apply_chat_template`，比构建脚本的计数恒定多 7 个 token（ChatML 控制符）。用 8,185 后实测 **9,977 条一条未丢（0.0%）**。

**为什么 47 篇被剔除**：单个对齐单元就超预算。根因是 `split_paragraphs` 按空行切分，而 Wikidot 的 `[[div]]` / `[[include]]` 块可以连续数万字符没有空行 —— `scp-7079` 的最长单"段落"有 92,006 字符（≈111,556 token）。这 47 篇合计约 1.45 M token，占语料 **2.88%**，是已知缺口，未修复。

## 4. 训练

```
GPU          RTX 4090 24 GB
torch        2.5.1+cu121      transformers 5.18.0
peft         0.21.2           trl 1.14.1        bitsandbytes 0.50.2

量化          4-bit NF4 + double quant，计算精度 bf16
LoRA          r=16, alpha=32, dropout=0.05, bias=none, rslora=false
target        q/k/v/o_proj, gate/up/down_proj
可训练参数     40,370,176 / 4,393,342,464 (0.9189%)
优化器        adamw_torch
```

| | 阶段 1（短集） | 阶段 2（长集，发布版） |
|---|---|---|
| 数据 | 30,000 条 | 9,977 条 |
| 上下文 | 4,096 | 8,192 |
| batch | 8 × accum 2 | 1 × accum 16 |
| 步数 | 1,853 | **624** |
| 耗时 | 11,406 s (3.17 h) | **37,253 s (10.35 h)** |
| train_loss | 1.0770 | 0.8670 |
| eval_loss | — | 0.8770 |
| token 准确率 | — | 78.23% |

阶段 2 起点是另一次 16k 长集运行的 `checkpoint-100`（用 `init_adapter` 而非 `resume_from_checkpoint` —— 后者会恢复优化器状态和步数，换数据集就是错的）。

**为什么 batch=1**：8k 上下文下 bs1 占 16.7 GB，bs2 约 28 GB，4090 装不下。
**为什么不用更大 batch**：实测 **bs1 比 bs2 快 39%** —— 动态 padding 会把 bs2 的批内最长样本拉高实际计算的 token 量，而 bs1 零 padding。大 batch 只减少优化步数，不减少 padding 浪费。

**评测轨迹**：

```
第  50 步  eval_loss 0.9051   准确率 77.68%
第 100 步             0.8977           77.83%
第 200 步             0.8891           77.97%
第 300 步             0.8826           78.12%
第 400 步             0.8791           78.17%
第 500 步             0.8770           78.22%   ← 开始平台化
第 624 步             0.8770           78.23%   ← 连续四次完全相同
```

## 5. 评测

### 5.1 设置

从 8k 长集验证集（187 条）中按 token 长度分层抽 18 条（短 ≤2000 / 中 2000–6000 / 长 >6000 各 6 条），贪心解码（可复现），`max_new_tokens=4096`。

指标：sacrebleu BLEU(`tokenize='zh'`)、chrF、ROUGE-L，以及本项目的 **Wikidot 结构验证器** —— 检查 `tag`（`[[include]]`/`[[module]]`/`[[div]]` 名称）、`param`（参数名）、`url`、`inline`（`**` `//` `__` `--` `##` 是否成对）、`code`、`heading`。

> 域名本地化（`scp-wiki.wikidot.com` → `scp-wiki-cn.wikidot.com`）视为**保留**，因为这是中文分部的正确写法，人工译文也如此处理。

### 5.2 结果

| 配置 | BLEU | chrF | ROUGE-L | Wikidot | 零错误率 | 错误/条 |
|---|---|---|---|---|---|---|
| 基座 | 42.36 | 46.00 | 71.33 | 0.8239 | 5.6% | 6.39 |
| 阶段 1 | 51.58 | 56.09 | 77.25 | 0.9117 | 22.2% | 4.33 |
| **阶段 2** | **51.93** | **58.90** | **80.07** | **0.9547** | 5.6% | **2.39** |

分类保留率：

| 配置 | tag | param | url | inline | code | heading |
|---|---|---|---|---|---|---|
| 基座 | 0.6798 | 0.8020 | 0.9246 | 0.9220 | 1.0000 | 1.0000 |
| 阶段 1 | 0.8497 | 0.9003 | 0.9495 | 0.9371 | 1.0000 | 1.0000 |
| **阶段 2** | **0.8812** | **0.9665** | **0.9907** | **0.9796** | 1.0000 | 1.0000 |

错误计数（18 条合计）：

| 配置 | tag_missing | param_missing | inline_missing | inline_unbalanced | url_missing |
|---|---|---|---|---|---|
| 基座 | 42 | 38 | 18 | 8 | 9 |
| 阶段 1 | 24 | 18 | 17 | 11 | 8 |
| **阶段 2** | **19** | **3** | **9** | 11 | **1** |

### 5.3 长文档（scp-643，46,708 字符 / 11,246 token，3 个窗口）

| | Wikidot | 错误 | tag | param | url | 汉字数 |
|---|---|---|---|---|---|---|
| **本模型** | **0.9721** | **3** | **1.0000** | **1.0000** | **1.0000** | 9,927 |
| 人工译文 | 0.9512 | 7 | 0.9787 | 1.0000 | 0.6000 | 10,322 |
| 基座 7B | 0.8744 | 12 | 0.6809 | 0.8947 | 1.0000 | — |

标记保留优于人工译文（人工译文丢了 1 个 `[[include]]` 和 2 个图片链接）。

## 6. 已知问题

### 6.1 中文表达的天花板（最重要）

对照实验：同一篇文档、同一套切窗，只换模型。

```
原文    Well, that's what you get for assuming.
基座    这是你假设的结果。
阶段 2  嗯，这就是你假设的结果。       ← 与本模型几乎相同
人工    你早该想到的。
```

**基座与微调在习语上的错误一致**，说明这是 7B 的容量边界，不是训练方式的问题，也无法通过微调解决。

| 类型 | 例子 |
|---|---|
| 习语直译 | `that's what you get for assuming` → `这就是你假设的结果` |
| 词义选错 | `That the place?` → `那就是地方吗？`（应为"那儿"） |
| 词汇塌缩 | `crates, chests, barrels` → `箱子、箱子、桶` |
| 幻觉 | `Not really.` → `不是真的。`（源文无此意） |
| 语法崩坏 | `恰好瞥见那块牌子上写了什么？`（陈述动词接问号） |
| 指代错误 | `Their eyes. They were dark.` → `它们很黑` |
| 内容遗漏 | 丢掉 `located in a wide alleyway in the Sewer District` |

说明文/技术描述尚可（术语固定），**对话与叙事明显吃力**。

### 6.2 语料本身的质量限制

crawl.db 是粉丝译文，本身带翻译腔（"的"密度 **43.6/千字**），且模型输出为 **37.2/千字** —— 模型比语料还收敛一点。语料中也有误译：

```
原文    Starvation doesn't play favorites, Surratt.
人工    饥饿会影响工作的，Surratt。      ← 误译
阶段 2  饥饿不会偏袒任何人，Surratt。    ← 生硬但正确
```

**语料的绝对水平就是模型的上限。**

### 6.3 其他

- **上下文 8k**，更长的文档必须切窗；切窗基于段落边界且 **`overlap=0`**，跨窗口语句可能被割裂；各窗口独立生成，**长文档术语一致性无法保证**
- **提示词锁定**：模型只见过一种 system prompt，实测加入"要地道自然、避免直译"等指令后输出**逐字不变**，无法通过 prompt 工程调整行为
- **零错误率仅 5.6%**：18 条中只有 1 条完全无标记错误，平均每条 2.39 处
- **`inline_unbalanced` 未改善**：`//` 不配对从基座的 8 次增至 11 次
- **47 篇文档（2.88% 语料）未进入训练**
- **6 篇文档未对齐**：`scp-5490` `6197` `8050` `9418` `9690` `9704`
- **`scp-001` 主动排除**（枢纽页）
- 训练时源码/译文长度比由文档本身决定（该语料中英比约 1 : 0.93）；推理时按 1 : 1 保守分配，因此**推理窗口比训练时的最大源码窗口更小**

## 7. 使用

```bash
# 短文档
python scripts/translate.py input.wikidot \
  --adapter outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter \
  -o output.wikidot --validate

# 长文档（任意长度，自动切窗）
python scripts/translate_long.py input.wikidot \
  --adapter outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter \
  -o output.wikidot --report report.json
```

`translate_long.py` 复用**训练时同一套**切段与打包逻辑，并实测 prompt 开销来切分预算：

```
源码窗口 + 输出上限 + 提示词开销  ≤  模型上下文
  3,978  +   4,096  +     110     =  8,184  ≤  8,192
```

**这个切分不能省。** 若把整个预算都给源码，输出只剩约 200 token 空间会被截断 —— 实测中这会让一篇文档丢掉 7/11 个引用块，表面上却像"模型丢标记"。

**环境变量**（离线环境必须设置，否则 `from_pretrained` 会因网络不可达反复重试超时）：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

**提示词格式必须与训练一致**：

```
system:  你是 SCP 维基的文本类型转换器。英文 SCP Wikidot 文本与中文 SCP Wikidot 文本 是同一篇
         文档的两种类型形式。你的任务是在这两种类型之间转换：Wikidot 标记、组件名、参数名、
         代码块、链接目标与 URL 一律逐字保持不变，只把正文转换为目标类型的语言。

user:    把下面的英文 SCP Wikidot 文本转换为中文 SCP Wikidot 文本：

         {源码}
```

## 8. 复现

```bash
python scripts/fetch_wikidot_docs.py                    # → data/raw/crawl.db
python scripts/setup_bge_m3.py
python scripts/align_scp_paragraphs.py --fresh          # → data/aligned/paragraph_alignments.jsonl

python scripts/build_datasets_from_alignment.py \
  --out-dir data/conversion_8k --only long \
  --max-tokens 8185 --overlap 0 --val-ratio 0.02 --exclude-docs scp-001

python scripts/train.py --config configs/train.convert.stage1.yaml
python scripts/train.py --config configs/train.convert.stage3.yaml

python scripts/eval_adapters.py --split data/conversion_8k/long.val.chat.jsonl \
  --limit 18 --strata 3 --run base \
  --run stage3=outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter
```

## 9. 结论

```
做到的：  Wikidot 标记保留率 0.8239 → 0.9547
          学会不翻译标签名、URL slug、组件名、参数名、等级名
          学会保留角色名、中文分部约定（:scp-wiki-cn: 前缀、域名本地化、图片名展开）
          BLEU 42.36 → 51.93

没做到的：提升中文表达的自然度与准确性（7B 容量限制）
          习语、词义、语感层面的判断
```

**适用**：译者初稿工具，省去标记处理与格式整理。
**不适用**：无人校对直接发布；文学性强的对话段落。

## 10. 许可

语料版权归 SCP 基金会及 SCP 中文分部各原作者所有，遵循 **CC BY-SA 3.0**。本适配器为在既有译文上训练的衍生作品，同样以 **CC BY-SA 3.0** 发布。

基座 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（Apache 2.0）；段落嵌入 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)（MIT）。

*本项目与 SCP 基金会官方无关。使用者需自行确认产出符合所在社区的发布规范。*
