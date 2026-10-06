# SCP 英中 Wikidot 转换模型（Qwen2.5-7B LoRA）

英文 SCP Wikidot 源码 → 中文 SCP Wikidot 源码。基座 `Qwen/Qwen2.5-7B-Instruct`，LoRA r=16，80.8 MB。

**它只解决了格式问题，没解决翻译质量问题。当初稿工具用，别直接发。**

## 对比原版 Qwen

```
原版 Qwen 会把标记也翻译掉：
  [[*user JakdragonX]]                →  [[*用户 JakdragonX]]
  [[[secure-facility-dossier-site-120|Site-120]]]
                                      →  [[安全设施档案-120|120号站点]]
  [[footnote]]                        →  [[脚注]]
  **Object Class:** --Safe-- Keter    →  **对象分类:** --安全-- 隐形

它保留：
  [[*user JakdragonX]]
  [[[secure-facility-dossier-site-120|Site-120]]]
  [[footnote]]
  **项目等级：**--Safe-- Keter
  :scp-wiki:component:X  →  :scp-wiki-cn:component:X
  scp-wiki.wikidot.com/X →  scp-wiki-cn.wikidot.com/X
```

## 但译文本身有翻译腔

```
原文    Well, that's what you get for assuming.
它译    嗯，这就是你假设的结果。        应为：你早该想到的。

原文    That the place?
它译    那就是地方吗？                 应为：就是那儿？

原文    many crates, chests, barrels
它译    许多箱子、箱子、桶              crates 和 chests 都成了"箱子"

原文    Not really. I wasn't trying to get smashed by it.
它译    不是真的。我没打算让它掉下来砸到我。   "不是真的"是它编的

原文    They were dark.（说眼睛）
它译    它们很黑。                      用"它们"指人的眼睛
```

对话和文学性段落最明显。

## 数据（18 条分层验证样本）

| | BLEU | chrF | 标记保留率 |
|---|---|---|---|
| 原版 Qwen2.5-7B | 42.36 | 46.00 | 0.8239 |
| **本模型** | **51.93** | **58.90** | **0.9547** |

标记保留细项：`tag` 0.68→0.88、`param` 0.80→0.97、`url` 0.92→0.99、`inline` 0.92→0.98。

**零错误率只有 5.6%**（18 条里 1 条），平均每条还有 2.39 处问题。

长文档实测（scp-643，46,708 字符）：标记保留 **0.9721**，比中文分部现有译文的 0.9512 还干净（现有译文丢了 1 个 `[[include]]` 和 2 个图片链接）。

## 怎么用

```bash
python scripts/translate.py 原文.wikidot --adapter final_adapter -o 译文.wikidot --validate
python scripts/translate_long.py 原文.wikidot --adapter final_adapter -o 译文.wikidot --report r.json
```

`--validate` 会逐项报告标记是否完整。推荐流程：**模型出初稿 → 看验证报告 → 人工只改正文**。

## 训练

语料：SCP 英文维基 + 中文分部 **7,467 篇**对照（英文 9,600 万字符 / 中文 4,330 万）。

```
按空行切段 → bge-m3 嵌入 → 单调 DP 对齐 → 383,201 个段落对齐单元（匹配率 93.4%）
           → 打包成 ≤8k token 的窗口 → 10,164 条训练样本

阶段 1  段落级  30,000 条  4k 上下文   3.2 小时
阶段 2  窗口级   9,977 条  8k 上下文  10.3 小时（RTX 4090）
```

## 为什么还是像机翻

**① 7B 的容量上限，不是训练没做好。** 拿未微调的原版跑同一篇：

```
原版    这是你假设的结果。
微调后  嗯，这就是你假设的结果。
```

同一个习语，两个都错，错得几乎一样。

**② 语料本身的质量就是天花板。** crawl.db 里是粉丝译文，本身带翻译腔（"的"密度 43.6/千字，模型输出 37.2/千字 —— 模型比语料还收敛）。语料里也有误译：

```
原文    Starvation doesn't play favorites.
现有译文 饥饿会影响工作的。         ← 译错
模型    饥饿不会偏袒任何人。        ← 生硬但正确
```

**③ 其他**：8k 上下文，长文要切窗；只见过一种提示词格式，**改 prompt 无效**；47 篇文档（2.88%）因单个标记块过长未进入训练。

## 翻译者重点检查清单

1. 习语和口语 —— 会逐词直译
2. 一词多义 —— `place` / `class` / `order` 常选错义项
3. 近义词 —— 几个词会被译成同一个（crates/chests → 箱子）
4. 偶发编造内容
5. 偶发语法不通的句子
6. 指代错误（人的部位用"它们"）
7. 漏译长定语从句
8. `//` 标记不配对（这个指标微调后反而变差：8 次 → 11 次）
9. 长文档术语不一致（各窗口独立翻译）

## 最后

如果想要"丢进去出成品"，用大模型 API 加一份写清规则的提示词，效果比这个 7B 好得多。

如果想要一个**离线、免费、帮你把标记和格式处理干净、你只管改正文**的工具，它有用。

翻译质量的吐槽我完全接受，上面都写了。

**文件**：`final_adapter/`（80.8 MB） · **许可**：CC BY-SA 3.0

*本项目与 SCP 基金会官方无关，语料版权归各原作者所有。*
