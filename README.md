# SCP 英中 Wikidot 转换

把英文 SCP Wikidot 源码转换为中文 SCP Wikidot 源码。

包含一条完整的数据与训练管线：爬取英中对照语料 → 段落对齐 → 构建训练集 → QLoRA 微调 → 推理与评测。

## 发布物

**模型权重就在本仓库里**（77 MiB）：

```
outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter/
  adapter_model.safetensors   LoRA r=16 权重
  adapter_config.json         架构声明
  README.md                   模型卡（含加载代码与已知问题）
  chat_template.jinja         训练时的 chat 模板
```

基座模型需要另外下载：`Qwen/Qwen2.5-7B-Instruct`。本目录不含 `tokenizer.json`，tokenizer 从基座加载。

**加载方式与提示词格式见模型卡** → [`final_adapter/README.md`](outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter/README.md)

**文档：**

- 技术说明（语料、对齐方法、训练配置、完整评测数据、已知问题）：[`docs/LORA_README.md`](docs/LORA_README.md)
- 面向社区的说明：[`docs/LORA_FORUM_POST.md`](docs/LORA_FORUM_POST.md)

## 一句话结论

模型改进了**格式遵从**（Wikidot 标记保留率 0.8239 → 0.9547），但**没有解决中文表达质量** —— 译文仍带翻译腔，习语与词义选择明显不如大模型。**当作译者初稿工具使用，不要直接发布未经校对的产出。**

## 管线

```
scripts/fetch_wikidot_docs.py            爬取 SCP 英文维基 + 中文分部 → data/raw/crawl.db
        ↓
scripts/setup_bge_m3.py                  准备 bge-m3 嵌入模型
        ↓
scripts/align_scp_paragraphs.py          段落切分 + bge-m3 嵌入 + 单调 DP 对齐
                                         → data/aligned/paragraph_alignments.jsonl
        ↓
scripts/build_datasets_from_alignment.py 按 token 预算把对齐单元打包成窗口
                                         → data/conversion_8k/
        ↓
scripts/train.py                         两阶段 QLoRA 微调
  configs/train.convert.stage1.yaml         · 短集（段落级）
  configs/train.convert.stage3.yaml         · 长集（窗口级）
        ↓
scripts/translate.py                     推理（短文档）
scripts/translate_long.py                推理（任意长度，自动切窗）
        ↓
scripts/eval_adapters.py                 多配置对比评测
scripts/evaluate.py                      常规评测
```

## 快速开始

```bash
# 环境变量（离线环境必须设置，否则 from_pretrained 会因网络不可达反复重试而卡住）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 短文档
./py scripts/translate.py 原文.wikidot \
  --adapter outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter \
  -o 译文.wikidot --validate

# 长文档（自动切窗）
./py scripts/translate_long.py 原文.wikidot \
  --adapter outputs/qwen2.5-7b-scp-convert-stage3-8k/final_adapter \
  -o 译文.wikidot --report report.json
```

## 核心指标

| | 基座 Qwen2.5-7B | 微调后 |
|---|---|---|
| BLEU | 42.36 | **51.93** |
| chrF | 46.00 | **58.90** |
| ROUGE-L | 71.33 | **80.07** |
| Wikidot 标记保留率 | 0.8239 | **0.9547** |

18 条按长度分层的验证样本，贪心解码。详见 [`docs/LORA_README.md`](docs/LORA_README.md) 第 5 节。

## 代码结构

```
src/align/         段落切分、bge-m3 嵌入、单调 DP 对齐、嵌入缓存
src/data/          chat 格式转换、文本清洗
src/inference/     模型加载、推理封装
src/training/      训练配置、数据集与 collator、Trainer 接线
src/evaluation/    指标（BLEU / chrF / ROUGE-L / Wikidot / 长度异常）
src/wikidot/       Wikidot 结构验证器
src/utils/         配置、IO、日志

scripts/           命令行入口（见上方管线图）
configs/           训练与推理配置
tests/             单元测试
docs/              模型说明文档
```

## 测试

```bash
./py tests/test_all.py     # 10 个测试，不需要 GPU
```

## 许可

本作品采用 **Creative Commons Attribution-ShareAlike 3.0 Unported**（CC BY-SA 3.0）许可协议。

- 完整法律文本：[`LICENSE`](LICENSE)
- 中文说明与语料来源：[`LICENSE-NOTICE.md`](LICENSE-NOTICE.md)

语料版权归 SCP 基金会及 SCP 中文分部各原作者所有，遵循 CC BY-SA 3.0。本项目的模型权重为在既有译文上训练的衍生作品，因此沿用同一许可协议。

*本项目与 SCP 基金会官方无关。*
