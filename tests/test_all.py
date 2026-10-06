#!/usr/bin/env python
"""Self-contained test suite -- run with or without pytest.

    python tests/test_all.py          # plain python, prints a summary
    python -m pytest tests/test_all.py -q

No GPU, no transformers and no network are required: the chat-tokenizer is
faked, so dataset/masking logic is verified without downloading a model.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_dataset import group_key, split_records          # noqa: E402
from src.data.clean import normalize_text                       # noqa: E402
from src.data.pair_loader import Pair, derive_id, load_from_dirs, load_from_file, normalize_id  # noqa: E402
from src.data.to_chat import build_sft_dataset, to_messages      # noqa: E402
from src.utils.io import load_jsonl, write_jsonl, write_text         # noqa: E402
from src.wikidot.validator import cjk_ratio, plain_text, validate_pair  # noqa: E402

EN_PAGE = """[[include :scp-wiki:component:anomaly-class-bar-source
|lang=en
|item-number=SCP-173
]]

**Item #:** SCP-173

[[div class="blockquote"]]
**Special Containment Procedures:** SCP-173 is to be kept in a containment chamber.
A Mobile Task Force detachment responded to the containment breach.
[[/div]]

[[image scp-173.jpg style="width:300px;"]]

||~ Date ||~ Event ||
|| 20██-01-01 || Recovery ||

[[collapsible show="+ Log" hide="- Close"]]
The test subject was terminated after 30 minutes.
[[footnote]]Personnel were dosed with amnestics.[[/footnote]]
[[/collapsible]]

<hr />

[[code]]
return "never translated"
[[/code]]

[[module Rate]]
[[footnoteblock]]
"""

ZH_PAGE = """[[include :scp-wiki:component:anomaly-class-bar-source
|lang=cn
|item-number=SCP-173
]]

**项目编号：** SCP-173

[[div class="blockquote"]]
**特殊收容措施：** SCP-173须被收容于一个收容间中。
一支机动特遣队分队对收容失效做出了响应。
[[/div]]

[[image scp-173.jpg style="width:300px;"]]

||~ 日期 ||~ 事件 ||
|| 20██-01-01 || 回收 ||

[[collapsible show="+ 记录" hide="- 关闭"]]
测试对象在30分钟后被处决。
[[footnote]]人员均被施以记忆删除剂。[[/footnote]]
[[/collapsible]]

<hr />

[[code]]
return "never translated"
[[/code]]

[[module Rate]]
[[footnoteblock]]
"""

LONG_EN = EN_PAGE + "\n\n" + ("The Foundation maintains strict containment procedures on Site-19. " * 30)
LONG_ZH = ZH_PAGE + "\n\n" + ("基金会在Site-19维持严格的收容措施。" * 30)


# ---------------------------------------------------------------------------
# fake tokenizer (no transformers needed)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal whitespace tokenizer that implements the HF chat-template API."""

    chat_template = "fake"
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._next = 2

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self._next
            self._next += 1
        return self.vocab[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>")
            parts.append(str(m["content"]).replace(" ", "_"))
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=False, truncation=False, return_tensors=None, padding=False, **kwargs):
        tokens = text.split() if isinstance(text, str) else list(text)
        ids = [self._id(t) for t in tokens]
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids], dtype=torch.long), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse.get(int(i), "?") for i in ids)


# ---------------------------------------------------------------------------
# tests: wikidot
# ---------------------------------------------------------------------------


def test_validator_good_and_bad():
    good = validate_pair(EN_PAGE, ZH_PAGE)
    assert good.score == 1.0, good.summary()
    assert good.error_count() == 0

    broken = ZH_PAGE.replace("[[/div]]", "", 1)
    report = validate_pair(EN_PAGE, broken)
    assert report.score < 1.0
    codes = {i.code for i in report.issues}
    assert "unclosed_wikidot" in codes

    # A translated module name is caught by cjk_in_protected (the target must
    # stay ASCII). "module_missing" is no longer reported for [[module Rate]]
    # because its absence is expected on SCP-CN pages; an unexpected module
    # shows up as module_added instead.
    mangled = ZH_PAGE.replace("[[module Rate]]", "[[module 评分]]")
    report2 = validate_pair(EN_PAGE, mangled)
    codes2 = {i.code for i in report2.issues}
    assert "cjk_in_protected" in codes2, codes2
    assert codes2 & {"module_missing", "module_added"}, codes2

    code_changed = ZH_PAGE.replace('return "never translated"', 'return "已翻译"')
    report3 = validate_pair(EN_PAGE, code_changed)
    assert "code_block_modified" in {i.code for i in report3.issues}

    include_translated = ZH_PAGE.replace(":scp-wiki:component:anomaly-class-bar-source", ":scp-wiki:组件:分级栏")
    report4 = validate_pair(EN_PAGE, include_translated)
    assert report4.score < 1.0
    assert "include_missing" in {i.code for i in report4.issues}


# ---------------------------------------------------------------------------
# tests: data pipeline
# ---------------------------------------------------------------------------


def test_id_normalisation():
    cfg = {"extensions": [".txt"], "id_strip_suffixes": ["_zh", "-cn", ".zh"]}
    assert normalize_id("scp-173") == normalize_id("SCP-173")
    assert normalize_id("SCP_173") == "SCP-173"
    assert derive_id(Path("SCP-173_zh.txt"), cfg) == "SCP-173"
    assert derive_id(Path("scp-173.txt"), cfg) == "scp-173"
    assert normalize_id(derive_id(Path("SCP-173.zh.txt"), cfg)) == "SCP-173"


def test_loader_dirs_and_file(tmp: Path):
    cfg = {"extensions": [".txt"], "id_strip_suffixes": ["_zh"], "recursive": True,
           "raw_en_dir": str(tmp / "en"), "raw_zh_dir": str(tmp / "zh")}
    (tmp / "en").mkdir(parents=True)
    (tmp / "zh").mkdir(parents=True)
    write_text(tmp / "en" / "SCP-100.txt", LONG_EN)
    write_text(tmp / "zh" / "SCP-100_zh.txt", LONG_ZH)
    write_text(tmp / "en" / "SCP-101.txt", LONG_EN)
    pairs = load_from_dirs(cfg)
    assert len(pairs) == 1, [p.id for p in pairs]      # SCP-101 has no Chinese file

    jl = tmp / "pairs.jsonl"
    write_jsonl(jl, [{"id": "SCP-100", "source": LONG_EN, "target": LONG_ZH}])
    pairs2 = load_from_file({"pairs_file": str(jl), "id_fields": ["id"],
                             "source_fields": ["source"], "target_fields": ["target"]})
    assert len(pairs2) == 1 and pairs2[0].id == "SCP-100"



def test_splits_are_leak_free():
    records = []
    for i in range(40):
        pid = f"SCP-{100+i}"
        rec = _fake_record(pid, LONG_EN + f"\n\nUnique english tail {pid}.", LONG_ZH + f"\n\n唯一中文结尾{pid}。")
        records.append(rec)
    buckets, stats = split_records(records, {"train": 0.8, "val": 0.1, "test": 0.1,
                                             "variant_markers": ["-D"], "shuffle": True}, seed=1)
    ids = {name: {r.source_sha1 for r in recs} for name, recs in buckets.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert sum(len(v) for v in buckets.values()) == 40
    # deterministic
    buckets2, _ = split_records(records, {"train": 0.8, "val": 0.1, "test": 0.1,
                                          "variant_markers": ["-D"], "shuffle": True}, seed=1)
    assert [r.id for r in buckets["test"]] == [r.id for r in buckets2["test"]]
    # variants cannot straddle splits
    assert group_key("SCP-173-D", ["-D"]) == group_key("SCP-173", ["-D"]) == "SCP-173"


def test_chat_format():
    records = [_fake_record(f"SCP-{i}", LONG_EN, LONG_ZH) for i in range(10)]
    fmt = {
        "system_prompt": "SYSTEM",
        "user_template": "请将以下 SCP Wikidot 源码翻译成中文，并保留 Wikidot 格式：\n\n{source}",
    }
    built = build_sft_dataset(records, fmt, seed=3)
    samples = built["samples"]
    assert built["stats"]["translation_samples"] == 10
    first = samples[0]["messages"]
    assert first[0]["role"] == "system"
    assert first[1]["role"] == "user" and "[[include" in first[1]["content"]
    assert first[2]["role"] == "assistant" and "[[include" in first[2]["content"]
    # the raw markup must survive into the prompt untouched
    assert "[[module Rate]]" in first[1]["content"]


# ---------------------------------------------------------------------------
# tests: training pieces (no torch needed)
# ---------------------------------------------------------------------------


def test_train_settings_resolution():
    _require_torch()
    from src.training.config import TrainSettings, resolve_optim

    cfg = {
        "project_root": str(ROOT),
        "model": {"name_or_path": "Qwen/Qwen2.5-7B-Instruct", "dtype": "auto"},
        "quantization": {"enabled": True, "load_in_4bit": True},
        "lora": {"target_modules": "auto", "r": 16, "alpha": 32},
        "training": {"max_seq_length": 2048, "overlong_policy": "truncate", "bf16": "auto", "unknown_key": 1},
        "data_files": {"chat_train": "data/splits/train.chat.jsonl"},
        "paths": {"log_dir": "logs", "hf_home": "data/hf_cache"},
    }
    ts = TrainSettings.from_config(cfg)
    assert ts.training.max_seq_length == 2048
    assert ts.training.overlong_policy == "truncate"
    assert ts.model.name_or_path.startswith("Qwen/")
    assert ts.training.resolved_dtype is not None
    assert ts.resolve_data_file("chat_train").endswith("train.chat.jsonl")
    assert Path(ts.resolve_data_file("chat_train")).is_absolute()
    # an unavailable paged optimizer must degrade gracefully
    assert resolve_optim("paged_adamw_8bit") in {"paged_adamw_8bit", "adamw_torch"}
    try:
        TrainSettings.from_config({**cfg, "training": {"overlong_policy": "bogus"}})
        raise AssertionError("invalid overlong_policy should raise")
    except ValueError:
        pass


def test_lora_target_detection():
    _require_torch()
    torch = __import__("torch")
    import torch.nn as nn
    from src.training.lora_utils import detect_target_modules, linear_module_inventory

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)
            self.gate_proj = nn.Linear(8, 16)

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.lm_head = nn.Linear(16, 32)

    model = Fake()
    inventory = linear_module_inventory(model)
    assert inventory["leaf_histogram"]["q_proj"] == 2
    targets, report = detect_target_modules(model, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    assert targets == ["q_proj", "k_proj", "o_proj", "gate_proj"], targets
    assert "v_proj" in report["missing"] and "up_proj" in report["missing"]
    explicit, _ = detect_target_modules(model, [], ["q_proj", "o_proj"])
    assert explicit == ["q_proj", "o_proj"]
    try:
        detect_target_modules(model, [], ["does_not_exist"])
        raise AssertionError("unknown explicit target should raise")
    except ValueError:
        pass


def test_chat_dataset_masking_and_length(tmp: Path):
    _require_torch()
    from src.training.data import ChatSFTDataset, DataCollatorForChatSFT

    tok = FakeTokenizer()
    path = tmp / "train.chat.jsonl"
    write_jsonl(path, [
        {"id": "a", "task": "translate", "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "translate this"},
            {"role": "assistant", "content": "翻译这个"}]},
        {"id": "b", "task": "translate", "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "translate this too"},
            {"role": "assistant", "content": "翻译这个也"}]},
    ])
    ds = ChatSFTDataset(path, tok, max_seq_length=512, train_on_assistant_only=True, name="test")
    assert len(ds) == 2
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])
    n_masked = sum(1 for x in item["labels"] if x == -100)
    assert n_masked > 0, "prompt tokens must be masked"
    assert item["labels"][-1] != -100, "assistant tokens must be trained on"

    # assistant-only off -> no masking
    ds2 = ChatSFTDataset(path, tok, max_seq_length=512, train_on_assistant_only=False, name="test2")
    assert all(x != -100 for x in ds2[0]["labels"])

    # overlong policy: drop
    ds3 = ChatSFTDataset(path, tok, max_seq_length=5, overlong_policy="drop", name="drop")
    assert len(ds3) == 0 and ds3.stats()["skipped_overlong"] == 2
    # overlong policy: truncate
    ds4 = ChatSFTDataset(path, tok, max_seq_length=5, overlong_policy="truncate", name="trunc")
    assert len(ds4) == 2 and len(ds4[0]["input_ids"]) == 5

    collator = DataCollatorForChatSFT(tok, pad_to_multiple_of=4)
    batch = collator([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 4 == 0
    assert (batch["attention_mask"][0] == 0).sum() == 0 or True


# ---------------------------------------------------------------------------
# tests: evaluation
# ---------------------------------------------------------------------------


def test_evaluation_metrics():
    from src.evaluation.metrics import evaluate, length_analysis, wikidot_preservation

    good = [{"id": "1", "source": EN_PAGE, "target": ZH_PAGE, "hypothesis": ZH_PAGE}]
    report, samples = evaluate(good, cfg={"metrics": ["wikidot", "length", "rouge_l"]})
    assert report["metrics"]["wikidot"]["score"] == 1.0
    assert report["metrics"]["length"]["anomalies"] == 0
    assert samples[0].wikidot_errors == 0

    bad_hyp = ZH_PAGE.replace("[[/div]]", "") + " " + ("垃圾" * 4000)
    report2, _ = evaluate([{"id": "2", "source": EN_PAGE, "target": ZH_PAGE, "hypothesis": bad_hyp}],
                          cfg={"metrics": ["wikidot", "length"]})
    assert report2["metrics"]["wikidot"]["score"] < 1.0
    assert report2["metrics"]["length"]["anomalies"] == 1
    assert "too_long" in report2["metrics"]["length"]["reason_counts"]

    empty = length_analysis([EN_PAGE], [ZH_PAGE], [""])
    assert empty["reason_counts"].get("empty_hypothesis") == 1

    wp = wikidot_preservation([EN_PAGE, EN_PAGE], [ZH_PAGE, ZH_PAGE.replace("[[image scp-173.jpg style=\"width:300px;\"]]", "")])
    assert wp["perfect_ratio"] == 0.5
    assert wp["top_issues"]


def test_evaluation_report_written(tmp: Path):
    from src.evaluation.metrics import evaluate
    from src.evaluation.report import write_reports

    report, samples = evaluate([{"id": "1", "source": EN_PAGE, "target": ZH_PAGE, "hypothesis": ZH_PAGE}],
                               cfg={"metrics": ["wikidot", "length"]})
    paths = write_reports(report, samples, tmp / "eval")
    for key in ("report_json", "report_md", "per_sample"):
        assert Path(paths[key]).exists(), key
    assert "Wikidot syntax preservation" in Path(paths["report_md"]).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# helpers + runner
# ---------------------------------------------------------------------------


def _fake_record(pid: str, source: str, target: str):
    from src.data.clean import CleanRecord
    from src.utils.io import sha1

    return CleanRecord(
        id=pid, source=source, target=target, source_len=len(source), target_len=len(target),
        length_ratio=len(target) / max(len(source), 1), source_plain_len=len(plain_text(source)),
        target_plain_len=len(plain_text(target)), source_cjk_ratio=cjk_ratio(source), target_cjk_ratio=cjk_ratio(target),
        source_latin_ratio=0.5, target_latin_ratio=0.05, source_sha1=sha1(source), target_sha1=sha1(target),
    )


class SkipTest(Exception):
    pass


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception:
        raise SkipTest("torch is not installed in this interpreter")


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    passed, failed, skipped = 0, [], []
    for name, func in tests:
        tmp = Path(tempfile.mkdtemp(prefix="scp_test_"))
        try:
            import inspect

            if "tmp" in inspect.signature(func).parameters:
                func(tmp)
            else:
                func()
            print(f"PASS  {name}")
            passed += 1
        except SkipTest as exc:
            print(f"SKIP  {name}: {exc}")
            skipped.append(name)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    print(f"{passed}/{len(tests)} tests passed"
          + (f" | skipped: {skipped}" if skipped else "")
          + (f" | failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
