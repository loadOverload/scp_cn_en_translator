#!/usr/bin/env python
"""Generate SYNTHETIC English/Chinese SCP pairs so the pipeline can be tested
before the real corpus is available.

These files are NOT training data. They live under ``data/samples/raw/`` and
exercise every construct the real corpus contains: [[include]], [[module]],
[[div]], [[footnote]], tables, images, links, HTML, code blocks, plus a
consistent EN->ZH wording so the sample corpus looks like real parallel data.

Usage
-----
    python scripts/make_sample_data.py --n 60
    python scripts/prepare_data.py --config configs/sample.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import add_common_args, load_config   # noqa: E402
from src.utils.io import write_jsonl, write_text            # noqa: E402
from src.utils.logging_utils import setup_logging           # noqa: E402

# ---------------------------------------------------------------------------
# Consistent EN -> ZH wording, so the synthetic corpus is self-consistent
# ---------------------------------------------------------------------------
GLOSSARY = {
    "Special Containment Procedures": "特殊收容措施",
    "Mobile Task Force": "机动特遣队",
    "Site Director": "站点主管",
    "Foundation personnel": "基金会人员",
    "Class D personnel": "D级人员",
    "containment chamber": "收容间",
    "anomalous object": "异常物体",
    "Foundation": "基金会",
    "research staff": "研究人员",
    "security personnel": "安保人员",
    "Euclid class": "Euclid级",
    "Keter class": "Keter级",
    "Safe class": "Safe级",
    "anomaly": "异常",
    "containment breach": "收容失效",
    "recovery team": "回收小组",
    "interview log": "访谈记录",
    "experiment log": "实验记录",
    "test subject": "测试对象",
    "hazardous materials": "危险材料",
    "observation deck": "观察室",
    "termination": "处决",
    "amnestics": "记忆删除剂",
    "Reality Bender": "现实扭曲者",
    "cognitohazard": "认知危害",
    "memetic hazard": "模因危害",
    "infohazard": "信息危害",
    "containment unit": "收容单元",
    "field agent": "外勤特工",
    "O5 Council": "O5议会",
}

EN_PARAGRAPHS = [
    "SCP-{n} was recovered on {date} from a decommissioned research facility in {place}. "
    "The object was transported to Site-{site} by a Mobile Task Force detachment, and all "
    "witnesses were administered amnestics by Foundation personnel.",
    "Foundation personnel assigned to SCP-{n} must remain outside the containment chamber "
    "unless level-3 authorization has been granted by the Site Director. Any deviation from "
    "the Special Containment Procedures is to be reported immediately.",
    "Under no circumstances should SCP-{n} be exposed to direct sunlight for more than "
    "{minutes} minutes. During such exposure the anomalous object emits a low-frequency tone "
    "that affects research staff within a {radius} metre radius.",
    "The containment chamber housing SCP-{n} is monitored by four remote cameras. A single "
    "Class D personnel is to be assigned to daily maintenance; the test subject is to be "
    "terminated and replaced every {days} days.",
    "Interview log {n}-A: the subject described a Class {cls} anomaly capable of altering "
    "local reality. The interview log was classified by the O5 Council and sealed pending "
    "review by the Site Director.",
    "Should a containment breach occur, the on-site Mobile Task Force is to establish a "
    "perimeter and deploy hazardous materials crews. Recovery team members must wear "
    "protective equipment at all times inside the observation deck.",
]

ZH_PARAGRAPHS = [
    "SCP-{n}于{date}在{place}一处已停用的研究设施中被回收。该物体由一支机动特遣队分队"
    "运送至Site-{site}，所有目击者均由基金会人员施以记忆删除剂。",
    "分配至SCP-{n}的基金会人员必须留在收容间之外，除非获得站点主管的三级授权。任何偏离"
    "特殊收容措施的行为都必须立即上报。",
    "在任何情况下，SCP-{n}都不得接受超过{minutes}分钟的阳光直射。在直射期间，该异常物体"
    "会发出一种低频声响，影响半径{radius}米内研究人员。",
    "收容SCP-{n}的收容间由四台远程摄像机监控。每日需指派一名D级人员执行维护；该测试对象"
    "应每{days}天被处决并更换。",
    "访谈记录{n}-A：对象描述了一个能够改变局部现实的{cls}级异常。该访谈记录已被O5议会"
    "列为机密，并封存以等待站点主管审核。",
    "一旦发生收容失效，现场机动特遣队须建立警戒线并部署危险材料处理小组。回收小组成员"
    "在观察室内必须全程穿戴防护装备。",
]

PLACES = [("Blackwood, Montana", "蒙大拿州布莱克伍德"), ("Arkhangelsk Oblast", "阿尔汉格尔斯克州"), ("Kyoto Prefecture", "京都府")]
CLASSES = ["Safe", "Euclid", "Keter"]


def make_pair(n: int, rng: random.Random) -> tuple[str, str]:
    scp_id = f"SCP-{100 + n}"
    params = {
        "n": 100 + n,
        "date": rng.choice(["19██-03-12", "20██-07-04", "19██-11-27"]),
        "place": PLACES[n % len(PLACES)][0],
        "place_zh": PLACES[n % len(PLACES)][1],
        "site": rng.choice([13, 19, 23, 45]),
        "minutes": rng.randint(5, 40),
        "radius": rng.randint(3, 30),
        "days": rng.choice([7, 14, 30]),
        "cls": CLASSES[n % len(CLASSES)],
    }
    zh_params = dict(params, cls={"Safe": "Safe", "Euclid": "Euclid", "Keter": "Keter"}[params["cls"]], place=params["place_zh"])

    # IMPORTANT: pick the SAME paragraph indices for both languages -- paragraph
    # i of EN_PARAGRAPHS is the translation of paragraph i of ZH_PARAGRAPHS.
    # Sampling independently would produce pairs that are not translations.
    chosen = sorted(rng.sample(range(len(EN_PARAGRAPHS)), 4))
    body_en = "\n\n".join(EN_PARAGRAPHS[i].format(**params) for i in chosen)
    body_zh = "\n\n".join(ZH_PARAGRAPHS[i].format(**zh_params) for i in chosen)

    en = f"""[[include :scp-wiki:component:anomaly-class-bar-source
|lang=en
|item-number={scp_id}
|clearance=3
|container-class={params['cls']}
]]

**Item #:** {scp_id}

**Object Class:** {params['cls']}

[[div class="blockquote"]]
**Special Containment Procedures:** {body_en}
[[/div]]

[[div class="scp-image-block block-right" style="width:300px;"]]
[[image {scp_id.lower()}.jpg style="width:300px;"]]
[[div class="scp-image-caption"]]
{scp_id} in its containment chamber.
[[/div]]
[[/div]]

**Description:** SCP-{params['n']} is an anomalous object recovered by the Foundation.
Refer to [[[scp-{params['n'] + 1}]]] for the related incident report.
External documentation is available at [https://scp-wiki.wikidot.com/scp-series the SCP wiki].

||~ Date ||~ Event ||~ Personnel ||
|| {params['date']} || Recovery || Mobile Task Force ||
|| 20██-08-01 || Containment breach || Class D personnel ||

[[collapsible show="+ Experiment log {scp_id}" hide="- Close"]]
**Experiment log {scp_id}-1**

> **Test subject:** D-{4000 + n}
> **Result:** The anomalous object reacted within {params['minutes']} minutes. The test subject was terminated.

[[footnote]]All research staff involved were dosed with amnestics.[[/footnote]]
[[/collapsible]]

<hr />

[[code]]
def contain(scp_id):
    return "contained"  # never translated
[[/code]]

[[module Rate]]

[[footnoteblock]]
"""

    zh = f"""[[include :scp-wiki:component:anomaly-class-bar-source
|lang=cn
|item-number={scp_id}
|clearance=3
|container-class={params['cls']}
]]

**项目编号：** {scp_id}

**项目等级：** {params['cls']}

[[div class="blockquote"]]
**特殊收容措施：** {body_zh}
[[/div]]

[[div class="scp-image-block block-right" style="width:300px;"]]
[[image {scp_id.lower()}.jpg style="width:300px;"]]
[[div class="scp-image-caption"]]
收容间中的{scp_id}。
[[/div]]
[[/div]]

**描述：** SCP-{params['n']}是基金会回收的一个异常物体。相关事故报告参见[[[scp-{params['n'] + 1}]]]。
外部文档可参见[https://scp-wiki.wikidot.com/scp-series SCP基金会维基]。

||~ 日期 ||~ 事件 ||~ 人员 ||
|| {params['date']} || 回收 || 机动特遣队 ||
|| 20██-08-01 || 收容失效 || D级人员 ||

[[collapsible show="+ 实验记录 {scp_id}" hide="- 关闭"]]
**实验记录 {scp_id}-1**

> **测试对象：** D-{4000 + n}
> **结果：** 该异常物体在{params['minutes']}分钟内发生反应。测试对象已被处决。

[[footnote]]所有参与的研究人员均被施以记忆删除剂。[[/footnote]]
[[/collapsible]]

<hr />

[[code]]
def contain(scp_id):
    return "contained"  # never translated
[[/code]]

[[module Rate]]

[[footnoteblock]]
"""
    return en, zh


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic SCP pairs for pipeline testing")
    add_common_args(parser)
    parser.add_argument("--n", type=int, default=60, help="number of pairs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None, help="output root (default: data/samples from config)")
    parser.add_argument("--jsonl", action="store_true", help="also emit a single JSONL file")
    parser.add_argument("--with-flaws", action="store_true", help="inject samples that the cleaner must reject")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    log = setup_logging("INFO", name="scp.samples")

    out_root = Path(args.out) if args.out else Path(cfg["paths"]["samples_dir"]) / "raw"
    en_dir, zh_dir = out_root / "en", out_root / "zh"
    en_dir.mkdir(parents=True, exist_ok=True)
    zh_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        en, zh = make_pair(i, rng)
        pid = f"SCP-{100 + i}"
        write_text(en_dir / f"{pid}.txt", en)
        write_text(zh_dir / f"{pid}.txt", zh)
        rows.append({"id": pid, "source": en, "target": zh})

    if args.with_flaws:
        flaws = [
            ("SCP-9001", "", "空英文", "empty source"),
            ("SCP-9002", "Short.", "太短的英文", "too-short source"),
            ("SCP-9003", "A" * 500, "This target is not Chinese at all.", "non-Chinese target"),
        ]
        for pid, en, zh, _ in flaws:
            if en:
                write_text(en_dir / f"{pid}.txt", en)
            write_text(zh_dir / f"{pid}.txt", zh)
        # exact duplicate of SCP-100
        write_text(en_dir / "SCP-9004.txt", rows[0]["source"])
        write_text(zh_dir / "SCP-9004.txt", rows[0]["target"])
        log.info("injected %d deliberately broken samples", len(flaws) + 1)

    if args.jsonl:
        write_jsonl(out_root / "pairs.jsonl", rows)
        log.info("wrote %s", out_root / "pairs.jsonl")

    log.info("wrote %d synthetic pairs to %s", args.n, out_root)
    log.info("try: python scripts/prepare_data.py --config configs/sample.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
