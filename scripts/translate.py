#!/usr/bin/env python
"""Phase 3/5 -- translate English SCP Wikidot source into Chinese SCP-CN source.

Usage
-----
    # one file
    python scripts/translate.py input.wikidot -o output.zh.wikidot

    # a whole directory (mirrors the tree into -o)
    python scripts/translate.py data/raw/en -o outputs/translated --adapter outputs/qwen2.5-7b-scp-qlora/final_adapter

    # inline text
    python scripts/translate.py --text "**Item #:** SCP-173"

    # from stdin
    cat page.wikidot | python scripts/translate.py --adapter outputs/..../final_adapter

    # inspect the prompt that would be sent (no model needed)
    python scripts/translate.py --text "..." --print-prompt --dry-run

    # validate the Wikidot structure of the produced translation
    python scripts/translate.py input.wikidot --adapter ... --validate

Terminology (Phase 5)
---------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import add_common_args, ensure_dirs, load_config   # noqa: E402
from src.utils.io import read_text, write_json, write_text               # noqa: E402
from src.utils.logging_utils import setup_logging                        # noqa: E402
from src.wikidot.validator import validate_pair                          # noqa: E402

TEXT_SUFFIXES = {".txt", ".wikidot", ".wiki", ".wtxt", ".md", ""}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Translate English SCP Wikidot source to Chinese")
    add_common_args(parser)
    parser.add_argument("input", nargs="?", help="input .wikidot file, or a directory of them")
    parser.add_argument("-o", "--output", help="output file (or directory when input is a directory)")
    parser.add_argument("--text", help="translate this string instead of a file")
    parser.add_argument("--model", help="base model name/path (or an adapter directory)")
    parser.add_argument("--adapter", help="LoRA adapter directory produced by scripts/train.py")
    parser.add_argument("--base-model", help="explicit base model for the adapter")
    parser.add_argument("--no-4bit", action="store_true", help="load the model in bf16/fp16 instead of 4-bit")


    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)

    parser.add_argument("--validate", action="store_true", help="run the Wikidot validator on the output")
    parser.add_argument("--print-prompt", action="store_true", help="print the exact prompt (and continue)")
    parser.add_argument("--dry-run", action="store_true", help="build prompts only, do not load the model")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES]
    return files


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)

    if not args.text and not args.input:
        if sys.stdin.isatty():
            print("error: give an input file, a directory, or --text (or pipe the source via stdin)", file=sys.stderr)
            return 2
        args.text = sys.stdin.read()

    # ------------------------------------------------------------------
    # prompt-only mode: no torch, no model download
    # ------------------------------------------------------------------
    if args.dry_run:
        from src.data.to_chat import build_user_prompt

        source = args.text if args.text is not None else read_text(Path(args.input))
        fmt_cfg = dict(cfg.get("format") or {})
        print("=== system prompt ===")
        print(fmt_cfg.get("system_prompt", ""))
        print("=== user prompt ===")
        print(build_user_prompt(source, fmt_cfg))
        return 0

    # ------------------------------------------------------------------
    # real translation
    # ------------------------------------------------------------------
    log = setup_logging("WARNING" if args.quiet else "INFO",
                        Path(cfg["paths"]["log_dir"]) / "translate.log", name="scp.translate")

    from src.inference.translator import GenerationSettings, SCPTranslator

    if not args.adapter and not args.model:
        default_adapter = Path(cfg["paths"]["output_dir"]) / "qwen2.5-7b-scp-qlora" / "final_adapter"
        if default_adapter.exists():
            args.adapter = str(default_adapter)
        else:
            args.model = str((cfg.get("model") or {}).get("name_or_path", "Qwen/Qwen2.5-7B-Instruct"))
            log.warning("no adapter found (looked at %s) -- translating with the base model %s", default_adapter, args.model)

    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    translator = SCPTranslator.from_config(
        cfg,
        model_path=args.model,
        adapter_path=args.adapter,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
        generation=generation,
        logger=log,
    )

    # ------------------------------------------------------------------
    # single text / single file
    # ------------------------------------------------------------------
    if args.text is not None or (args.input and Path(args.input).is_file()):
        source = args.text if args.text is not None else read_text(Path(args.input))
        messages = translator.build_messages(source)
        if args.print_prompt:
            print("=" * 70)
            print(translator._render_prompt(messages))
            print("=" * 70)
        result = translator.translate(source)
        report = None
        if args.validate:

            report = validate_pair(source, result.text, (cfg.get("evaluation") or {}).get("wikidot", {}))
            log.info("wikidot validation:\n%s", report.summary())

        if args.output:
            write_text(args.output, result.text)
            log.info("wrote %s", args.output)
            if report is not None:
                write_json(str(args.output) + ".validation.json", report.to_dict())
        if args.json:
            payload = result.to_dict()
            if report is not None:
                payload["validation"] = report.to_dict()
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if not args.output:
                print(result.text)
            if args.validate and report is not None:
                print(f"\n# wikidot score={report.score:.3f} errors={report.error_count()} warnings={report.warning_count()}", file=sys.stderr)
        return 0 if (report is None or report.ok) else 1

    # ------------------------------------------------------------------
    # directory mode
    # ------------------------------------------------------------------
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(cfg["paths"]["output_dir"]) / "translated"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = collect_inputs(input_dir)
    log.info("translating %d files from %s -> %s", len(files), input_dir, output_dir)


    index = []
    for i, path in enumerate(files, 1):
        source = read_text(path)
        result = translator.translate(source)
        relative = path.relative_to(input_dir)
        target_path = output_dir / relative
        write_text(target_path, result.text)
        report = validate_pair(source, result.text, (cfg.get("evaluation") or {}).get("wikidot", {}))
        index.append(
            {
                "source": str(path),
                "output": str(target_path),
                "n_input_tokens": result.n_input_tokens,
                "n_output_tokens": result.n_output_tokens,
                "wikidot_score": round(report.score, 4),
                "wikidot_errors": report.error_count(),
            }
        )
        log.info("[%d/%d] %s -> %s (score %.3f, %d errors)", i, len(files), path.name, target_path.name, report.score, report.error_count())

    write_json(output_dir / "translation_index.json", index)
    avg = sum(item["wikidot_score"] for item in index) / max(len(index), 1)
    log.info("done: %d files, average wikidot score %.4f", len(index), avg)
    print(f"translated {len(index)} files into {output_dir} (average wikidot score {avg:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
