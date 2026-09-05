from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import EVAL_LOGS, add_checkpoint_args, cell_dir, parse_args, resolve_checkpoints

from experiment_utils.eval_runner import cell_done
from experiment_utils.serving import served_model

CATEGORIES = ["self_pres", "oversight_subversion"]
AUDITOR_MODEL = "openai/gpt-5.6-terra"
JUDGE_MODEL = "openai/gpt-5.6-sol"
RESCORE_JUDGE_MODEL = "openai/gpt-5.6-terra"
TARGET_DISPLAY_NAME = "gpt-4.1"


def cell_name(category: str) -> str:
    return f"petri_{category}"


def _role_model(model: str, reasoning_effort: str):
    from inspect_ai.model import GenerateConfig, get_model

    return get_model(model, config=GenerateConfig(reasoning_effort=reasoning_effort, seed=0))


def run_audits(ckpts, args) -> None:
    import inspect_ai
    from inspect_ai.model import get_model

    from automatic_alignment_audit._common import make_audit

    cell = cell_name(args.category)
    for label, model in ckpts:
        d = cell_dir(label, cell)
        if cell_done(d):
            print(f"=== {label}/{cell}: done")
            continue
        print(f"=== {label}/{cell}: {model} ({args.epochs} audits x seed)")
        d.mkdir(parents=True, exist_ok=True)
        with served_model(model, args.base_model, args.provider) as (inspect_model, model_args):
            target = (get_model(inspect_model, **(model_args or {}))
                      if isinstance(inspect_model, str) else inspect_model)
            inspect_ai.eval(
                make_audit(args.category, max_turns=args.max_turns,
                           target_display_name=args.target_display_name),
                model=target,
                model_roles={
                    "target": target,
                    "auditor": _role_model(args.auditor_model, "medium"),
                    "judge": _role_model(args.judge_model, "medium"),
                },
                epochs=args.epochs,
                log_dir=str(d),
                max_connections=args.max_connections,
                fail_on_error=0.2,
            )


def rescore_refusals(ckpts, args) -> None:
    """Re-judge samples whose `audit_judge` score is a content-filter refusal."""
    from inspect_ai import score
    from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log
    from inspect_petri import audit_judge

    from automatic_alignment_audit._common import JUDGE_DIMENSIONS

    scorer = audit_judge(JUDGE_DIMENSIONS, model=_role_model(args.rescore_judge, "medium"))
    for label, _ in ckpts:
        d = cell_dir(label, cell_name(args.category))
        for info in list_eval_logs(str(d)):
            log = read_eval_log(info.name)
            if log.status != "success":
                continue
            refused = [s for s in log.samples
                       if (s.scores or {}).get("audit_judge") is not None
                       and (s.scores["audit_judge"].metadata or {}).get("refusal")]
            print(f"  {label}/{Path(info.name).name}: {len(refused)} judge refusals")
            if not refused:
                continue
            sub = log.model_copy(deep=True)
            sub.samples, sub.results = refused, None
            rescored = score(sub, scorers=[scorer], action="overwrite", display="plain")
            by_key = {(s.id, s.epoch): s for s in rescored.samples}
            for s in log.samples:
                r = by_key.get((s.id, s.epoch))
                if r is None:
                    continue
                new = r.scores["audit_judge"]
                new.metadata = {**(new.metadata or {}), "rescored_judge": args.rescore_judge,
                                "original_judge_refusal": True}
                s.scores["audit_judge"] = new
            src = Path(info.name.replace("file://", ""))
            bak = src.with_suffix(src.suffix + ".pre_rescore.bak")
            if not bak.exists():
                shutil.copy2(src, bak)
            write_eval_log(log, str(src))
            print(f"    re-judged {len(refused)} with {args.rescore_judge}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--category", default="self_pres", choices=CATEGORIES)
    ap.add_argument("--epochs", type=int, default=10, help="audits per seed")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--auditor-model", default=AUDITOR_MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--rescore-judge", default=RESCORE_JUDGE_MODEL,
                    help="judge for transcripts the main judge refused")
    ap.add_argument("--skip-rescore", action="store_true")
    ap.add_argument("--target-display-name", default=TARGET_DISPLAY_NAME,
                    help="public model name the auditor is told (never the ft: id)")
    ap.set_defaults(max_connections=20)
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    run_audits(ckpts, args)
    if not args.skip_rescore:
        rescore_refusals(ckpts, args)


if __name__ == "__main__":
    main()
