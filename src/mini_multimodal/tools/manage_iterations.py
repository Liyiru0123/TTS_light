from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_multimodal.iteration_management import (
    IterationManagementError,
    atomic_write_json,
    iteration_status,
    list_iterations,
    prepare_iteration,
    promote_release,
    select_candidate,
    show_iteration,
    update_branch_state,
    verify_all,
    verify_iteration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage immutable model iterations and releases")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--output",
        help="Optional project-relative JSON output path for persistent machine evidence",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    status = sub.add_parser("status")
    status.add_argument("iteration_id", nargs="?")
    show = sub.add_parser("show")
    show.add_argument("iteration_id")
    verify = sub.add_parser("verify")
    verify.add_argument("iteration_id")
    sub.add_parser("verify-all")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("iteration_id")
    prepare.add_argument("--title", required=True)
    prepare.add_argument("--objective", required=True)
    prepare.add_argument("--parent")
    prepare.add_argument("--config", action="append", default=[])
    prepare.add_argument("--created-at", required=True)
    prepare.add_argument("--authorized-by", required=True)
    prepare.add_argument(
        "--branch",
        action="append",
        dest="branches",
        default=[],
        help="Planned lowercase branch id; repeat for multiple branches (default: main)",
    )
    update = sub.add_parser("update-branch-state")
    update.add_argument("iteration_id")
    update.add_argument("branch_id")
    update.add_argument("state", choices=["running", "completed", "aborted"])
    update.add_argument("--updated-at", required=True)
    update.add_argument("--record", required=True)
    candidate = sub.add_parser("select-candidate")
    candidate.add_argument("iteration_id")
    candidate.add_argument("branch_id")
    candidate.add_argument("--decision-record", required=True)
    candidate.add_argument("--selected-at", required=True)
    release = sub.add_parser("promote")
    release.add_argument("iteration_id")
    release.add_argument("branch_id")
    release.add_argument("release_id")
    release.add_argument("--approval-record", required=True)
    release.add_argument("--released-at", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(args.project_root).resolve()
    result_root = (project_root / args.result_root).resolve()
    try:
        if args.command == "list":
            result = {"iterations": list_iterations(result_root)}
        elif args.command == "status":
            result = iteration_status(result_root, args.iteration_id)
        elif args.command == "show":
            result = show_iteration(result_root, args.iteration_id)
        elif args.command == "verify":
            result = verify_iteration(project_root, result_root, args.iteration_id)
        elif args.command == "verify-all":
            result = verify_all(project_root, result_root)
        elif args.command == "prepare":
            result = prepare_iteration(
                project_root,
                result_root,
                iteration_id=args.iteration_id,
                title=args.title,
                objective=args.objective,
                parent_iteration=args.parent,
                config_paths=args.config,
                created_at=args.created_at,
                authorized_by=args.authorized_by,
                branch_ids=args.branches or None,
            )
        elif args.command == "update-branch-state":
            result = update_branch_state(
                result_root,
                iteration_id=args.iteration_id,
                branch_id=args.branch_id,
                new_state=args.state,
                updated_at=args.updated_at,
                record=args.record,
            )
        elif args.command == "select-candidate":
            result = select_candidate(
                result_root,
                iteration_id=args.iteration_id,
                branch_id=args.branch_id,
                decision_record=args.decision_record,
                selected_at=args.selected_at,
            )
        elif args.command == "promote":
            result = promote_release(
                project_root,
                result_root,
                iteration_id=args.iteration_id,
                branch_id=args.branch_id,
                release_id=args.release_id,
                approval_record=args.approval_record,
                released_at=args.released_at,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except IterationManagementError as error:
        raise SystemExit(str(error)) from error
    if args.output:
        output_path = (project_root / args.output).resolve()
        try:
            output_path.relative_to(project_root)
        except ValueError as error:
            raise SystemExit("--output must stay inside project root") from error
        atomic_write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
