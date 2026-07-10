"""Run evaluation.py over multiple solver/model result directories in parallel.

Examples:
    python scripts/batch_evaluate.py
    python scripts/batch_evaluate.py -j 3 --models gpt-5.4 gpt-5.4-mini gemma3-12b
    python scripts/batch_evaluate.py --solvers nl2p_1 nl2p_1_ablation nl2p_1_coref --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVERS = ("nl2p_1", "nl2p_1_ablation", "nl2p_1_coref", "gpt3_to_plan")
DEFAULT_MODELS = (
    "gpt-5.4",
    "gpt-5.4-mini",
    "gemma3-12b",
    "gemma3-27b",
    "llama3-70b",
)


@dataclass(frozen=True)
class EvaluationJob:
    solver: str
    model: str
    result_dir: Path


@dataclass(frozen=True)
class EvaluationResult:
    job: EvaluationJob
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    duration: float


@dataclass
class RunningJob:
    job: EvaluationJob
    command: list[str]
    process: subprocess.Popen
    stdout_file: BinaryIO
    stderr_file: BinaryIO
    started_at: float


def build_jobs(prefix: Path, solvers: list[str], models: list[str], strict: bool = False) -> tuple[list[EvaluationJob], list[Path]]:
    jobs = []
    missing = []
    for solver in solvers:
        for model in models:
            result_dir = prefix / solver / model
            if result_dir.is_dir():
                jobs.append(EvaluationJob(solver=solver, model=model, result_dir=result_dir))
            else:
                missing.append(result_dir)

    if strict and missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing result directories:\n{missing_text}")

    return jobs, missing


def evaluation_command(python: str, evaluator: Path, job: EvaluationJob, diagnostics: bool) -> list[str]:
    command = [python, str(evaluator), "-d", str(job.result_dir)]
    if diagnostics:
        command.append("--diagnostics")
    return command


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def run_job(job: EvaluationJob, python: str, evaluator: Path, diagnostics: bool) -> EvaluationResult:
    command = evaluation_command(python, evaluator, job, diagnostics)
    started_at = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - started_at
    return EvaluationResult(
        job=job,
        returncode=completed.returncode,
        command=command,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
        duration=duration,
    )


def print_result(result: EvaluationResult) -> None:
    job = result.job
    label = f"{job.solver}/{job.model}"
    if result.returncode == 0:
        print(f"[OK] {label} finished in {format_duration(result.duration)}")
    else:
        print(f"[FAIL] {label} exited with {result.returncode} after {format_duration(result.duration)}")
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)


def start_job(job: EvaluationJob, python: str, evaluator: Path, diagnostics: bool) -> RunningJob:
    command = evaluation_command(python, evaluator, job, diagnostics)
    stdout_file = tempfile.TemporaryFile(mode="w+b")
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    running = RunningJob(
        job=job,
        command=command,
        process=process,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        started_at=time.monotonic(),
    )
    print(f"[START] {job.solver}/{job.model}: {command_text(command)}")
    return running


def finish_job(running: RunningJob) -> EvaluationResult:
    returncode = running.process.wait()
    duration = time.monotonic() - running.started_at
    running.stdout_file.seek(0)
    running.stderr_file.seek(0)
    stdout = running.stdout_file.read().decode("utf-8", errors="replace")
    stderr = running.stderr_file.read().decode("utf-8", errors="replace")
    running.stdout_file.close()
    running.stderr_file.close()
    return EvaluationResult(
        job=running.job,
        returncode=returncode,
        command=running.command,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
    )


def print_progress(running_jobs: list[RunningJob]) -> None:
    now = time.monotonic()
    for running in running_jobs:
        elapsed = format_duration(now - running.started_at)
        print(f"[RUNNING] {running.job.solver}/{running.job.model} elapsed {elapsed}: {command_text(running.command)}")


def run_jobs(
    jobs: list[EvaluationJob],
    python: str,
    evaluator: Path,
    diagnostics: bool,
    max_workers: int,
    progress_interval: int,
) -> list[EvaluationResult]:
    pending = list(jobs)
    running: list[RunningJob] = []
    results: list[EvaluationResult] = []
    next_progress_at = time.monotonic() + progress_interval

    def fill_slots() -> None:
        while pending and len(running) < max_workers:
            running.append(start_job(pending.pop(0), python, evaluator, diagnostics))

    fill_slots()
    while running:
        completed = [item for item in running if item.process.poll() is not None]
        if completed:
            for item in completed:
                running.remove(item)
                result = finish_job(item)
                results.append(result)
                print_result(result)
            fill_slots()
            next_progress_at = time.monotonic() + progress_interval
            continue

        now = time.monotonic()
        if now >= next_progress_at:
            print_progress(running)
            next_progress_at = now + progress_interval
        time.sleep(min(0.5, max(0.1, next_progress_at - time.monotonic())))

    return results


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        default=str(ROOT / "results"),
        help="Root containing <solver>/<model> result directories.",
    )
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS), help="Solver directories to evaluate.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model directories to evaluate.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=positive_int,
        default=min(5, os.cpu_count() or 1),
        help="Maximum concurrent evaluation.py processes.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run evaluation.py.")
    parser.add_argument("--evaluator", default=str(ROOT / "evaluation.py"), help="Path to evaluation.py.")
    parser.add_argument(
        "--progress-interval",
        type=positive_int,
        default=30,
        help="Seconds between running-process progress reports.",
    )
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not pass --diagnostics to evaluation.py.")
    parser.add_argument("--strict", action="store_true", help="Fail if any requested solver/model directory is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prefix = Path(args.prefix)
    evaluator = Path(args.evaluator)
    diagnostics = not args.no_diagnostics
    jobs, missing = build_jobs(prefix, args.solvers, args.models, strict=args.strict)

    for path in missing:
        print(f"[SKIP] missing results dir: {path}")

    if not jobs:
        print("No evaluation jobs to run.")
        return 1 if args.strict else 0

    print(f"Prepared {len(jobs)} evaluation job(s), running up to {args.jobs} at once.")
    if args.dry_run:
        for job in jobs:
            print(" ".join(evaluation_command(args.python, evaluator, job, diagnostics)))
        return 0

    results = run_jobs(
        jobs,
        python=args.python,
        evaluator=evaluator,
        diagnostics=diagnostics,
        max_workers=args.jobs,
        progress_interval=args.progress_interval,
    )
    failures = [result for result in results if result.returncode != 0]

    if failures:
        print(f"{len(failures)} evaluation job(s) failed.", file=sys.stderr)
        return 1
    print("All evaluation jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
