import pytest
import tempfile

from scripts import batch_evaluate


def test_build_jobs_skips_missing_directories(tmp_path):
    existing = tmp_path / "nl2p_1" / "gpt-5.4"
    existing.mkdir(parents=True)

    jobs, missing = batch_evaluate.build_jobs(
        tmp_path,
        solvers=["nl2p_1", "nl2p_1_coref"],
        models=["gpt-5.4"],
    )

    assert [job.result_dir for job in jobs] == [existing]
    assert missing == [tmp_path / "nl2p_1_coref" / "gpt-5.4"]


def test_build_jobs_strict_raises_for_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing result directories"):
        batch_evaluate.build_jobs(
            tmp_path,
            solvers=["nl2p_1_coref"],
            models=["gpt-5.4"],
            strict=True,
        )


def test_evaluation_command_includes_diagnostics_flag(tmp_path):
    job = batch_evaluate.EvaluationJob(
        solver="nl2p_1",
        model="gpt-5.4",
        result_dir=tmp_path / "results" / "nl2p_1" / "gpt-5.4",
    )

    command = batch_evaluate.evaluation_command("python", tmp_path / "evaluation.py", job, diagnostics=True)

    assert command == [
        "python",
        str(tmp_path / "evaluation.py"),
        "-d",
        str(job.result_dir),
        "--diagnostics",
    ]


def test_format_duration():
    assert batch_evaluate.format_duration(9.8) == "9s"
    assert batch_evaluate.format_duration(65) == "1m 05s"
    assert batch_evaluate.format_duration(3661) == "1h 01m 01s"


def test_finish_job_replaces_invalid_output_bytes(tmp_path):
    class ProcessStub:
        def wait(self):
            return 1

    stdout_file = tempfile.TemporaryFile(mode="w+b")
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    stdout_file.write(b"ok\n")
    stderr_file.write(b"bad byte: \xa8\n")

    running = batch_evaluate.RunningJob(
        job=batch_evaluate.EvaluationJob("nl2p_1", "gpt-5.4", tmp_path),
        command=["python", "evaluation.py"],
        process=ProcessStub(),
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        started_at=0,
    )

    result = batch_evaluate.finish_job(running)

    assert result.returncode == 1
    assert result.stdout == "ok\n"
    assert "bad byte: �" in result.stderr


def test_main_dry_run_uses_requested_matrix(tmp_path, capsys):
    (tmp_path / "nl2p_1" / "gpt-5.4").mkdir(parents=True)
    (tmp_path / "nl2p_1_coref" / "gpt-5.4").mkdir(parents=True)

    exit_code = batch_evaluate.main(
        [
            "--prefix",
            str(tmp_path),
            "--solvers",
            "nl2p_1",
            "nl2p_1_coref",
            "--models",
            "gpt-5.4",
            "--python",
            "python",
            "--evaluator",
            "evaluation.py",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Prepared 2 evaluation job(s)" in out
    assert f"python evaluation.py -d {tmp_path / 'nl2p_1' / 'gpt-5.4'} --diagnostics" in out
    assert f"python evaluation.py -d {tmp_path / 'nl2p_1_coref' / 'gpt-5.4'} --diagnostics" in out
