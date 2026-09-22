"""Data-safety tests (brief Task 1.1, Section 2.3).

Covers the restricted-data checker, the pre-commit hook, the .gitignore allow-list,
the $LCP_DATA_ROOT configuration and the external-API gate on the GPT script.
Checker and hook cases run inside a throwaway ``git init`` repository under
``tmp_path`` so the real repository's index is never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_cong_predict import config

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check_no_restricted_data.py"
HOOK = REPO / "scripts" / "hooks" / "pre-commit"
GPT_SCRIPT = REPO / "scripts" / "get_gpt_embeddings.py"


# ------------------------------------------------------------------- helpers --

def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo, capture_output=True, text=True, check=check,
    )


def _run_checker(repo: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--repo", str(repo), *extra],
        capture_output=True, text=True,
    )


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def _stage(repo: Path, rel: str, content: bytes = b"synthetic\n") -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    _git(repo, "add", "-f", rel)  # -f: bypass .gitignore, as a careless user might


# ------------------------------------------------------------------- checker --

def test_checker_passes_on_current_tracked_files():
    """The checker accepts every file currently tracked (git ls-files)."""
    result = _run_checker(REPO, "--all")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_checker_rejects_staged_tab_file(scratch_repo: Path):
    """A staged fake .tab file (a UK Data Service tab-delimited export) is refused."""
    _stage(scratch_repo, "ncds_sweep.tab")
    result = _run_checker(scratch_repo)
    assert result.returncode == 1
    assert "ncds_sweep.tab" in result.stderr and "data-like file type" in result.stderr


def test_checker_rejects_staged_parquet_under_data_embeddings(scratch_repo: Path):
    """data/embeddings/x.parquet is the old default output folder of
    scripts/get_gpt_embeddings.py (brief F8.6); staging it is refused."""
    _stage(scratch_repo, "data/embeddings/x.parquet")
    result = _run_checker(scratch_repo)
    assert result.returncode == 1
    assert "data/embeddings/x.parquet" in result.stderr
    assert "not one of the public reference files" in result.stderr


@pytest.mark.parametrize(
    "rel, reason",
    [
        ("predictions.csv", "outside tests/ and docs/"),
        ("notes/participants.txt", "outside tests/ and docs/"),
        ("src/pkg/scores.dta", "data-like file type"),
        ("data/camsis/sub/x.dta", "not one of the public reference files"),
        ("data/new_table.xlsx", "not one of the public reference files"),
        ("reference/llm_paper/_targets.R", "third-party reference source"),
    ],
)
def test_checker_rejects_other_forbidden_paths(scratch_repo: Path, rel: str, reason: str):
    """Each rule of scripts/check_no_restricted_data.py refuses its case."""
    _stage(scratch_repo, rel)
    result = _run_checker(scratch_repo)
    assert result.returncode == 1
    assert rel in result.stderr and reason in result.stderr


def test_checker_rejects_file_over_5_mb(scratch_repo: Path):
    """Any staged file over 5,000,000 bytes is refused, whatever its type."""
    _stage(scratch_repo, "src/big_module.py", b"#" * 5_000_001)
    result = _run_checker(scratch_repo)
    assert result.returncode == 1
    assert "over the 5,000,000-byte limit" in result.stderr


def test_checker_accepts_allow_list_and_permitted_text(scratch_repo: Path):
    """The public reference files and CSV/TXT under tests/ and docs/ pass."""
    for rel in [
        "data/variables.xlsx",
        "data/occupation_aspiration_mapping.xlsx",
        "data/camsis/gb71co70.dta",
        "tests/fixtures/synthetic.csv",
        "docs/reference/table.txt",
        "requirements-dev.txt",
        "src/pkg/module.py",
    ]:
        _stage(scratch_repo, rel)
    result = _run_checker(scratch_repo)
    assert result.returncode == 0, result.stderr


def test_pre_commit_hook_blocks_commit(scratch_repo: Path):
    """With core.hooksPath=scripts/hooks, a commit that stages a .tab file fails."""
    (scratch_repo / "scripts" / "hooks").mkdir(parents=True)
    shutil.copy(CHECKER, scratch_repo / "scripts" / "check_no_restricted_data.py")
    shutil.copy(HOOK, scratch_repo / "scripts" / "hooks" / "pre-commit")
    os.chmod(scratch_repo / "scripts" / "hooks" / "pre-commit", 0o755)
    _git(scratch_repo, "config", "core.hooksPath", "scripts/hooks")
    _git(scratch_repo, "add", "scripts")
    _git(scratch_repo, "commit", "-q", "-m", "hook")  # allowed: only Python + shell
    _stage(scratch_repo, "export.tab")
    result = _git(scratch_repo, "commit", "-q", "-m", "should fail", check=False)
    assert result.returncode != 0
    assert "export.tab" in result.stderr


# ----------------------------------------------------------------- gitignore --

@pytest.mark.parametrize(
    "rel",
    [
        "data/embeddings/embeddings_gpt35.parquet",  # brief F8.6
        "data/raw/ncds.tab",  # brief F8.6
        "predictions.csv",  # brief F8.6: predictions CSV at the repository root
        "data/ncds_1_2_3/ncds0123.dta",
        "outputs/table.csv",
        "results/fits.pkl",
        "reference/llm_paper/_targets.R",
        "anywhere/else/file.dta",
    ],
)
def test_gitignore_ignores_restricted_paths(rel: str):
    """.gitignore allow-list: restricted paths are ignored (git check-ignore exit 0)."""
    result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
    assert result.returncode == 0


@pytest.mark.parametrize(
    "rel",
    ["data/variables.xlsx", "data/camsis/gb71co70.dta", "src/llm_cong_predict/outputs/x.py",
     "tests/fixtures/x.csv"],
)
def test_gitignore_does_not_ignore_allowed_paths(rel: str):
    """Public reference files, and package folders named like ignored dirs, stay visible."""
    result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO)
    assert result.returncode == 1


# -------------------------------------------------------------------- config --

def test_data_root_unset_raises(monkeypatch):
    """Reading restricted data without $LCP_DATA_ROOT raises a clear error (no default)."""
    monkeypatch.delenv(config.LCP_DATA_ROOT_ENV, raising=False)
    with pytest.raises(config.DataRootError, match="LCP_DATA_ROOT is not set"):
        config.restricted_path("ncds_4")


def test_data_root_inside_repository_raises(monkeypatch):
    """A data root inside the repository is refused."""
    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(REPO / "data"))
    with pytest.raises(config.DataRootError, match="inside the repository"):
        config.data_root()


def test_data_root_symlink_into_repository_raises(monkeypatch, tmp_path: Path):
    """A path outside the repository that is a symlink into it is refused (resolve())."""
    link = tmp_path / "looks_outside"
    link.symlink_to(REPO / "data", target_is_directory=True)
    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(link))
    with pytest.raises(config.DataRootError, match="inside the repository"):
        config.data_root()


def test_data_root_missing_directory_raises(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(tmp_path / "absent"))
    with pytest.raises(config.DataRootError, match="does not exist"):
        config.data_root()


def test_restricted_and_output_paths_built_under_data_root(monkeypatch, tmp_path: Path):
    """Restricted inputs follow the _targets.R layout (llm_paper/_targets.R:L45, L100)
    under $LCP_DATA_ROOT; derived/, fits/ and logs/ are created under it."""
    monkeypatch.setenv(config.LCP_DATA_ROOT_ENV, str(tmp_path))
    root = tmp_path.resolve()
    assert config.restricted_path("ncds_1_2_3") == root / "ncds_1_2_3" / "ncds0123.dta"
    assert config.restricted_path("spelling_mistakes") == root / "spelling_mistakes.csv"
    for fn, name in [(config.derived_dir, "derived"), (config.fits_dir, "fits"),
                     (config.logs_dir, "logs")]:
        path = fn()
        assert path == root / name and path.is_dir()
    assert config.CAMSIS_FILE.is_relative_to(REPO / "data")  # public files stay in data/


def test_importing_config_never_touches_the_data_root(monkeypatch):
    """Only asking for a restricted path raises; importing the package does not."""
    monkeypatch.delenv(config.LCP_DATA_ROOT_ENV, raising=False)
    code = "import llm_cong_predict.config as c; print(c.VARIABLES_XLSX.name)"
    env = {k: v for k, v in os.environ.items() if k != config.LCP_DATA_ROOT_ENV}
    env["PYTHONPATH"] = str(REPO / "src")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and "variables.xlsx" in result.stdout


# --------------------------------------------------- external-API gate (GPT) --

_GATE_PROBE = r"""
import runpy, sys
sys.argv = ["get_gpt_embeddings.py"] + sys.argv[1:]
try:
    runpy.run_path({script!r}, run_name="__main__")
    code = 0
except SystemExit as exc:
    code = exc.code
print("EXIT", code)
print("OPENAI_IMPORTED", "openai" in sys.modules)
print("READERS_IMPORTED", "llm_cong_predict.io.readers" in sys.modules)
"""


@pytest.mark.parametrize(
    "flag, allow_env",
    [(False, False), (True, False), (False, True)],
    ids=["neither", "flag_only", "env_only"],
)
def test_gpt_script_refuses_without_double_opt_in(flag: bool, allow_env: bool):
    """R: llm_paper/R/get_gpt_embeddings.R:L19–27 sends essay text to OpenAI. The port
    refuses unless both the confirmation flag and LCP_ALLOW_EXTERNAL_API=1 are given,
    prints the project rule, and neither reads essays nor imports openai."""
    env = {k: v for k, v in os.environ.items() if k != "LCP_ALLOW_EXTERNAL_API"}
    if allow_env:
        env["LCP_ALLOW_EXTERNAL_API"] = "1"
    env["OPENAI_API_KEY"] = "sk-not-a-real-key"
    args = ["--i-confirm-the-data-licence-permits-external-processing"] if flag else []
    result = subprocess.run(
        [sys.executable, "-c", _GATE_PROBE.format(script=str(GPT_SCRIPT)), *args],
        capture_output=True, text=True, env=env,
    )
    assert "EXIT 3" in result.stdout
    assert "OPENAI_IMPORTED False" in result.stdout
    assert "READERS_IMPORTED False" in result.stdout
    assert "must never be sent to an external API" in result.stderr


def test_embeddings_extra_does_not_include_openai():
    """The local embedding extra must not pull in the OpenAI client (owner decision)."""
    import tomllib

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert not any(dep.startswith("openai") for dep in extras["embeddings"])
    assert any(dep.startswith("openai") for dep in extras["external-api"])
