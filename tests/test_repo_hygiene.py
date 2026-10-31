"""Publication-gate checks on the working tree (everything that is not git-ignored or generated locally)."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "raw", "processed", "local_dictionary_run"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cff", ".json", ".csv", ".cfg", ".gitignore"}


def tracked_files():
    """Files git tracks; without git (e.g. a source archive), the working tree minus generated folders."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True, encoding="utf-8"
        ).stdout.splitlines()
        for name in out:
            p = ROOT / name
            if p.is_file():
                yield p
        return
    except (OSError, subprocess.CalledProcessError):
        pass
    for p in ROOT.rglob("*"):
        parts = p.relative_to(ROOT).parts[:-1]
        if not p.is_file() or set(parts) & SKIP_DIRS or any(x.startswith(".venv") for x in parts):
            continue
        if p.suffix == ".log" or p.name == "_upload.csv":
            continue
        yield p


def text_files():
    for p in tracked_files():
        if p.name == Path(__file__).name:  # the scanner contains the patterns it searches for
            continue
        if p.suffix in TEXT_SUFFIXES or p.name in {".gitignore", "LICENSE"}:
            yield p, p.read_text(encoding="utf-8", errors="replace")


def test_no_email_addresses_other_than_noreply():
    pat = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    bad = [
        (str(p.relative_to(ROOT)), m)
        for p, t in text_files()
        for m in pat.findall(t)
        if "users.noreply.github.com" not in m and not m.endswith(("@example.com",))
    ]
    assert not bad, bad


def test_no_local_paths():
    pat = re.compile(r"[A-Za-z]:\\\\?(?:Users|thy-chantha|AI-Finance)|/Users/[a-z]|/home/[a-z]|AppData", re.I)
    bad = [str(p.relative_to(ROOT)) for p, t in text_files() if pat.search(t)]
    assert not bad, bad


def test_no_secret_patterns():
    pat = re.compile(
        r"(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*['\"][^'\"]{6,}['\"]|AIza[0-9A-Za-z_-]{20,}|ghp_[0-9A-Za-z]{20,}"
    )
    bad = [str(p.relative_to(ROOT)) for p, t in text_files() if pat.search(t)]
    assert not bad, bad


def test_no_third_party_team_materials_by_name():
    banned = {"FIle_Stopwords.csv", "khmer_stopwords_broad_classes.csv", "news_text_file_150k.txt"}
    present = [p.name for p in tracked_files() if p.name in banned or p.name.startswith("khmer_stopwords_")]
    assert not present, present


def test_only_expected_csv_files_are_present():
    allowed_roots = {("data", "samples"), ("data", "manifest"), ("reports", "experiments"), ("configs",)}
    for p in tracked_files():
        if p.suffix == ".csv":
            rel = p.relative_to(ROOT).parts
            assert any(rel[: len(a)] == a for a in allowed_roots), p


def test_no_raw_wikipedia_text_is_tracked():
    for p in tracked_files():
        assert p.suffix not in {".jsonl", ".gz"}, p


def test_sources_have_no_literal_invisible_characters():
    bad_chars = {chr(c) for c in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060, 0xAD)}
    for p, t in text_files():
        if p.suffix == ".py":
            assert not (set(t) & bad_chars), p


def test_licence_files_and_data_licence_notice_exist():
    for name in (
        "LICENSE",
        "CITATION.cff",
        "data/README.md",
        "docs/dataset_card.md",
        "docs/NEXT_STEPS.md",
        "docs/limitations.md",
    ):
        assert (ROOT / name).exists(), name
    assert "CC BY-SA" in (ROOT / "data" / "README.md").read_text(encoding="utf-8")


def test_the_synthetic_demo_dictionary_is_labelled_synthetic():
    readme = (ROOT / "data" / "samples" / "README.md").read_text(encoding="utf-8").replace("*", "")
    assert "synthetic" in readme.lower() and "not a research dictionary" in readme.lower()
