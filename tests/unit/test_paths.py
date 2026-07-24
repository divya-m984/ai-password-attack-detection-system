"""Unit tests for password_attack_detector.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from password_attack_detector.exceptions import ConfigurationError
from password_attack_detector.paths import (
    ensure_dir,
    find_repo_root,
    get_artifacts_dir,
    get_configs_dir,
    get_data_dir,
    get_models_dir,
    get_reports_dir,
    reset_repo_root_cache,
)


class TestFindRepoRoot:
    def test_finds_real_repo_root(self) -> None:
        root = find_repo_root()
        assert (root / "pyproject.toml").exists()
        assert root.name == "ai-password-attack-detection-system"

    def test_start_override_finds_fake_root(self, fake_repo_root: Path) -> None:
        # Pass start= inside the fake repo to bypass the cache entirely.
        result = find_repo_root(start=fake_repo_root / "data")
        assert result == fake_repo_root

    def test_start_override_at_root_itself(self, fake_repo_root: Path) -> None:
        result = find_repo_root(start=fake_repo_root)
        assert result == fake_repo_root

    def test_raises_when_no_pyproject_found(self, tmp_path: Path) -> None:
        no_marker = tmp_path / "a" / "b" / "c"
        no_marker.mkdir(parents=True)
        with pytest.raises(ConfigurationError, match=r"pyproject\.toml"):
            find_repo_root(start=no_marker)

    def test_caches_result_on_second_call(self) -> None:
        first = find_repo_root()
        second = find_repo_root()
        assert first is second  # same object — cached

    def test_start_override_does_not_pollute_cache(
        self, fake_repo_root: Path, tmp_path: Path
    ) -> None:
        # Using start= should NOT overwrite the module-level cache.
        _ = find_repo_root(start=fake_repo_root)
        # Cache is still None (no-arg path never called), so next call
        # without start= should discover the real repo.
        real_root = find_repo_root()
        assert (real_root / "pyproject.toml").exists()
        assert real_root.name == "ai-password-attack-detection-system"

    def test_reset_cache_forces_rediscovery(self) -> None:
        _ = find_repo_root()
        reset_repo_root_cache()
        again = find_repo_root()
        assert (again / "pyproject.toml").exists()


class TestEnsureDir:
    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "nested" / "dir"
        assert not target.exists()
        result = ensure_dir(target)
        assert target.exists()
        assert result == target

    def test_idempotent_on_existing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "already_exists"
        target.mkdir()
        result = ensure_dir(target)  # must not raise
        assert result == target

    def test_returns_the_same_path(self, tmp_path: Path) -> None:
        target = tmp_path / "some_dir"
        result = ensure_dir(target)
        assert result == target


class TestDirectoryHelpers:
    def test_get_data_dir(self) -> None:
        result = get_data_dir()
        assert result.name == "data"
        assert result.parent == find_repo_root()

    def test_get_artifacts_dir(self) -> None:
        result = get_artifacts_dir()
        assert result.name == "artifacts"
        assert result.parent == find_repo_root()

    def test_get_models_dir(self) -> None:
        result = get_models_dir()
        assert result.name == "models"
        assert result.parent == find_repo_root()

    def test_get_reports_dir(self) -> None:
        result = get_reports_dir()
        assert result.name == "reports"
        assert result.parent == find_repo_root()

    def test_get_configs_dir(self) -> None:
        result = get_configs_dir()
        assert result.name == "configs"
        assert result.parent == find_repo_root()

    def test_no_mkdir_side_effects_on_import(self, tmp_path: Path) -> None:
        """Importing paths must not create any directories."""
        import importlib

        import password_attack_detector.paths as paths_module

        before = set(tmp_path.rglob("*"))
        importlib.reload(paths_module)
        after = set(tmp_path.rglob("*"))
        assert before == after
