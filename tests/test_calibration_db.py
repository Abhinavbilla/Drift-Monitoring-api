"""
Tests for Step 2's DB persistence layer (db/crud.py): the calibration_config
column, its self-healing migration, and -- critically -- that it survives
re-fits rather than being silently wiped by INSERT OR REPLACE (verified as
a real risk in db/crud.py, not assumed).

Uses a throwaway sqlite file so these tests never touch the real drift.db.

Run:
    python -m pytest tests/test_calibration_db.py -v
"""

import os

import pytest

import db.crud as crud


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_drift.db")
    monkeypatch.setattr(crud, "DB_PATH", db_path)
    crud.init_db()
    yield db_path


class TestMigration:
    def test_calibration_config_column_exists_after_init(self):
        conn = crud.get_connection()
        cols = [row[1] for row in conn.execute("PRAGMA table_info(baselines)").fetchall()]
        conn.close()
        assert "calibration_config" in cols

    def test_running_init_db_twice_is_a_noop(self):
        """Self-healing migration must not error on a DB that already has
        the column (the try/except sqlite3.OperationalError pattern)."""
        crud.init_db()  # should not raise
        conn = crud.get_connection()
        cols = [row[1] for row in conn.execute("PRAGMA table_info(baselines)").fetchall()]
        conn.close()
        assert cols.count("calibration_config") == 1


class TestGetBaselineCalibrationConfig:
    def test_existing_project_no_config_returns_none(self):
        """A project fit with no calibration_config argument (the default
        for every existing caller) must come back as None -- resolves to
        legacy via CalibrationConfig.from_dict(None)."""
        crud.insert_baseline("proj_legacy", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]})
        state = crud.get_baseline("proj_legacy")
        assert state["calibration_config"] is None

    def test_explicit_config_at_fit_time(self):
        cfg = {"decision_mode": "calibrated", "alpha": 0.05}
        crud.insert_baseline("proj_calibrated", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]},
                              calibration_config=cfg)
        state = crud.get_baseline("proj_calibrated")
        assert state["calibration_config"] == cfg


class TestConfigSurvivesRefit:
    def test_refit_without_config_arg_preserves_existing_config(self):
        """THE critical regression test: INSERT OR REPLACE deletes and
        re-inserts the row, which (verified separately) wipes any column
        not named in the INSERT back to its default. A caller re-fitting a
        project (e.g. new reference data) without touching calibration
        settings must not silently revert that project to legacy."""
        cfg = {"decision_mode": "calibrated", "alpha": 0.05, "effect_floors": {"ks_d": 0.03}}
        crud.insert_baseline("proj_refit", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]},
                              calibration_config=cfg)
        assert crud.get_baseline("proj_refit")["calibration_config"] == cfg

        # Re-fit with new data, NOT passing calibration_config at all --
        # this is exactly what main.py's /fit handler does today.
        crud.insert_baseline("proj_refit", {"x": "continuous"}, {"x": [10, 20, 30, 40, 50]})

        state = crud.get_baseline("proj_refit")
        assert state["calibration_config"] == cfg, "calibration_config was wiped by re-fit"
        assert state["reference_data"]["x"] == [10, 20, 30, 40, 50]  # new data did apply

    def test_refit_can_explicitly_clear_config(self):
        cfg = {"decision_mode": "calibrated"}
        crud.insert_baseline("proj_clear", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]},
                              calibration_config=cfg)
        crud.insert_baseline("proj_clear", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]},
                              calibration_config=None)
        assert crud.get_baseline("proj_clear")["calibration_config"] is None

    def test_embedding_baseline_refit_preserves_config(self):
        cfg = {"decision_mode": "calibrated"}
        crud.insert_baseline("proj_embed", {}, {})  # ensure row exists first
        crud.set_calibration_config("proj_embed", cfg)
        crud.insert_embedding_baseline("proj_embed", "text", [[0.1, 0.2], [0.3, 0.4]], "test-model")
        assert crud.get_baseline("proj_embed")["calibration_config"] == cfg

    def test_joint_baseline_refit_preserves_config(self):
        cfg = {"decision_mode": "calibrated"}
        crud.insert_baseline("proj_joint", {}, {})
        crud.set_calibration_config("proj_joint", cfg)
        crud.insert_joint_baseline("proj_joint", [[0.1, 0.2]], {"a": {"mean": 0, "std": 1}}, "joint-v1")
        assert crud.get_baseline("proj_joint")["calibration_config"] == cfg


class TestSetCalibrationConfig:
    def test_set_then_get(self):
        crud.insert_baseline("proj_set", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]})
        cfg = {"decision_mode": "calibrated", "multiple_testing": "bh"}
        crud.set_calibration_config("proj_set", cfg)
        assert crud.get_baseline("proj_set")["calibration_config"] == cfg

    def test_set_none_clears(self):
        crud.insert_baseline("proj_clear2", {"x": "continuous"}, {"x": [1, 2, 3, 4, 5]},
                              calibration_config={"decision_mode": "calibrated"})
        crud.set_calibration_config("proj_clear2", None)
        assert crud.get_baseline("proj_clear2")["calibration_config"] is None
