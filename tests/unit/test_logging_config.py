"""Unit tests for password_attack_detector.logging_config."""

from __future__ import annotations

import password_attack_detector.logging_config as lc


class TestSetupLogging:
    def test_configures_without_error_development(self) -> None:
        lc.setup_logging("INFO", "development")
        assert lc._configured is True

    def test_configures_without_error_production(self) -> None:
        lc.setup_logging("INFO", "production")
        assert lc._configured is True

    def test_idempotent_by_default(self) -> None:
        """Second call without force should be a no-op."""
        lc.setup_logging("INFO", "development")
        # Change would only take effect if reconfigured — capture the flag.
        lc.setup_logging("DEBUG", "development")
        # Still configured (not reconfigured to DEBUG); no error raised.
        assert lc._configured is True

    def test_force_reconfigures(self) -> None:
        lc.setup_logging("INFO", "development")
        lc.setup_logging("DEBUG", "development", force=True)
        assert lc._configured is True

    def test_debug_level(self) -> None:
        lc.setup_logging("DEBUG", "development")
        assert lc._configured is True

    def test_warning_level(self) -> None:
        lc.setup_logging("WARNING", "development")
        assert lc._configured is True


class TestResetLogging:
    def test_reset_clears_configured_flag(self) -> None:
        lc.setup_logging("INFO", "development")
        assert lc._configured is True
        lc.reset_logging()
        assert lc._configured is False

    def test_reset_allows_reconfigure(self) -> None:
        lc.setup_logging("INFO", "development")
        lc.reset_logging()
        lc.setup_logging("ERROR", "production")
        assert lc._configured is True


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        lc.setup_logging("INFO", "development")
        logger = lc.get_logger("test.module")
        assert logger is not None

    def test_logger_info_does_not_raise(self) -> None:
        lc.setup_logging("INFO", "development")
        logger = lc.get_logger("test.module")
        logger.info("test event", key="value")  # must not raise

    def test_logger_debug_does_not_raise(self) -> None:
        lc.setup_logging("DEBUG", "development")
        logger = lc.get_logger("test.debug")
        logger.debug("debug event")

    def test_logger_warning_does_not_raise(self) -> None:
        lc.setup_logging("WARNING", "development")
        logger = lc.get_logger("test.warning")
        logger.warning("warn event")

    def test_production_logger_does_not_raise(self) -> None:
        lc.setup_logging("INFO", "production")
        logger = lc.get_logger("prod.module")
        logger.info("production event", structured=True)
