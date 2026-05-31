"""Logging setup is idempotent and writes to disk."""

import logging

import pytest

import logging_setup


def _root_handler_count() -> int:
    return len(logging.getLogger().handlers)


@pytest.fixture
def _isolated_logging():
    """Snapshot + restore root logger state so these tests don't leak
    open file handles or the temp log directory into subsequent tests."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    try:
        yield
    finally:
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


def test_setup_is_idempotent(tmp_path, _isolated_logging):
    log = tmp_path / "a.log"
    logging_setup.setup("INFO", str(log))
    n1 = _root_handler_count()
    logging_setup.setup("DEBUG", str(log))
    n2 = _root_handler_count()
    assert n1 == n2 == 2          # one console + one rotating file


def test_log_messages_reach_disk(tmp_path, _isolated_logging):
    log = tmp_path / "b.log"
    logging_setup.setup("INFO", str(log))
    logging.getLogger("test").info("hello-disk-marker")
    for h in logging.getLogger().handlers:
        h.flush()
    contents = log.read_text()
    assert "hello-disk-marker" in contents


def test_setup_creates_log_dir_if_missing(tmp_path, _isolated_logging):
    nested = tmp_path / "subdir" / "nested" / "c.log"
    logging_setup.setup("INFO", str(nested))
    assert nested.parent.is_dir()
