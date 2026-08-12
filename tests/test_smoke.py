"""Repository smoke tests (Phase 1)."""

import bioneural


def test_version():
    assert bioneural.__version__ == "0.1.0-alpha"


def test_license():
    assert bioneural.__license__ == "MIT"


def test_author():
    assert "Bhandari" in bioneural.__author__
