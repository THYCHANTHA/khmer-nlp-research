"""Every Streamlit page must load without an exception, with or without generated results."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py", *sorted(f"pages/{p.name}" for p in (APP / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_exception(page):
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_there_are_eight_pages():
    assert len(PAGES) == 8


def test_preprocessing_page_handles_empty_text():
    at = AppTest.from_file(str(APP / "pages/1_Text_Preprocessing.py"), default_timeout=120).run()
    at.text_area[0].set_value("").run()
    assert not at.exception


def test_preprocessing_page_handles_mixed_text():
    at = AppTest.from_file(str(APP / "pages/1_Text_Preprocessing.py"), default_timeout=120).run()
    at.text_area[0].set_value("Hello ខ្ញុំ 123 ។ ។").run()
    assert not at.exception
