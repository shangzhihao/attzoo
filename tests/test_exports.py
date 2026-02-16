"""Public package export tests."""

from attzoo import CombinedAttention, GatedSelfAttention


def test_package_exports_include_combined_and_gated() -> None:
    """Ensure documented classes are importable from top-level package."""
    assert CombinedAttention.__name__ == "CombinedAttention"
    assert GatedSelfAttention.__name__ == "GatedSelfAttention"
