"""Tests for sector constraint enforcement (Phase 2.1)."""

import pytest

from python.data.sectors import (
    DEFAULT_MAX_SECTOR_WEIGHT,
    SECTOR_MAP,
    enforce_sector_constraints,
    get_sector,
    get_sector_map,
    get_sector_weights,
)


def test_get_sector_known_ticker():
    assert get_sector("AAPL") == "Technology"
    assert get_sector("JPM") == "Financials"
    assert get_sector("XOM") == "Energy"


def test_get_sector_unknown_ticker():
    assert get_sector("ZZZZZ") == "Unknown"


def test_get_sector_map():
    tickers = ["AAPL", "MSFT", "JPM", "ZZZZZ"]
    smap = get_sector_map(tickers)
    assert smap["AAPL"] == "Technology"
    assert smap["MSFT"] == "Technology"
    assert smap["JPM"] == "Financials"
    assert smap["ZZZZZ"] == "Unknown"


def test_get_sector_weights():
    weights = {"AAPL": 0.3, "MSFT": 0.2, "JPM": 0.3, "XOM": 0.2}
    sw = get_sector_weights(weights)
    assert sw["Technology"] == pytest.approx(0.5)
    assert sw["Financials"] == pytest.approx(0.3)
    assert sw["Energy"] == pytest.approx(0.2)


def test_enforce_sector_constraints_no_violation():
    """When no sector exceeds the limit, weights should be unchanged."""
    weights = {"AAPL": 0.1, "JPM": 0.1, "XOM": 0.1, "UNH": 0.1, "PG": 0.1}
    # Sum per sector: Tech=0.1, Fin=0.1, Energy=0.1, HC=0.1, Staples=0.1
    result = enforce_sector_constraints(weights, max_sector_weight=0.25)
    # Should be unchanged (renormalized to same proportions)
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=0.01)
    # Proportions preserved
    for t in weights:
        assert result[t] == pytest.approx(1.0 / len(weights), abs=0.02)


def test_enforce_sector_constraints_with_violation():
    """When a sector exceeds 25%, it should be scaled down."""
    # 50% in Technology — should be reduced to 25%
    # Need enough sectors so constraints are feasible (4 * 25% = 100%)
    weights = {
        "AAPL": 0.25,
        "MSFT": 0.25,
        "JPM": 0.15,
        "XOM": 0.15,
        "UNH": 0.10,
        "PG": 0.10,
    }
    result = enforce_sector_constraints(weights, max_sector_weight=0.25)
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=0.01)

    # Technology sector should be at or below 25%
    tech_weight = sum(v for k, v in result.items() if get_sector(k) == "Technology")
    assert tech_weight <= 0.26  # Small tolerance for floating point

    # Other sectors should have picked up the redistributed weight
    fin_weight = sum(v for k, v in result.items() if get_sector(k) == "Financials")
    assert fin_weight > 0.15  # Financials should have grown from redistribution


def test_enforce_sector_constraints_preserves_normalization():
    """Weights should sum to ~1.0 after constraint enforcement."""
    weights = {
        "AAPL": 0.2,
        "MSFT": 0.2,
        "NVDA": 0.2,
        "JPM": 0.2,
        "XOM": 0.2,
    }
    result = enforce_sector_constraints(weights, max_sector_weight=0.25)
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_enforce_sector_constraints_filters_dust():
    """Weights below 0.001 should be removed."""
    weights = {
        "AAPL": 0.9,
        "JPM": 0.05,
        "XOM": 0.05,
    }
    result = enforce_sector_constraints(weights, max_sector_weight=0.25)
    for v in result.values():
        assert v > 0.001


def test_enforce_sector_constraints_unknown_sector():
    """Unknown sector tickers should not be constrained."""
    weights = {
        "AAPL": 0.1,
        "ZZZZZ": 0.9,  # Unknown sector
    }
    result = enforce_sector_constraints(weights, max_sector_weight=0.25)
    # Technology is only 10%, unknown is 90% — only tech would be checked
    # but tech is not overweight. Unknown is skipped.
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_default_max_sector_weight():
    assert DEFAULT_MAX_SECTOR_WEIGHT == 0.25


def test_sector_map_coverage():
    """SECTOR_MAP should cover the major sectors."""
    sectors = set(SECTOR_MAP.values())
    expected_sectors = {
        "Technology",
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Health Care",
        "Financials",
        "Industrials",
        "Energy",
        "Utilities",
        "Real Estate",
        "Materials",
    }
    assert sectors == expected_sectors
