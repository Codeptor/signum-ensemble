"""Sector classification for S&P 500 stocks.

Provides GICS sector mapping for portfolio constraints. Uses a built-in
mapping for the major S&P 500 constituents, with dynamic yfinance lookup
for tickers not in the static map (results are cached for the session).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for dynamic yfinance lookups (persists for the process lifetime)
_dynamic_sector_cache: dict[str, str] = {}

# GICS Sector mapping for major S&P 500 constituents.
# This covers ~200 of the most liquid names. Unknown tickers
# are classified as "Unknown" and excluded from sector constraints.
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "CSCO": "Technology",
    "ADBE": "Technology",
    "ACN": "Technology",
    "IBM": "Technology",
    "TXN": "Technology",
    "QCOM": "Technology",
    "INTU": "Technology",
    "AMD": "Technology",
    "AMAT": "Technology",
    "NOW": "Technology",
    "ISRG": "Health Care",
    "ADI": "Technology",
    "LRCX": "Technology",
    "KLAC": "Technology",
    "SNPS": "Technology",
    "CDNS": "Technology",
    "PANW": "Technology",
    "MCHP": "Technology",
    "FTNT": "Technology",
    "NXPI": "Technology",
    "MPWR": "Technology",
    "ON": "Technology",
    "ANSS": "Technology",
    "KEYS": "Technology",
    "INTC": "Technology",
    "HPQ": "Technology",
    "HPE": "Technology",
    "DELL": "Technology",
    "MSI": "Technology",
    # Communication Services
    "META": "Communication Services",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "TMUS": "Communication Services",
    "CHTR": "Communication Services",
    "EA": "Communication Services",
    "TTWO": "Communication Services",
    "WBD": "Communication Services",
    "OMC": "Communication Services",
    "MTCH": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",
    "CMG": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary",
    "AZO": "Consumer Discretionary",
    "MAR": "Consumer Discretionary",
    "HLT": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "ROST": "Consumer Discretionary",
    "DHI": "Consumer Discretionary",
    "LEN": "Consumer Discretionary",
    "YUM": "Consumer Discretionary",
    "DRI": "Consumer Discretionary",
    "POOL": "Consumer Discretionary",
    "APTV": "Consumer Discretionary",
    "BBY": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "CL": "Consumer Staples",
    "KMB": "Consumer Staples",
    "GIS": "Consumer Staples",
    "K": "Consumer Staples",
    "SJM": "Consumer Staples",
    "HSY": "Consumer Staples",
    "STZ": "Consumer Staples",
    "ADM": "Consumer Staples",
    "SYY": "Consumer Staples",
    "KR": "Consumer Staples",
    "EL": "Consumer Staples",
    "MKC": "Consumer Staples",
    "CHD": "Consumer Staples",
    "CAG": "Consumer Staples",
    "TSN": "Consumer Staples",
    "HRL": "Consumer Staples",
    # Health Care
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "DHR": "Health Care",
    "PFE": "Health Care",
    "AMGN": "Health Care",
    "BMY": "Health Care",
    "MDT": "Health Care",
    "GILD": "Health Care",
    "VRTX": "Health Care",
    "SYK": "Health Care",
    "ELV": "Health Care",
    "CI": "Health Care",
    "ZTS": "Health Care",
    "BDX": "Health Care",
    "BSX": "Health Care",
    "REGN": "Health Care",
    "HCA": "Health Care",
    "DXCM": "Health Care",
    "IQV": "Health Care",
    "IDXX": "Health Care",
    "EW": "Health Care",
    "A": "Health Care",
    "BIIB": "Health Care",
    "HOLX": "Health Care",
    "MTD": "Health Care",
    "BAX": "Health Care",
    "ALGN": "Health Care",
    # Financials
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "SPGI": "Financials",
    "BLK": "Financials",
    "C": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",
    "CB": "Financials",
    "PGR": "Financials",
    "MMC": "Financials",
    "ICE": "Financials",
    "AON": "Financials",
    "CME": "Financials",
    "MCO": "Financials",
    "MET": "Financials",
    "AIG": "Financials",
    "TRV": "Financials",
    "ALL": "Financials",
    "AFL": "Financials",
    "PRU": "Financials",
    "MSCI": "Financials",
    "AMP": "Financials",
    "USB": "Financials",
    "PNC": "Financials",
    "TFC": "Financials",
    "COF": "Financials",
    "BK": "Financials",
    "FIS": "Financials",
    "FISV": "Financials",
    "PYPL": "Financials",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "UNP": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "RTX": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "ADP": "Industrials",
    "MMM": "Industrials",
    "GD": "Industrials",
    "ITW": "Industrials",
    "NOC": "Industrials",
    "WM": "Industrials",
    "CSX": "Industrials",
    "NSC": "Industrials",
    "EMR": "Industrials",
    "ETN": "Industrials",
    "PCAR": "Industrials",
    "FDX": "Industrials",
    "PH": "Industrials",
    "TT": "Industrials",
    "CTAS": "Industrials",
    "CARR": "Industrials",
    "OTIS": "Industrials",
    "ROK": "Industrials",
    "FAST": "Industrials",
    "VRSK": "Industrials",
    "IR": "Industrials",
    "GWW": "Industrials",
    "CPRT": "Industrials",
    "AME": "Industrials",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "EOG": "Energy",
    "SLB": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "PXD": "Energy",
    "OXY": "Energy",
    "WMB": "Energy",
    "KMI": "Energy",
    "HAL": "Energy",
    "DVN": "Energy",
    "FANG": "Energy",
    "HES": "Energy",
    "BKR": "Energy",
    "CTRA": "Energy",
    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "AEP": "Utilities",
    "EXC": "Utilities",
    "SRE": "Utilities",
    "XEL": "Utilities",
    "ED": "Utilities",
    "WEC": "Utilities",
    "ES": "Utilities",
    "AWK": "Utilities",
    "DTE": "Utilities",
    "EIX": "Utilities",
    "PPL": "Utilities",
    "FE": "Utilities",
    "AEE": "Utilities",
    "CMS": "Utilities",
    "CNP": "Utilities",
    "ETR": "Utilities",
    "PEG": "Utilities",
    # Real Estate
    "PLD": "Real Estate",
    "AMT": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",
    "O": "Real Estate",
    "WELL": "Real Estate",
    "DLR": "Real Estate",
    "SPG": "Real Estate",
    "VICI": "Real Estate",
    "AVB": "Real Estate",
    "EQR": "Real Estate",
    "SBAC": "Real Estate",
    "WY": "Real Estate",
    "ARE": "Real Estate",
    "MAA": "Real Estate",
    "ESS": "Real Estate",
    "UDR": "Real Estate",
    "VTR": "Real Estate",
    "PEAK": "Real Estate",
    "KIM": "Real Estate",
    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "FCX": "Materials",
    "ECL": "Materials",
    "NEM": "Materials",
    "NUE": "Materials",
    "DOW": "Materials",
    "DD": "Materials",
    "VMC": "Materials",
    "MLM": "Materials",
    "PPG": "Materials",
    "EMN": "Materials",
    "IFF": "Materials",
    "ALB": "Materials",
    "CE": "Materials",
    "CF": "Materials",
    "PKG": "Materials",
    "IP": "Materials",
    "BLL": "Materials",
}

# Maximum weight per sector (25% cap prevents concentration risk)
DEFAULT_MAX_SECTOR_WEIGHT = 0.25


def _yf_lookup_sector(ticker: str) -> str:
    """Look up GICS sector from yfinance for a single ticker.

    Returns the sector string or 'Unknown' on failure.
    Result is cached in _dynamic_sector_cache for the session.
    """
    if ticker in _dynamic_sector_cache:
        return _dynamic_sector_cache[ticker]

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown") if info else "Unknown"
        # Normalize to our GICS names (yfinance uses slightly different labels)
        sector_normalization = {
            "Information Technology": "Technology",
            "Communication Services": "Communication Services",
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Defensive": "Consumer Staples",
            "Healthcare": "Health Care",
            "Financial Services": "Financials",
            "Basic Materials": "Materials",
        }
        sector = sector_normalization.get(sector, sector)
        _dynamic_sector_cache[ticker] = sector
        return sector
    except Exception as e:
        logger.debug(f"yfinance sector lookup failed for {ticker}: {e}")
        _dynamic_sector_cache[ticker] = "Unknown"
        return "Unknown"


def _batch_yf_lookup(tickers: list[str]) -> dict[str, str]:
    """Look up sectors for multiple tickers via yfinance (batched).

    Only looks up tickers not already in SECTOR_MAP or _dynamic_sector_cache.
    """
    unknown = [t for t in tickers if t not in SECTOR_MAP and t not in _dynamic_sector_cache]
    if not unknown:
        return {}

    logger.info(f"Looking up sectors for {len(unknown)} unmapped tickers via yfinance...")
    results: dict[str, str] = {}
    for ticker in unknown:
        results[ticker] = _yf_lookup_sector(ticker)

    found = sum(1 for v in results.values() if v != "Unknown")
    logger.info(f"Sector lookup complete: {found}/{len(unknown)} resolved")
    return results


def get_sector(ticker: str) -> str:
    """Get GICS sector for a ticker.

    Checks built-in SECTOR_MAP first, then dynamic cache, then
    falls back to yfinance lookup. Returns 'Unknown' only if all fail.
    """
    if ticker in SECTOR_MAP:
        return SECTOR_MAP[ticker]
    if ticker in _dynamic_sector_cache:
        return _dynamic_sector_cache[ticker]
    # Dynamic lookup — result is cached for future calls
    return _yf_lookup_sector(ticker)


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """Get sector mapping for a list of tickers.

    Performs batch yfinance lookup for any tickers not in the static map.

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dict mapping ticker -> sector.
    """
    # Batch lookup all unknown tickers at once
    _batch_yf_lookup(tickers)
    return {t: get_sector(t) for t in tickers}


def get_sector_weights(
    weights: dict[str, float],
    sector_map: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Calculate aggregate sector weights from position weights.

    Args:
        weights: Dict of ticker -> portfolio weight.
        sector_map: Optional custom mapping. Uses SECTOR_MAP if None.

    Returns:
        Dict of sector -> aggregate weight.
    """
    smap = sector_map or SECTOR_MAP
    sector_weights: dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = smap.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return sector_weights


def enforce_sector_constraints(
    weights: dict[str, float],
    max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
    sector_map: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Enforce sector weight constraints by scaling down overweight sectors.

    Iteratively scales down stocks in overweight sectors and redistributes
    to underweight positions. Preserves total weight normalization.

    Args:
        weights: Dict of ticker -> portfolio weight.
        max_sector_weight: Maximum allowed weight per sector.
        sector_map: Optional custom mapping. Uses SECTOR_MAP if None.

    Returns:
        Adjusted weights dict with all sectors within limits.
    """
    import pandas as pd

    smap = sector_map or SECTOR_MAP
    w = pd.Series(weights, dtype=float)

    # Always renormalize input to sum to 1.0
    total = w.sum()
    if total > 0:
        w = w / total

    # R3-S-1 fix: capped redistribution — only redistribute to sectors
    # that can absorb excess without exceeding the cap themselves.
    capped_sectors: set[str] = set()

    for iteration in range(50):  # generous limit for convergence
        # Calculate current sector weights
        sw: dict[str, float] = {}
        for ticker, weight in w.items():
            sector = smap.get(ticker, "Unknown")
            sw[sector] = sw.get(sector, 0.0) + weight

        # Find overweight sectors (skip "Unknown" — no constraint)
        violations = {
            s: wt for s, wt in sw.items() if wt > max_sector_weight + 1e-9 and s != "Unknown"
        }

        if not violations:
            break

        # Scale down ALL overweight sectors to the cap
        excess = 0.0
        for sector, sector_wt in violations.items():
            scale = max_sector_weight / sector_wt
            sector_tickers = [t for t in w.index if smap.get(t, "Unknown") == sector]
            for t in sector_tickers:
                old = w[t]
                w[t] *= scale
                excess += old - w[t]
            capped_sectors.add(sector)
            if iteration == 0:
                logger.info(
                    f"Sector '{sector}' overweight ({sector_wt:.1%} > {max_sector_weight:.1%}). "
                    f"Scaled down {len(sector_tickers)} positions by {scale:.2f}x"
                )

        # Redistribute excess ONLY to sectors that still have room
        free_tickers = [
            t
            for t in w.index
            if smap.get(t, "Unknown") not in capped_sectors and smap.get(t, "Unknown") != "Unknown"
        ]
        # Also include Unknown tickers as absorption targets
        free_tickers += [t for t in w.index if smap.get(t, "Unknown") == "Unknown"]

        if not free_tickers or excess <= 0:
            break

        # Compute how much each free sector can still absorb
        free_sector_room: dict[str, float] = {}
        free_sector_current: dict[str, float] = {}
        for t in free_tickers:
            s = smap.get(t, "Unknown")
            free_sector_current[s] = free_sector_current.get(s, 0.0) + w[t]
        for s, cur in free_sector_current.items():
            if s == "Unknown":
                free_sector_room[s] = float("inf")
            else:
                free_sector_room[s] = max(0.0, max_sector_weight - cur)

        total_room = sum(r for r in free_sector_room.values() if r != float("inf"))
        # If infinite room (only unknown tickers), distribute proportionally
        has_inf = any(r == float("inf") for r in free_sector_room.values())
        if total_room <= 0 and not has_inf:
            break  # No room anywhere — stop

        # Distribute proportionally to free tickers, capped by sector room
        free_total = sum(w[t] for t in free_tickers)
        if free_total > 0:
            distributable = min(excess, total_room) if not has_inf else excess
            scale_up = (free_total + distributable) / free_total
            for t in free_tickers:
                w[t] *= scale_up

    # Final renormalization for numerical stability
    total = w.sum()
    if total > 0:
        w = w / total

    # R3-S-2 fix: filter small weights THEN renormalize to maintain sum-to-1
    w = w[w > 0.001]
    total = w.sum()
    if total > 0:
        w = w / total

    return {t: float(v) for t, v in w.items()}
