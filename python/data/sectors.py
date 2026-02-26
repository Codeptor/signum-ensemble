"""Sector classification for S&P 500 stocks.

Provides GICS sector mapping for portfolio constraints. Uses a built-in
mapping for the major S&P 500 constituents, with optional dynamic lookup
via yfinance for unknown tickers.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
    "ISRG": "Technology",
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
    "DARDEN": "Consumer Discretionary",
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


def get_sector(ticker: str) -> str:
    """Get GICS sector for a ticker.

    Returns 'Unknown' if the ticker is not in the built-in mapping.
    """
    return SECTOR_MAP.get(ticker, "Unknown")


def get_sector_map(tickers: list[str]) -> dict[str, str]:
    """Get sector mapping for a list of tickers.

    Args:
        tickers: List of ticker symbols.

    Returns:
        Dict mapping ticker -> sector.
    """
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

    for _ in range(20):  # Iterate until convergence
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

        # Identify capped (overweight) and free (underweight) tickers
        capped_sectors = set(violations.keys())
        free_tickers = [t for t in w.index if smap.get(t, "Unknown") not in capped_sectors]

        # Scale down all overweight sectors to the cap
        excess = 0.0
        for sector, sector_wt in violations.items():
            scale = max_sector_weight / sector_wt
            sector_tickers = [t for t in w.index if smap.get(t, "Unknown") == sector]
            for t in sector_tickers:
                old = w[t]
                w[t] *= scale
                excess += old - w[t]
            logger.info(
                f"Sector '{sector}' overweight ({sector_wt:.1%} > {max_sector_weight:.1%}). "
                f"Scaled down {len(sector_tickers)} positions by {scale:.2f}x"
            )

        # Redistribute excess to non-capped tickers proportionally
        free_total = sum(w[t] for t in free_tickers)
        if free_total > 0 and excess > 0:
            scale_up = (free_total + excess) / free_total
            for t in free_tickers:
                w[t] *= scale_up

    # Final renormalization for numerical stability
    total = w.sum()
    if total > 0:
        w = w / total

    return {t: float(v) for t, v in w.items() if v > 0.001}
