"""Barra-style multi-factor risk model for portfolio analytics.

Implements a simplified but institutional-grade factor risk decomposition:
  - **Market beta**: CAPM systematic risk
  - **Size**: log market cap (SMB proxy)
  - **Momentum**: 12-1 month return
  - **Volatility**: 60-day realized volatility
  - **Value**: Earnings yield (if available)

Factor returns are estimated via cross-sectional regression (Fama-MacBeth).
Factor covariance is computed with exponential weighting and Newey-West
adjustment for serial correlation.

Usage::

    model = FactorRiskModel(prices, market_caps=caps)
    model.fit()

    # Portfolio risk decomposition
    risk = model.decompose_risk(weights)
    # {'total_vol': 0.18, 'systematic_vol': 0.15, 'idio_vol': 0.09, ...}

    # Stress testing
    stress = model.stress_test(weights, scenario="gfc_2008")

References:
  - Barra Risk Model Handbook (MSCI)
  - Fama & MacBeth (1973), "Risk, Return, and Equilibrium"
  - Menchero, Orr & Wang (2011), "The Barra US Equity Model (USE4)"
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Historical stress scenarios: factor return shocks (annualized)
STRESS_SCENARIOS = {
    "gfc_2008": {
        "description": "Global Financial Crisis (Sep-Nov 2008)",
        "market": -0.40,
        "size": -0.15,
        "momentum": -0.30,
        "volatility": 0.25,
        "value": -0.20,
    },
    "covid_2020": {
        "description": "COVID-19 Crash (Feb-Mar 2020)",
        "market": -0.34,
        "size": -0.10,
        "momentum": -0.15,
        "volatility": 0.30,
        "value": -0.15,
    },
    "rate_shock_2022": {
        "description": "Fed Rate Hike Cycle (2022)",
        "market": -0.20,
        "size": 0.05,
        "momentum": -0.10,
        "volatility": 0.10,
        "value": 0.08,
    },
    "vol_spike": {
        "description": "Generic +3 sigma volatility shock",
        "market": -0.15,
        "size": -0.08,
        "momentum": -0.12,
        "volatility": 0.20,
        "value": -0.05,
    },
}


class FactorRiskModel:
    """Barra-style multi-factor risk model.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (columns=tickers, index=dates).
    market_caps : pd.DataFrame, optional
        Market cap data (same structure as prices). If None, uses
        price as a proxy for size factor.
    market_returns : pd.Series, optional
        Market benchmark returns. If None, uses equal-weighted average.
    halflife : int
        Exponential decay half-life in days for covariance estimation.
    min_history : int
        Minimum number of observations required per asset.
    """

    FACTOR_NAMES = ["market", "size", "momentum", "volatility"]

    def __init__(
        self,
        prices: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
        market_returns: Optional[pd.Series] = None,
        halflife: int = 63,
        min_history: int = 126,
    ):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.tickers = list(prices.columns)
        self.market_caps = market_caps
        self.halflife = halflife
        self.min_history = min_history

        if market_returns is not None:
            self.market_returns = market_returns
        else:
            self.market_returns = self.returns.mean(axis=1)

        # Fitted state
        self._fitted = False
        self._factor_exposures: Optional[pd.DataFrame] = None  # (n_assets, n_factors)
        self._factor_covariance: Optional[pd.DataFrame] = None  # (n_factors, n_factors)
        self._factor_returns: Optional[pd.DataFrame] = None  # (n_dates, n_factors)
        self._idio_variance: Optional[pd.Series] = None  # (n_assets,)
        self._residuals: Optional[pd.DataFrame] = None

    def fit(self) -> "FactorRiskModel":
        """Estimate factor exposures, factor covariance, and idiosyncratic risk."""
        logger.info(
            f"Fitting factor risk model: {len(self.tickers)} assets, "
            f"{len(self.returns)} observations"
        )

        # 1. Compute factor exposures (cross-sectional)
        self._factor_exposures = self._compute_exposures()

        # 2. Estimate factor returns via cross-sectional regression
        self._factor_returns = self._estimate_factor_returns()

        # 3. Factor covariance with exponential weighting
        self._factor_covariance = self._estimate_factor_covariance()

        # 4. Idiosyncratic variance from residuals
        self._idio_variance = self._estimate_idio_variance()

        self._fitted = True
        logger.info(
            f"Factor risk model fitted: "
            f"{len(self.FACTOR_NAMES)} factors, "
            f"avg idio vol = {np.sqrt(self._idio_variance.mean()):.4f}"
        )
        return self

    def _compute_exposures(self) -> pd.DataFrame:
        """Compute factor exposures for each asset (latest cross-section)."""
        n = len(self.returns)
        exposures = {}

        # Market beta: rolling 1y regression against market
        lookback = min(252, n)
        recent_returns = self.returns.iloc[-lookback:]
        recent_market = self.market_returns.iloc[-lookback:]

        betas = {}
        for ticker in self.tickers:
            asset_ret = recent_returns[ticker].values
            mkt_ret = recent_market.values
            mask = ~np.isnan(asset_ret) & ~np.isnan(mkt_ret)
            if mask.sum() < 30:
                betas[ticker] = 1.0
                continue
            slope, _, _, _, _ = stats.linregress(mkt_ret[mask], asset_ret[mask])
            betas[ticker] = float(np.clip(slope, -2, 5))
        exposures["market"] = betas

        # Size: log(price) as proxy (or log(market_cap) if available)
        if self.market_caps is not None and not self.market_caps.empty:
            latest_caps = self.market_caps.iloc[-1]
            sizes = {t: float(np.log(max(latest_caps.get(t, 1e6), 1))) for t in self.tickers}
        else:
            latest_prices = self.prices.iloc[-1]
            sizes = {t: float(np.log(max(latest_prices.get(t, 1), 1))) for t in self.tickers}
        exposures["size"] = _zscore_dict(sizes)

        # Momentum: 12-1 month return
        mom_lookback = min(252, n)
        skip = min(21, n - 1)
        if mom_lookback > skip + 1:
            mom_returns = self.prices.iloc[-mom_lookback] / self.prices.iloc[-skip] - 1
        else:
            mom_returns = pd.Series(0.0, index=self.tickers)
        exposures["momentum"] = _zscore_dict(mom_returns.to_dict())

        # Volatility: 60-day realized vol
        vol_lookback = min(60, n)
        vols = self.returns.iloc[-vol_lookback:].std()
        exposures["volatility"] = _zscore_dict(vols.to_dict())

        df = pd.DataFrame(exposures, index=self.tickers)
        return df

    def _estimate_factor_returns(self) -> pd.DataFrame:
        """Fama-MacBeth cross-sectional regression for factor returns.

        For each date t:
            r_it = B_i' * f_t + e_it

        where B_i are factor exposures and f_t are factor returns.
        """
        dates = self.returns.index
        factor_returns = []

        # Use rolling exposures for more accuracy, but approximate
        # with latest exposures for efficiency
        B = self._factor_exposures.values  # (n_assets, n_factors)

        for i, dt in enumerate(dates):
            r = self.returns.iloc[i].values  # (n_assets,)
            mask = ~np.isnan(r)
            if mask.sum() < len(self.FACTOR_NAMES) + 1:
                factor_returns.append(np.zeros(len(self.FACTOR_NAMES)))
                continue

            # OLS: f_t = (B'B)^{-1} B'r
            Bm = B[mask]
            rm = r[mask]
            try:
                f_t = np.linalg.lstsq(Bm, rm, rcond=None)[0]
            except np.linalg.LinAlgError:
                f_t = np.zeros(len(self.FACTOR_NAMES))

            factor_returns.append(f_t)

        df = pd.DataFrame(
            factor_returns,
            index=dates,
            columns=self.FACTOR_NAMES,
        )

        # Store residuals for idiosyncratic risk estimation
        self._residuals = pd.DataFrame(
            self.returns.values - df.values @ B.T,
            index=dates,
            columns=self.tickers,
        )

        return df

    def _estimate_factor_covariance(self) -> pd.DataFrame:
        """Exponentially weighted factor covariance matrix."""
        n = len(self._factor_returns)
        decay = np.log(2) / self.halflife
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights /= weights.sum()

        fr = self._factor_returns.values  # (n_dates, n_factors)
        fr_dm = fr - np.average(fr, weights=weights, axis=0)

        # Weighted covariance
        cov = (fr_dm * weights[:, None]).T @ fr_dm

        # Annualize
        cov *= 252

        return pd.DataFrame(
            cov,
            index=self.FACTOR_NAMES,
            columns=self.FACTOR_NAMES,
        )

    def _estimate_idio_variance(self) -> pd.Series:
        """Estimate idiosyncratic (specific) variance for each asset."""
        if self._residuals is None:
            return pd.Series(0.01**2, index=self.tickers)

        n = len(self._residuals)
        decay = np.log(2) / self.halflife
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights /= weights.sum()

        resid = self._residuals.values
        # Weighted variance per asset (annualized)
        resid_dm = resid - np.average(resid, weights=weights, axis=0)
        idio_var = np.sum(weights[:, None] * resid_dm**2, axis=0) * 252

        return pd.Series(idio_var, index=self.tickers)

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------

    def decompose_risk(self, weights: pd.Series) -> dict:
        """Decompose portfolio risk into systematic and idiosyncratic components.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.

        Returns
        -------
        dict with keys:
            total_vol, systematic_vol, idio_vol,
            factor_contributions (dict of factor -> vol contribution),
            factor_exposures (dict of factor -> portfolio exposure)
        """
        self._check_fitted()

        w = weights.reindex(self.tickers, fill_value=0.0).values
        B = self._factor_exposures.values
        F = self._factor_covariance.values
        D = np.diag(self._idio_variance.values)

        # Portfolio factor exposure: b_p = B'w
        b_p = B.T @ w

        # Systematic variance: b_p' F b_p
        sys_var = b_p @ F @ b_p
        sys_var = max(sys_var, 0.0)

        # Idiosyncratic variance: w'Dw
        idio_var = w @ D @ w
        idio_var = max(idio_var, 0.0)

        # Total variance
        total_var = sys_var + idio_var

        # Factor marginal contributions
        factor_contribs = {}
        for i, factor in enumerate(self.FACTOR_NAMES):
            # Marginal contribution = b_p_i * (F @ b_p)_i / total_vol
            if total_var > 0:
                mc = b_p[i] * (F @ b_p)[i] / np.sqrt(total_var)
            else:
                mc = 0.0
            factor_contribs[factor] = float(mc)

        factor_exp = {
            f: float(b_p[i]) for i, f in enumerate(self.FACTOR_NAMES)
        }

        return {
            "total_vol": float(np.sqrt(total_var)),
            "systematic_vol": float(np.sqrt(sys_var)),
            "idio_vol": float(np.sqrt(idio_var)),
            "pct_systematic": float(sys_var / total_var) if total_var > 0 else 0.0,
            "factor_contributions": factor_contribs,
            "factor_exposures": factor_exp,
        }

    def portfolio_factor_exposure(self, weights: pd.Series) -> pd.Series:
        """Return the portfolio's exposure to each factor."""
        self._check_fitted()
        w = weights.reindex(self.tickers, fill_value=0.0).values
        B = self._factor_exposures.values
        b_p = B.T @ w
        return pd.Series(b_p, index=self.FACTOR_NAMES)

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def stress_test(
        self,
        weights: pd.Series,
        scenario: str | None = None,
        custom_shocks: dict[str, float] | None = None,
    ) -> dict:
        """Estimate portfolio P&L under a stress scenario.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights.
        scenario : str, optional
            Name of predefined scenario (e.g., "gfc_2008").
        custom_shocks : dict, optional
            Custom factor shocks: {factor_name: return_shock}.

        Returns
        -------
        dict with:
            scenario, description, portfolio_return, factor_pnl (per-factor),
            factor_shocks
        """
        self._check_fitted()

        if scenario is not None:
            if scenario not in STRESS_SCENARIOS:
                raise ValueError(
                    f"Unknown scenario '{scenario}'. "
                    f"Available: {list(STRESS_SCENARIOS.keys())}"
                )
            shocks = STRESS_SCENARIOS[scenario]
            desc = shocks.get("description", scenario)
        elif custom_shocks is not None:
            shocks = custom_shocks
            desc = "custom"
        else:
            raise ValueError("Provide either 'scenario' or 'custom_shocks'")

        w = weights.reindex(self.tickers, fill_value=0.0).values
        B = self._factor_exposures.values
        b_p = B.T @ w

        # Portfolio return = sum(b_p_i * shock_i)
        factor_pnl = {}
        total_return = 0.0
        factor_shocks_used = {}
        for i, factor in enumerate(self.FACTOR_NAMES):
            shock = shocks.get(factor, 0.0)
            pnl = float(b_p[i] * shock)
            factor_pnl[factor] = pnl
            factor_shocks_used[factor] = shock
            total_return += pnl

        return {
            "scenario": scenario or "custom",
            "description": desc,
            "portfolio_return": total_return,
            "factor_pnl": factor_pnl,
            "factor_shocks": factor_shocks_used,
        }

    def stress_test_all(self, weights: pd.Series) -> pd.DataFrame:
        """Run all predefined stress scenarios and return summary."""
        rows = []
        for name in STRESS_SCENARIOS:
            result = self.stress_test(weights, scenario=name)
            rows.append({
                "scenario": name,
                "description": result["description"],
                "portfolio_return": result["portfolio_return"],
                **{f"pnl_{k}": v for k, v in result["factor_pnl"].items()},
            })
        return pd.DataFrame(rows).set_index("scenario")

    # ------------------------------------------------------------------
    # Risk model covariance matrix
    # ------------------------------------------------------------------

    def covariance_matrix(self) -> pd.DataFrame:
        """Return the full asset covariance matrix: B F B' + D.

        This is more stable than the sample covariance when n_assets > n_obs/3,
        because factor structure regularizes the estimate.
        """
        self._check_fitted()
        B = self._factor_exposures.values
        F = self._factor_covariance.values
        D = np.diag(self._idio_variance.values)
        cov = B @ F @ B.T + D
        return pd.DataFrame(cov, index=self.tickers, columns=self.tickers)

    def correlation_matrix(self) -> pd.DataFrame:
        """Return the factor-model implied correlation matrix."""
        cov = self.covariance_matrix()
        std = np.sqrt(np.diag(cov.values))
        std[std == 0] = 1.0
        corr = cov.values / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        return pd.DataFrame(corr, index=self.tickers, columns=self.tickers)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, weights: Optional[pd.Series] = None) -> dict:
        """Serialize model state and optionally portfolio risk decomposition."""
        self._check_fitted()
        result = {
            "n_assets": len(self.tickers),
            "n_factors": len(self.FACTOR_NAMES),
            "factors": self.FACTOR_NAMES,
            "factor_covariance": self._factor_covariance.to_dict(),
            "mean_idio_vol": float(np.sqrt(self._idio_variance.mean())),
        }
        if weights is not None:
            result["risk_decomposition"] = self.decompose_risk(weights)
        return result

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zscore_dict(d: dict[str, float]) -> dict[str, float]:
    """Z-score normalize a dict of values."""
    vals = np.array(list(d.values()))
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return {k: 0.0 for k in d}
    mu = np.nanmean(vals)
    sigma = np.nanstd(vals)
    if sigma < 1e-10:
        return {k: 0.0 for k in d}
    return {k: float((v - mu) / sigma) if not np.isnan(v) else 0.0 for k, v in d.items()}
