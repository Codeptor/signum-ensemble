"""Signum — Quantitative Trading Cockpit.

Dark-themed monitoring dashboard for paper trading and backtest review.
Two modes:
- **Backtest** tab: historical strategy performance from data/processed/
- **Live** tab: real-time Alpaca paper trading monitoring with regime beacon

JSON API (LLM-friendly):
- ``GET /api``            — index of all available endpoints
- ``GET /api/status``     — system overview (regime, bot state, account)
- ``GET /api/account``    — Alpaca account details
- ``GET /api/positions``  — current open positions with P&L
- ``GET /api/regime``     — market regime (VIX, SPY drawdown, exposure)
- ``GET /api/backtest``   — backtest metrics and risk summary
- ``GET /api/equity``     — equity history time-series
- ``GET /api/risk``       — full risk engine summary from backtest
- ``GET /api/drift``      — latest feature drift report
- ``GET /api/bot``        — bot state (last trade, shutdown reason, etc.)
- ``GET /api/logs``       — latest bot log lines (?lines=N, default 80, max 500)

Public API (unchanged):
- ``create_dashboard(...)``  — standalone backtest-only app
- ``create_tabbed_dashboard()`` — full two-tab cockpit
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from flask import Response, jsonify

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/processed")
STATE_FILE = Path("data/bot_state.json")
BOT_LOG_FILE = Path(os.getenv("LOG_DIR", "/var/log/signum")) / "bot.log"
# Fallback: check for local log file when not on VPS
_LOCAL_LOG = Path("live_bot.log")

# ═══════════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════════

# Surfaces — 4-tier dark elevation, same blue-gray hue, lightness only
_VOID = "#08080d"  # deepest background (behind everything)
_CANVAS = "#0e0e14"  # app background
_SURFACE = "#151520"  # cards, panels
_ELEVATED = "#1c1c2a"  # dropdowns, overlays, hover states
_INSET = "#0b0b10"  # recessed areas (inputs, code blocks)

# Text hierarchy — 4 levels
_TEXT_PRIMARY = "#e2e2ea"
_TEXT_SECONDARY = "#8b8ba0"
_TEXT_MUTED = "#555568"
_TEXT_FAINT = "#3a3a4a"

# Borders — rgba for blending
_BORDER = "rgba(255, 255, 255, 0.06)"
_BORDER_STRONG = "rgba(255, 255, 255, 0.10)"
_BORDER_SUBTLE = "rgba(255, 255, 255, 0.03)"

# Semantic — functional color only, never decorative
_HEALTHY = "#22c55e"
_CAUTION = "#f59e0b"
_DANGER = "#ef4444"
_ACCENT = "#4488ff"  # interactive elements only

# Semantic muted (for backgrounds/fills)
_HEALTHY_DIM = "rgba(34, 197, 94, 0.08)"
_CAUTION_DIM = "rgba(245, 158, 11, 0.08)"
_DANGER_DIM = "rgba(239, 68, 68, 0.08)"

# Typography
_FONT_MONO = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Cascadia Code', monospace"
_FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

# Spacing scale (8px base)
_SP_1 = "4px"
_SP_2 = "8px"
_SP_3 = "12px"
_SP_4 = "16px"
_SP_5 = "24px"
_SP_6 = "32px"
_SP_7 = "48px"

# Radii — sharp-technical
_RADIUS_SM = "4px"
_RADIUS_MD = "6px"
_RADIUS_LG = "8px"

# Chart colors — muted palette for multi-series
_CHART_PALETTE = [
    "#4488ff",
    "#22c55e",
    "#f59e0b",
    "#a78bfa",
    "#f472b6",
    "#06b6d4",
    "#fb923c",
    "#34d399",
    "#818cf8",
    "#e879f9",
]

# ─── Plotly chart template (shared across all figures) ──────────
_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_INSET,
    font=dict(family=_FONT_MONO, size=11, color=_TEXT_SECONDARY),
    title=dict(font=dict(family=_FONT_SANS, size=13, color=_TEXT_PRIMARY), x=0, xanchor="left"),
    margin=dict(l=48, r=16, t=40, b=36),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=10),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color=_TEXT_MUTED),
        orientation="h",
        y=-0.15,
    ),
    hoverlabel=dict(
        bgcolor=_ELEVATED,
        bordercolor=_BORDER_STRONG,
        font=dict(family=_FONT_MONO, size=11, color=_TEXT_PRIMARY),
    ),
    height=300,
)


def _fig(**overrides) -> go.Figure:
    """Create a pre-themed empty figure."""
    layout = {**_CHART_LAYOUT, **overrides}
    return go.Figure(layout=layout)


# ═══════════════════════════════════════════════════════════════════
# Component Primitives
# ═══════════════════════════════════════════════════════════════════


def _metric(label: str, value: str, *, color: str | None = None, mono: bool = True) -> html.Div:
    """Single metric readout — label above, value below."""
    val_style: dict = {
        "fontSize": "20px",
        "fontWeight": "600",
        "letterSpacing": "-0.02em",
        "lineHeight": "1.2",
        "fontFamily": _FONT_MONO if mono else _FONT_SANS,
        "fontFeatureSettings": "'tnum' 1" if mono else "normal",
    }
    if color:
        val_style["color"] = color
    else:
        val_style["color"] = _TEXT_PRIMARY

    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize": "11px",
                    "fontFamily": _FONT_SANS,
                    "fontWeight": "500",
                    "color": _TEXT_MUTED,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                    "marginBottom": _SP_1,
                },
            ),
            html.Div(value, style=val_style),
        ],
        style={
            "padding": f"{_SP_3} {_SP_4}",
            "minWidth": "100px",
        },
    )


def _metric_colored(label: str, value: str, color: str) -> html.Div:
    """Metric with semantic color on the value."""
    return _metric(label, value, color=color)


def _panel(*children, **style_overrides) -> html.Div:
    """Surface-level panel with border."""
    base_style = {
        "backgroundColor": _SURFACE,
        "border": f"1px solid {_BORDER}",
        "borderRadius": _RADIUS_MD,
        "padding": _SP_4,
    }
    base_style.update(style_overrides)
    return html.Div(list(children), style=base_style)


def _section_label(text: str) -> html.Div:
    """Section heading — small caps, muted."""
    return html.Div(
        text,
        style={
            "fontSize": "11px",
            "fontFamily": _FONT_SANS,
            "fontWeight": "600",
            "color": _TEXT_MUTED,
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "marginBottom": _SP_3,
        },
    )


def _row(*children, gap: str = _SP_4) -> html.Div:
    """Horizontal flex row."""
    return html.Div(
        list(children),
        style={"display": "flex", "gap": gap, "flexWrap": "wrap"},
    )


def _grid(*children, cols: str = "1fr 1fr", gap: str = _SP_4) -> html.Div:
    """CSS Grid container."""
    return html.Div(
        list(children),
        style={
            "display": "grid",
            "gridTemplateColumns": cols,
            "gap": gap,
        },
    )


def _status_dot(color: str) -> html.Span:
    """Tiny colored indicator dot."""
    return html.Span(
        style={
            "display": "inline-block",
            "width": "8px",
            "height": "8px",
            "borderRadius": "50%",
            "backgroundColor": color,
            "marginRight": _SP_2,
            "verticalAlign": "middle",
            "boxShadow": f"0 0 6px {color}40",
        }
    )


def _empty_state(message: str) -> html.Div:
    """Centered muted message for missing data."""
    return html.Div(
        message,
        style={
            "padding": _SP_7,
            "textAlign": "center",
            "color": _TEXT_MUTED,
            "fontSize": "13px",
            "fontFamily": _FONT_SANS,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Regime Beacon — the signature element
# ═══════════════════════════════════════════════════════════════════


def _regime_beacon(regime_state: dict | None) -> html.Div:
    """Persistent status strip at the top — master caution light.

    Always visible. Glows green/amber/red based on market regime.
    """
    if regime_state is None:
        return html.Div(
            _row(
                _status_dot(_TEXT_MUTED),
                html.Span(
                    "REGIME DATA UNAVAILABLE",
                    style={
                        "fontSize": "11px",
                        "fontFamily": _FONT_MONO,
                        "fontWeight": "600",
                        "color": _TEXT_MUTED,
                        "letterSpacing": "0.06em",
                    },
                ),
            ),
            style={
                "padding": f"{_SP_2} {_SP_4}",
                "backgroundColor": _VOID,
                "borderBottom": f"1px solid {_BORDER}",
            },
        )

    regime = regime_state.get("regime", "unknown")
    vix = regime_state.get("vix", 0)
    spy_dd = regime_state.get("spy_drawdown", 0)
    exposure = regime_state.get("exposure_multiplier", 1.0)

    color_map = {"normal": _HEALTHY, "caution": _CAUTION, "halt": _DANGER}
    bg_map = {"normal": _HEALTHY_DIM, "caution": _CAUTION_DIM, "halt": _DANGER_DIM}
    color = color_map.get(regime, _TEXT_MUTED)
    bg = bg_map.get(regime, _VOID)

    return html.Div(
        html.Div(
            [
                _status_dot(color),
                html.Span(
                    regime.upper(),
                    style={
                        "fontSize": "11px",
                        "fontFamily": _FONT_MONO,
                        "fontWeight": "700",
                        "color": color,
                        "letterSpacing": "0.08em",
                        "marginRight": _SP_5,
                    },
                ),
                html.Span(
                    f"VIX {vix:.1f}",
                    style={
                        "fontSize": "11px",
                        "fontFamily": _FONT_MONO,
                        "color": _TEXT_SECONDARY,
                        "marginRight": _SP_4,
                    },
                ),
                html.Span(
                    f"SPY DD {spy_dd:.1%}",
                    style={
                        "fontSize": "11px",
                        "fontFamily": _FONT_MONO,
                        "color": _DANGER if spy_dd < -0.10 else _TEXT_SECONDARY,
                        "marginRight": _SP_4,
                    },
                ),
                html.Span(
                    f"EXPOSURE {exposure:.0%}",
                    style={
                        "fontSize": "11px",
                        "fontFamily": _FONT_MONO,
                        "color": _CAUTION if exposure < 1.0 else _TEXT_SECONDARY,
                    },
                ),
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        style={
            "padding": f"{_SP_2} {_SP_4}",
            "backgroundColor": bg,
            "borderBottom": f"1px solid {_BORDER}",
            "borderLeft": f"3px solid {color}",
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Chart Builders — Backtest
# ═══════════════════════════════════════════════════════════════════


def _chart_cumulative(cumulative: pd.Series) -> go.Figure:
    fig = _fig(title=dict(text="Cumulative Returns"))
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=_ACCENT, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(68, 136, 255, 0.06)",
        )
    )
    fig.update_yaxes(title_text="Growth of $1")
    return fig


def _chart_drawdown(drawdown: pd.Series) -> go.Figure:
    fig = _fig(title=dict(text="Drawdown"))
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=_DANGER, width=1),
            fillcolor="rgba(239, 68, 68, 0.10)",
        )
    )
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
    return fig


def _chart_rolling_sharpe(rolling_sharpe: pd.Series) -> go.Figure:
    fig = _fig(title=dict(text="Rolling Sharpe (60d)"))
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode="lines",
            name="Sharpe",
            line=dict(color=_ACCENT, width=1.5),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=_TEXT_FAINT, line_width=1)
    fig.update_yaxes(title_text="Sharpe Ratio")
    return fig


def _chart_weights_bar(weights: pd.Series) -> go.Figure:
    weights = weights.sort_values(ascending=True)
    fig = _fig(title=dict(text="Portfolio Weights"))
    fig.add_trace(
        go.Bar(
            x=weights.values,
            y=weights.index,
            orientation="h",
            marker=dict(color=_ACCENT, line=dict(width=0)),
        )
    )
    fig.update_xaxes(title_text="Weight", tickformat=".0%")
    fig.update_layout(height=max(250, len(weights) * 28 + 60))
    return fig


def _chart_turnover(turnover: pd.Series) -> go.Figure:
    fig = _fig(title=dict(text="Rebalance Turnover"))
    avg = turnover.mean()
    fig.add_trace(
        go.Bar(
            x=turnover.index,
            y=turnover.values,
            name="Turnover",
            marker=dict(color="rgba(68, 136, 255, 0.5)", line=dict(width=0)),
        )
    )
    fig.add_hline(
        y=avg,
        line_dash="dot",
        line_color=_CAUTION,
        line_width=1,
        annotation_text=f"avg {avg:.1%}",
        annotation_font=dict(size=10, color=_CAUTION),
    )
    fig.update_yaxes(title_text="Turnover", tickformat=".0%")
    return fig


def _chart_concentration(weights: pd.Series) -> go.Figure:
    """Effective number of bets — horizontal gauge style."""
    hhi = (weights**2).sum()
    eff_n = 1.0 / hhi if hhi > 0 else 0
    n_assets = len(weights)

    fig = _fig(title=dict(text="Concentration"))
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=eff_n,
            number=dict(
                font=dict(family=_FONT_MONO, size=28, color=_TEXT_PRIMARY),
                suffix=f" / {n_assets}",
            ),
            title=dict(text="Effective Bets", font=dict(size=11, color=_TEXT_MUTED)),
            gauge=dict(
                axis=dict(range=[0, n_assets], tickfont=dict(size=9, color=_TEXT_FAINT)),
                bar=dict(color=_ACCENT),
                bgcolor=_INSET,
                borderwidth=0,
                steps=[
                    {"range": [0, n_assets * 0.3], "color": "rgba(239, 68, 68, 0.15)"},
                    {
                        "range": [n_assets * 0.3, n_assets * 0.7],
                        "color": "rgba(245, 158, 11, 0.10)",
                    },
                    {"range": [n_assets * 0.7, n_assets], "color": "rgba(34, 197, 94, 0.10)"},
                ],
                shape="bullet",
            ),
        )
    )
    fig.update_layout(height=160, margin=dict(l=48, r=16, t=48, b=16))
    return fig


def _chart_risk_attribution(risk_contributions: dict) -> go.Figure:
    """Horizontal bar chart of risk contribution by asset."""
    sorted_items = sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig = _fig(title=dict(text="Risk Contribution"))
    fig.add_trace(
        go.Bar(
            x=values,
            y=tickers,
            orientation="h",
            marker=dict(
                color=[_DANGER if v > 0.25 else _CAUTION if v > 0.15 else _ACCENT for v in values],
                line=dict(width=0),
            ),
        )
    )
    fig.update_xaxes(title_text="Contribution", tickformat=".0%")
    fig.update_layout(height=max(220, len(tickers) * 26 + 60))
    return fig


def _chart_rolling_var(rolling_var: pd.Series) -> go.Figure:
    fig = _fig(title=dict(text="Rolling VaR (95%, 63d)"))
    fig.add_trace(
        go.Scatter(
            x=rolling_var.index,
            y=(rolling_var * 100).tolist(),
            mode="lines",
            fill="tozeroy",
            name="VaR",
            line=dict(color=_DANGER, width=1),
            fillcolor="rgba(239, 68, 68, 0.08)",
        )
    )
    fig.update_yaxes(title_text="VaR (%)")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Chart Builders — Live
# ═══════════════════════════════════════════════════════════════════


def _chart_positions_value(positions: list) -> go.Figure:
    """Horizontal bar of position market values."""
    positions = sorted(positions, key=lambda p: p.market_value)
    symbols = [p.symbol for p in positions]
    values = [p.market_value for p in positions]
    colors = [_ACCENT if v >= 0 else _DANGER for v in values]

    fig = _fig(title=dict(text="Position Values"))
    fig.add_trace(
        go.Bar(x=values, y=symbols, orientation="h", marker=dict(color=colors, line=dict(width=0)))
    )
    fig.update_xaxes(title_text="Market Value ($)")
    fig.update_layout(height=max(220, len(symbols) * 28 + 60))
    return fig


def _chart_pnl_bar(positions: list) -> go.Figure:
    """Horizontal bar of unrealized P&L per position."""
    positions = sorted(positions, key=lambda p: p.unrealized_pl)
    symbols = [p.symbol for p in positions]
    pnls = [p.unrealized_pl for p in positions]
    colors = [_HEALTHY if v >= 0 else _DANGER for v in pnls]

    fig = _fig(title=dict(text="Unrealized P&L"))
    fig.add_trace(
        go.Bar(x=pnls, y=symbols, orientation="h", marker=dict(color=colors, line=dict(width=0)))
    )
    fig.update_xaxes(title_text="P&L ($)")
    fig.update_layout(height=max(220, len(symbols) * 28 + 60))
    return fig


def _chart_allocation_ring(weights_dict: dict) -> go.Figure:
    """Donut chart of portfolio allocation."""
    labels = list(weights_dict.keys())
    values = list(weights_dict.values())

    fig = _fig(title=dict(text="Allocation"))
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(
                colors=_CHART_PALETTE[: len(labels)],
                line=dict(color=_SURFACE, width=2),
            ),
            textinfo="label+percent",
            textfont=dict(size=10, family=_FONT_MONO),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=8, r=8, t=40, b=8),
    )
    return fig


def _chart_equity_curve(equity_data: pd.Series) -> go.Figure:
    """Equity curve with simple area fill."""
    fig = _fig(title=dict(text="Equity Curve"))
    fig.add_trace(
        go.Scatter(
            x=equity_data.index,
            y=equity_data.values,
            mode="lines",
            name="Equity",
            line=dict(color=_ACCENT, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(68, 136, 255, 0.06)",
        )
    )
    fig.update_yaxes(title_text="Equity ($)", tickprefix="$", tickformat=",.0f")
    fig.update_layout(height=260)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Positions Table
# ═══════════════════════════════════════════════════════════════════


def _positions_table(positions: list) -> html.Table:
    """Compact monospace table of open positions."""
    cols = ["Symbol", "Qty", "Avg Entry", "Mkt Value", "P&L ($)", "P&L (%)"]

    header_style = {
        "padding": f"{_SP_2} {_SP_3}",
        "textAlign": "right",
        "fontSize": "10px",
        "fontFamily": _FONT_SANS,
        "fontWeight": "600",
        "color": _TEXT_MUTED,
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "borderBottom": f"1px solid {_BORDER_STRONG}",
    }
    header = html.Tr(
        [
            html.Th(c, style={**header_style, "textAlign": "left" if i == 0 else "right"})
            for i, c in enumerate(cols)
        ]
    )

    rows = []
    cell_base = {
        "padding": f"{_SP_2} {_SP_3}",
        "fontSize": "12px",
        "fontFamily": _FONT_MONO,
        "fontFeatureSettings": "'tnum' 1",
        "color": _TEXT_PRIMARY,
        "borderBottom": f"1px solid {_BORDER_SUBTLE}",
    }
    for p in sorted(positions, key=lambda x: abs(x.market_value), reverse=True):
        pnl_color = _HEALTHY if p.unrealized_pl >= 0 else _DANGER
        rows.append(
            html.Tr(
                [
                    html.Td(
                        p.symbol,
                        style={**cell_base, "textAlign": "left", "fontWeight": "600"},
                    ),
                    html.Td(f"{p.qty:.0f}", style={**cell_base, "textAlign": "right"}),
                    html.Td(
                        f"${p.avg_entry_price:,.2f}", style={**cell_base, "textAlign": "right"}
                    ),
                    html.Td(f"${p.market_value:,.2f}", style={**cell_base, "textAlign": "right"}),
                    html.Td(
                        f"${p.unrealized_pl:+,.2f}",
                        style={**cell_base, "textAlign": "right", "color": pnl_color},
                    ),
                    html.Td(
                        f"{p.unrealized_plpc:+.2%}",
                        style={**cell_base, "textAlign": "right", "color": pnl_color},
                    ),
                ]
            )
        )

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "tableLayout": "fixed",
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Tab Styles
# ═══════════════════════════════════════════════════════════════════

_TAB_STYLE = {
    "padding": f"{_SP_2} {_SP_5}",
    "border": "none",
    "borderBottom": "2px solid transparent",
    "background": "transparent",
    "cursor": "pointer",
    "fontFamily": _FONT_SANS,
    "fontWeight": "500",
    "fontSize": "12px",
    "color": _TEXT_MUTED,
    "letterSpacing": "0.02em",
    "textTransform": "uppercase",
}
_TAB_SELECTED_STYLE = {
    **_TAB_STYLE,
    "color": _TEXT_PRIMARY,
    "fontWeight": "600",
    "borderBottom": f"2px solid {_ACCENT}",
}


# ═══════════════════════════════════════════════════════════════════
# Backtest Tab
# ═══════════════════════════════════════════════════════════════════


def _build_backtest_tab() -> html.Div:
    """Backtest results: metrics strip + charts grid."""
    result = _load_backtest_results()
    if result is None:
        return _empty_state("No backtest results found. Run 'make backtest' to generate data.")

    portfolio_returns, weights, risk_summary, rolling_sharpe, turnover, metrics = result

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    # ── Metrics strip ──
    sharpe = risk_summary.get("sharpe_ratio", risk_summary.get("sharpe", 0))
    sharpe_color = _HEALTHY if sharpe > 0.5 else _CAUTION if sharpe > 0 else _DANGER

    ann_ret = risk_summary.get("annualized_return", 0)
    ret_color = _HEALTHY if ann_ret > 0.08 else _CAUTION if ann_ret > 0 else _DANGER

    max_dd = risk_summary.get("max_drawdown", 0)
    dd_color = _HEALTHY if max_dd > -0.15 else _CAUTION if max_dd > -0.25 else _DANGER

    kpis = _panel(
        _row(
            _metric_colored("Sharpe", f"{sharpe:.2f}", sharpe_color),
            _metric("Sortino", f"{risk_summary.get('sortino_ratio', 0):.2f}"),
            _metric("Calmar", f"{risk_summary.get('calmar_ratio', 0):.2f}"),
            _metric_colored("Max DD", f"{max_dd:.1%}", dd_color),
            _metric_colored("Ann. Return", f"{ann_ret:.1%}", ret_color),
            _metric("VaR 95%", f"{risk_summary.get('var_95_historical', 0):.1%}"),
            _metric("CVaR 95%", f"{risk_summary.get('cvar_95', 0):.1%}"),
            _metric("Omega", f"{risk_summary.get('omega_ratio', 0):.2f}"),
            *(
                [_metric("Avg Turn", f"{metrics['avg_turnover']:.1%}")]
                if metrics and "avg_turnover" in metrics
                else []
            ),
            *(
                [_metric("Cost", f"{metrics['total_cost_bps']:.0f} bps", mono=True)]
                if metrics and "total_cost_bps" in metrics
                else []
            ),
        ),
        padding=f"{_SP_3} {_SP_2}",
    )

    # ── Charts ──
    charts_top = _grid(
        _panel(dcc.Graph(figure=_chart_cumulative(cumulative), config={"displayModeBar": False})),
        _panel(dcc.Graph(figure=_chart_drawdown(drawdown), config={"displayModeBar": False})),
    )
    charts_mid = _grid(
        _panel(
            dcc.Graph(
                figure=_chart_rolling_sharpe(rolling_sharpe), config={"displayModeBar": False}
            )
        ),
        _panel(dcc.Graph(figure=_chart_weights_bar(weights), config={"displayModeBar": False})),
    )

    extra_rows = []
    if turnover is not None:
        _turnover: pd.Series = turnover if isinstance(turnover, pd.Series) else pd.Series(turnover)
        extra_rows.append(
            _grid(
                _panel(
                    dcc.Graph(figure=_chart_turnover(_turnover), config={"displayModeBar": False})
                ),
                _panel(
                    dcc.Graph(
                        figure=_chart_concentration(weights), config={"displayModeBar": False}
                    )
                ),
            )
        )

    return html.Div(
        [
            kpis,
            html.Div(style={"height": _SP_4}),
            charts_top,
            html.Div(style={"height": _SP_4}),
            charts_mid,
        ]
        + [html.Div(style={"height": _SP_4})] * len(extra_rows)
        + extra_rows,
    )


# ═══════════════════════════════════════════════════════════════════
# Live Tab
# ═══════════════════════════════════════════════════════════════════


def _build_live_tab() -> html.Div:
    """Live paper trading: account → positions → equity history."""

    # ── Credentials check ──
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        return _panel(
            _section_label("Configuration Required"),
            html.Div(
                "Set ALPACA_API_KEY and ALPACA_API_SECRET to enable live monitoring.",
                style={
                    "color": _TEXT_SECONDARY,
                    "fontSize": "13px",
                    "fontFamily": _FONT_SANS,
                    "marginBottom": _SP_3,
                },
            ),
            html.Pre(
                "export ALPACA_API_KEY=your_key\nexport ALPACA_API_SECRET=your_secret",
                style={
                    "fontFamily": _FONT_MONO,
                    "fontSize": "12px",
                    "color": _TEXT_SECONDARY,
                    "backgroundColor": _INSET,
                    "padding": _SP_3,
                    "borderRadius": _RADIUS_SM,
                    "border": f"1px solid {_BORDER}",
                    "margin": "0",
                },
            ),
        )

    # ── Connect to broker ──
    account = None
    positions = []
    open_orders = []
    broker_error = None

    try:
        from python.brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper_trading=True)
        broker.connect()
        account = broker.get_account()
        positions = broker.list_positions()
        open_orders = broker.list_orders(status="open")
    except Exception as e:
        broker_error = str(e)
        logger.warning(f"Failed to connect to Alpaca: {e}")

    bot_state = _load_bot_state()

    # ── Timestamp ──
    timestamp = datetime.now().strftime("%H:%M:%S")
    children: list = [
        html.Div(
            f"Updated {timestamp}",
            style={
                "fontSize": "10px",
                "fontFamily": _FONT_MONO,
                "color": _TEXT_FAINT,
                "marginBottom": _SP_3,
            },
        ),
    ]

    # ── Broker error ──
    if broker_error:
        children.append(
            html.Div(
                [_status_dot(_DANGER), html.Span(f"Broker: {broker_error}")],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "backgroundColor": _DANGER_DIM,
                    "border": "1px solid rgba(239, 68, 68, 0.20)",
                    "borderRadius": _RADIUS_MD,
                    "padding": f"{_SP_3} {_SP_4}",
                    "marginBottom": _SP_4,
                    "fontSize": "12px",
                    "fontFamily": _FONT_SANS,
                    "color": _TEXT_PRIMARY,
                },
            )
        )

    # ── Account metrics ──
    if account is not None:
        # Compute day P&L from bot state
        day_pnl = None
        if bot_state and "last_equity" in bot_state:
            try:
                day_pnl = float(account.equity) - float(bot_state["last_equity"])
            except (ValueError, TypeError):
                pass

        acct_row_items = [
            _metric("Equity", f"${account.equity:,.0f}"),
            _metric("Cash", f"${account.cash:,.0f}"),
            _metric("Buying Power", f"${account.buying_power:,.0f}"),
            _metric("Positions", str(len(positions)), mono=False),
            _metric("Open Orders", str(len(open_orders)), mono=False),
        ]
        if day_pnl is not None:
            pnl_color = _HEALTHY if day_pnl >= 0 else _DANGER
            acct_row_items.insert(2, _metric_colored("Day P&L", f"${day_pnl:+,.0f}", pnl_color))

        children.append(
            _panel(_row(*acct_row_items), padding=f"{_SP_3} {_SP_2}", marginBottom=_SP_4)
        )

    # ── Positions ──
    children.append(_section_label("Positions"))

    if not positions:
        children.append(_panel(_empty_state("No open positions"), marginBottom=_SP_4))
    else:
        # Summary metrics
        total_pnl = sum(p.unrealized_pl for p in positions)
        total_mv = sum(p.market_value for p in positions)
        pnl_color = _HEALTHY if total_pnl >= 0 else _DANGER

        children.append(
            _panel(
                _row(
                    _metric("Market Value", f"${total_mv:,.0f}"),
                    _metric_colored("Unrealized P&L", f"${total_pnl:+,.2f}", pnl_color),
                ),
                padding=f"{_SP_3} {_SP_2}",
                marginBottom=_SP_3,
            )
        )

        # Table
        children.append(
            _panel(
                _positions_table(positions),
                padding=_SP_3,
                marginBottom=_SP_3,
            )
        )

        # Position charts
        children.append(
            _grid(
                _panel(
                    dcc.Graph(
                        figure=_chart_positions_value(positions), config={"displayModeBar": False}
                    )
                ),
                _panel(
                    dcc.Graph(figure=_chart_pnl_bar(positions), config={"displayModeBar": False})
                ),
                gap=_SP_3,
            )
        )

        # Allocation ring
        if account is not None and float(account.equity) > 0:
            equity_val = float(account.equity)
            weights_dict = {p.symbol: abs(p.market_value) / equity_val for p in positions}
            cash_frac = max(0, float(account.cash) / equity_val)
            if cash_frac > 0.005:
                weights_dict["Cash"] = cash_frac
            children.append(
                html.Div(
                    _panel(
                        dcc.Graph(
                            figure=_chart_allocation_ring(weights_dict),
                            config={"displayModeBar": False},
                        ),
                    ),
                    style={"maxWidth": "400px", "marginTop": _SP_3},
                )
            )

    # ── Equity history ──
    children.append(html.Div(style={"height": _SP_5}))
    children.append(_section_label("Equity History"))

    equity_data = _load_equity_history()
    if equity_data is not None and len(equity_data) > 1:
        children.append(
            _panel(
                dcc.Graph(figure=_chart_equity_curve(equity_data), config={"displayModeBar": False})
            )
        )
    elif bot_state:
        info_parts = []
        if "last_equity" in bot_state:
            info_parts.append(f"Last equity: ${bot_state['last_equity']:,.2f}")
        if "last_trade_date" in bot_state:
            info_parts.append(f"Last trade: {bot_state['last_trade_date']}")
        children.append(
            _panel(
                html.Div(
                    " | ".join(info_parts)
                    if info_parts
                    else "Bot state loaded, no equity history yet.",
                    style={"fontSize": "12px", "fontFamily": _FONT_MONO, "color": _TEXT_SECONDARY},
                ),
                padding=_SP_4,
            )
        )
    else:
        children.append(_panel(_empty_state("No trading history yet.")))

    # ── Bot Log ──
    children.append(html.Div(style={"height": _SP_5}))
    children.append(_section_label("Bot Log"))

    log_text = _read_log_tail(80)
    children.append(
        _panel(
            html.Pre(
                log_text,
                style={
                    "fontFamily": _FONT_MONO,
                    "fontSize": "11px",
                    "lineHeight": "1.6",
                    "color": _TEXT_SECONDARY,
                    "backgroundColor": _INSET,
                    "padding": _SP_4,
                    "borderRadius": _RADIUS_SM,
                    "border": f"1px solid {_BORDER}",
                    "margin": "0",
                    "maxHeight": "500px",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "wordBreak": "break-all",
                },
            ),
            padding=_SP_3,
        )
    )

    return html.Div(children)


# ═══════════════════════════════════════════════════════════════════
# Public API — Standalone Backtest Dashboard
# ═══════════════════════════════════════════════════════════════════


def create_dashboard(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    risk_summary: dict,
    rolling_sharpe: pd.Series,
    turnover: pd.Series | None = None,
    metrics: dict | None = None,
    risk_contributions: dict | None = None,
    rolling_var: pd.Series | None = None,
) -> dash.Dash:
    """Create standalone backtest dashboard (backward-compatible API)."""
    app = dash.Dash(__name__)

    cumulative = (1 + portfolio_returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

    # ── Metrics strip ──
    sharpe = risk_summary.get("sharpe_ratio", risk_summary.get("sharpe", 0))
    sharpe_color = _HEALTHY if sharpe > 0.5 else _CAUTION if sharpe > 0 else _DANGER
    ann_ret = risk_summary.get("annualized_return", 0)
    ret_color = _HEALTHY if ann_ret > 0.08 else _CAUTION if ann_ret > 0 else _DANGER
    max_dd = risk_summary.get("max_drawdown", 0)
    dd_color = _HEALTHY if max_dd > -0.15 else _CAUTION if max_dd > -0.25 else _DANGER

    kpi_items = [
        _metric_colored("Sharpe", f"{sharpe:.2f}", sharpe_color),
        _metric("Sortino", f"{risk_summary.get('sortino_ratio', 0):.2f}"),
        _metric("Calmar", f"{risk_summary.get('calmar_ratio', 0):.2f}"),
        _metric_colored("Max DD", f"{max_dd:.1%}", dd_color),
        _metric_colored("Ann. Return", f"{ann_ret:.1%}", ret_color),
        _metric("VaR 95%", f"{risk_summary.get('var_95_historical', 0):.1%}"),
        _metric("CVaR 95%", f"{risk_summary.get('cvar_95', 0):.1%}"),
        _metric("Omega", f"{risk_summary.get('omega_ratio', 0):.2f}"),
    ]
    if metrics:
        if "avg_turnover" in metrics:
            kpi_items.append(_metric("Avg Turn", f"{metrics['avg_turnover']:.1%}"))
        if "optimizer_method" in metrics:
            kpi_items.append(
                _metric(
                    "Optimizer", metrics["optimizer_method"].replace("_", " ").title(), mono=False
                )
            )
        if "total_cost_bps" in metrics:
            kpi_items.append(_metric("Cost", f"{metrics['total_cost_bps']:.0f} bps"))

    # ── Charts ──
    chart_rows = [
        _grid(
            _panel(
                dcc.Graph(figure=_chart_cumulative(cumulative), config={"displayModeBar": False})
            ),
            _panel(dcc.Graph(figure=_chart_weights_bar(weights), config={"displayModeBar": False})),
        ),
        html.Div(style={"height": _SP_4}),
        _grid(
            _panel(dcc.Graph(figure=_chart_drawdown(drawdown), config={"displayModeBar": False})),
            _panel(
                dcc.Graph(
                    figure=_chart_rolling_sharpe(rolling_sharpe), config={"displayModeBar": False}
                )
            ),
        ),
    ]

    if turnover is not None:
        chart_rows.extend(
            [
                html.Div(style={"height": _SP_4}),
                _grid(
                    _panel(
                        dcc.Graph(
                            figure=_chart_turnover(turnover), config={"displayModeBar": False}
                        )
                    ),
                    _panel(
                        dcc.Graph(
                            figure=_chart_concentration(weights), config={"displayModeBar": False}
                        )
                    ),
                ),
            ]
        )

    if risk_contributions is not None:
        extra = [
            _panel(
                dcc.Graph(
                    figure=_chart_risk_attribution(risk_contributions),
                    config={"displayModeBar": False},
                )
            ),
        ]
        if rolling_var is not None:
            extra.append(
                _panel(
                    dcc.Graph(
                        figure=_chart_rolling_var(rolling_var), config={"displayModeBar": False}
                    )
                ),
            )
        chart_rows.extend([html.Div(style={"height": _SP_4}), _grid(*extra)])

    app.layout = html.Div(
        [
            # Header
            html.Div(
                "SIGNUM",
                style={
                    "fontSize": "14px",
                    "fontFamily": _FONT_MONO,
                    "fontWeight": "700",
                    "color": _TEXT_PRIMARY,
                    "letterSpacing": "0.12em",
                    "padding": f"{_SP_4} {_SP_5}",
                    "borderBottom": f"1px solid {_BORDER}",
                },
            ),
            # Content
            html.Div(
                [
                    _panel(_row(*kpi_items), padding=f"{_SP_3} {_SP_2}"),
                    html.Div(style={"height": _SP_4}),
                    *chart_rows,
                ],
                style={"padding": _SP_5},
            ),
        ],
        style={
            "backgroundColor": _CANVAS,
            "color": _TEXT_PRIMARY,
            "fontFamily": _FONT_SANS,
            "minHeight": "100vh",
        },
    )

    register_api_routes(app)
    return app


# ═══════════════════════════════════════════════════════════════════
# Public API — Tabbed Dashboard (Backtest + Live)
# ═══════════════════════════════════════════════════════════════════


def create_tabbed_dashboard() -> dash.Dash:
    """Create the full two-tab cockpit with regime beacon.

    The backtest tab renders historical performance.
    The live tab connects to Alpaca and auto-refreshes every 60s.
    """
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Fetch regime once for the beacon (refreshed with live tab)
    regime_state = _fetch_regime_state()

    app.layout = html.Div(
        [
            # ── Regime Beacon (always visible) ──
            _regime_beacon(regime_state),
            # ── Header bar ──
            html.Div(
                [
                    html.Span(
                        "SIGNUM",
                        style={
                            "fontSize": "14px",
                            "fontFamily": _FONT_MONO,
                            "fontWeight": "700",
                            "color": _TEXT_PRIMARY,
                            "letterSpacing": "0.12em",
                            "marginRight": _SP_7,
                        },
                    ),
                    dcc.Tabs(
                        id="main-tabs",
                        value="live",
                        children=[
                            dcc.Tab(
                                label="Live",
                                value="live",
                                style=_TAB_STYLE,
                                selected_style=_TAB_SELECTED_STYLE,
                            ),
                            dcc.Tab(
                                label="Backtest",
                                value="backtest",
                                style=_TAB_STYLE,
                                selected_style=_TAB_SELECTED_STYLE,
                            ),
                        ],
                        style={"display": "inline-flex", "border": "none"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": f"{_SP_3} {_SP_5}",
                    "borderBottom": f"1px solid {_BORDER}",
                },
            ),
            # ── Tab content ──
            html.Div(id="tab-content", style={"padding": _SP_5}),
            # ── Auto-refresh interval ──
            dcc.Interval(id="live-refresh", interval=60_000, n_intervals=0, disabled=True),
        ],
        style={
            "backgroundColor": _CANVAS,
            "color": _TEXT_PRIMARY,
            "fontFamily": _FONT_SANS,
            "minHeight": "100vh",
        },
    )

    @app.callback(
        Output("tab-content", "children"),
        Output("live-refresh", "disabled"),
        Input("main-tabs", "value"),
        Input("live-refresh", "n_intervals"),
    )
    def render_tab(tab: str, _n_intervals: int):
        if tab == "live":
            return _build_live_tab(), False
        return _build_backtest_tab(), True

    register_api_routes(app)
    return app


# ═══════════════════════════════════════════════════════════════════
# Data Loaders (unchanged logic, cleaned up)
# ═══════════════════════════════════════════════════════════════════


def _load_backtest_results():
    """Load persisted backtest results from data/processed/."""
    returns_path = RESULTS_DIR / "backtest_returns.parquet"
    if not returns_path.exists():
        logger.warning("No backtest results found. Run 'make backtest' first.")
        return None

    returns_df = pd.read_parquet(returns_path)
    portfolio_returns = returns_df["return"]

    weights = pd.read_json(RESULTS_DIR / "backtest_weights.json", typ="series")

    from python.portfolio.risk import RiskEngine

    returns_matrix = pd.DataFrame({"portfolio": portfolio_returns})
    engine = RiskEngine(returns_matrix, pd.Series({"portfolio": 1.0}))
    risk_summary = engine.summary()
    rolling_sharpe = engine.rolling_sharpe(window=60)

    turnover = None
    turnover_path = RESULTS_DIR / "backtest_turnover.parquet"
    if turnover_path.exists():
        turnover = pd.read_parquet(turnover_path)["turnover"]

    metrics = None
    metrics_path = RESULTS_DIR / "backtest_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return portfolio_returns, weights, risk_summary, rolling_sharpe, turnover, metrics


def _load_bot_state() -> dict:
    """Load bot state from data/bot_state.json."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load bot state: {e}")
        return {}


def _load_equity_history() -> pd.Series | None:
    """Load equity history from data/equity_history.json.

    Falls back to bot_state.json for a minimal single-point series.
    """
    history_path = Path("data/equity_history.json")
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                records = json.load(f)
            if records and isinstance(records, list):
                dates = [r.get("date", r.get("timestamp", "")) for r in records]
                values = [r.get("equity", 0) for r in records]
                idx = pd.to_datetime(dates, errors="coerce")
                series = pd.Series(values, index=idx).dropna()
                if len(series) > 1:
                    return series
        except Exception as e:
            logger.warning(f"Could not load equity history: {e}")

    state = _load_bot_state()
    if state and "last_equity" in state and "last_trade_date" in state:
        try:
            dt = pd.to_datetime(state["last_trade_date"])
            return pd.Series([state["last_equity"]], index=[dt])
        except Exception:
            pass

    return None


def _read_log_tail(n_lines: int = 80) -> str:
    """Read the last N lines from the bot log file.

    Checks the systemd log path first, then the local log fallback.
    """
    for log_path in [BOT_LOG_FILE, _LOCAL_LOG]:
        if log_path.exists():
            try:
                with open(log_path, "rb") as f:
                    # Seek from end to find last N lines efficiently
                    f.seek(0, 2)
                    size = f.tell()
                    # Read last 64KB max (enough for ~80 lines)
                    chunk_size = min(size, 65536)
                    f.seek(max(0, size - chunk_size))
                    data = f.read().decode("utf-8", errors="replace")
                    lines = data.splitlines()
                    return "\n".join(lines[-n_lines:])
            except Exception as e:
                logger.warning(f"Could not read log {log_path}: {e}")
    return "No log file found. Bot may not have started yet."


def _fetch_regime_state() -> dict | None:
    """Fetch current market regime (VIX + SPY drawdown)."""
    try:
        from python.monitoring.regime import (
            RegimeDetector,
            fetch_spy_drawdown,
            fetch_vix,
        )

        vix = fetch_vix()
        spy_dd = fetch_spy_drawdown()

        if vix is None or spy_dd is None:
            return None

        detector = RegimeDetector()
        regime_state = detector.get_regime_state(vix, spy_dd)
        return {
            "regime": regime_state.regime,
            "vix": regime_state.vix,
            "spy_drawdown": regime_state.spy_drawdown,
            "exposure_multiplier": regime_state.exposure_multiplier,
            "message": regime_state.message,
        }
    except Exception as e:
        logger.warning(f"Could not fetch regime state: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# JSON API — LLM-friendly endpoints
# ═══════════════════════════════════════════════════════════════════

_API_ENDPOINTS = {
    "/api": "Index of all available API endpoints",
    "/api/status": "System overview: regime, bot state, account, position count",
    "/api/account": "Alpaca account details (equity, cash, buying power, status)",
    "/api/positions": "Current open positions with P&L and weight",
    "/api/regime": "Market regime state (VIX, SPY drawdown, exposure multiplier)",
    "/api/backtest": "Backtest performance metrics and risk summary",
    "/api/equity": "Equity history time-series [{date, equity}, ...]",
    "/api/risk": "Full risk engine summary (Sharpe, VaR, drawdowns, etc.)",
    "/api/drift": "Latest ML feature drift report (KS stat, PSI per feature)",
    "/api/bot": "Bot state (last trade date, shutdown reason, positions count)",
    "/api/logs": "Latest bot log lines (default 80, ?lines=N to customize)",
}


def _json_response(data: dict, status: int = 200) -> Response:
    """Return a JSON response with CORS headers for broad access."""
    resp = jsonify(data)
    resp.status_code = status
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce to float safely."""
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def register_api_routes(app: dash.Dash) -> None:
    """Register JSON API routes on the Dash app's Flask server.

    Every endpoint returns ``Content-Type: application/json`` with
    ``Access-Control-Allow-Origin: *`` so any LLM, script, or browser
    can fetch data without auth (paper trading — no secrets exposed).
    """
    server = app.server

    # ── GET /api — endpoint index ──
    @server.route("/api")
    def api_index():
        return _json_response(
            {
                "name": "Signum Trading Cockpit API",
                "description": (
                    "JSON API for the Signum quantitative trading system. "
                    "All endpoints return structured JSON. "
                    "No authentication required (paper trading)."
                ),
                "timestamp": datetime.now().isoformat(),
                "endpoints": {path: desc for path, desc in _API_ENDPOINTS.items()},
            }
        )

    # ── GET /api/status — system overview ──
    @server.route("/api/status")
    def api_status():
        regime = _fetch_regime_state()
        bot = _load_bot_state()
        account_data = _fetch_account_json()
        positions_data = _fetch_positions_json()

        return _json_response(
            {
                "timestamp": datetime.now().isoformat(),
                "regime": regime,
                "bot_state": bot if bot else None,
                "account": account_data,
                "positions_count": len(positions_data.get("positions", []))
                if positions_data
                else 0,
                "has_backtest_results": (RESULTS_DIR / "backtest_returns.parquet").exists(),
            }
        )

    # ── GET /api/account — Alpaca account ──
    @server.route("/api/account")
    def api_account():
        data = _fetch_account_json()
        if data is None:
            return _json_response({"error": "Could not connect to Alpaca. Check credentials."}, 503)
        return _json_response(data)

    # ── GET /api/positions — open positions ──
    @server.route("/api/positions")
    def api_positions():
        data = _fetch_positions_json()
        if data is None:
            return _json_response({"error": "Could not fetch positions from Alpaca."}, 503)
        return _json_response(data)

    # ── GET /api/regime — market regime ──
    @server.route("/api/regime")
    def api_regime():
        regime = _fetch_regime_state()
        if regime is None:
            return _json_response(
                {"error": "Could not fetch regime data (VIX/SPY unavailable)."}, 503
            )
        return _json_response({"timestamp": datetime.now().isoformat(), **regime})

    # ── GET /api/backtest — backtest metrics ──
    @server.route("/api/backtest")
    def api_backtest():
        result = _load_backtest_results()
        if result is None:
            return _json_response({"error": "No backtest results. Run 'make backtest' first."}, 404)

        portfolio_returns, weights, risk_summary, rolling_sharpe, turnover, metrics = result

        cumulative = (1 + portfolio_returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()

        return _json_response(
            {
                "timestamp": datetime.now().isoformat(),
                "risk_summary": {
                    k: _safe_float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in risk_summary.items()
                },
                "metrics": metrics,
                "weights": {k: _safe_float(v) for k, v in weights.to_dict().items()},
                "latest_cumulative_return": _safe_float(cumulative.iloc[-1])
                if len(cumulative) > 0
                else None,
                "current_drawdown": _safe_float(drawdown.iloc[-1]) if len(drawdown) > 0 else None,
                "n_periods": len(portfolio_returns),
            }
        )

    # ── GET /api/equity — equity history ──
    @server.route("/api/equity")
    def api_equity():
        equity = _load_equity_history()
        if equity is None or len(equity) == 0:
            return _json_response({"error": "No equity history available."}, 404)

        records = [
            {"date": str(d), "equity": _safe_float(v)} for d, v in zip(equity.index, equity.values)
        ]
        return _json_response(
            {
                "timestamp": datetime.now().isoformat(),
                "count": len(records),
                "history": records,
            }
        )

    # ── GET /api/risk — full risk engine summary ──
    @server.route("/api/risk")
    def api_risk():
        result = _load_backtest_results()
        if result is None:
            return _json_response({"error": "No backtest results for risk computation."}, 404)

        _, _, risk_summary, _, _, _ = result
        clean = {}
        for k, v in risk_summary.items():
            if isinstance(v, (int, float, np.floating)):
                clean[k] = _safe_float(v)
            elif isinstance(v, dict):
                clean[k] = {
                    kk: _safe_float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                    for kk, vv in v.items()
                }
            else:
                clean[k] = v

        return _json_response({"timestamp": datetime.now().isoformat(), "risk": clean})

    # ── GET /api/drift — ML feature drift ──
    @server.route("/api/drift")
    def api_drift():
        try:
            from python.alpha.predict import _last_drift_report

            if _last_drift_report is None:
                return _json_response(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "drift_report": None,
                        "message": "No drift report available. Run the ML pipeline first.",
                    }
                )

            # Serialize the drift report
            clean_report = {}
            for feature, info in _last_drift_report.items():
                clean_report[feature] = {
                    k: _safe_float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in (info if isinstance(info, dict) else {"value": info}).items()
                }

            n_drifted = sum(
                1
                for info in clean_report.values()
                if isinstance(info, dict) and info.get("drifted", False)
            )

            return _json_response(
                {
                    "timestamp": datetime.now().isoformat(),
                    "n_features": len(clean_report),
                    "n_drifted": n_drifted,
                    "features": clean_report,
                }
            )
        except ImportError:
            return _json_response(
                {
                    "timestamp": datetime.now().isoformat(),
                    "drift_report": None,
                    "message": "Drift module not available.",
                }
            )

    # ── GET /api/bot — bot state ──
    @server.route("/api/bot")
    def api_bot():
        state = _load_bot_state()
        if not state:
            return _json_response(
                {
                    "timestamp": datetime.now().isoformat(),
                    "bot_state": None,
                    "message": "No bot state file found. Bot hasn't run yet.",
                }
            )
        return _json_response({"timestamp": datetime.now().isoformat(), "bot_state": state})

    # ── GET /api/logs — bot log tail ──
    @server.route("/api/logs")
    def api_logs():
        from flask import request

        n_lines = min(int(request.args.get("lines", 80)), 500)
        log_text = _read_log_tail(n_lines)
        lines = log_text.splitlines()
        return _json_response(
            {
                "timestamp": datetime.now().isoformat(),
                "n_lines": len(lines),
                "log": lines,
            }
        )


def _fetch_account_json() -> dict | None:
    """Fetch Alpaca account data as a dict. Returns None on failure."""
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        return None
    try:
        from python.brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper_trading=True)
        broker.connect()
        acct = broker.get_account()
        return {
            "timestamp": datetime.now().isoformat(),
            "equity": _safe_float(acct.equity),
            "cash": _safe_float(acct.cash),
            "portfolio_value": _safe_float(acct.portfolio_value),
            "buying_power": _safe_float(acct.buying_power),
            "status": str(acct.status),
            "currency": "USD",
        }
    except Exception as e:
        logger.warning(f"API: could not fetch account: {e}")
        return None


def _fetch_positions_json() -> dict | None:
    """Fetch Alpaca positions as a dict. Returns None on failure."""
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        return None
    try:
        from python.brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(paper_trading=True)
        broker.connect()
        positions = broker.list_positions()
        acct = broker.get_account()
        equity = _safe_float(acct.equity) if acct else 0.0

        pos_list = []
        total_pnl = 0.0
        total_mv = 0.0
        for p in sorted(positions, key=lambda x: abs(x.market_value), reverse=True):
            weight = abs(p.market_value) / equity if equity > 0 else 0
            pos_list.append(
                {
                    "symbol": p.symbol,
                    "qty": _safe_float(p.qty),
                    "avg_entry_price": _safe_float(p.avg_entry_price),
                    "market_value": _safe_float(p.market_value),
                    "unrealized_pl": _safe_float(p.unrealized_pl),
                    "unrealized_plpc": _safe_float(p.unrealized_plpc),
                    "weight": round(weight, 4),
                }
            )
            total_pnl += _safe_float(p.unrealized_pl)
            total_mv += _safe_float(p.market_value)

        return {
            "timestamp": datetime.now().isoformat(),
            "count": len(pos_list),
            "total_market_value": round(total_mv, 2),
            "total_unrealized_pnl": round(total_pnl, 2),
            "positions": pos_list,
        }
    except Exception as e:
        logger.warning(f"API: could not fetch positions: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    has_alpaca = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"))
    has_backtest = (RESULTS_DIR / "backtest_returns.parquet").exists()

    if has_alpaca or has_backtest:
        app = create_tabbed_dashboard()
    else:
        # Synthetic demo data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
        weights = pd.Series(
            {"AAPL": 0.25, "MSFT": 0.20, "GOOG": 0.18, "AMZN": 0.15, "META": 0.12, "NVDA": 0.10}
        )
        risk = {
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.8,
            "calmar_ratio": 2.1,
            "max_drawdown": -0.15,
            "annualized_return": 0.12,
            "var_95_historical": -0.018,
            "cvar_95": -0.025,
            "omega_ratio": 1.4,
        }
        rolling = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        app = create_dashboard(returns, weights, risk, rolling)

    app.run(debug=True, port=8050)
