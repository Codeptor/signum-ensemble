"""Meta-labeling and triple barrier method (de Prado 2018).

Implements the meta-labeling framework from AFML:
  1. Triple barrier labeling — generates labels from price paths using
     profit-taking, stop-loss, and time barriers.
  2. Meta-labeling — secondary model predicts bet size given primary signal.
  3. Sample uniqueness — average uniqueness for concurrency-aware weighting.
  4. Sequential bootstrap — IID-safe resampling respecting label overlaps.

The key insight: separate the *side* decision (primary model) from the
*size* decision (meta model). This allows a simple directional signal
to be combined with a sophisticated sizing model.

Usage::

    labels = triple_barrier_labels(prices, events, pt_sl=[1.5, 1.0])
    meta = MetaLabeler(primary_model, meta_model)
    meta.fit(X_train, y_side, y_meta)
    positions = meta.predict(X_test)

References:
  - de Prado (2018), "Advances in Financial Machine Learning", Ch. 3-4
  - de Prado (2018), ibid., Ch. 4 — "Sample Weights"
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triple Barrier Labels
# ---------------------------------------------------------------------------


@dataclass
class BarrierEvent:
    """Defines a labeling event window."""

    t_start: int  # Index of the event start
    t_end: int  # Maximum horizon (vertical barrier)
    pt_level: float  # Profit-taking barrier (in return space)
    sl_level: float  # Stop-loss barrier (in return space, positive)
    side: int  # +1 (long) or -1 (short) — from primary model


@dataclass
class BarrierLabel:
    """Result of applying triple barrier to a single event."""

    t_start: int
    t_end: int  # Actual exit time
    ret: float  # Return achieved
    label: int  # +1 (profit), -1 (loss), 0 (time expiry)
    barrier_type: str  # "pt", "sl", or "time"


def triple_barrier_labels(
    prices: np.ndarray,
    events: list[BarrierEvent],
) -> list[BarrierLabel]:
    """Apply triple barrier method to generate labels.

    For each event:
      - If price hits profit-taking barrier first → label = +1
      - If price hits stop-loss barrier first → label = -1
      - If neither hit by t_end → label based on sign of return

    Parameters
    ----------
    prices : np.ndarray
        Price series.
    events : list[BarrierEvent]
        Events to label.

    Returns
    -------
    list[BarrierLabel]
    """
    results = []
    T = len(prices)

    for ev in events:
        if ev.t_start >= T:
            continue

        entry_price = prices[ev.t_start]
        if entry_price <= 0:
            continue

        t_exit = min(ev.t_end, T - 1)
        exit_ret = 0.0
        barrier_type = "time"

        for t in range(ev.t_start + 1, t_exit + 1):
            ret = ev.side * (prices[t] / entry_price - 1)

            if ev.pt_level > 0 and ret >= ev.pt_level:
                exit_ret = ret
                t_exit = t
                barrier_type = "pt"
                break
            if ev.sl_level > 0 and ret <= -ev.sl_level:
                exit_ret = ret
                t_exit = t
                barrier_type = "sl"
                break

        if barrier_type == "time":
            exit_ret = ev.side * (prices[t_exit] / entry_price - 1)

        if barrier_type == "pt":
            label = 1
        elif barrier_type == "sl":
            label = -1
        else:
            label = 1 if exit_ret > 0 else (-1 if exit_ret < 0 else 0)

        results.append(BarrierLabel(
            t_start=ev.t_start,
            t_end=t_exit,
            ret=exit_ret,
            label=label,
            barrier_type=barrier_type,
        ))

    return results


# ---------------------------------------------------------------------------
# Sample Uniqueness
# ---------------------------------------------------------------------------


def compute_concurrency(
    events: list[BarrierLabel],
    n_samples: int,
) -> np.ndarray:
    """Compute concurrency: how many labels overlap at each time step.

    Parameters
    ----------
    events : list[BarrierLabel]
    n_samples : int
        Total number of time steps.

    Returns
    -------
    np.ndarray (n_samples,)
        Count of concurrent labels at each time.
    """
    concurrency = np.zeros(n_samples, dtype=int)
    for ev in events:
        concurrency[ev.t_start : ev.t_end + 1] += 1
    return concurrency


def average_uniqueness(
    events: list[BarrierLabel],
    n_samples: int,
) -> np.ndarray:
    """Compute average uniqueness per event.

    Uniqueness at time t for event i = 1 / concurrency(t)
    Average uniqueness for event i = mean of uniqueness over its span.

    Parameters
    ----------
    events : list[BarrierLabel]
    n_samples : int

    Returns
    -------
    np.ndarray (n_events,)
        Average uniqueness per event, in [0, 1].
    """
    concurrency = compute_concurrency(events, n_samples)
    concurrency = np.maximum(concurrency, 1)  # avoid div by zero

    uniqueness = np.zeros(len(events))
    for i, ev in enumerate(events):
        span = concurrency[ev.t_start : ev.t_end + 1]
        uniqueness[i] = float(np.mean(1.0 / span))

    return uniqueness


# ---------------------------------------------------------------------------
# Sample Weights
# ---------------------------------------------------------------------------


def sample_weights_by_return(
    events: list[BarrierLabel],
    n_samples: int,
) -> np.ndarray:
    """Compute sample weights proportional to absolute return attribution.

    Each event's weight is its absolute return divided by concurrency,
    normalized so weights sum to number of events.

    Parameters
    ----------
    events : list[BarrierLabel]
    n_samples : int

    Returns
    -------
    np.ndarray (n_events,)
    """
    concurrency = compute_concurrency(events, n_samples)
    concurrency = np.maximum(concurrency, 1)

    weights = np.zeros(len(events))
    for i, ev in enumerate(events):
        # Attribution: abs(return) / mean(concurrency over span)
        span_conc = concurrency[ev.t_start : ev.t_end + 1]
        avg_conc = float(np.mean(span_conc))
        weights[i] = abs(ev.ret) / max(avg_conc, 1e-12)

    # Normalize
    total = weights.sum()
    if total > 0:
        weights *= len(events) / total

    return weights


# ---------------------------------------------------------------------------
# Sequential Bootstrap
# ---------------------------------------------------------------------------


def sequential_bootstrap(
    events: list[BarrierLabel],
    n_samples: int,
    n_draws: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Sequential bootstrap for IID-safe resampling.

    Standard bootstrap draws independently, which produces highly
    redundant samples when labels overlap. Sequential bootstrap
    draws each sample with probability proportional to its average
    uniqueness given already-drawn samples.

    Parameters
    ----------
    events : list[BarrierLabel]
    n_samples : int
        Total time steps in the dataset.
    n_draws : int, optional
        Number of bootstrap samples. Default = len(events).
    seed : int

    Returns
    -------
    np.ndarray
        Indices into events for the bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    n_events = len(events)
    if n_draws is None:
        n_draws = n_events

    # Build indicator matrix: (n_events, n_samples) — which time steps each event spans
    indicators = np.zeros((n_events, n_samples), dtype=bool)
    for i, ev in enumerate(events):
        indicators[i, ev.t_start : min(ev.t_end + 1, n_samples)] = True

    drawn = []
    concurrency = np.zeros(n_samples, dtype=float)

    for _ in range(n_draws):
        # Compute average uniqueness for each candidate
        probs = np.zeros(n_events)
        for i in range(n_events):
            span = indicators[i]
            temp_conc = concurrency[span] + 1  # if we add this event
            probs[i] = float(np.mean(1.0 / temp_conc))

        # Normalize to probabilities
        total_prob = probs.sum()
        if total_prob > 0:
            probs /= total_prob
        else:
            probs = np.ones(n_events) / n_events

        choice = rng.choice(n_events, p=probs)
        drawn.append(choice)
        concurrency[indicators[choice]] += 1

    return np.array(drawn)


# ---------------------------------------------------------------------------
# Meta-Labeling
# ---------------------------------------------------------------------------


@dataclass
class MetaLabelResult:
    """Result from meta-labeling predictions."""

    sides: np.ndarray  # +1/-1 from primary model
    sizes: np.ndarray  # [0, 1] from meta model
    positions: np.ndarray  # side * size
    meta_accuracy: float  # Meta model accuracy on validation


class MetaLabeler:
    """Meta-labeling framework.

    Primary model decides the side (+1 long, -1 short).
    Meta model decides the size (0 = no bet, 1 = full size).

    Parameters
    ----------
    threshold : float
        Probability threshold for meta model to take a bet.
    max_position : float
        Maximum position size.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_position: float = 1.0,
    ):
        self.threshold = threshold
        self.max_position = max_position
        self._meta_weights = None
        self._meta_bias = None

    def fit(
        self,
        X: np.ndarray,
        primary_side: np.ndarray,
        meta_labels: np.ndarray,
    ) -> "MetaLabeler":
        """Fit the meta model.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            Features for meta model.
        primary_side : np.ndarray (n_samples,)
            Side predictions from primary model (+1/-1).
        meta_labels : np.ndarray (n_samples,)
            Binary labels: 1 if primary was correct, 0 if not.

        Returns
        -------
        self
        """
        # Simple logistic regression via iterative reweighted least squares
        n, d = X.shape
        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])
        w = np.zeros(d + 1)

        y = meta_labels.astype(float)

        for _ in range(100):
            logits = X_aug @ w
            logits = np.clip(logits, -30, 30)
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-8, 1 - 1e-8)

            # Gradient
            grad = X_aug.T @ (p - y) / n
            # Hessian diagonal approximation
            diag_H = p * (1 - p)
            H = (X_aug.T * diag_H) @ X_aug / n + np.eye(d + 1) * 1e-4

            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w -= step

            if np.max(np.abs(step)) < 1e-6:
                break

        self._meta_bias = w[0]
        self._meta_weights = w[1:]
        return self

    def predict(
        self,
        X: np.ndarray,
        primary_side: np.ndarray,
    ) -> MetaLabelResult:
        """Predict position sizes using meta model.

        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
        primary_side : np.ndarray (n_samples,)
            Side predictions from primary model.

        Returns
        -------
        MetaLabelResult
        """
        if self._meta_weights is None:
            raise RuntimeError("MetaLabeler not fitted yet")

        logits = X @ self._meta_weights + self._meta_bias
        logits = np.clip(logits, -30, 30)
        probs = 1.0 / (1.0 + np.exp(-logits))

        sizes = np.where(probs >= self.threshold, probs, 0.0)
        sizes = np.clip(sizes, 0, self.max_position)

        positions = primary_side * sizes

        # Accuracy: fraction where meta model correctly predicts whether to bet
        meta_pred = (probs >= self.threshold).astype(int)
        # We don't have ground truth here, so report mean probability
        meta_acc = float(np.mean(probs))

        return MetaLabelResult(
            sides=primary_side,
            sizes=sizes,
            positions=positions,
            meta_accuracy=meta_acc,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw probabilities from meta model."""
        if self._meta_weights is None:
            raise RuntimeError("MetaLabeler not fitted yet")
        logits = X @ self._meta_weights + self._meta_bias
        logits = np.clip(logits, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))
