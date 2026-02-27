use matching_engine::orderbook::OrderBook;
use matching_engine::types::{Order, OrderType, Side, Timestamp};
use pyo3::prelude::*;

/// Python-visible Side enum.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PySide {
    Buy = 0,
    Sell = 1,
}

impl From<PySide> for Side {
    fn from(s: PySide) -> Self {
        match s {
            PySide::Buy => Side::Buy,
            PySide::Sell => Side::Sell,
        }
    }
}

/// Python-visible OrderType enum.
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyOrderType {
    Limit = 0,
    Market = 1,
    ImmediateOrCancel = 2,
    FillOrKill = 3,
}

impl From<PyOrderType> for OrderType {
    fn from(ot: PyOrderType) -> Self {
        match ot {
            PyOrderType::Limit => OrderType::Limit,
            PyOrderType::Market => OrderType::Market,
            PyOrderType::ImmediateOrCancel => OrderType::ImmediateOrCancel,
            PyOrderType::FillOrKill => OrderType::FillOrKill,
        }
    }
}

/// A single execution (trade) from the matching engine.
#[pyclass]
#[derive(Clone)]
pub struct PyExecution {
    #[pyo3(get)]
    pub buy_order_id: u64,
    #[pyo3(get)]
    pub sell_order_id: u64,
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub quantity: u64,
    #[pyo3(get)]
    pub timestamp: u64,
}

#[pymethods]
impl PyExecution {
    fn __repr__(&self) -> String {
        format!(
            "Execution(buy={}, sell={}, price={:.4}, qty={}, ts={})",
            self.buy_order_id, self.sell_order_id, self.price, self.quantity, self.timestamp
        )
    }
}

/// Level-2 order book snapshot: lists of (price, quantity) at each level.
#[pyclass]
#[derive(Clone)]
pub struct PyL2Snapshot {
    #[pyo3(get)]
    pub bids: Vec<(f64, u64)>,
    #[pyo3(get)]
    pub asks: Vec<(f64, u64)>,
}

#[pymethods]
impl PyL2Snapshot {
    fn __repr__(&self) -> String {
        format!(
            "L2Snapshot(bids={} levels, asks={} levels)",
            self.bids.len(),
            self.asks.len()
        )
    }

    /// Best bid price, or None if empty.
    #[getter]
    fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|(p, _)| *p)
    }

    /// Best ask price, or None if empty.
    #[getter]
    fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|(p, _)| *p)
    }

    /// Spread in price units, or None if one side is empty.
    #[getter]
    fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    /// Mid price, or None if one side is empty.
    #[getter]
    fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((a + b) / 2.0),
            _ => None,
        }
    }
}

/// High-performance order book backed by the Rust matching engine.
///
/// Supports limit, market, IOC, and FOK order types with price-time priority.
///
/// Example:
///     from matching_engine_py import PyOrderBook, PySide, PyOrderType
///     book = PyOrderBook()
///     execs = book.submit(1, PySide.Buy, 100.0, 10, PyOrderType.Limit)
#[pyclass]
pub struct PyOrderBook {
    inner: OrderBook,
    next_ts: Timestamp,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    fn new() -> Self {
        Self {
            inner: OrderBook::new(),
            next_ts: 0,
        }
    }

    /// Submit an order and return a list of resulting executions.
    ///
    /// Args:
    ///     order_id: Unique order identifier.
    ///     side: PySide.Buy or PySide.Sell.
    ///     price: Limit price (ignored for Market orders).
    ///     quantity: Number of shares.
    ///     order_type: PyOrderType variant.
    ///
    /// Returns:
    ///     List of PyExecution objects for any fills.
    #[pyo3(signature = (order_id, side, price, quantity, order_type=PyOrderType::Limit))]
    fn submit(
        &mut self,
        order_id: u64,
        side: PySide,
        price: f64,
        quantity: u64,
        order_type: PyOrderType,
    ) -> Vec<PyExecution> {
        let order = Order::new(
            order_id,
            side.into(),
            price,
            quantity,
            order_type.into(),
            self.next_ts,
        );
        self.next_ts += 1;

        let execs = self.inner.submit(order, self.next_ts);
        execs
            .into_iter()
            .map(|e| PyExecution {
                buy_order_id: e.buy_order_id,
                sell_order_id: e.sell_order_id,
                price: e.price.into_inner(),
                quantity: e.quantity,
                timestamp: e.timestamp,
            })
            .collect()
    }

    /// Cancel a resting order by ID. Returns True if found and cancelled.
    fn cancel(&mut self, order_id: u64) -> bool {
        self.inner.cancel(order_id)
    }

    /// Get the best bid price, or None if the bid side is empty.
    #[getter]
    fn best_bid(&self) -> Option<f64> {
        self.inner.best_bid().map(|p| p.into_inner())
    }

    /// Get the best ask price, or None if the ask side is empty.
    #[getter]
    fn best_ask(&self) -> Option<f64> {
        self.inner.best_ask().map(|p| p.into_inner())
    }

    /// Get the bid-ask spread, or None if one side is empty.
    #[getter]
    fn spread(&self) -> Option<f64> {
        self.inner.spread()
    }

    /// Get a level-2 snapshot with the given depth (number of price levels).
    #[pyo3(signature = (depth=10))]
    fn snapshot(&self, depth: usize) -> PyL2Snapshot {
        let snap = self.inner.snapshot(depth);
        PyL2Snapshot {
            bids: snap
                .bids
                .into_iter()
                .map(|(p, q)| (p.into_inner(), q))
                .collect(),
            asks: snap
                .asks
                .into_iter()
                .map(|(p, q)| (p.into_inner(), q))
                .collect(),
        }
    }

    fn __repr__(&self) -> String {
        let snap = self.inner.snapshot(1);
        let bid_str = snap
            .bids
            .first()
            .map(|(p, q)| format!("{:.2}x{}", p.into_inner(), q))
            .unwrap_or_else(|| "empty".to_string());
        let ask_str = snap
            .asks
            .first()
            .map(|(p, q)| format!("{:.2}x{}", p.into_inner(), q))
            .unwrap_or_else(|| "empty".to_string());
        format!("OrderBook(bid={}, ask={})", bid_str, ask_str)
    }
}

/// Matching engine Python bindings.
#[pymodule]
fn matching_engine_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrderBook>()?;
    m.add_class::<PySide>()?;
    m.add_class::<PyOrderType>()?;
    m.add_class::<PyExecution>()?;
    m.add_class::<PyL2Snapshot>()?;
    Ok(())
}
