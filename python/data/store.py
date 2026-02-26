"""Data storage layer. Uses TimescaleDB in production, SQLite for testing."""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from python.data.models import Base, OHLCVBar

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, connection_string: str = "sqlite:///data/quant.db") -> None:
        self.engine = create_engine(connection_string)
        self._dialect = self.engine.dialect.name  # "sqlite", "postgresql", etc.

    def init_db(self) -> None:
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def upsert_ohlcv(self, df: pd.DataFrame) -> int:
        """Insert or update OHLCV bars from a DataFrame.

        Expects columns: ticker, open, high, low, close, volume
        with a DatetimeIndex.

        Uses bulk INSERT ... ON CONFLICT DO UPDATE (Fix #26) instead of
        row-by-row SELECT queries.  Falls back to per-row merge for
        dialects that don't support ``on_conflict_do_update``.
        """
        if df.empty:
            return 0

        rows = [
            {
                "ticker": row["ticker"],
                "timestamp": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for ts, row in df.iterrows()
        ]

        update_cols = {"open", "high", "low", "close", "volume"}

        if self._dialect == "sqlite":
            self._upsert_sqlite(rows, update_cols)
        elif self._dialect == "postgresql":
            self._upsert_postgresql(rows, update_cols)
        else:
            # Generic fallback — row-by-row merge (same as original)
            self._upsert_generic(rows)

        logger.info(f"Upserted {len(rows)} OHLCV bars")
        return len(rows)

    # --- dialect-specific bulk upserts ---

    def _upsert_sqlite(self, rows: list[dict], update_cols: set[str]) -> None:
        """SQLite INSERT ... ON CONFLICT DO UPDATE in batches of 500."""
        batch_size = 500
        with Session(self.engine) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                stmt = sqlite_insert(OHLCVBar).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker", "timestamp"],
                    set_={col: stmt.excluded[col] for col in update_cols},
                )
                session.execute(stmt)
            session.commit()

    def _upsert_postgresql(self, rows: list[dict], update_cols: set[str]) -> None:
        """PostgreSQL INSERT ... ON CONFLICT DO UPDATE in batches of 500."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        batch_size = 500
        with Session(self.engine) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                stmt = pg_insert(OHLCVBar).values(batch)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_ticker_timestamp",
                    set_={col: stmt.excluded[col] for col in update_cols},
                )
                session.execute(stmt)
            session.commit()

    def _upsert_generic(self, rows: list[dict]) -> None:
        """Row-by-row fallback for unknown dialects."""
        with Session(self.engine) as session:
            for row in rows:
                existing = session.execute(
                    select(OHLCVBar).where(
                        OHLCVBar.ticker == row["ticker"],
                        OHLCVBar.timestamp == row["timestamp"],
                    )
                ).scalar_one_or_none()
                if existing:
                    existing.open = row["open"]
                    existing.high = row["high"]
                    existing.low = row["low"]
                    existing.close = row["close"]
                    existing.volume = row["volume"]
                else:
                    session.add(OHLCVBar(**row))
            session.commit()

    def get_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Retrieve OHLCV bars for a ticker within a date range."""
        with Session(self.engine) as session:
            stmt = (
                select(OHLCVBar)
                .where(
                    OHLCVBar.ticker == ticker,
                    OHLCVBar.timestamp >= datetime.fromisoformat(start_date),
                    OHLCVBar.timestamp <= datetime.fromisoformat(end_date),
                )
                .order_by(OHLCVBar.timestamp)
            )
            rows = session.execute(stmt).scalars().all()

        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "ticker": r.ticker,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in rows
            ]
        )
