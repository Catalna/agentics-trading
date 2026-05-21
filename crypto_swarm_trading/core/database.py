"""
core/database.py — SQLite Database for Swarm Trading with production-grade schemas.
"""

import sqlite3
import threading
from datetime import datetime, timezone
import json
import logging
from typing import Dict, Any, List, Optional
import os

class DBManager:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        self.logger = logging.getLogger("DBManager")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        with self._get_conn() as conn:
            # Main trading tables
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS swarm_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                sender TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL,
                vote_distribution TEXT,
                features_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_signals_ts ON swarm_signals(ts);

            CREATE TABLE IF NOT EXISTS swarm_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_ts TEXT NOT NULL,
                exit_ts TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                sl_price REAL,
                tp_price REAL,
                pnl_usdt REAL,
                pnl_pct REAL,
                duration_min REAL,
                exit_reason TEXT,
                status TEXT DEFAULT 'OPEN'
            );
            CREATE INDEX IF NOT EXISTS idx_trades_status ON swarm_trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_entry ON swarm_trades(entry_ts);

            -- Agent monitoring tables
            CREATE TABLE IF NOT EXISTS agent_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                uptime_seconds REAL,
                last_heartbeat REAL,
                error_count INTEGER,
                success_count INTEGER,
                avg_exec_ms REAL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_health_name ON agent_health(agent_name);

            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                date TEXT NOT NULL,
                total_calls INTEGER,
                success_calls INTEGER,
                error_calls INTEGER,
                avg_exec_ms REAL,
                accuracy_pct REAL,
                UNIQUE(agent_name, date)
            );

            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                agent_name TEXT,
                message TEXT,
                severity TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sys_events_ts_sev ON system_events(ts, severity);
            """)

    def upsert_agent_health(self, agent_name: str, status: str, uptime: float, error_count: int, success_count: int, avg_exec_ms: float):
        ts = datetime.now(timezone.utc).isoformat()
        last_hb = datetime.now(timezone.utc).timestamp()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO agent_health 
                (agent_name, status, uptime_seconds, last_heartbeat, error_count, success_count, avg_exec_ms, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_name) DO UPDATE SET
                    status=excluded.status,
                    uptime_seconds=excluded.uptime_seconds,
                    last_heartbeat=excluded.last_heartbeat,
                    error_count=excluded.error_count,
                    success_count=excluded.success_count,
                    avg_exec_ms=excluded.avg_exec_ms,
                    updated_at=excluded.updated_at
            """, (agent_name, status, uptime, last_hb, error_count, success_count, avg_exec_ms, ts))

    def log_system_event(self, event_type: str, agent_name: str, message: str, severity: str = 'INFO'):
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO system_events (ts, event_type, agent_name, message, severity) VALUES (?, ?, ?, ?, ?)",
                (ts, event_type, agent_name, message, severity)
            )

    def log_signal(self, sender: str, direction: str, confidence: float, vote_distribution: Dict, features: Dict):
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO swarm_signals (ts, sender, direction, confidence, vote_distribution, features_json) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, sender, direction, confidence, json.dumps(vote_distribution), json.dumps(features))
            )

    def open_trade(self, trade_id: str, symbol: str, side: str, entry_price: float, quantity: float, sl_price: float, tp_price: float):
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO swarm_trades 
                (trade_id, symbol, side, entry_ts, entry_price, quantity, sl_price, tp_price, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (trade_id, symbol, side, ts, entry_price, quantity, sl_price, tp_price))

    def close_trade(self, trade_id: str, exit_price: float, pnl_usdt: float, pnl_pct: float, duration_min: float, exit_reason: str):
        ts = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE swarm_trades
                SET exit_ts=?, exit_price=?, pnl_usdt=?, pnl_pct=?, duration_min=?, exit_reason=?, status='CLOSED'
                WHERE trade_id=? AND status='OPEN'
            """, (ts, exit_price, pnl_usdt, pnl_pct, duration_min, exit_reason, trade_id))

    def get_daily_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            row = conn.execute("SELECT SUM(pnl_usdt) as pnl FROM swarm_trades WHERE date(exit_ts)=? AND status='CLOSED'", (today,)).fetchone()
            return float(row['pnl']) if row and row['pnl'] else 0.0

    def get_recent_trades(self, n: int = 50) -> List[Dict]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM swarm_trades ORDER BY entry_ts DESC LIMIT ?", (n,)).fetchall()
            return [dict(r) for r in rows]

    def cleanup_old_logs(self, days: int = 30):
        # Implementation for cleaning up logs older than `days`
        pass
