"""
dashboard/streamlit_app.py — Real-time Swarm Trading System Monitor
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swarm Trading Dashboard",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── DB Path ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'swarm_trading.db')

def get_conn():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)

@st.cache_data(ttl=5)
def load_agent_health():
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query("SELECT * FROM agent_health ORDER BY agent_name", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=5)
def load_recent_trades(n=20):
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM swarm_trades ORDER BY entry_ts DESC LIMIT {n}", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=5)
def load_signals(n=50):
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM swarm_signals ORDER BY ts DESC LIMIT {n}", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=5)
def load_system_events(n=30):
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM system_events ORDER BY ts DESC LIMIT {n}", conn)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-healthy { color: #48bb78; font-weight: bold; }
    .status-error { color: #fc8181; font-weight: bold; }
    .status-stopped { color: #a0aec0; font-weight: bold; }
    .sidebar-title { font-size: 1.2em; font-weight: bold; color: #68d391; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐝 Swarm Control Panel")
    st.markdown("---")
    st.markdown("**System Config**")
    st.info("TESTNET: True\nDRY_RUN: Configurable\nSymbol: BTC/USDT")
    st.markdown("---")
    st.markdown("**Refresh**")
    auto_refresh = st.checkbox("Auto-Refresh (5s)", value=True)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

# ─── Header ───────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("# 🐝 Swarm Agent Trading System")
    st.markdown("*6-Layer Distributed Intelligence | BTC/USDT Futures*")
with col_status:
    st.markdown("")
    db_exists = os.path.exists(DB_PATH)
    if db_exists:
        st.success("🟢 DB Connected")
    else:
        st.error("🔴 DB Not Found")

st.markdown("---")

# ─── Top Metrics ──────────────────────────────────────────────────────────────
trades_df = load_recent_trades(100)

total_trades = len(trades_df) if not trades_df.empty else 0
closed_trades = trades_df[trades_df['status'] == 'CLOSED'] if not trades_df.empty else pd.DataFrame()
total_pnl = closed_trades['pnl_usdt'].sum() if not closed_trades.empty else 0.0
win_rate = (len(closed_trades[closed_trades['pnl_usdt'] > 0]) / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0
open_trades = len(trades_df[trades_df['status'] == 'OPEN']) if not trades_df.empty else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Total PnL (USDT)", f"${total_pnl:+.2f}")
m2.metric("📊 Total Trades", total_trades)
m3.metric("🎯 Win Rate", f"{win_rate:.1f}%")
m4.metric("🔓 Open Positions", open_trades)

st.markdown("---")

# ─── Agent Status + Consensus ─────────────────────────────────────────────────
col_agents, col_consensus = st.columns([3, 2])

with col_agents:
    st.markdown("### 🤖 Agent Health Status")
    health_df = load_agent_health()
    if health_df.empty:
        st.info("No agent health data yet. Start the swarm to populate.")
        # Show placeholder
        placeholder_data = {
            'agent_name': ['market_data', 'technical_analysis', 'scalping_strategy', 
                          'position_sizer', 'circuit_breaker', 'order_router'],
            'status': ['IDLE'] * 6,
            'uptime_seconds': [0] * 6,
            'error_count': [0] * 6,
            'avg_exec_ms': [0.0] * 6
        }
        health_df = pd.DataFrame(placeholder_data)
        st.dataframe(health_df, use_container_width=True)
    else:
        # Color code status
        def color_status(val):
            if val == 'IDLE': return 'color: #68d391'
            elif val == 'RUNNING': return 'color: #63b3ed'
            elif val == 'ERROR': return 'color: #fc8181'
            return 'color: #a0aec0'
        
        display_cols = ['agent_name', 'status', 'uptime_seconds', 'error_count', 'avg_exec_ms']
        available_cols = [c for c in display_cols if c in health_df.columns]
        st.dataframe(health_df[available_cols], use_container_width=True)

with col_consensus:
    st.markdown("### 🗳️ Latest Consensus")
    signals_df = load_signals(10)
    if signals_df.empty:
        st.info("No signals yet.")
        # Placeholder
        fig = go.Figure(go.Bar(
            x=['LONG', 'SHORT', 'HOLD'],
            y=[0, 0, 0],
            marker_color=['#48bb78', '#fc8181', '#a0aec0']
        ))
        fig.update_layout(
            title="Signal Distribution (No Data)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        signal_counts = signals_df['direction'].value_counts().reset_index()
        signal_counts.columns = ['Signal', 'Count']
        
        color_map = {'LONG': '#48bb78', 'SHORT': '#fc8181', 'HOLD': '#a0aec0'}
        colors = [color_map.get(s, '#a0aec0') for s in signal_counts['Signal']]
        
        fig = go.Figure(go.Bar(
            x=signal_counts['Signal'],
            y=signal_counts['Count'],
            marker_color=colors
        ))
        fig.update_layout(
            title="Signal Distribution (Last 10)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if not signals_df.empty:
            latest = signals_df.iloc[0]
            sig = latest.get('direction', 'N/A')
            conf = latest.get('confidence', 0)
            color = '#48bb78' if sig == 'LONG' else ('#fc8181' if sig == 'SHORT' else '#a0aec0')
            st.markdown(f"**Last Signal:** <span style='color:{color};font-size:1.2em'>{sig}</span> "
                       f"(conf: {float(conf):.1%})" if conf else f"**Last Signal:** {sig}", 
                       unsafe_allow_html=True)

st.markdown("---")

# ─── PnL Chart ────────────────────────────────────────────────────────────────
st.markdown("### 📈 Equity Curve")
if closed_trades.empty or 'pnl_usdt' not in closed_trades.columns:
    st.info("No closed trade history yet. Start trading to see equity curve.")
else:
    closed_trades_sorted = closed_trades.sort_values('exit_ts')
    closed_trades_sorted['cumulative_pnl'] = closed_trades_sorted['pnl_usdt'].cumsum()
    
    fig_equity = px.area(
        closed_trades_sorted, x='exit_ts', y='cumulative_pnl',
        title="Cumulative PnL Over Time",
        color_discrete_sequence=['#68d391']
    )
    fig_equity.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#2d3748'),
        yaxis=dict(gridcolor='#2d3748')
    )
    st.plotly_chart(fig_equity, use_container_width=True)

# ─── Recent Trades ────────────────────────────────────────────────────────────
col_trades, col_events = st.columns([3, 2])

with col_trades:
    st.markdown("### 📋 Recent Trades")
    trades_display = load_recent_trades(20)
    if trades_display.empty:
        st.info("No trades yet.")
    else:
        display_cols = ['trade_id', 'side', 'entry_ts', 'entry_price', 'pnl_usdt', 'status']
        avail = [c for c in display_cols if c in trades_display.columns]
        st.dataframe(trades_display[avail], use_container_width=True)

with col_events:
    st.markdown("### 📡 System Events")
    events_df = load_system_events(20)
    if events_df.empty:
        st.info("No system events yet.")
    else:
        display_cols = ['ts', 'event_type', 'agent_name', 'message', 'severity']
        avail = [c for c in display_cols if c in events_df.columns]
        st.dataframe(events_df[avail], use_container_width=True)

# ─── Auto Refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
