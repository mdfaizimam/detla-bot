# --- detla-bot/dashboard.py ---
# WORLD CLASS VISUALIZATION DASHBOARD
# Run with: streamlit run detla-bot/dashboard.py

import streamlit as st
import redis
import orjson
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="Delta Algo Dashboard", layout="wide", page_icon="📊")
st.title("🚀 Delta Algo: World Class Dashboard")

# --- Redis Connection ---
@st.cache_resource
def get_redis():
    return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

r = get_redis()

# --- Sidebar ---
symbol = st.sidebar.selectbox("Select Asset", ["BTCUSD", "ETHUSD", "SOLUSD"])
auto_refresh = st.sidebar.checkbox("Auto Refresh (1s)", value=True)

# --- Data Fetching ---
def get_latest_data(sym):
    key = f"latest:enriched:{sym}"
    data = r.get(key)
    if data:
        return orjson.loads(data)
    return None

# --- Main Loop ---
placeholder = st.empty()

while True:
    data = get_latest_data(symbol)
    
    with placeholder.container():
        if not data:
            st.error(f"No data found for {symbol}. Is the bot running?")
            time.sleep(2)
            continue

        # Extract Metrics
        tas = data.get("tas", {}).get("5m", {})
        price = data.get("mid_price", 0)
        ker = tas.get("ker", 0)
        rsi = tas.get("rsi_14", 0)
        bb_width = tas.get("bb_width", 0)
        tfi = data.get("tfi", 0)
        
        # Regime Logic
        regime = "🛑 CHOPPY"
        regime_color = "orange"
        if ker > 0.25:
            regime = "🚀 TRENDING"
            regime_color = "green"

        # --- KPI Row ---
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Price", f"${price:,.2f}")
        kpi2.metric("Regime (KER)", f"{ker:.2f}", delta=regime, delta_color="off")
        kpi3.metric("RSI (5m)", f"{rsi:.1f}")
        kpi4.metric("Volatility (BBW)", f"{bb_width:.4f}")
        kpi5.metric("Order Flow (TFI)", f"{tfi:.4f}")

        # --- Charts ---
        # Note: For a full chart, we would need history from Redis Streams or CSV.
        # For this MVP, we show the current state relative to BB.
        
        fig = go.Figure()
        
        # Gauge Chart for Regime
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = ker,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Efficiency (KER)"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 0.25], 'color': "red"},
                    {'range': [0.25, 1.0], 'color': "green"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.25}
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)

        # Signals Log (Last 5)
        st.subheader("📝 Recent Signal Logs")
        # (In a real app, you'd pull this from a Redis List/Stream)
        st.info("Connect this component to 'delta:signals' Redis Stream for live trade history.")

        st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

    if not auto_refresh:
        break
    time.sleep(1)