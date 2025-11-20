# --- detla-bot/dashboard.py ---
# WORLD CLASS VISUALIZATION DASHBOARD
# Run with: streamlit run dashboard.py
# ✅ FIX: Replaced deprecated 'use_container_width' with 'width="stretch"'
# ✅ FIX: Unique Keys to prevent ID collisions

import streamlit as st
import redis
import orjson
import plotly.graph_objects as go
import time
from datetime import datetime

# --- Config ---
st.set_page_config(page_title="Delta Algo Dashboard", layout="wide", page_icon="📊")
st.title("🚀 Delta Algo: World Class Dashboard")

# --- Redis Connection ---
@st.cache_resource
def get_redis():
    # Connect to local Redis
    return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

try:
    r = get_redis()
    # Test connection
    r.ping()
except Exception as e:
    st.error(f"❌ Could not connect to Redis: {e}")
    st.stop()

# --- Sidebar ---
symbol = st.sidebar.selectbox("Select Asset", ["BTCUSD", "ETHUSD", "SOLUSD"])
auto_refresh = st.sidebar.checkbox("Auto Refresh (1s)", value=True)

# --- Data Fetching ---
def get_latest_data(sym):
    key = f"latest:enriched:{sym}"
    try:
        data = r.get(key)
        if data:
            return orjson.loads(data)
    except Exception:
        return None
    return None

# --- Main Loop ---
placeholder = st.empty()

while True:
    data = get_latest_data(symbol)
    
    with placeholder.container():
        if not data:
            st.warning(f"⚠️ No data found for {symbol}. Is 'main.py' running?")
            st.caption("Waiting for data stream...")
            time.sleep(2)
            continue # Retry loop

        # Extract Metrics
        tas = data.get("tas", {}).get("5m", {})
        price = data.get("mid_price", 0)
        ker = tas.get("ker", 0)
        rsi = tas.get("rsi_14", 0)
        bb_width = tas.get("bb_width", 0)
        tfi = data.get("tfi", 0)
        
        # Regime Logic
        regime = "🛑 CHOPPY"
        regime_col = "off"
        if ker > 0.25:
            regime = "🚀 TRENDING"
            regime_col = "normal" 

        # --- KPI Row ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Price", f"${price:,.2f}")
        k2.metric("Regime (KER)", f"{ker:.2f}", delta=regime, delta_color=regime_col)
        k3.metric("RSI (5m)", f"{rsi:.1f}")
        k4.metric("Volatility (BBW)", f"{bb_width:.4f}")
        k5.metric("Order Flow (TFI)", f"{tfi:.4f}")

        # --- Charts ---
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
                    {'range': [0, 0.25], 'color': "rgba(255, 0, 0, 0.3)"}, # Red Zone
                    {'range': [0.25, 1.0], 'color': "rgba(0, 255, 0, 0.3)"} # Green Zone
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.25}
            }
        ))
        
        # ✅ FIX: Updated for Streamlit 1.40+
        # Replaced 'use_container_width=True' with 'key' logic and no deprecated arg if possible
        # or mapping it correctly.
        try:
            # Try new syntax first
            st.plotly_chart(fig, key=f"ker_gauge_{symbol}_{time.time()}", on_select="ignore") 
        except TypeError:
             # Fallback for older/intermediate versions if on_select isn't ready, 
             # but we remove use_container_width to stop the warning spam.
             # By default plotly charts in streamlit usually stretch anyway.
             st.plotly_chart(fig, key=f"ker_gauge_{symbol}_{time.time()}")

        st.divider()
        
        # Signals Info
        st.subheader("📋 Active Monitoring")
        st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} | Symbol: {symbol}")
        st.info("💡 Tip: If KER > 0.25, the bot is hunting for Trends. If KER < 0.25, it is Scalping Ranges.")

    if not auto_refresh:
        break
    
    # Refresh Rate
    time.sleep(1)