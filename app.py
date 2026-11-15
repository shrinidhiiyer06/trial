"""
KisanMandi AI - Agricultural Price Intelligence & Farmer Action Advisory
A robust, clean, classic-meets-modern decision intelligence platform for Indian Mandis.
Built with Streamlit, Plotly, and LightGBM using official Agmarknet data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(__file__))

from src.data_pipeline import load_and_clean_data, get_daily_series, engineer_features
from src.models import MandiForecaster
from src.decision_engine import optimize_sell_vs_hold, calculate_spatial_arbitrage, assess_cobweb_volatility, STORAGE_PROFILES

# Page configuration
st.set_page_config(
    page_title="KisanMandi AI | Agricultural Market Advisory",
    page_icon="🌾",
    layout="wide"
)

# Custom Styling that works harmoniously with Streamlit themes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4332;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .step-box {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #1B4332;
        background-color: rgba(27, 67, 50, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(show_spinner=False)
def get_cached_raw_data():
    return load_and_clean_data()

# Hero Header
st.markdown('<div class="main-title">🌾 KISAN MANDI AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Action-Oriented Decision Advisory & Market Price Intelligence for Farmers</div>', unsafe_allow_html=True)

# Load data
with st.spinner("Connecting to Agricultural Market Intelligence..."):
    df_raw = get_cached_raw_data()

# ========================================================
# SIDEBAR CONTROLS
# ========================================================
st.sidebar.header("📍 Farm & Crop Settings")

# 1. State, Commodity, District
available_states = sorted(df_raw['STATE_CLEAN'].unique())
default_state_idx = available_states.index('Tamil Nadu') if 'Tamil Nadu' in available_states else 0
selected_state = st.sidebar.selectbox("State", available_states, index=default_state_idx)

state_df = df_raw[df_raw['STATE_CLEAN'] == selected_state]
available_commodities = sorted(state_df['Commodity_Clean'].unique())
selected_commodity = st.sidebar.selectbox("Commodity / Crop", available_commodities, index=0)

comm_df = state_df[state_df['Commodity_Clean'] == selected_commodity]
available_districts = ['All Districts'] + sorted(comm_df['District_Clean'].unique())
default_dist_idx = available_districts.index('Thanjavur') if 'Thanjavur' in available_districts else 0
selected_district = st.sidebar.selectbox("Nearest District Mandi", available_districts, index=default_dist_idx)

st.sidebar.markdown("---")
st.sidebar.header("🚜 Harvest & Farmer Expectations")

# 2. Lot Size and Status
quantity_qtl = st.sidebar.slider("Lot Quantity to Sell (Quintals / 100 kg bags)", min_value=5, max_value=250, value=50, step=5)
st.sidebar.caption(f"📦 Total produce: **{quantity_qtl * 100:,} kg** ({quantity_qtl} bags)")

harvest_status = st.sidebar.radio(
    "Crop Readiness",
    ["Already Harvested & Packed", "Standing in Field (Harvesting Soon)"],
    index=0
)

# 3. Farmer Goal & Target Price
farmer_goal = st.sidebar.radio(
    "Your Primary Priority",
    [
        "💰 Maximize Profit (Can wait 1–3 weeks for peak price)",
        "⚡ Urgent Cash (Need money in 48–72 hours for debts/labor)",
        "🛡️ Play Safe (Avoid risk of sudden market price crash)"
    ],
    index=0
)

expected_price_kg = st.sidebar.number_input(
    "Your Expected Target Price (₹ / kg)",
    min_value=5.0,
    max_value=150.0,
    value=45.0,
    step=1.0,
    help="Enter the minimum price per kg you are hoping to get."
)
expected_price_qtl = expected_price_kg * 100

st.sidebar.markdown("---")
st.sidebar.header("🚚 Storage & Transport")
selected_facility = st.sidebar.selectbox(
    "Where is Produce Stored?",
    list(STORAGE_PROFILES.keys()),
    index=0
)

transport_mode = st.sidebar.radio(
    "Transport Availability",
    ["Can Hire Mini-Truck / Auto", "Own Tractor / Cart", "No Vehicle (Sell Locally Only)"],
    index=0
)

# ========================================================
# DATA PROCESSING & MODEL INFERENCE
# ========================================================
district_filter = None if selected_district == 'All Districts' else selected_district
daily_series = get_daily_series(df_raw, state=selected_state, district=district_filter, commodity=selected_commodity)

if len(daily_series) < 30:
    st.error(f"Insufficient historical mandi data for {selected_commodity} in {selected_district}. Please select another combination.")
    st.stop()

feat_df = engineer_features(daily_series)

# Forecaster caching
@st.cache_resource(show_spinner=False)
def train_forecaster_cached(df_features, horizon='15d'):
    forecaster = MandiForecaster(horizon=horizon)
    metrics, importances, test_preds = forecaster.train_and_benchmark(df_features)
    return forecaster, metrics, importances, test_preds

with st.spinner("Analyzing market trends..."):
    forecaster, metrics, importances, test_preds = train_forecaster_cached(feat_df, horizon='15d')

# Current Price Stats
latest_row = daily_series.iloc[-1]
current_modal = float(latest_row['Modal_Price'])
current_modal_kg = current_modal / 100.0
prev_7d_price = float(daily_series.iloc[-8]['Modal_Price']) if len(daily_series) >= 8 else current_modal
velocity_7d_pct = ((current_modal - prev_7d_price) / prev_7d_price) * 100

# Predictions
future_pred = forecaster.predict_future(feat_df, days_ahead=15)
pred_15d_modal = future_pred['p50_modal_forecast']
pred_15d_modal_kg = pred_15d_modal / 100.0
pred_15d_p10 = future_pred['p10_bearish']
pred_15d_p90 = future_pred['p90_bullish']
forecast_change_pct = ((pred_15d_modal - current_modal) / current_modal) * 100

# Decision calculations
decision_result = optimize_sell_vs_hold(
    current_price=current_modal,
    forecasted_price_30d=pred_15d_modal * 1.05,
    quantity_qtl=quantity_qtl,
    commodity=selected_commodity,
    facility=selected_facility
)

# Spatial Arbitrage calculations
arbitrage_df, origin_p = calculate_spatial_arbitrage(
    df_raw,
    origin_district=selected_district if selected_district != 'All Districts' else 'Thanjavur',
    commodity=selected_commodity,
    quantity_qtl=quantity_qtl
)

# Volatility Risk
volatility_assessment = assess_cobweb_volatility(daily_series)

# ========================================================
# ACTION LOGIC
# ========================================================
price_gap_kg = expected_price_kg - current_modal_kg
if price_gap_kg > 5:
    target_verdict = f"Your target of ₹{expected_price_kg:.1f}/kg is ₹{price_gap_kg:.1f}/kg above today's market rate (₹{current_modal_kg:.1f}/kg). Holding may bring you closer to this target."
elif abs(price_gap_kg) <= 5:
    target_verdict = f"Your target of ₹{expected_price_kg:.1f}/kg is realistic and well-matched with market levels (₹{current_modal_kg:.1f}/kg)."
else:
    target_verdict = f"Good news! Today's mandi rate (₹{current_modal_kg:.1f}/kg) is already higher than your target expectation (₹{expected_price_kg:.1f}/kg)!"

if "Urgent Cash" in farmer_goal:
    action_1_title = "Sell Within 48 Hours"
    action_1_desc = f"Because you need immediate cash flow, do not risk waiting. Liquidate your {quantity_qtl} bags within 2 days at the best local spot rate."
    recommended_hold_days = 0
elif "Play Safe" in farmer_goal:
    if volatility_assessment['risk_score'] >= 50 or forecast_change_pct < 0:
        action_1_title = "Sell Today (Distress Prevention)"
        action_1_desc = f"Market volatility is elevated. Sell immediately at current rate of ₹{current_modal:,.0f}/Qtl (₹{current_modal_kg:.1f}/kg) before prices soften."
        recommended_hold_days = 0
    else:
        action_1_title = "Hold for 4–6 Days"
        action_1_desc = f"Market is steady. A short holding period of 4–6 days safely yields an extra margin with minimal risk."
        recommended_hold_days = 5
else: # Maximize Profit
    if decision_result['best_day'] > 2 and decision_result['max_net_gain'] > 1000:
        action_1_title = f"Hold Produce for {decision_result['best_day']} Days"
        action_1_desc = f"Prices are trending upward (+{forecast_change_pct:.1f}%). Holding for {decision_result['best_day']} days will net you an estimated extra ₹{decision_result['max_net_gain']:,.0f} in hand, even after produce moisture loss."
        recommended_hold_days = decision_result['best_day']
    else:
        action_1_title = "Sell Immediately Today"
        action_1_desc = f"Holding produce will incur storage fees and weight shrinkage that outweigh minor price gains. Selling today gives you maximum total money."
        recommended_hold_days = 0

if len(arbitrage_df) > 0 and arbitrage_df.iloc[0]['Is_Profitable'] and "No Vehicle" not in transport_mode:
    best_mandi = arbitrage_df.iloc[0]['Destination_Mandi']
    best_dist = arbitrage_df.iloc[0]['Distance_KM']
    best_extra = arbitrage_df.iloc[0]['Net_Profit_Gain']
    action_2_title = f"Transport to {best_mandi} Mandi ({best_dist} km away)"
    action_2_desc = f"Transporting to {best_mandi} nets an estimated extra ₹{best_extra:,.0f} profit after deducting truck fuel (₹24/km) and mandi cess."
else:
    action_2_title = f"Sell at Local {selected_district} Mandi"
    action_2_desc = f"Local price in {selected_district} is strong. Traveling to other districts does not justify fuel and handling costs."

expected_gross = current_modal * quantity_qtl if recommended_hold_days == 0 else decision_result['optimal_revenue']

# ========================================================
# 1. ACTION PLAN CONTAINER (DIRECT FARMER GUIDANCE)
# ========================================================
with st.container(border=True):
    st.subheader(f"🎯 Recommended Action Plan for {selected_commodity} in {selected_district}")
    st.caption(f"Personalized for **{quantity_qtl} bags ({quantity_qtl*100:,} kg)** • Priority: *{farmer_goal.split('(')[0].strip()}*")

    st.markdown(f"""
    <div class="step-box">
        <b>Step 1 (Timing): {action_1_title}</b><br>
        {action_1_desc}
    </div>
    <div class="step-box">
        <b>Step 2 (Location): {action_2_title}</b><br>
        {action_2_desc}
    </div>
    <div class="step-box">
        <b>Step 3 (Money in Hand): Approx. ₹{expected_gross:,.0f}</b><br>
        {target_verdict}
    </div>
    """, unsafe_allow_html=True)

# Top KPI Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric(
        label=f"Today's Rate ({selected_district})",
        value=f"₹{current_modal_kg:.1f} / kg",
        delta=f"₹{current_modal:,.0f} / Qtl"
    )
with kpi_col2:
    st.metric(
        label="7-Day Price Momentum",
        value=f"{velocity_7d_pct:+.1f}%",
        delta="Trailing 7 days"
    )
with kpi_col3:
    st.metric(
        label="15-Day AI Forecast",
        value=f"₹{pred_15d_modal_kg:.1f} / kg",
        delta=f"{forecast_change_pct:+.1f}% projected"
    )
with kpi_col4:
    st.metric(
        label="Market Crash Risk",
        value=volatility_assessment['risk_level'].split('(')[0].strip(),
        delta=f"Score: {volatility_assessment['risk_score']}/100",
        delta_color="inverse" if volatility_assessment['risk_score'] >= 50 else "normal"
    )

st.write("")

# ========================================================
# NAVIGATION TABS
# ========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💡 Plain-English AI Explanation",
    "📈 Visual Price Chart",
    "🚚 Neighboring Mandi Comparison",
    "⚖️ Storage & Spoilage Calculator",
    "📊 Academic Verification (Evaluators)"
])

# TAB 1: PLAIN-ENGLISH AI EXPLANATION
with tab1:
    st.subheader("💡 Why is the AI Giving This Recommendation?")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        with st.container(border=True):
            st.markdown("#### 1. What is Happening to Crop Prices?")
            st.write(
                f"In **{selected_district}**, wholesale prices for **{selected_commodity}** have moved by **{velocity_7d_pct:+.1f}%** over the past 7 days. "
                f"Our model analyzed 2 years of daily market trading patterns and projects prices to head toward **₹{pred_15d_modal_kg:.1f} / kg** over the next 15 days."
            )
            st.markdown(f"""
            - **Safe Conservative Rate ($P_{{10}}$):** ₹{pred_15d_p10/100:.1f} / kg (90% confidence floor)
            - **Expected Rate ($P_{{50}}$):** ₹{pred_15d_modal_kg:.1f} / kg
            - **Optimistic Ceiling ($P_{{90}}$):** ₹{pred_15d_p90/100:.1f} / kg
            """)

        with st.container(border=True):
            st.markdown("#### 2. Target Price vs. Market Reality")
            st.write(f"You entered an expected selling price of **₹{expected_price_kg:.1f} / kg**.")
            st.info(target_verdict)

    with col_e2:
        with st.container(border=True):
            st.markdown("#### 3. Why This Timing & Storage Advice?")
            st.write(
                f"Your produce is currently stored in **{selected_facility}**. According to ICAR agricultural benchmarks, "
                f"{selected_commodity} gradually loses weight through moisture respiration."
            )
            shrinkage_qtl = abs(decision_result['trajectory'].loc[recommended_hold_days, 'Effective_Qty_Qtl'] - quantity_qtl)
            st.write(f"• Estimated moisture shrinkage: **~{shrinkage_qtl:.1f} Quintals ({shrinkage_qtl*100:.0f} kg)** over {recommended_hold_days} days.")
            st.write(f"• {decision_result['action_detail']}")

        with st.container(border=True):
            st.markdown("#### 4. Market Risk & Volatility")
            st.write(f"**Current Risk Index:** {volatility_assessment['risk_score']} / 100 ({volatility_assessment['risk_level']})")
            st.write(volatility_assessment['advice'])

# TAB 2: VISUAL PRICE CHART
with tab2:
    st.subheader(f"📈 Historical & Forecasted Price Trajectory ({selected_commodity} in {selected_district})")
    
    plot_df = daily_series.tail(120).copy()
    last_date = plot_df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=15, freq='D')
    
    smooth_p50 = np.linspace(current_modal, pred_15d_modal, 15)
    smooth_p10 = np.linspace(current_modal, pred_15d_p10, 15)
    smooth_p90 = np.linspace(current_modal, pred_15d_p90, 15)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['Date'],
        y=plot_df['Modal_Price'],
        mode='lines',
        name='Actual Past Mandi Price',
        line=dict(color='#1B4332', width=2.5),
        customdata=plot_df['Modal_Price'] / 100.0,
        hovertemplate='<b>Date:</b> %{x|%d %b %Y}<br><b>Price:</b> ₹%{y:,.0f}/Qtl (₹%{customdata:.1f}/kg)<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=smooth_p90,
        mode='lines',
        line=dict(color='rgba(197, 160, 89, 0.4)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=smooth_p10,
        mode='lines',
        line=dict(color='rgba(197, 160, 89, 0.4)', width=0),
        fill='tonexty',
        fillcolor='rgba(197, 160, 89, 0.25)',
        name='Expected Range (Safe to Best)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=smooth_p50,
        mode='lines+markers',
        name='Projected Future Price',
        line=dict(color='#C5A059', width=3, dash='dash'),
        marker=dict(size=4),
        customdata=smooth_p50 / 100.0,
        hovertemplate='<b>Date:</b> %{x|%d %b %Y}<br><b>Forecast:</b> ₹%{y:,.0f}/Qtl (₹%{customdata:.1f}/kg)<extra></extra>'
    ))
    fig.add_hline(
        y=expected_price_qtl,
        line_dash="dot",
        line_color="#2980B9",
        annotation_text=f"Your Target: ₹{expected_price_kg:.1f}/kg",
        annotation_position="bottom right"
    )
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Wholesale Price (₹ / Quintal)", showgrid=True)
    )
    st.plotly_chart(fig)

# TAB 3: NEIGHBORING MANDI COMPARISON
with tab3:
    st.subheader(f"🚚 Inter-Mandi Transport Profitability for {quantity_qtl} Bags")
    if len(arbitrage_df) > 0:
        col_a1, col_a2 = st.columns([1.5, 1])
        with col_a1:
            fig_arb = px.bar(
                arbitrage_df.head(8),
                x='Destination_Mandi',
                y='Net_Profit_Gain',
                color='Is_Profitable',
                color_discrete_map={True: '#2D6A4F', False: '#C0392B'},
                labels={'Net_Profit_Gain': 'Net Profit Delta (₹)', 'Destination_Mandi': 'Mandi Name'},
                title=f"Net Extra Profit vs Selling Locally in {selected_district}"
            )
            fig_arb.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', height=340, showlegend=False)
            st.plotly_chart(fig_arb)

        with col_a2:
            st.markdown("#### Mandi Ranking")
            for _, r in arbitrage_df.head(5).iterrows():
                status_icon = "✅ Profitable" if r['Is_Profitable'] else "❌ Not Worth"
                with st.container(border=True):
                    st.write(f"**{r['Destination_Mandi']} Mandi** ({r['Distance_KM']} km)")
                    st.write(f"Rate: **₹{r['Dest_Price_Rs_Qtl']/100:.1f} / kg** • Net: **₹{r['Net_Profit_Gain']:+,.0f}** ({status_icon})")
    else:
        st.info("No neighboring mandis have reported prices on this date for comparison.")

# TAB 4: STORAGE & SPOILAGE CALCULATOR
with tab4:
    st.subheader("⚖️ Produce Weight Loss & Net Revenue Curve")
    st.caption("Shows how moisture loss and warehouse storage fees affect net money in hand over time.")
    
    traj_df = decision_result['trajectory']
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(
        x=traj_df['Day'],
        y=traj_df['Net_Profit_Delta'],
        mode='lines+markers',
        name='Extra Money vs Selling Today (₹)',
        line=dict(color='#1B4332', width=3),
        marker=dict(size=4)
    ))
    fig_dec.add_hline(y=0, line_dash="dash", line_color="#C0392B", annotation_text="Break-even Baseline")

    best_d = decision_result['best_day']
    fig_dec.add_trace(go.Scatter(
        x=[best_d],
        y=[traj_df.loc[best_d, 'Net_Profit_Delta']],
        mode='markers',
        marker=dict(color='#C5A059', size=13, symbol='star'),
        name=f'Optimal Day to Sell (Day {best_d})'
    ))
    fig_dec.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Holding Duration (Days)", dtick=2),
        yaxis=dict(title="Net Additional Profit (₹)", showgrid=True)
    )
    st.plotly_chart(fig_dec)

# TAB 5: ACADEMIC VERIFICATION
with tab5:
    st.subheader("📊 Academic Model Verification (For Faculty Evaluators)")
    st.caption("Verification metrics for SASTRA CSE 111 AI Fundamentals evaluation rubrics.")

    st.markdown("#### 1. Algorithm Benchmark Matrix")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.highlight_min(subset=['RMSE', 'MAE'], color='#D4EDDA').highlight_max(subset=['R2 Score', 'Directional Accuracy (%)'], color='#D4EDDA'))

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### 2. Top Feature Drivers (LightGBM)")
        fig_imp = px.bar(
            importances.head(10).sort_values('Importance', ascending=True),
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#C5A059', '#1B4332']
        )
        fig_imp.update_layout(template='plotly_white', height=330, coloraxis_showscale=False)
        st.plotly_chart(fig_imp)

    with col_v2:
        st.markdown("#### 3. Out-of-Time Test Set Tracking")
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=test_preds['Date'], y=test_preds['Actual'], mode='lines', name='Actual Price', line=dict(color='#2C3531', width=1.8)))
        fig_val.add_trace(go.Scatter(x=test_preds['Date'], y=test_preds['Pred_LGBM'], mode='lines', name='LightGBM Forecast', line=dict(color='#C5A059', width=2, dash='dot')))
        fig_val.update_layout(template='plotly_white', height=330, yaxis=dict(title="Modal Price (₹/Qtl)"))
        st.plotly_chart(fig_val)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.85rem;'>"
    "<b>KisanMandi AI Platform</b> • CSE 111 Project • SASTRA Deemed University<br>"
    "Official Agmarknet Telemetry • Action-Oriented Decision Support for Indian Agriculture"
    "</div>",
    unsafe_allow_html=True
)
