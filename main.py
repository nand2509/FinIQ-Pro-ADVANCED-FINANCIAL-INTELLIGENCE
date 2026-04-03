# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import yfinance as yf
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from io import StringIO
# import warnings
# warnings.filterwarnings("ignore")
#
# # ──────────────────────────────────────────────────────────
# # PAGE CONFIG
# # ──────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="FinIQ — Portfolio Intelligence",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # ──────────────────────────────────────────────────────────
# # THEME CONSTANTS
# # ──────────────────────────────────────────────────────────
# BG       = "#080c14"
# CARD_BG  = "#0d1220"
# BORDER   = "#1e2d45"
# FONT     = "#94a3b8"
# WHITE    = "#f0f6ff"
# C1       = "#22d3ee"
# C2       = "#3b82f6"
# C3       = "#a78bfa"
# C4       = "#4ade80"
# C5       = "#f87171"
# C6       = "#fb923c"
# C7       = "#fbbf24"
# PIE_COLS = [C1, C2, C3, C4, C5, C6, C7, "#34d399"]
#
# def dark_fig(fig, h=380):
#     fig.update_layout(
#         paper_bgcolor=BG, plot_bgcolor=BG,
#         font=dict(color=FONT, family="IBM Plex Mono, monospace"),
#         height=h,
#         margin=dict(l=16, r=16, t=40, b=16),
#         xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
#         yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
#         legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1, font_size=11),
#     )
#     return fig
#
# # ──────────────────────────────────────────────────────────
# # CSS
# # ──────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
# html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:#080c14; color:#e2e8f0; }
# .stApp { background:#080c14; }
# section[data-testid="stSidebar"] { background:#0d1220; border-right:1px solid #1e2d45; }
#
# .dash-header { background:linear-gradient(135deg,#0d1220,#111827); border:1px solid #1e3a5f;
#   border-radius:12px; padding:22px 28px; margin-bottom:20px; }
# .dash-header h1 { font-size:1.6rem; font-weight:700; color:#f0f6ff; margin:0; }
# .dash-header p  { color:#64748b; margin:4px 0 0; font-size:0.84rem; }
#
# .kpi-card { background:#0d1220; border:1px solid #1e2d45; border-radius:10px;
#   padding:16px 18px; position:relative; overflow:hidden; margin-bottom:8px; }
# .kpi-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%;
#   background:linear-gradient(180deg,#22d3ee,#3b82f6); }
# .kpi-label { font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;
#   margin-bottom:5px; font-family:'IBM Plex Mono',monospace; }
# .kpi-value { font-size:1.45rem; font-weight:700; color:#f0f6ff; font-family:'IBM Plex Mono',monospace; }
# .kpi-delta-pos { font-size:0.76rem; color:#4ade80; margin-top:3px; font-family:'IBM Plex Mono',monospace; }
# .kpi-delta-neg { font-size:0.76rem; color:#f87171; margin-top:3px; font-family:'IBM Plex Mono',monospace; }
# .kpi-delta-neu { font-size:0.76rem; color:#94a3b8; margin-top:3px; font-family:'IBM Plex Mono',monospace; }
#
# .sec { font-size:0.74rem; font-weight:600; text-transform:uppercase; letter-spacing:2px;
#   color:#22d3ee; margin:24px 0 12px; display:flex; align-items:center; gap:8px; }
# .sec::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1e3a5f,transparent); }
#
# .ratio-cell { background:#0d1220; border:1px solid #1e2d45; border-radius:8px;
#   padding:12px 14px; margin-bottom:8px; }
# .ratio-name { font-size:0.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }
# .ratio-val  { font-size:1.1rem; font-weight:600; color:#f0f6ff; font-family:'IBM Plex Mono',monospace; }
# .ratio-bench { font-size:0.68rem; color:#475569; margin-top:2px; }
#
# .stTabs [data-baseweb="tab-list"] { background:#0d1220; border-bottom:1px solid #1e2d45; }
# .stTabs [data-baseweb="tab"] { color:#64748b; font-size:0.82rem; padding:10px 18px; }
# .stTabs [aria-selected="true"] { color:#22d3ee !important; border-bottom:2px solid #22d3ee !important; background:transparent; }
#
# .info-box { background:#0d2035; border:1px solid #1e3a5f; border-radius:8px;
#   padding:12px 16px; color:#94a3b8; font-size:0.83rem; margin:8px 0; }
# .good-box { background:#0d2b1d; border:1px solid #166534; border-radius:8px;
#   padding:12px 16px; color:#4ade80; font-size:0.83rem; margin:8px 0; }
# </style>
# """, unsafe_allow_html=True)
#
# # ──────────────────────────────────────────────────────────
# # HEADER
# # ──────────────────────────────────────────────────────────
# st.markdown("""
# <div class="dash-header">
#   <h1>📊 FinIQ — Financial Portfolio Intelligence</h1>
#   <p>Upload client financial data · Auto-detect columns · Compute ratios · Build portfolio · Forecast</p>
# </div>
# """, unsafe_allow_html=True)
#
# # ──────────────────────────────────────────────────────────
# # SIDEBAR
# # ──────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("### 📂 Data Source")
#     source = st.radio("Input Type", ["Upload File", "Paste CSV", "Stock Tickers"])
#     st.markdown("---")
#     st.markdown("### 👤 Client Config")
#     client_name  = st.text_input("Client Name", "Client")
#     risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
#     benchmark    = st.selectbox("Benchmark", ["Nifty 50", "S&P 500", "MSCI World"])
#     st.markdown("---")
#     st.markdown("### 📈 Stock Tickers")
#     tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
#     period        = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
#
# # ──────────────────────────────────────────────────────────
# # DATA INGESTION
# # ──────────────────────────────────────────────────────────
# df = None
#
# if source == "Upload File":
#     file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
#     if file:
#         try:
#             df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
#             st.success(f"Loaded {len(df)} rows x {len(df.columns)} columns")
#         except Exception as e:
#             st.error(f"Load error: {e}")
#
# elif source == "Paste CSV":
#     text = st.text_area("Paste CSV data here", height=140)
#     if text.strip():
#         try:
#             df = pd.read_csv(StringIO(text))
#             st.success(f"Loaded {len(df)} rows x {len(df.columns)} columns")
#         except Exception as e:
#             st.error(f"Parse error: {e}")
#
# elif source == "Stock Tickers":
#     tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
#     if tickers:
#         raw_dfs = {}
#         for t in tickers:
#             try:
#                 raw_dfs[t] = yf.Ticker(t).history(period=period)
#             except Exception:
#                 pass
#         if raw_dfs:
#             closes = pd.DataFrame({t: d["Close"] for t, d in raw_dfs.items()}).reset_index()
#             df = closes
#             st.success(f"Loaded {len(tickers)} tickers")
#
#             st.markdown('<div class="sec">Candlestick Charts</div>', unsafe_allow_html=True)
#             stock_tabs = st.tabs(tickers)
#             for i, t in enumerate(tickers):
#                 with stock_tabs[i]:
#                     if t in raw_dfs:
#                         h = raw_dfs[t].reset_index()
#                         fig = go.Figure(go.Candlestick(
#                             x=h["Date"], open=h["Open"], high=h["High"],
#                             low=h["Low"], close=h["Close"], name=t,
#                             increasing_line_color=C4, decreasing_line_color=C5
#                         ))
#                         fig = dark_fig(fig, 400)
#                         fig.update_layout(xaxis_rangeslider_visible=False, title=f"{t} Price")
#                         st.plotly_chart(fig, use_container_width=True)
#
# # ──────────────────────────────────────────────────────────
# # MAIN
# # ──────────────────────────────────────────────────────────
# if df is not None:
#
#     df.columns = df.columns.str.strip()
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     all_cols     = df.columns.tolist()
#
#     if not numeric_cols:
#         st.error("No numeric columns found. Please check your data.")
#         st.stop()
#
#     # ── Auto-detect ───────────────────────────────────────
#     def detect(keywords, pool):
#         for kw in keywords:
#             for c in pool:
#                 if kw.lower() in c.lower():
#                     return c
#         return "(none)"
#
#     auto = {
#         "date":    detect(["year","date","period","month","quarter","fy","time"], all_cols),
#         "revenue": detect(["revenue","sales","turnover","net sales","total revenue","amount"], numeric_cols),
#         "cost":    detect(["cost","expense","cogs","expenditure","opex"], numeric_cols),
#         "asset":   detect(["total asset","asset"], numeric_cols),
#         "liab":    detect(["liabilit","debt","borrowing"], numeric_cols),
#         "equity":  detect(["equity","net worth","shareholder"], numeric_cols),
#         "ni":      detect(["net income","net profit","profit after","pat","earnings","ebit"], numeric_cols),
#         "ca":      detect(["current asset"], numeric_cols),
#         "cl":      detect(["current liab"], numeric_cols),
#     }
#
#     def idx_of(val, pool):
#         opts = ["(none)"] + pool
#         return opts.index(val) if val in opts else 0
#
#     # ── Column Mapping ────────────────────────────────────
#     with st.expander("🔧 Column Mapping — auto-detected, adjust if needed", expanded=True):
#         st.caption("FinIQ matched your columns automatically. Override any dropdown if incorrect.")
#         r1a, r1b, r1c = st.columns(3)
#         date_col    = r1a.selectbox("Date / Year",          ["(none)"]+all_cols,     index=idx_of(auto["date"],   all_cols))
#         revenue_col = r1b.selectbox("Revenue / Sales",      ["(none)"]+numeric_cols, index=idx_of(auto["revenue"],numeric_cols))
#         cost_col    = r1c.selectbox("Cost / Expenses",      ["(none)"]+numeric_cols, index=idx_of(auto["cost"],   numeric_cols))
#
#         r2a, r2b, r2c = st.columns(3)
#         asset_col   = r2a.selectbox("Total Assets",         ["(none)"]+numeric_cols, index=idx_of(auto["asset"],  numeric_cols))
#         liab_col    = r2b.selectbox("Total Liabilities",    ["(none)"]+numeric_cols, index=idx_of(auto["liab"],   numeric_cols))
#         equity_col  = r2c.selectbox("Equity / Net Worth",   ["(none)"]+numeric_cols, index=idx_of(auto["equity"], numeric_cols))
#
#         r3a, r3b, r3c = st.columns(3)
#         ni_col      = r3a.selectbox("Net Income / Profit",  ["(none)"]+numeric_cols, index=idx_of(auto["ni"],     numeric_cols))
#         ca_col      = r3b.selectbox("Current Assets",       ["(none)"]+numeric_cols, index=idx_of(auto["ca"],     numeric_cols))
#         cl_col      = r3c.selectbox("Current Liabilities",  ["(none)"]+numeric_cols, index=idx_of(auto["cl"],     numeric_cols))
#
#     def C(name):
#         return None if name == "(none)" else name
#
#     rev   = C(revenue_col)
#     cost  = C(cost_col)
#     ni    = C(ni_col)
#     asset = C(asset_col)
#     liab  = C(liab_col)
#     eq    = C(equity_col)
#     ca    = C(ca_col)
#     cl    = C(cl_col)
#     dax   = C(date_col)
#
#     def xax():
#         if dax:
#             return df[dax].astype(str).tolist()
#         return list(range(len(df)))
#
#     # ── Raw Data ──────────────────────────────────────────
#     st.markdown('<div class="sec">Data Preview</div>', unsafe_allow_html=True)
#     st.dataframe(df, use_container_width=True, height=200)
#
#     # ── TABS ──────────────────────────────────────────────
#     tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#         "🏦 KPI Overview",
#         "📐 Financial Ratios",
#         "📁 Portfolio Builder",
#         "📈 Trend & Forecast",
#         "⚖️ Risk Analysis",
#         "📋 Client Report"
#     ])
#
#     # ══════════════════════════════════════════════════════
#     # TAB 1 — KPI OVERVIEW
#     # ══════════════════════════════════════════════════════
#     with tab1:
#         st.markdown('<div class="sec">Key Performance Indicators</div>', unsafe_allow_html=True)
#
#         if not rev:
#             st.markdown('<div class="info-box">Select a <strong>Revenue / Sales</strong> column in the mapping above to populate this tab.</div>', unsafe_allow_html=True)
#         else:
#             r_series = df[rev].astype(float)
#             latest   = r_series.iloc[-1]
#             avg      = r_series.mean()
#             total    = r_series.sum()
#             growth   = ((r_series.iloc[-1] - r_series.iloc[-2]) / r_series.iloc[-2] * 100) if len(r_series) > 1 and r_series.iloc[-2] != 0 else 0
#             cagr     = ((r_series.iloc[-1] / r_series.iloc[0]) ** (1 / max(len(r_series)-1, 1)) - 1) * 100 if r_series.iloc[0] != 0 and len(r_series) > 1 else 0
#
#             kpis = [
#                 ("Latest Revenue",  f"{latest:,.0f}",  f"{'▲' if growth>=0 else '▼'} {abs(growth):.1f}% vs prior", growth >= 0),
#                 ("Total Revenue",   f"{total:,.0f}",   f"{len(r_series)} periods",                                  True),
#                 ("Avg Revenue",     f"{avg:,.0f}",     "per period",                                                True),
#                 ("Revenue CAGR",    f"{cagr:.2f}%",    "annualised growth rate",                                    cagr >= 0),
#             ]
#
#             if cost:
#                 gp_v = float(df[rev].astype(float).iloc[-1] - df[cost].astype(float).iloc[-1])
#                 gm_v = gp_v / latest * 100 if latest != 0 else 0
#                 kpis.append(("Gross Profit", f"{gp_v:,.0f}", f"margin: {gm_v:.1f}%", gm_v > 0))
#
#             if ni:
#                 ni_v = float(df[ni].astype(float).iloc[-1])
#                 nm_v = ni_v / latest * 100 if latest != 0 else 0
#                 kpis.append(("Net Income", f"{ni_v:,.0f}", f"margin: {nm_v:.1f}%", nm_v > 0))
#
#             cols = st.columns(min(len(kpis), 4))
#             for i, (label, val, delta, positive) in enumerate(kpis):
#                 dcls = "kpi-delta-pos" if positive else "kpi-delta-neg"
#                 cols[i % 4].markdown(
#                     f'<div class="kpi-card">'
#                     f'<div class="kpi-label">{label}</div>'
#                     f'<div class="kpi-value">{val}</div>'
#                     f'<div class="{dcls}">{delta}</div>'
#                     f'</div>',
#                     unsafe_allow_html=True
#                 )
#
#             # Revenue bar chart
#             st.markdown('<div class="sec">Revenue Overview</div>', unsafe_allow_html=True)
#             fig = go.Figure()
#             fig.add_trace(go.Bar(x=xax(), y=df[rev].astype(float), name="Revenue", marker_color=C1, opacity=0.85))
#             if cost:
#                 fig.add_trace(go.Bar(x=xax(), y=df[cost].astype(float), name="Cost", marker_color=C5, opacity=0.75))
#             if ni:
#                 fig.add_trace(go.Scatter(x=xax(), y=df[ni].astype(float), name="Net Income",
#                                          mode="lines+markers", line=dict(color=C4, width=2.5), marker=dict(size=7)))
#             fig.update_layout(barmode="group", title="Revenue vs Cost vs Net Income")
#             fig = dark_fig(fig)
#             st.plotly_chart(fig, use_container_width=True)
#
#             # Margin trends
#             if cost or ni:
#                 st.markdown('<div class="sec">Margin Trends</div>', unsafe_allow_html=True)
#                 fig2 = go.Figure()
#                 if cost:
#                     gm_s = (df[rev].astype(float) - df[cost].astype(float)) / df[rev].astype(float).replace(0, np.nan) * 100
#                     fig2.add_trace(go.Scatter(x=xax(), y=gm_s, name="Gross Margin %",
#                                               mode="lines+markers", line=dict(color=C1, width=2),
#                                               fill="tozeroy", fillcolor="rgba(34,211,238,0.08)"))
#                 if ni:
#                     nm_s = df[ni].astype(float) / df[rev].astype(float).replace(0, np.nan) * 100
#                     fig2.add_trace(go.Scatter(x=xax(), y=nm_s, name="Net Margin %",
#                                               mode="lines+markers", line=dict(color=C4, width=2),
#                                               fill="tozeroy", fillcolor="rgba(74,222,128,0.08)"))
#                 fig2.update_layout(title="Margin % Over Time", yaxis_title="%")
#                 fig2 = dark_fig(fig2)
#                 st.plotly_chart(fig2, use_container_width=True)
#
#             # YoY growth
#             if len(df) > 1:
#                 st.markdown('<div class="sec">Year-on-Year Growth %</div>', unsafe_allow_html=True)
#                 yoy = r_series.pct_change() * 100
#                 colors = [C4 if v >= 0 else C5 for v in yoy]
#                 fig3 = go.Figure(go.Bar(x=xax(), y=yoy, marker_color=colors, name="YoY %"))
#                 fig3.update_layout(title="Revenue Growth % (Period-on-Period)", yaxis_title="%")
#                 fig3 = dark_fig(fig3, 300)
#                 st.plotly_chart(fig3, use_container_width=True)
#
#     # ══════════════════════════════════════════════════════
#     # TAB 2 — FINANCIAL RATIOS
#     # ══════════════════════════════════════════════════════
#     with tab2:
#         st.markdown('<div class="sec">Financial Ratios — Latest Period</div>', unsafe_allow_html=True)
#
#         def last(col_name):
#             return float(df[col_name].astype(float).iloc[-1]) if col_name else None
#
#         rv  = last(rev);  cs = last(cost); ns = last(ni)
#         at  = last(asset); lb = last(liab); ev = last(eq)
#         cv  = last(ca);   clv = last(cl)
#
#         ratios = {}
#
#         if rv and cs and rv != 0:
#             gm = (rv - cs) / rv * 100
#             ratios["Gross Margin %"]      = (f"{gm:.2f}%",  "Profitability", ">40%",   gm > 40)
#         if ns and rv and rv != 0:
#             nm = ns / rv * 100
#             ratios["Net Profit Margin %"] = (f"{nm:.2f}%",  "Profitability", ">10%",   nm > 10)
#         if ns and at and at != 0:
#             roa = ns / at * 100
#             ratios["Return on Assets %"]  = (f"{roa:.2f}%", "Profitability", ">5%",    roa > 5)
#         if ns and ev and ev != 0:
#             roe = ns / ev * 100
#             ratios["Return on Equity %"]  = (f"{roe:.2f}%", "Profitability", ">15%",   roe > 15)
#         if cv and clv and clv != 0:
#             cr = cv / clv
#             qr = (cv * 0.8) / clv
#             ratios["Current Ratio"]       = (f"{cr:.2f}x",  "Liquidity",     "1.5-3x", 1.5 <= cr <= 3)
#             ratios["Quick Ratio"]         = (f"{qr:.2f}x",  "Liquidity",     ">1x",    qr > 1)
#         if lb and at and at != 0:
#             da = lb / at
#             ratios["Debt to Assets"]      = (f"{da:.2f}x",  "Leverage",      "<0.5",   da < 0.5)
#         if lb and ev and ev != 0:
#             de = lb / ev
#             ratios["Debt to Equity"]      = (f"{de:.2f}x",  "Leverage",      "<1x",    de < 1)
#         if at and ev and ev != 0:
#             em = at / ev
#             ratios["Equity Multiplier"]   = (f"{em:.2f}x",  "Leverage",      "1-3",    1 <= em <= 3)
#         if rv and at and at != 0:
#             ato = rv / at
#             ratios["Asset Turnover"]      = (f"{ato:.2f}x", "Efficiency",    ">1x",    ato > 1)
#
#         if not ratios:
#             st.markdown('<div class="info-box">Map Revenue, Assets, Liabilities, and Net Income columns to compute financial ratios.</div>', unsafe_allow_html=True)
#         else:
#             for cat in sorted(set(v[1] for v in ratios.values())):
#                 st.markdown(f'<div class="sec">{cat}</div>', unsafe_allow_html=True)
#                 items = {k: v for k, v in ratios.items() if v[1] == cat}
#                 cols  = st.columns(min(len(items), 4))
#                 for i, (name, (val, _, bench, good)) in enumerate(items.items()):
#                     icon = "🟢" if good else "🔴"
#                     cols[i % 4].markdown(
#                         f'<div class="ratio-cell">'
#                         f'<div class="ratio-name">{name}</div>'
#                         f'<div class="ratio-val">{icon} {val}</div>'
#                         f'<div class="ratio-bench">Benchmark: {bench}</div>'
#                         f'</div>',
#                         unsafe_allow_html=True
#                     )
#
#             st.markdown('<div class="sec">Health Radar</div>', unsafe_allow_html=True)
#             labels = list(ratios.keys())
#             scores = [85 if v[3] else 25 for v in ratios.values()]
#             fig_r  = go.Figure(go.Scatterpolar(
#                 r=scores + [scores[0]], theta=labels + [labels[0]],
#                 fill="toself", fillcolor="rgba(34,211,238,0.12)",
#                 line=dict(color=C1, width=2), name="Health"
#             ))
#             fig_r.update_layout(
#                 polar=dict(
#                     bgcolor=CARD_BG,
#                     radialaxis=dict(visible=True, range=[0,100], gridcolor=BORDER, color=FONT),
#                     angularaxis=dict(color=FONT)
#                 ),
#                 paper_bgcolor=BG, font=dict(color=FONT, family="IBM Plex Mono"),
#                 height=420, margin=dict(l=40, r=40, t=40, b=40)
#             )
#             st.plotly_chart(fig_r, use_container_width=True)
#
#     # ══════════════════════════════════════════════════════
#     # TAB 3 — PORTFOLIO BUILDER
#     # ══════════════════════════════════════════════════════
#     with tab3:
#         st.markdown('<div class="sec">Asset Allocation</div>', unsafe_allow_html=True)
#         left, right = st.columns(2)
#
#         with left:
#             n_assets    = st.number_input("Number of Asset Classes", min_value=2, max_value=10, value=5)
#             def_names   = ["Equity","Debt/Bonds","Gold","Real Estate","Cash"]
#             def_weights = [40, 30, 10, 15, 5]
#             def_rets    = [12.0, 7.0, 8.0, 9.0, 4.0]
#             a_names, a_weights, a_rets = [], [], []
#             for i in range(n_assets):
#                 pc1, pc2, pc3 = st.columns([2, 1, 1])
#                 nm  = pc1.text_input("Asset",   def_names[i]   if i < 5 else f"Asset {i+1}", key=f"nm{i}")
#                 wt  = pc2.number_input("Wt %",  min_value=0,   max_value=100,  value=def_weights[i] if i < 5 else 10,  key=f"wt{i}")
#                 ret = pc3.number_input("Ret %", min_value=0.0, max_value=50.0, value=def_rets[i]    if i < 5 else 8.0, key=f"rt{i}")
#                 a_names.append(nm); a_weights.append(wt); a_rets.append(ret)
#             total_w = sum(a_weights)
#             if total_w != 100:
#                 st.warning(f"Weights sum to {total_w}% — must equal 100%")
#
#         with right:
#             fig_pie = go.Figure(go.Pie(
#                 labels=a_names, values=a_weights, hole=0.52,
#                 marker_colors=PIE_COLS[:n_assets],
#                 textfont=dict(color=WHITE, size=12),
#                 hovertemplate="%{label}: %{value}%<extra></extra>"
#             ))
#             fig_pie.update_layout(
#                 paper_bgcolor=BG, font=dict(color=FONT), height=340,
#                 margin=dict(l=0,r=0,t=30,b=0),
#                 legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1),
#                 annotations=[dict(text=client_name, x=0.5, y=0.5,
#                                   font_size=13, showarrow=False, font_color="#94a3b8")]
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
#
#         st.markdown('<div class="sec">Portfolio Metrics</div>', unsafe_allow_html=True)
#         if total_w > 0:
#             wts      = np.array(a_weights) / total_w
#             port_ret = float(np.dot(wts, a_rets))
#             vol_map  = {"Conservative": 6.0, "Moderate": 10.0, "Aggressive": 16.0}
#             vol      = vol_map[risk_profile]
#             sharpe   = (port_ret - 6.0) / vol
#
#             mc1, mc2, mc3, mc4 = st.columns(4)
#             for co, lbl, val in [
#                 (mc1, "Expected Return", f"{port_ret:.2f}%"),
#                 (mc2, "Est. Volatility",  f"{vol:.1f}%"),
#                 (mc3, "Sharpe Ratio",    f"{sharpe:.2f}"),
#                 (mc4, "Risk Profile",    risk_profile),
#             ]:
#                 co.markdown(
#                     f'<div class="kpi-card">'
#                     f'<div class="kpi-label">{lbl}</div>'
#                     f'<div class="kpi-value">{val}</div>'
#                     f'</div>',
#                     unsafe_allow_html=True
#                 )
#
#             st.markdown('<div class="sec">10-Year Growth Projection</div>', unsafe_allow_html=True)
#             invest = st.number_input("Initial Investment (Rs)", min_value=10000, max_value=100000000, value=1000000, step=50000)
#             sip    = st.number_input("Monthly SIP (Rs)", min_value=0, max_value=1000000, value=10000, step=1000)
#
#             proj_years = list(range(11))
#             proj_vals  = [invest]
#             v = float(invest)
#             for _ in range(10):
#                 v = v * (1 + port_ret / 100) + sip * 12
#                 proj_vals.append(v)
#
#             fig_proj = go.Figure()
#             fig_proj.add_trace(go.Scatter(
#                 x=proj_years, y=proj_vals, mode="lines+markers",
#                 fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
#                 line=dict(color=C2, width=2.5), marker=dict(size=7, color=C2),
#                 hovertemplate="Year %{x}: Rs %{y:,.0f}<extra></extra>", name="Portfolio Value"
#             ))
#             fig_proj.update_layout(title=f"Projected Growth @ {port_ret:.1f}% p.a.",
#                                    xaxis_title="Years", yaxis_title="Value (Rs)")
#             fig_proj = dark_fig(fig_proj)
#             st.plotly_chart(fig_proj, use_container_width=True)
#
#             invested = invest + sip * 12 * 10
#             gain     = proj_vals[-1] - invested
#             st.markdown(
#                 f'<div class="good-box">'
#                 f'Projected corpus after 10 years: <strong>Rs {proj_vals[-1]:,.0f}</strong> | '
#                 f'Total invested: Rs {invested:,.0f} | '
#                 f'Wealth created: Rs {gain:,.0f}'
#                 f'</div>',
#                 unsafe_allow_html=True
#             )
#
#     # ══════════════════════════════════════════════════════
#     # TAB 4 — TREND & FORECAST
#     # ══════════════════════════════════════════════════════
#     with tab4:
#         if not rev:
#             st.markdown('<div class="info-box">Select a Revenue column in the mapping to enable trend analysis.</div>', unsafe_allow_html=True)
#         elif len(df) < 3:
#             st.markdown('<div class="info-box">Need at least 3 rows for trend analysis.</div>', unsafe_allow_html=True)
#         else:
#             y_vals = df[rev].astype(float).values
#             x_vals = xax()
#
#             st.markdown('<div class="sec">Moving Averages</div>', unsafe_allow_html=True)
#             ma3 = pd.Series(y_vals).rolling(3, min_periods=1).mean().values
#             ma5 = pd.Series(y_vals).rolling(5, min_periods=1).mean().values
#
#             fig_ma = go.Figure()
#             fig_ma.add_trace(go.Scatter(x=x_vals, y=y_vals, name="Actual", line=dict(color=C1, width=2.5)))
#             fig_ma.add_trace(go.Scatter(x=x_vals, y=ma3,    name="MA-3",   line=dict(color=C3, width=1.5, dash="dot")))
#             fig_ma.add_trace(go.Scatter(x=x_vals, y=ma5,    name="MA-5",   line=dict(color=C4, width=1.5, dash="dash")))
#             fig_ma.update_layout(title="Revenue with Moving Averages")
#             fig_ma = dark_fig(fig_ma)
#             st.plotly_chart(fig_ma, use_container_width=True)
#
#             st.markdown('<div class="sec">Polynomial Regression Forecast</div>', unsafe_allow_html=True)
#             deg = st.slider("Polynomial Degree", 1, 3, 2)
#             hor = st.slider("Forecast Horizon (periods)", 1, 10, 5)
#
#             idx_arr  = np.arange(len(y_vals)).reshape(-1, 1)
#             poly     = PolynomialFeatures(degree=deg)
#             Xp       = poly.fit_transform(idx_arr)
#             mdl      = LinearRegression().fit(Xp, y_vals)
#             fitted   = mdl.predict(Xp)
#             fut_idx  = np.arange(len(y_vals), len(y_vals)+hor).reshape(-1, 1)
#             preds    = mdl.predict(poly.transform(fut_idx))
#             std_err  = float(np.std(y_vals - fitted))
#             upper    = preds + 1.65 * std_err
#             lower    = preds - 1.65 * std_err
#             fut_x    = [f"+{i+1}" for i in range(hor)]
#
#             fig_fc = go.Figure()
#             fig_fc.add_trace(go.Scatter(x=list(range(len(y_vals))), y=y_vals,  name="Actual",   line=dict(color=C1, width=2.5)))
#             fig_fc.add_trace(go.Scatter(x=list(range(len(y_vals))), y=fitted,  name="Fitted",   line=dict(color=C2, width=1.5, dash="dot")))
#             fig_fc.add_trace(go.Scatter(x=fut_x, y=preds, name="Forecast",
#                                         mode="lines+markers", line=dict(color=C5, width=2.5, dash="dash"), marker=dict(size=8)))
#             band_x = fut_x + fut_x[::-1]
#             band_y = list(upper) + list(lower[::-1])
#             fig_fc.add_trace(go.Scatter(x=band_x, y=band_y, fill="toself",
#                                         fillcolor="rgba(248,113,113,0.10)",
#                                         line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
#             fig_fc.update_layout(title="Revenue Forecast with Confidence Interval")
#             fig_fc = dark_fig(fig_fc)
#             st.plotly_chart(fig_fc, use_container_width=True)
#
#             st.dataframe(
#                 pd.DataFrame({"Period": fut_x, "Forecast": preds, "Lower (90%)": lower, "Upper (90%)": upper})
#                 .style.format({"Forecast":"{:,.0f}","Lower (90%)":"{:,.0f}","Upper (90%)":"{:,.0f}"}),
#                 use_container_width=True
#             )
#
#     # ══════════════════════════════════════════════════════
#     # TAB 5 — RISK ANALYSIS
#     # ══════════════════════════════════════════════════════
#     with tab5:
#         if not rev or len(df) < 3:
#             st.markdown('<div class="info-box">Select a Revenue column with at least 3 rows for risk analysis.</div>', unsafe_allow_html=True)
#         else:
#             pct = pd.Series(df[rev].astype(float).values).pct_change().dropna() * 100
#
#             st.markdown('<div class="sec">Return Distribution & Volatility</div>', unsafe_allow_html=True)
#             rc1, rc2 = st.columns(2)
#
#             with rc1:
#                 fig_h = go.Figure(go.Histogram(x=pct, nbinsx=20, marker_color=C2, opacity=0.8))
#                 fig_h.update_layout(title="Distribution of Period Returns", xaxis_title="%")
#                 fig_h = dark_fig(fig_h, 320)
#                 st.plotly_chart(fig_h, use_container_width=True)
#
#             with rc2:
#                 rvol = pct.rolling(3, min_periods=1).std()
#                 fig_v = go.Figure(go.Scatter(
#                     y=rvol, fill="tozeroy",
#                     fillcolor="rgba(167,139,250,0.12)",
#                     line=dict(color=C3, width=2), name="Rolling Vol"
#                 ))
#                 fig_v.update_layout(title="Rolling 3-Period Volatility", yaxis_title="%")
#                 fig_v = dark_fig(fig_v, 320)
#                 st.plotly_chart(fig_v, use_container_width=True)
#
#             st.markdown('<div class="sec">Risk Statistics</div>', unsafe_allow_html=True)
#             stats = {
#                 "Mean Return":         f"{pct.mean():.2f}%",
#                 "Std Deviation":       f"{pct.std():.2f}%",
#                 "Max Drawdown":        f"{pct.min():.2f}%",
#                 "Best Period":         f"{pct.max():.2f}%",
#                 "Skewness":            f"{pct.skew():.2f}",
#                 "Kurtosis":            f"{pct.kurtosis():.2f}",
#                 "VaR (95%)":           f"{pct.quantile(0.05):.2f}%",
#                 "Positive Periods":    f"{(pct>0).sum()}/{len(pct)}",
#             }
#             sc = st.columns(4)
#             for i, (k, v) in enumerate(stats.items()):
#                 sc[i%4].markdown(
#                     f'<div class="ratio-cell">'
#                     f'<div class="ratio-name">{k}</div>'
#                     f'<div class="ratio-val">{v}</div>'
#                     f'</div>',
#                     unsafe_allow_html=True
#                 )
#
#             if len(numeric_cols) >= 2:
#                 st.markdown('<div class="sec">Correlation Heatmap</div>', unsafe_allow_html=True)
#                 corr   = df[numeric_cols].astype(float).corr()
#                 fig_c  = px.imshow(corr, text_auto=".2f",
#                                    color_continuous_scale="RdBu_r",
#                                    color_continuous_midpoint=0)
#                 fig_c  = dark_fig(fig_c, 420)
#                 st.plotly_chart(fig_c, use_container_width=True)
#
#     # ══════════════════════════════════════════════════════
#     # TAB 6 — CLIENT REPORT
#     # ══════════════════════════════════════════════════════
#     with tab6:
#         st.markdown('<div class="sec">Client Portfolio Report</div>', unsafe_allow_html=True)
#         rev_status = (f'<span style="color:#4ade80">Yes — {rev}</span>' if rev
#                       else '<span style="color:#f87171">Not mapped</span>')
#         ni_status  = (f'<span style="color:#4ade80">Yes — {ni}</span>' if ni
#                       else '<span style="color:#f87171">Not mapped</span>')
#         st.markdown(
#             f'<div style="background:{CARD_BG};border:1px solid #1e3a5f;border-radius:12px;padding:28px 32px;">'
#             f'<h2 style="color:{C1};font-size:1.3rem;margin:0 0 4px;">Investment Portfolio Report</h2>'
#             f'<p style="color:#64748b;margin:0 0 20px;font-size:0.82rem;">FinIQ Portfolio Intelligence Platform</p>'
#             f'<table style="width:100%;font-size:0.84rem;border-collapse:collapse;">'
#             f'<tr><td style="color:#64748b;padding:5px 0;width:180px;">Client</td><td style="color:{WHITE};font-weight:600;">{client_name}</td>'
#             f'<td style="color:#64748b;padding:5px 0;width:180px;">Risk Profile</td><td style="color:{WHITE};font-weight:600;">{risk_profile}</td></tr>'
#             f'<tr><td style="color:#64748b;padding:5px 0;">Benchmark</td><td style="color:{WHITE};font-weight:600;">{benchmark}</td>'
#             f'<td style="color:#64748b;padding:5px 0;">Data</td><td style="color:{WHITE};font-weight:600;">{len(df):,} rows x {len(df.columns)} cols</td></tr>'
#             f'<tr><td style="color:#64748b;padding:5px 0;">Revenue Col</td><td>{rev_status}</td>'
#             f'<td style="color:#64748b;padding:5px 0;">Net Income Col</td><td>{ni_status}</td></tr>'
#             f'</table>'
#             f'<hr style="border-color:#1e2d45;margin:18px 0;">'
#             f'<p style="color:#475569;font-size:0.74rem;line-height:1.65;">'
#             f'This report is for informational purposes only and does not constitute financial advice. '
#             f'Past performance is not indicative of future results. Consult a SEBI-registered advisor before any investment decision.'
#             f'</p></div>',
#             unsafe_allow_html=True
#         )
#         st.markdown('<div class="sec">Descriptive Statistics</div>', unsafe_allow_html=True)
#         st.dataframe(
#             df[numeric_cols].astype(float).describe().T.style.format("{:,.2f}").background_gradient(cmap="Blues"),
#             use_container_width=True
#         )
#
# # ──────────────────────────────────────────────────────────
# # FOOTER
# # ──────────────────────────────────────────────────────────
# st.markdown(
#     f'<div style="text-align:center;padding:28px 0 12px;color:#334155;font-size:0.73rem;'
#     f'font-family:IBM Plex Mono,monospace;border-top:1px solid #1e2d45;margin-top:36px;">'
#     f'FinIQ — Financial Portfolio Intelligence | Streamlit + Plotly | For advisory use only'
#     f'</div>',
#     unsafe_allow_html=True
# )