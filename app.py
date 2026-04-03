# -*- coding: utf-8 -*-
"""
FinIQ Pro \u2014 Advanced Financial Intelligence Platform
All data from LIVE sources:
  - yfinance  : stock prices, fundamentals, FX, commodities
  - FRED CSV  : CPI, Fed Funds, GDP, Unemployment (no API key needed)
  - Yahoo RSS : news + sentiment
All 20 custom-upload tabs fully implemented.
FIX: 'charmap' codec error on Windows \u2014 all open() calls use encoding='utf-8'
FIX: nonlocal score/max_score replaced with mutable list _sc
FIX: Styler non-unique index \u2014 safe_style() resets index before styling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy import stats
from io import StringIO
import hashlib, json, os, datetime, time, re, math, warnings
import ta
from textblob import TextBlob
import feedparser
import requests

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FinIQ Pro \u2014 Advanced Financial Intelligence",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded"
)

BG      = "#080c14"
CARD_BG = "#0d1220"
BORDER  = "#1e2d45"
FONT    = "#94a3b8"
WHITE   = "#f0f6ff"
C1,C2,C3,C4,C5,C6,C7 = "#22d3ee","#3b82f6","#a78bfa","#4ade80","#f87171","#fb923c","#fbbf24"
PIE_COLS = [C1,C2,C3,C4,C5,C6,C7,"#34d399","#e879f9","#38bdf8","#f472b6","#a3e635"]

# \u2500\u2500 DATABASE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Always write UTF-8. On read, try UTF-8 first, then fall back through latin-1
# (latin-1 maps bytes 0-255 one-to-one, so it NEVER raises UnicodeDecodeError).
# After a successful latin-1 fallback the file is immediately re-written as
# proper UTF-8, so the problem self-heals after one run.
DB_FILE = "finiq_pro_users.json"

def _default_db() -> dict:
    return {
        "users": {"admin": {
            "password": hash_pw("Admin@123"), "role": "admin",
            "name": "System Admin", "email": "admin@finiq.pro",
            "created": str(datetime.datetime.now()), "last_login": None,
            "login_count": 0, "watchlist": [], "portfolios": [],
            "notes": [], "clients": [], "activity_log": []
        }},
        "settings": {"allow_reg": True, "app_name": "FinIQ Pro"}
    }

def load_db() -> dict:
    # Missing OR empty file -> start fresh
    if not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0:
        db = _default_db()
        save_db(db)
        return db

    # Try encodings in order; latin-1 is the guaranteed fallback (maps all 256 bytes)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(DB_FILE, "r", encoding=enc, errors="strict") as f:
                db = json.load(f)
            if enc != "utf-8":
                save_db(db)   # re-write as clean UTF-8 so this only runs once
            return db
        except (UnicodeDecodeError, ValueError):
            # ValueError covers json.JSONDecodeError (its base class)
            continue

    # Last resort: raw bytes, replace bad chars, attempt parse
    try:
        raw = open(DB_FILE, "rb").read().decode("utf-8", errors="replace")
        db = json.loads(raw)
        save_db(db)
        return db
    except Exception:
        pass

    # Completely unreadable — delete and recreate clean
    try:
        os.remove(DB_FILE)
    except OSError:
        pass
    db = _default_db()
    save_db(db)
    return db

def save_db(db: dict) -> None:
    """Always write as UTF-8 so Windows never hits a charmap error again."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, default=str, ensure_ascii=False)

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()
def verify_pw(p, h): return hash_pw(p) == h

def log_act(user, action):
    db = load_db()
    if user in db["users"]:
        db["users"][user].setdefault("activity_log", []).insert(0, {
            "time": str(datetime.datetime.now()), "action": action})
        db["users"][user]["activity_log"] = db["users"][user]["activity_log"][:50]
        save_db(db)

# \u2500\u2500 LIVE DATA FETCHERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
@st.cache_data(ttl=300)
def fetch_ticker(symbol, period="1y"):
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period=period)
        info = {}
        try: info = tk.info
        except: pass
        return hist, info
    except:
        return pd.DataFrame(), {}

@st.cache_data(ttl=600)
def fetch_fred_series(series_id: str):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            for line in reversed(resp.text.strip().splitlines()[1:]):
                parts = line.split(",")
                if len(parts) == 2 and parts[1].strip() not in ("", "."):
                    try: return round(float(parts[1].strip()), 2)
                    except ValueError: continue
    except: pass
    return None

@st.cache_data(ttl=300)
def fetch_market_indicator(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="3d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except: pass
    return None

@st.cache_data(ttl=600)
def fetch_economic_data() -> dict:
    results = {}
    for name, sym in {
        "VIX (Fear Index)":"^VIX","USD Index (DXY)":"DX-Y.NYB",
        "Crude Oil ($/bbl)":"CL=F","10Y Treasury Yield (%)":"^TNX",
        "Gold ($/oz)":"GC=F","S&P 500":"^GSPC",
        "Nasdaq 100":"^NDX","Bitcoin (USD)":"BTC-USD"}.items():
        v = fetch_market_indicator(sym)
        results[name] = v if v is not None else "N/A"
    for name, sid in {
        "US CPI (YoY %)":"CPIAUCSL","Fed Funds Rate (%)":"FEDFUNDS",
        "Unemployment Rate (%)":"UNRATE","US GDP Growth (%)":"A191RL1Q225SBEA",
        "Core PCE (%)":"PCEPILFE"}.items():
        v = fetch_fred_series(sid)
        results[name] = v if v is not None else "N/A"
    return results

@st.cache_data(ttl=3600)
def fetch_news_sentiment(query):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:10]:
            title = e.get("title",""); summary = e.get("summary","")
            blob = TextBlob(title+" "+summary)
            s = blob.sentiment.polarity
            items.append({"title":title[:120],"url":e.get("link","#"),
                          "published":e.get("published","")[:20],"sentiment":s,
                          "label":"\U0001f7e2 Positive" if s>0.05 else "\U0001f534 Negative" if s<-0.05 else "\u26aa Neutral"})
        return items
    except: return []

# \u2500\u2500 HELPERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def safe_style(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)

def dark_fig(fig, h=380):
    axes = (go.Bar,go.Scatter,go.Histogram,go.Heatmap,go.Candlestick,
            go.Box,go.Violin,go.Waterfall,go.Funnel)
    upd = dict(paper_bgcolor=BG, font=dict(color=FONT,family="IBM Plex Mono,monospace"),
               height=h, margin=dict(l=16,r=16,t=44,b=16),
               legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1,font_size=11))
    if any(isinstance(t,axes) for t in fig.data):
        upd.update(plot_bgcolor=BG,
                   xaxis=dict(gridcolor=BORDER,linecolor=BORDER,zerolinecolor=BORDER),
                   yaxis=dict(gridcolor=BORDER,linecolor=BORDER,zerolinecolor=BORDER))
    fig.update_layout(**upd)
    return fig

def kpi(col, label, value, delta="", pos=True):
    dcls = "kpi-delta-pos" if pos else "kpi-delta-neg"
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div>'
                 f'<div class="{dcls}">{delta}</div></div>', unsafe_allow_html=True)

def sec(title): st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)

def ibox(msg, kind="info"):
    cls = {"info":"info-box","good":"good-box","warn":"warn-box",
           "error":"error-box","blue":"blue-box"}[kind]
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)


# \u2500\u2500 CSS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;background:#080c14;color:#e2e8f0;}
.stApp{background:#080c14;}
section[data-testid="stSidebar"]{background:#0a0f1c;border-right:1px solid #1e2d45;}
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] p{color:#94a3b8!important;}
.dash-header{background:linear-gradient(135deg,#0d1220,#0f172a);border:1px solid #1e3a5f;border-radius:14px;padding:20px 28px;margin-bottom:20px;position:relative;overflow:hidden;}
.dash-header h1{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;color:#f0f6ff;margin:0;letter-spacing:-0.5px;}
.dash-header p{color:#475569;margin:4px 0 0;font-size:0.82rem;}
.kpi-card{background:#0d1220;border:1px solid #1e2d45;border-radius:12px;padding:16px 18px;position:relative;overflow:hidden;margin-bottom:8px;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#22d3ee,#3b82f6);}
.kpi-label{font-size:0.67rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;font-family:'IBM Plex Mono',monospace;}
.kpi-value{font-size:1.38rem;font-weight:700;color:#f0f6ff;font-family:'IBM Plex Mono',monospace;line-height:1.2;}
.kpi-delta-pos{font-size:0.73rem;color:#4ade80;margin-top:4px;font-family:'IBM Plex Mono',monospace;}
.kpi-delta-neg{font-size:0.73rem;color:#f87171;margin-top:4px;font-family:'IBM Plex Mono',monospace;}
.sec{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:2.5px;color:#22d3ee;margin:22px 0 10px;display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;}
.sec::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,#1e3a5f,transparent);}
.ratio-cell{background:#0d1220;border:1px solid #1e2d45;border-radius:9px;padding:11px 14px;margin-bottom:7px;}
.ratio-name{font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
.ratio-val{font-size:1.05rem;font-weight:600;color:#f0f6ff;font-family:'IBM Plex Mono',monospace;}
.ratio-bench{font-size:0.65rem;color:#334155;margin-top:2px;}
.stTabs [data-baseweb="tab-list"]{background:#0d1220;border-bottom:1px solid #1e2d45;gap:0;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{color:#475569;font-size:0.79rem;padding:9px 14px;}
.stTabs [aria-selected="true"]{color:#22d3ee!important;border-bottom:2px solid #22d3ee!important;background:transparent;}
.info-box{background:#0d2035;border:1px solid #1e3a5f;border-radius:8px;padding:11px 15px;color:#94a3b8;font-size:0.82rem;margin:7px 0;}
.good-box{background:#0d2b1d;border:1px solid #166534;border-radius:8px;padding:11px 15px;color:#4ade80;font-size:0.82rem;margin:7px 0;}
.warn-box{background:#2b1e0d;border:1px solid #92400e;border-radius:8px;padding:11px 15px;color:#fbbf24;font-size:0.82rem;margin:7px 0;}
.error-box{background:#2b0d0d;border:1px solid #7f1d1d;border-radius:8px;padding:11px 15px;color:#f87171;font-size:0.82rem;margin:7px 0;}
.blue-box{background:#0d1e35;border:1px solid #1e3a6f;border-radius:8px;padding:11px 15px;color:#93c5fd;font-size:0.82rem;margin:7px 0;}
.news-card{background:#0d1220;border:1px solid #1e2d45;border-radius:9px;padding:13px 16px;margin-bottom:8px;}
.news-title{color:#e2e8f0;font-size:0.85rem;font-weight:500;line-height:1.4;}
.news-meta{color:#475569;font-size:0.72rem;margin-top:5px;font-family:'IBM Plex Mono',monospace;}
.pos-sentiment{color:#4ade80;}.neg-sentiment{color:#f87171;}.neu-sentiment{color:#94a3b8;}
.signal-buy{background:#0d2b1d;border:1px solid #166534;color:#4ade80;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-block;}
.signal-sell{background:#2b0d0d;border:1px solid #7f1d1d;color:#f87171;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-block;}
.signal-hold{background:#2b1e0d;border:1px solid #92400e;color:#fbbf24;padding:3px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;display:inline-block;}
.badge-admin{background:#1e3a5f;color:#22d3ee;padding:2px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}
.badge-user{background:#1a2d1a;color:#4ade80;padding:2px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;}
.auth-glow{box-shadow:0 0 40px rgba(34,211,238,0.08);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# \u2500\u2500 SESSION STATE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
for k,v in [("authenticated",False),("username",None),("role",None),
            ("user_name",None),("auth_page","login"),
            ("login_attempts",0),("lockout_time",None),
            ("selected_tickers",["AAPL","MSFT","GOOGL"])]:
    if k not in st.session_state: st.session_state[k] = v

# \u2500\u2500 AUTH \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def render_login():
    _,c2,_ = st.columns([1,1.4,1])
    with c2:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 16px;">
          <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;
            background:linear-gradient(135deg,#22d3ee,#3b82f6,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-2px;">FinIQ Pro</div>
          <div style="color:#334155;font-size:0.75rem;letter-spacing:3px;font-family:'IBM Plex Mono',monospace;margin-top:4px;">
            ADVANCED FINANCIAL INTELLIGENCE</div>
        </div>
        <div style="background:#0d1220;border:1px solid #1e3a5f;border-radius:16px;padding:32px;" class="auth-glow">
        """, unsafe_allow_html=True)
        if st.session_state.lockout_time:
            elapsed = time.time()-st.session_state.lockout_time
            if elapsed < 300:
                ibox(f"Locked for {int((300-elapsed)/60)+1} more minute(s).", "error"); return
            else:
                st.session_state.login_attempts = 0; st.session_state.lockout_time = None
        st.markdown("#### Sign In")
        with st.form("lf"):
            username = st.text_input("Username"); password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                if not username or not password: st.error("Fill all fields.")
                else:
                    db = load_db()
                    if username in db["users"]:
                        u = db["users"][username]
                        if verify_pw(password, u["password"]):
                            st.session_state.update({"authenticated":True,"username":username,
                                "role":u["role"],"user_name":u["name"],"login_attempts":0})
                            db["users"][username]["last_login"] = str(datetime.datetime.now())
                            db["users"][username]["login_count"] = u.get("login_count",0)+1
                            save_db(db); log_act(username,"Logged in"); st.rerun()
                        else:
                            st.session_state.login_attempts += 1
                            if st.session_state.login_attempts >= 5:
                                st.session_state.lockout_time = time.time(); st.error("Locked 5 min.")
                            else: st.error(f"Wrong password. {5-st.session_state.login_attempts} left.")
                    else: st.error("User not found.")
        ca,cb = st.columns(2)
        if ca.button("Register", use_container_width=True):
            st.session_state.auth_page="register"; st.rerun()
        if cb.button("Reset Password", use_container_width=True):
            st.session_state.auth_page="reset"; st.rerun()
        st.markdown('<div style="text-align:center;margin-top:12px;color:#334155;font-size:0.73rem;">Default: <code style="color:#22d3ee">admin</code> / <code style="color:#22d3ee">Admin@123</code></div></div>', unsafe_allow_html=True)

def render_register():
    _,c2,_ = st.columns([1,1.4,1])
    with c2:
        st.markdown("### Create Account")
        db = load_db()
        if not db["settings"].get("allow_reg",True):
            ibox("Registration disabled.","error")
            if st.button("Back"): st.session_state.auth_page="login"; st.rerun()
            return
        with st.form("rf"):
            r1,r2=st.columns(2); full=r1.text_input("Full Name *"); uname=r2.text_input("Username *")
            r3,r4=st.columns(2); email=r3.text_input("Email *"); org=r4.text_input("Organization")
            r5,r6=st.columns(2); pw=r5.text_input("Password *",type="password"); cpw=r6.text_input("Confirm *",type="password")
            agree=st.checkbox("I agree to Terms")
            if st.form_submit_button("Register", use_container_width=True):
                errs=[]
                if not all([full,uname,email,pw,cpw]): errs.append("All * fields required.")
                if uname and uname in db["users"]: errs.append("Username taken.")
                if uname and not re.match(r"^[a-z0-9_]{3,20}$",uname): errs.append("Username: 3-20 lowercase chars.")
                if email and not re.match(r"[^@]+@[^@]+\.[^@]+",email): errs.append("Invalid email.")
                if pw!=cpw: errs.append("Passwords don't match.")
                if pw and len(pw)<8: errs.append("Min 8 chars.")
                if not agree: errs.append("Must agree to Terms.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    db["users"][uname]={"password":hash_pw(pw),"role":"user","name":full,"email":email,
                        "organization":org,"created":str(datetime.datetime.now()),"last_login":None,
                        "login_count":0,"watchlist":[],"portfolios":[],"notes":[],"clients":[],"activity_log":[]}
                    save_db(db); st.success("Account created!"); time.sleep(1)
                    st.session_state.auth_page="login"; st.rerun()
        if st.button("Back to Login"): st.session_state.auth_page="login"; st.rerun()

def render_reset():
    _,c2,_ = st.columns([1,1.4,1])
    with c2:
        st.markdown("### Reset Password")
        with st.form("resetf"):
            uname=st.text_input("Username"); email=st.text_input("Registered Email")
            np1=st.text_input("New Password",type="password"); np2=st.text_input("Confirm",type="password")
            if st.form_submit_button("Reset Password", use_container_width=True):
                db=load_db()
                if uname not in db["users"]: st.error("User not found.")
                elif db["users"][uname].get("email","").lower()!=email.lower(): st.error("Email mismatch.")
                elif np1!=np2: st.error("Don't match.")
                elif len(np1)<8: st.error("Min 8 chars.")
                else:
                    db["users"][uname]["password"]=hash_pw(np1)
                    save_db(db); st.success("Password reset!"); time.sleep(1)
                    st.session_state.auth_page="login"; st.rerun()
        if st.button("Back"): st.session_state.auth_page="login"; st.rerun()

if not st.session_state.authenticated:
    {"login":render_login,"register":render_register,"reset":render_reset}[st.session_state.auth_page]()
    st.stop()


# \u2500\u2500 MAIN APP \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
db = load_db()
cur_user = db["users"].get(st.session_state.username, {})
IS_ADMIN = st.session_state.role == "admin"

with st.sidebar:
    initials = "".join(n[0] for n in st.session_state.user_name.split()[:2]).upper()
    st.markdown(f"""<div style="padding:14px;background:#0d1220;border:1px solid #1e3a5f;border-radius:10px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#22d3ee,#3b82f6);
          display:flex;align-items:center;justify-content:center;font-weight:700;color:#080c14;font-size:0.95rem;">{initials}</div>
        <div><div style="color:#f0f6ff;font-weight:600;font-size:0.88rem;">{st.session_state.user_name}</div>
        <div style="color:#475569;font-size:0.7rem;">@{st.session_state.username} {"\u00b7 Admin" if IS_ADMIN else "\u00b7 User"}</div></div>
      </div></div>""", unsafe_allow_html=True)

    page = st.selectbox("Navigation", [
        "\U0001f4ca Market Dashboard","\U0001f4c8 Stock Analyzer","\U0001f916 ML Forecasting",
        "\u26a1 Technical Analysis","\U0001f4c9 Portfolio Optimizer","\U0001f504 Monte Carlo Sim",
        "\U0001f9e0 AI Insights & Signals","\U0001f4f0 News & Sentiment","\U0001f4c1 Custom Data Upload",
        "\U0001f3af Goal & What-If Planner","\U0001f464 My Profile / Watchlist","\U0001f4cb Client Manager",
        *(["\u2699\ufe0f Admin Panel"] if IS_ADMIN else [])
    ])
    st.markdown("---")
    currency_sym = st.selectbox("Currency", ["INR","USD","EUR","GBP"])
    curr = currency_sym
    risk_profile = st.selectbox("Risk Profile", ["Conservative","Moderate","Aggressive"])
    tickers_raw = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,TSLA")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    period  = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3)
    st.markdown("---")
    if st.button("Sign Out", use_container_width=True):
        log_act(st.session_state.username,"Logged out")
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# \u2500\u2500 ADMIN PANEL \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
if page == "\u2699\ufe0f Admin Panel" and IS_ADMIN:
    st.markdown('<div class="dash-header"><h1>Admin Panel</h1><p>System management</p></div>', unsafe_allow_html=True)
    ta1,ta2,ta3 = st.tabs(["Users","Platform Stats","Settings"])
    with ta1:
        for uname,ud in db["users"].items():
            with st.expander(f"{'Admin' if ud.get('role')=='admin' else 'User'} {ud.get('name')} (@{uname})"):
                u1,u2,u3,u4=st.columns(4)
                u1.metric("Logins",ud.get("login_count",0)); u2.metric("Watchlist",len(ud.get("watchlist",[])))
                u3.metric("Portfolios",len(ud.get("portfolios",[]))); u4.metric("Last Login",str(ud.get("last_login","\u2014"))[:10])
                c1,c2,c3=st.columns(3)
                nr=c1.selectbox("Role",["user","admin"],index=0 if ud.get("role")!="admin" else 1,key=f"r_{uname}")
                nn=c2.text_input("Name",ud.get("name",""),key=f"n_{uname}")
                if c3.button("Save",key=f"s_{uname}"):
                    db["users"][uname]["role"]=nr; db["users"][uname]["name"]=nn; save_db(db); st.success("Saved!")
                if uname not in ["admin",st.session_state.username]:
                    if st.button(f"Delete {uname}",key=f"d_{uname}"):
                        del db["users"][uname]; save_db(db); st.rerun()
    with ta2:
        all_u=db["users"]; c1,c2,c3,c4=st.columns(4)
        kpi(c1,"Users",len(all_u)); kpi(c2,"Admins",sum(1 for u in all_u.values() if u.get("role")=="admin"))
        kpi(c3,"Logins",sum(u.get("login_count",0) for u in all_u.values()))
        kpi(c4,"Portfolios",sum(len(u.get("portfolios",[])) for u in all_u.values()))
    with ta3:
        with st.form("ss"):
            ar=st.checkbox("Allow Registration",db["settings"].get("allow_reg",True))
            an=st.text_input("App Name",db["settings"].get("app_name","FinIQ Pro"))
            if st.form_submit_button("Save"):
                db["settings"]["allow_reg"]=ar; db["settings"]["app_name"]=an; save_db(db); st.success("Saved!")
    st.stop()

# \u2500\u2500 PROFILE / WATCHLIST \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f464 My Profile / Watchlist":
    st.markdown('<div class="dash-header"><h1>My Profile & Watchlist</h1></div>', unsafe_allow_html=True)
    pt1,pt2,pt3,pt4 = st.tabs(["Profile","Watchlist","Saved Portfolios","Security"])
    with pt1:
        c1,c2=st.columns([1,2])
        with c2:
            with st.form("epf"):
                nn=st.text_input("Name",cur_user.get("name","")); ne=st.text_input("Email",cur_user.get("email",""))
                no=st.text_input("Organization",cur_user.get("organization","")); nb=st.text_area("Bio",cur_user.get("bio",""),height=80)
                if st.form_submit_button("Save Profile"):
                    db["users"][st.session_state.username].update({"name":nn,"email":ne,"organization":no,"bio":nb})
                    st.session_state.user_name=nn; save_db(db); st.success("Saved!")
    with pt2:
        wl=cur_user.get("watchlist",[])
        c1,c2=st.columns([3,1]); add_sym=c1.text_input("Add symbol")
        if c2.button("Add") and add_sym:
            sym=add_sym.strip().upper()
            if sym not in wl:
                db["users"][st.session_state.username].setdefault("watchlist",[]).append(sym)
                save_db(db); st.rerun()
        for i,sym in enumerate(wl):
            hd,_=fetch_ticker(sym,"5d"); r1,r2,_,_,r5=st.columns([2,2,2,2,1])
            r1.markdown(f"**{sym}**")
            if not hd.empty and len(hd)>=2:
                lp=hd["Close"].iloc[-1]; pp=hd["Close"].iloc[-2]; chg=(lp-pp)/pp*100
                r2.metric("Price",f"${lp:.2f}",f"{chg:+.2f}%")
            if r5.button("Del",key=f"wl_{i}"):
                db["users"][st.session_state.username]["watchlist"].pop(i); save_db(db); st.rerun()
    with pt4:
        with st.form("cpwf"):
            op=st.text_input("Current Password",type="password")
            np1=st.text_input("New Password",type="password"); np2=st.text_input("Confirm",type="password")
            if st.form_submit_button("Update"):
                if not verify_pw(op,cur_user["password"]): st.error("Wrong current password.")
                elif np1!=np2: st.error("Don't match.")
                elif len(np1)<8: st.error("Min 8 chars.")
                else:
                    db["users"][st.session_state.username]["password"]=hash_pw(np1); save_db(db); st.success("Updated!")
    st.stop()

# \u2500\u2500 CLIENT MANAGER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4cb Client Manager":
    st.markdown('<div class="dash-header"><h1>Client Manager</h1></div>', unsafe_allow_html=True)
    ct1,ct2=st.tabs(["Clients","Add Client"])
    with ct1:
        clients=cur_user.get("clients",[])
        if not clients: ibox("No clients yet.","info")
        else:
            s_q=st.text_input("Search")
            for i,c in enumerate(clients):
                if s_q and s_q.lower() not in json.dumps(c).lower(): continue
                with st.expander(f"{c.get('name')} \u2014 {c.get('risk_profile','N/A')}"):
                    cc1,cc2,cc3=st.columns(3)
                    cc1.write(f"**Email:** {c.get('email','\u2014')}"); cc2.write(f"**Goal:** {c.get('goal','\u2014')}")
                    cc3.write(f"**Added:** {c.get('added','')[:10]}")
                    if st.button("Remove",key=f"rc_{i}"):
                        db["users"][st.session_state.username]["clients"].pop(i); save_db(db); st.rerun()
    with ct2:
        with st.form("acf"):
            r1,r2=st.columns(2); cn=r1.text_input("Client Name *"); ce=r2.text_input("Email")
            r3,r4=st.columns(2); cr=r3.selectbox("Risk",["Conservative","Moderate","Aggressive"]); cc_v=r4.selectbox("Currency",["INR","USD","EUR"])
            r5,r6=st.columns(2); ca_v=r5.number_input("AUM",0,10**12,1000000,50000); cg=r6.text_input("Goal")
            cn2=st.text_area("Notes",height=70)
            if st.form_submit_button("Add Client"):
                if not cn: st.error("Name required.")
                else:
                    db["users"][st.session_state.username].setdefault("clients",[]).append({
                        "name":cn,"email":ce,"risk_profile":cr,"currency":cc_v,"aum":ca_v,
                        "goal":cg,"notes":cn2,"added":str(datetime.datetime.now())})
                    save_db(db); st.success(f"{cn} added!")
    st.stop()


# \u2500\u2500 MARKET DASHBOARD \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4ca Market Dashboard":
    st.markdown('<div class="dash-header"><h1>Market Dashboard</h1><p>Live data: yfinance + FRED public API</p></div>', unsafe_allow_html=True)
    sec("Live Global Market & Macro Indicators")
    with st.spinner("Fetching live data\u2026"):
        eco = fetch_economic_data()
    cols = st.columns(4)
    for i,(k,v) in enumerate(eco.items()):
        if v=="N/A": kpi(cols[i%4],k,"N/A","unavailable",True); continue
        pos=True
        if "VIX" in k: pos=float(v)<20
        if "Unemployment" in k: pos=float(v)<4.5
        if "CPI" in k: pos=float(v)<3.0
        kpi(cols[i%4],k,str(v),"",pos)
    ibox("Data: yfinance (real-time delayed) \u00b7 FRED public CSV (macro releases)","blue")
    sec("Multi-Stock Comparison")
    raw_dfs={}
    with st.spinner("Fetching stock data\u2026"):
        for t in tickers[:6]:
            h,info=fetch_ticker(t,period)
            if not h.empty: raw_dfs[t]=(h,info)
    if raw_dfs:
        kc=st.columns(min(len(raw_dfs),4))
        for i,(t,(h,_)) in enumerate(raw_dfs.items()):
            if len(h)>=2:
                lp=h["Close"].iloc[-1]; pp=h["Close"].iloc[-2]; chg=(lp-pp)/pp*100
                kpi(kc[i%4],t,f"${lp:.2f}",f"{'up' if chg>=0 else 'dn'} {abs(chg):.2f}%",chg>=0)
        fig_norm=go.Figure()
        for t,(h,_) in raw_dfs.items():
            cl=h["Close"].astype(float); norm=(cl/cl.iloc[0])*100
            fig_norm.add_trace(go.Scatter(x=h.index.astype(str).str[:10],y=norm,name=t,mode="lines"))
        fig_norm.update_layout(title="Normalised Return (Base=100)")
        st.plotly_chart(dark_fig(fig_norm,400),use_container_width=True)
        if len(raw_dfs)>=2:
            close_df=pd.DataFrame({t:h["Close"] for t,(h,_) in raw_dfs.items()}).dropna()
            fig_c=px.imshow(close_df.corr(),text_auto=".2f",color_continuous_scale="RdBu_r",color_continuous_midpoint=0,title="Correlation Matrix")
            st.plotly_chart(dark_fig(fig_c,360),use_container_width=True)

# \u2500\u2500 STOCK ANALYZER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4c8 Stock Analyzer":
    st.markdown('<div class="dash-header"><h1>Deep Stock Analysis</h1><p>Candlestick, MAs, risk metrics & fundamentals</p></div>', unsafe_allow_html=True)
    sym=st.selectbox("Select Ticker",tickers if tickers else ["AAPL"])
    h,info=fetch_ticker(sym,period)
    if h.empty: ibox(f"No data for {sym}.","error"); st.stop()
    h.index=pd.to_datetime(h.index)
    d1,d2=st.columns(2)
    date_from=d1.date_input("From",h.index[0].date()); date_to=d2.date_input("To",h.index[-1].date())
    h=h[(h.index.date>=date_from)&(h.index.date<=date_to)]
    if h.empty: ibox("No data in range.","error"); st.stop()
    dates=h.index.astype(str).str[:10].tolist()
    closes=h["Close"].astype(float); rets=closes.pct_change().dropna()
    lp=closes.iloc[-1]; pp=closes.iloc[-2] if len(closes)>1 else lp
    day_chg=(lp-pp)/pp*100; vol_52=closes.rolling(252,min_periods=1).std().iloc[-1]*math.sqrt(252)*100
    max_dd=float(((closes/closes.cummax())-1).min()*100)
    sharpe=float(rets.mean()/rets.std()*math.sqrt(252)) if rets.std()!=0 else 0
    total_ret=(closes.iloc[-1]/closes.iloc[0]-1)*100
    c1,c2,c3,c4,c5,c6=st.columns(6)
    kpi(c1,"Price",f"${lp:.2f}",f"{'up' if day_chg>=0 else 'dn'} {abs(day_chg):.2f}%",day_chg>=0)
    kpi(c2,"Period Return",f"{total_ret:+.2f}%","",total_ret>=0)
    kpi(c3,"Ann.Vol",f"{vol_52:.1f}%","",vol_52<25); kpi(c4,"Max DD",f"{max_dd:.2f}%","",max_dd>-20)
    kpi(c5,"Sharpe",f"{sharpe:.2f}","",sharpe>1)
    if info.get("marketCap"): kpi(c6,"Mkt Cap",f"${info['marketCap']/1e9:.1f}B","",True)
    sec("OHLC Candlestick + Bollinger Bands")
    fig_cs=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.75,0.25],vertical_spacing=0.03)
    fig_cs.add_trace(go.Candlestick(x=dates,open=h["Open"],high=h["High"],low=h["Low"],close=h["Close"],
                                     increasing_line_color=C4,decreasing_line_color=C5),row=1,col=1)
    for w,c in [(20,C2),(50,C3),(200,C7)]:
        if len(closes)>=w:
            fig_cs.add_trace(go.Scatter(x=dates,y=closes.rolling(w).mean(),name=f"MA{w}",
                                         line=dict(color=c,width=1.5,dash="dot")),row=1,col=1)
    vc=[C4 if closes.iloc[i]>=closes.iloc[i-1] else C5 for i in range(len(closes))]
    fig_cs.add_trace(go.Bar(x=dates,y=h["Volume"],marker_color=vc,opacity=0.7),row=2,col=1)
    if len(closes)>=20:
        ma20=closes.rolling(20).mean(); s20=closes.rolling(20).std()
        fig_cs.add_trace(go.Scatter(x=dates,y=ma20+2*s20,line=dict(color="#475569",dash="dash",width=1),showlegend=False),row=1,col=1)
        fig_cs.add_trace(go.Scatter(x=dates,y=ma20-2*s20,line=dict(color="#475569",dash="dash",width=1),
                                     fill="tonexty",fillcolor="rgba(71,85,105,0.06)",showlegend=False),row=1,col=1)
    fig_cs.update_layout(xaxis_rangeslider_visible=False,height=520,paper_bgcolor=BG,plot_bgcolor=BG,
                          font=dict(color=FONT),margin=dict(l=16,r=16,t=44,b=16),
                          xaxis2=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER),yaxis2=dict(gridcolor=BORDER))
    st.plotly_chart(fig_cs,use_container_width=True)
    if info:
        sec("Fundamental Data (live)")
        fund_items={"P/E":info.get("trailingPE","\u2014"),"Fwd P/E":info.get("forwardPE","\u2014"),
                    "P/B":info.get("priceToBook","\u2014"),"EPS":info.get("trailingEps","\u2014"),
                    "Revenue":f"${info.get('totalRevenue',0)/1e9:.2f}B" if info.get("totalRevenue") else "\u2014",
                    "Net Inc":f"${info.get('netIncomeToCommon',0)/1e9:.2f}B" if info.get("netIncomeToCommon") else "\u2014",
                    "Gross Mgn":f"{info.get('grossMargins',0)*100:.1f}%" if info.get("grossMargins") else "\u2014",
                    "D/E":info.get("debtToEquity","\u2014"),"ROE":f"{info.get('returnOnEquity',0)*100:.1f}%" if info.get("returnOnEquity") else "\u2014",
                    "Beta":info.get("beta","\u2014"),"52W Hi":f"${info.get('fiftyTwoWeekHigh','\u2014')}","52W Lo":f"${info.get('fiftyTwoWeekLow','\u2014')}"}
        fc=st.columns(4)
        for i,(k,v) in enumerate(fund_items.items()):
            fc[i%4].markdown(f'<div class="ratio-cell"><div class="ratio-name">{k}</div><div class="ratio-val">{v}</div></div>',unsafe_allow_html=True)
    sec("Risk Metrics")
    var_95=float(rets.quantile(0.05)*100)
    cvar_95=float(rets[rets<=rets.quantile(0.05)].mean()*100) if not rets.empty else 0
    calmar=float(total_ret/abs(max_dd)) if max_dd!=0 else 0
    neg_r=rets[rets<0]; sortino=(rets.mean()*252)/(neg_r.std()*math.sqrt(252)) if len(neg_r)>0 and neg_r.std()!=0 else 0
    for i,(k,v) in enumerate([("VaR 95%",f"{var_95:.3f}%"),("CVaR 95%",f"{cvar_95:.3f}%"),
                               ("Calmar",f"{calmar:.2f}"),("Sortino",f"{sortino:.2f}"),
                               ("Skew",f"{float(rets.skew()):.3f}"),("Kurt",f"{float(rets.kurtosis()):.3f}"),
                               ("Max DD",f"{max_dd:.2f}%"),("Ann.Vol",f"{vol_52:.2f}%")]):
        [st.columns(4)[j] for j in range(4)][i%4].markdown(
            f'<div class="ratio-cell"><div class="ratio-name">{k}</div><div class="ratio-val">{v}</div></div>',unsafe_allow_html=True)


# \u2500\u2500 ML FORECASTING \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f916 ML Forecasting":
    st.markdown('<div class="dash-header"><h1>ML Price Forecasting</h1><p>RF, XGBoost, GB, Linear \u2014 auto model selection</p></div>', unsafe_allow_html=True)
    sym=st.selectbox("Select Ticker",tickers if tickers else ["AAPL"])
    h,_=fetch_ticker(sym,"2y")
    if h.empty: ibox(f"No data for {sym}.","error"); st.stop()
    h=h.copy()
    for col,fn in [("Return_1d",lambda x:x["Close"].pct_change()),
                   ("Return_5d",lambda x:x["Close"].pct_change(5)),
                   ("Return_20d",lambda x:x["Close"].pct_change(20)),
                   ("MA_10",lambda x:x["Close"].rolling(10).mean()),
                   ("MA_20",lambda x:x["Close"].rolling(20).mean()),
                   ("MA_50",lambda x:x["Close"].rolling(50).mean()),
                   ("Vol_10",lambda x:x["Close"].pct_change().rolling(10).std()),
                   ("High_Low",lambda x:x["High"]-x["Low"]),
                   ("Close_Open",lambda x:x["Close"]-x["Open"]),
                   ("Lag_1",lambda x:x["Close"].shift(1)),
                   ("Lag_5",lambda x:x["Close"].shift(5)),
                   ("Lag_10",lambda x:x["Close"].shift(10))]:
        h[col]=fn(h)
    h=h.dropna()
    features=["Return_1d","Return_5d","Return_20d","MA_10","MA_20","MA_50","Vol_10","High_Low","Close_Open","Lag_1","Lag_5","Lag_10"]
    X=h[features].values; y=h["Close"].values; dates_ml=h.index.astype(str).str[:10].tolist()
    c1,c2,c3=st.columns(3)
    model_choice=c1.selectbox("Model",["Auto (Best)","Random Forest","XGBoost","Gradient Boosting","Linear Regression"])
    n_forecast=c2.slider("Forecast Days",5,30,10); test_size=c3.slider("Test Size %",10,40,20)
    scaler=StandardScaler(); X_sc=scaler.fit_transform(X)
    split=int(len(X_sc)*(1-test_size/100))
    X_tr,X_te,y_tr,y_te=X_sc[:split],X_sc[split:],y[:split],y[split:]
    models_dict={"Random Forest":RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1),
                 "Gradient Boosting":GradientBoostingRegressor(n_estimators=200,random_state=42),
                 "Linear Regression":Ridge(alpha=1.0)}
    if HAS_XGB: models_dict["XGBoost"]=XGBRegressor(n_estimators=200,random_state=42,verbosity=0)
    with st.spinner("Training\u2026"):
        results={}
        for name,mdl in models_dict.items():
            mdl.fit(X_tr,y_tr); pred=mdl.predict(X_te)
            results[name]={"model":mdl,"pred":pred,"rmse":math.sqrt(mean_squared_error(y_te,pred)),"r2":r2_score(y_te,pred)}
        if model_choice=="Auto (Best)":
            best_name=min(results,key=lambda k:results[k]["rmse"])
            ibox(f"Auto-selected: <strong>{best_name}</strong> (RMSE:{results[best_name]['rmse']:.2f}, R2:{results[best_name]['r2']:.3f})","good")
        else: best_name=model_choice if model_choice in results else list(results.keys())[0]
    sec("Model Comparison")
    comp_df=pd.DataFrame({"Model":list(results.keys()),"RMSE":[v["rmse"] for v in results.values()],"R2":[v["r2"] for v in results.values()]}).sort_values("RMSE")
    st.dataframe(safe_style(comp_df).style.format({"RMSE":"{:.2f}","R2":"{:.4f}"}).background_gradient(cmap="RdYlGn",subset=["R2"]),use_container_width=True)
    best_mdl=results[best_name]["model"]; best_pred=results[best_name]["pred"]
    sec(f"{best_name} \u2014 Actual vs Predicted")
    fig_ml=go.Figure()
    fig_ml.add_trace(go.Scatter(x=dates_ml[:split],y=y_tr,name="Train",line=dict(color=C1)))
    fig_ml.add_trace(go.Scatter(x=dates_ml[split:],y=y_te,name="Test Actual",line=dict(color=C4)))
    fig_ml.add_trace(go.Scatter(x=dates_ml[split:],y=best_pred,name="Predicted",line=dict(color=C5,dash="dash")))
    st.plotly_chart(dark_fig(fig_ml,380),use_container_width=True)
    sec(f"{n_forecast}-Day Forecast")
    last_date=pd.to_datetime(dates_ml[-1]); curr_feat=X_sc[-1:].copy()
    future_preds=[]; fut_dates=[]
    for i in range(n_forecast):
        fp=best_mdl.predict(curr_feat)[0]; future_preds.append(fp)
        fut_dates.append((last_date+datetime.timedelta(days=i+1)).strftime("%Y-%m-%d"))
        nf=curr_feat.copy(); nf[0,9]=fp; curr_feat=nf
    fig_fut=go.Figure()
    fig_fut.add_trace(go.Scatter(x=dates_ml[-60:],y=y[-60:],name="Historical",line=dict(color=C1,width=2)))
    fig_fut.add_trace(go.Scatter(x=fut_dates,y=future_preds,name=f"Forecast ({n_forecast}d)",
                                  mode="lines+markers",line=dict(color=C5,width=2.5,dash="dash")))
    ci=float(np.std(future_preds)*1.96)
    fig_fut.add_trace(go.Scatter(x=fut_dates+fut_dates[::-1],
                                  y=[p+ci for p in future_preds]+[p-ci for p in future_preds][::-1],
                                  fill="toself",fillcolor="rgba(248,113,113,0.10)",
                                  line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
    st.plotly_chart(dark_fig(fig_fut,380),use_container_width=True)
    if hasattr(best_mdl,"feature_importances_"):
        imp=pd.Series(best_mdl.feature_importances_,index=features).sort_values(ascending=True)
        fig_fi=go.Figure(go.Bar(x=imp.values,y=imp.index,orientation="h",marker_color=C2,opacity=0.85))
        st.plotly_chart(dark_fig(fig_fi,360),use_container_width=True)

# \u2500\u2500 TECHNICAL ANALYSIS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\u26a1 Technical Analysis":
    st.markdown('<div class="dash-header"><h1>Technical Analysis</h1><p>RSI, MACD, BB, Stoch, ATR, OBV</p></div>', unsafe_allow_html=True)
    sym=st.selectbox("Select Ticker",tickers if tickers else ["AAPL"])
    h,_=fetch_ticker(sym,period)
    if h.empty: ibox(f"No data for {sym}.","error"); st.stop()
    h=h.copy(); h.index=pd.to_datetime(h.index)
    close=h["Close"].astype(float); high=h["High"].astype(float)
    low=h["Low"].astype(float); vol=h["Volume"].astype(float)
    dates=h.index.astype(str).str[:10].tolist()
    rsi=ta.momentum.RSIIndicator(close,window=14).rsi()
    macd_obj=ta.trend.MACD(close); macd_line=macd_obj.macd(); macd_signal=macd_obj.macd_signal(); macd_hist=macd_obj.macd_diff()
    bb_obj=ta.volatility.BollingerBands(close,window=20,window_dev=2)
    bb_high=bb_obj.bollinger_hband(); bb_low=bb_obj.bollinger_lband()
    stoch=ta.momentum.StochasticOscillator(high,low,close); stoch_k=stoch.stoch(); stoch_d=stoch.stoch_signal()
    atr=ta.volatility.AverageTrueRange(high,low,close,window=14).average_true_range()
    obv=ta.volume.OnBalanceVolumeIndicator(close,vol).on_balance_volume()
    ema_9=ta.trend.EMAIndicator(close,window=9).ema_indicator()
    ema_21=ta.trend.EMAIndicator(close,window=21).ema_indicator()
    cci=ta.trend.CCIIndicator(high,low,close,window=20).cci()
    wr=ta.momentum.WilliamsRIndicator(high,low,close,lbp=14).williams_r()
    lr=rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    lm=macd_line.iloc[-1]; ls=macd_signal.iloc[-1]
    lc=close.iloc[-1]; lbh=bb_high.iloc[-1]; lbl=bb_low.iloc[-1]
    lstk=stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
    signals=[("RSI","BUY" if lr<30 else "SELL" if lr>70 else "HOLD",f"RSI={lr:.1f}"),
             ("MACD","BUY" if lm>ls else "SELL","MACD vs Signal"),
             ("BB","BUY" if lc<lbl else "SELL" if lc>lbh else "HOLD","Price vs Bands"),
             ("Stoch","BUY" if lstk<20 else "SELL" if lstk>80 else "HOLD",f"Stoch={lstk:.1f}")]
    buys=sum(1 for _,s,_ in signals if s=="BUY"); sells=sum(1 for _,s,_ in signals if s=="SELL")
    overall="BUY" if buys>sells else "SELL" if sells>buys else "HOLD"
    sec("Signal Summary")
    o1,o2,o3,o4=st.columns(4)
    badge={"BUY":"signal-buy","SELL":"signal-sell","HOLD":"signal-hold"}[overall]
    o1.markdown(f'<div class="kpi-card"><div class="kpi-label">Overall</div><div class="kpi-value"><span class="{badge}">{overall}</span></div></div>',unsafe_allow_html=True)
    kpi(o2,"Buy",buys,"",True); kpi(o3,"Sell",sells,"",False); kpi(o4,"Hold",len(signals)-buys-sells,"",True)
    sc=st.columns(len(signals))
    for i,(ind,sig,reason) in enumerate(signals):
        b={"BUY":"signal-buy","SELL":"signal-sell","HOLD":"signal-hold"}[sig]
        sc[i].markdown(f'<div class="ratio-cell"><div class="ratio-name">{ind}</div><div class="ratio-val"><span class="{b}">{sig}</span></div><div class="ratio-bench">{reason}</div></div>',unsafe_allow_html=True)
    sec("RSI & MACD")
    fig_tech=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.5,0.25,0.25],vertical_spacing=0.04)
    fig_tech.add_trace(go.Scatter(x=dates,y=close,name="Close",line=dict(color=C1,width=2)),row=1,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=bb_high,line=dict(color="#475569",dash="dash",width=1),showlegend=False),row=1,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=bb_low,line=dict(color="#475569",dash="dash",width=1),fill="tonexty",fillcolor="rgba(71,85,105,0.06)",showlegend=False),row=1,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=ema_9,name="EMA9",line=dict(color=C4,width=1.5,dash="dot")),row=1,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=ema_21,name="EMA21",line=dict(color=C6,width=1.5,dash="dot")),row=1,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=rsi,name="RSI",line=dict(color=C3,width=1.8)),row=2,col=1)
    fig_tech.add_hline(y=70,line_color=C5,line_dash="dot",row=2,col=1)
    fig_tech.add_hline(y=30,line_color=C4,line_dash="dot",row=2,col=1)
    hc=[C4 if v>=0 else C5 for v in macd_hist.fillna(0)]
    fig_tech.add_trace(go.Bar(x=dates,y=macd_hist,marker_color=hc,opacity=0.7),row=3,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=macd_line,name="MACD",line=dict(color=C2,width=1.5)),row=3,col=1)
    fig_tech.add_trace(go.Scatter(x=dates,y=macd_signal,name="Signal",line=dict(color=C5,width=1.5)),row=3,col=1)
    fig_tech.update_layout(height=580,paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=FONT),
                            margin=dict(l=16,r=16,t=44,b=16),legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1,font_size=10))
    for row in [1,2,3]:
        fig_tech.update_xaxes(gridcolor=BORDER,linecolor=BORDER,row=row,col=1)
        fig_tech.update_yaxes(gridcolor=BORDER,linecolor=BORDER,row=row,col=1)
    st.plotly_chart(fig_tech,use_container_width=True)
    sec("Stochastic, ATR & OBV")
    fig_more=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.04)
    fig_more.add_trace(go.Scatter(x=dates,y=stoch_k,name="Stoch %K",line=dict(color=C1,width=1.5)),row=1,col=1)
    fig_more.add_trace(go.Scatter(x=dates,y=stoch_d,name="Stoch %D",line=dict(color=C6,width=1.5,dash="dot")),row=1,col=1)
    fig_more.add_trace(go.Scatter(x=dates,y=atr,name="ATR",line=dict(color=C7,width=1.8)),row=2,col=1)
    fig_more.add_trace(go.Scatter(x=dates,y=obv,name="OBV",line=dict(color=C3,width=1.5)),row=3,col=1)
    fig_more.update_layout(height=460,paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=FONT),
                            margin=dict(l=16,r=16,t=44,b=16),legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1))
    for row in [1,2,3]:
        fig_more.update_xaxes(gridcolor=BORDER,linecolor=BORDER,row=row,col=1)
        fig_more.update_yaxes(gridcolor=BORDER,linecolor=BORDER,row=row,col=1)
    st.plotly_chart(fig_more,use_container_width=True)


# \u2500\u2500 PORTFOLIO OPTIMIZER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4c9 Portfolio Optimizer":
    st.markdown('<div class="dash-header"><h1>Portfolio Optimizer</h1><p>MPT \u2014 Efficient Frontier, Min Variance, Max Sharpe</p></div>', unsafe_allow_html=True)
    if len(tickers)<2: ibox("Enter at least 2 tickers.","warn"); st.stop()
    close_dfs={}
    with st.spinner("Loading\u2026"):
        for t in tickers[:8]:
            h,_=fetch_ticker(t,period)
            if not h.empty: close_dfs[t]=h["Close"].astype(float)
    if len(close_dfs)<2: ibox("Not enough data.","error"); st.stop()
    prices=pd.DataFrame(close_dfs).dropna(); returns=prices.pct_change().dropna()
    n=len(prices.columns); symbols=list(prices.columns)
    mu_ann=returns.mean().values*252; cov_ann=returns.cov().values*252
    def ps(w):
        w=np.array(w); r=np.dot(w,mu_ann); v=math.sqrt(np.dot(w.T,np.dot(cov_ann,w)))
        return r,v,(r-0.05)/v if v>0 else 0
    np.random.seed(42); mc_rets,mc_vols,mc_sh=[],[],[]
    for _ in range(4000):
        w=np.random.random(n); w/=w.sum(); r,v,s=ps(w)
        mc_rets.append(r*100); mc_vols.append(v*100); mc_sh.append(s)
    cons=[{"type":"eq","fun":lambda w:np.sum(w)-1}]; bounds=[(0.01,0.99)]*n; w0=np.array([1/n]*n)
    msr=minimize(lambda w:-ps(w)[2],w0,method="SLSQP",bounds=bounds,constraints=cons)
    mvr=minimize(lambda w:ps(w)[1],w0,method="SLSQP",bounds=bounds,constraints=cons)
    msw=msr.x; mvw=mvr.x
    ms_ret,ms_vol,ms_sh=ps(msw); mv_ret,mv_vol,mv_sh=ps(mvw); eq_ret,eq_vol,eq_sh=ps(w0)
    fig_ef=go.Figure()
    fig_ef.add_trace(go.Scatter(x=mc_vols,y=mc_rets,mode="markers",
                                 marker=dict(color=mc_sh,colorscale="Viridis",size=4,opacity=0.5,
                                             colorbar=dict(title="Sharpe",tickfont=dict(color=FONT))),name="Portfolios"))
    fig_ef.add_trace(go.Scatter(x=[ms_vol*100],y=[ms_ret*100],mode="markers+text",
                                 marker=dict(size=18,color=C4,symbol="star"),text=["Max Sharpe"],textposition="top center"))
    fig_ef.add_trace(go.Scatter(x=[mv_vol*100],y=[mv_ret*100],mode="markers+text",
                                 marker=dict(size=18,color=C1,symbol="diamond"),text=["Min Var"],textposition="top center"))
    fig_ef.update_layout(title="Efficient Frontier",xaxis_title="Volatility %",yaxis_title="Return %")
    st.plotly_chart(dark_fig(fig_ef,480),use_container_width=True)
    pc1,pc2=st.columns(2)
    fig_ms=go.Figure(go.Pie(labels=symbols,values=msw*100,hole=0.5,marker_colors=PIE_COLS[:n]))
    fig_ms.update_layout(paper_bgcolor=BG,font=dict(color=FONT),height=300,margin=dict(l=0,r=0,t=40,b=0),
                          title="Max Sharpe",legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1))
    pc1.plotly_chart(fig_ms,use_container_width=True)
    fig_mv=go.Figure(go.Pie(labels=symbols,values=mvw*100,hole=0.5,marker_colors=PIE_COLS[:n]))
    fig_mv.update_layout(paper_bgcolor=BG,font=dict(color=FONT),height=300,margin=dict(l=0,r=0,t=40,b=0),
                          title="Min Variance",legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1))
    pc2.plotly_chart(fig_mv,use_container_width=True)

# \u2500\u2500 MONTE CARLO \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f504 Monte Carlo Sim":
    st.markdown('<div class="dash-header"><h1>Monte Carlo Simulation</h1></div>', unsafe_allow_html=True)
    sym=st.selectbox("Select Ticker",tickers if tickers else ["AAPL"])
    h,_=fetch_ticker(sym,"2y")
    if h.empty: ibox(f"No data.","error"); st.stop()
    close=h["Close"].astype(float); daily_rets=close.pct_change().dropna()
    mc1,mc2,mc3,mc4=st.columns(4)
    n_sims=mc1.select_slider("Simulations",[500,1000,2000,5000,10000],2000)
    mc_days=mc2.slider("Forecast Days",5,252,90)
    mc_invest=mc3.number_input(f"Investment ({curr})",1000,100000000,100000,5000)
    mc_method=mc4.selectbox("Method",["GBM","Historical Bootstrap","Parametric"])
    mu=float(daily_rets.mean()); sigma=float(daily_rets.std()); S0=float(close.iloc[-1])
    with st.spinner(f"Running {n_sims:,} simulations\u2026"):
        np.random.seed(42); paths=np.zeros((n_sims,mc_days+1)); paths[:,0]=S0
        for t in range(1,mc_days+1):
            if mc_method=="GBM":
                z=np.random.standard_normal(n_sims)
                paths[:,t]=paths[:,t-1]*np.exp((mu-0.5*sigma**2)+sigma*z)
            elif mc_method=="Historical Bootstrap":
                paths[:,t]=paths[:,t-1]*(1+np.random.choice(daily_rets.values,n_sims))
            else:
                paths[:,t]=paths[:,t-1]*(1+np.random.normal(mu,sigma,n_sims))
        port=paths/S0*mc_invest; fv=port[:,-1]
        p5,p25,p50,p75,p95=np.percentile(fv,[5,25,50,75,95])
    x_days=list(range(mc_days+1))
    fig_mc=go.Figure()
    for i in np.random.choice(n_sims,min(200,n_sims),replace=False):
        fig_mc.add_trace(go.Scatter(x=x_days,y=port[i],mode="lines",line=dict(color="rgba(59,130,246,0.02)",width=1),showlegend=False))
    for pv,col,nm in [(p95,C4,"P95"),(p50,C7,"Median"),(p5,C5,"P5")]:
        idx=np.argmin(np.abs(fv-pv))
        fig_mc.add_trace(go.Scatter(x=x_days,y=port[idx],mode="lines",line=dict(color=col,width=2.5),name=f"{nm}: {curr}{pv:,.0f}"))
    fig_mc.add_hline(y=mc_invest,line_dash="dash",line_color=WHITE,annotation_text=f"Initial: {curr}{mc_invest:,.0f}")
    fig_mc.update_layout(title=f"Monte Carlo: {sym}",xaxis_title="Days",yaxis_title=f"Value ({curr})")
    st.plotly_chart(dark_fig(fig_mc,480),use_container_width=True)
    sec("Statistics")
    prob_profit=float((fv>mc_invest).mean()*100)
    sc_m=st.columns(5)
    for i,(k,v) in enumerate([("Median",f"{curr}{p50:,.0f}"),("P95",f"{curr}{p95:,.0f}"),
                               ("P5",f"{curr}{p5:,.0f}"),("Prob Profit",f"{prob_profit:.1f}%"),
                               ("Daily Vol",f"{sigma*100:.3f}%")]):
        sc_m[i].markdown(f'<div class="ratio-cell"><div class="ratio-name">{k}</div><div class="ratio-val">{v}</div></div>',unsafe_allow_html=True)

# \u2500\u2500 AI INSIGHTS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f9e0 AI Insights & Signals":
    st.markdown('<div class="dash-header"><h1>AI Insights & Signals</h1><p>Anomaly detection, trend, backtesting</p></div>', unsafe_allow_html=True)
    sym=st.selectbox("Select Ticker",tickers if tickers else ["AAPL"])
    h,info=fetch_ticker(sym,period)
    if h.empty: ibox(f"No data.","error"); st.stop()
    close=h["Close"].astype(float); rets=close.pct_change().dropna()
    dates=h.index.astype(str).str[:10].tolist(); lp=close.iloc[-1]
    sec("Anomaly Detection")
    z_scores=np.abs(stats.zscore(rets.fillna(0))); anomalies=rets[z_scores>2.5]
    fig_anom=go.Figure()
    fig_anom.add_trace(go.Scatter(x=dates[1:],y=rets*100,name="Returns",line=dict(color=C2,width=1.5)))
    if not anomalies.empty:
        fig_anom.add_trace(go.Scatter(x=anomalies.index.astype(str).str[:10].tolist(),y=anomalies*100,
                                       mode="markers",marker=dict(color=C5,size=10,symbol="x"),name="Anomaly"))
    st.plotly_chart(dark_fig(fig_anom,320),use_container_width=True)
    ibox(f"{'Detected '+str(len(anomalies))+' anomalous days.' if not anomalies.empty else 'No anomalies detected.'}",
         "warn" if not anomalies.empty else "good")
    sec("Trend Classification")
    ma20=close.rolling(20).mean(); ma50=close.rolling(50).mean(); ma200=close.rolling(200).mean()
    ts=0; tr=[]
    if close.iloc[-1]>ma20.iloc[-1]: ts+=1; tr.append("Price > MA20")
    else: ts-=1; tr.append("Price < MA20")
    if close.iloc[-1]>ma50.iloc[-1]: ts+=1; tr.append("Price > MA50")
    else: ts-=1; tr.append("Price < MA50")
    if not pd.isna(ma200.iloc[-1]):
        if close.iloc[-1]>ma200.iloc[-1]: ts+=2; tr.append("Price > MA200 (bullish)")
        else: ts-=2; tr.append("Price < MA200 (bearish)")
    tl="Strongly Bullish" if ts>=4 else "Bullish" if ts>=2 else "Neutral" if ts>=0 else "Bearish" if ts>=-2 else "Strongly Bearish"
    ibox(f"<strong>{tl}</strong> (Score: {ts})","good" if ts>=2 else "warn" if ts==0 else "error")
    tc=st.columns(min(len(tr),3))
    for i,r in enumerate(tr): tc[i%3].markdown(f'<div class="ratio-cell"><div class="ratio-val" style="font-size:0.82rem;">{r}</div></div>',unsafe_allow_html=True)
    sec("MA Crossover Backtest")
    bt1,bt2,bt3=st.columns(3)
    fma=bt1.number_input("Fast MA",5,50,10); sma=bt2.number_input("Slow MA",20,200,50)
    ic=bt3.number_input(f"Capital ({curr})",10000,10000000,100000,10000)
    maf=close.rolling(fma).mean(); mas=close.rolling(sma).mean()
    pos=pd.Series(0,index=close.index); pos[maf>mas]=1; pos=pos.shift(1).fillna(0)
    sr=pos*rets; cs=(1+sr).cumprod(); cb=(1+rets).cumprod()
    fig_bt=go.Figure()
    fig_bt.add_trace(go.Scatter(x=dates[1:],y=cs*ic,name=f"Strategy MA{fma}/{sma}",line=dict(color=C4,width=2.5)))
    fig_bt.add_trace(go.Scatter(x=dates[1:],y=cb*ic,name="Buy & Hold",line=dict(color=C2,width=2,dash="dash")))
    st.plotly_chart(dark_fig(fig_bt,360),use_container_width=True)
    b1,b2,b3,b4=st.columns(4)
    kpi(b1,"Strategy Ret",f"{(cs.iloc[-1]-1)*100:+.2f}%","",cs.iloc[-1]>1)
    kpi(b2,"B&H Ret",f"{(cb.iloc[-1]-1)*100:+.2f}%","",cb.iloc[-1]>1)
    kpi(b3,"Strategy Sharpe",f"{float(sr.mean()/sr.std()*math.sqrt(252)) if sr.std()!=0 else 0:.3f}","",True)
    kpi(b4,"B&H Sharpe",f"{float(rets.mean()/rets.std()*math.sqrt(252)) if rets.std()!=0 else 0:.3f}","",True)

# \u2500\u2500 NEWS & SENTIMENT \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4f0 News & Sentiment":
    st.markdown('<div class="dash-header"><h1>News & Sentiment</h1><p>Yahoo Finance RSS + NLP</p></div>', unsafe_allow_html=True)
    sym=st.selectbox("Ticker",tickers if tickers else ["AAPL"])
    with st.spinner("Fetching news\u2026"): news=fetch_news_sentiment(sym)
    if news:
        avg=np.mean([n["sentiment"] for n in news]); pos_c=sum(1 for n in news if n["sentiment"]>0.05)
        neg_c=sum(1 for n in news if n["sentiment"]<-0.05)
        sn1,sn2,sn3,sn4=st.columns(4)
        kpi(sn1,"Avg Sentiment",f"{avg:+.3f}","",avg>0); kpi(sn2,"Positive",f"\U0001f7e2 {pos_c}","",True)
        kpi(sn3,"Negative",f"\U0001f534 {neg_c}","",neg_c==0); kpi(sn4,"Neutral",f"\u26aa {len(news)-pos_c-neg_c}","",True)
        fig_s=go.Figure(go.Bar(x=[n["title"][:40]+"..." for n in news],y=[n["sentiment"] for n in news],
                                marker_color=[C4 if n["sentiment"]>0.05 else C5 if n["sentiment"]<-0.05 else "#64748b" for n in news]))
        st.plotly_chart(dark_fig(fig_s,360),use_container_width=True)
        for n in news:
            sc="pos-sentiment" if n["sentiment"]>0.05 else "neg-sentiment" if n["sentiment"]<-0.05 else "neu-sentiment"
            st.markdown(f'<div class="news-card"><a href="{n["url"]}" target="_blank" style="text-decoration:none;"><div class="news-title">{n["title"]}</div></a>'
                        f'<div class="news-meta"><span class="{sc}">{n["label"]}</span> | {n["sentiment"]:+.3f} | {n["published"]}</div></div>',unsafe_allow_html=True)
    else: ibox("Could not fetch news.","warn")


# \u2500\u2500 CUSTOM DATA UPLOAD \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f4c1 Custom Data Upload":
    st.markdown('<div class="dash-header"><h1>Custom Financial Data Analysis</h1><p>All 20 tabs analyse YOUR uploaded data only \u2014 no external sources</p></div>', unsafe_allow_html=True)
    source=st.radio("Input Method",["Upload File","Paste CSV"],horizontal=True)
    df=None
    if source=="Upload File":
        file=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx"])
        if file:
            try:
                df=pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
                df.columns=df.columns.str.strip()
                ibox(f"Loaded <strong>{len(df):,}</strong> rows x <strong>{len(df.columns)}</strong> columns","good")
            except Exception as e: ibox(f"Error: {e}","error")
    else:
        txt=st.text_area("Paste CSV",height=140)
        if txt.strip():
            try:
                df=pd.read_csv(StringIO(txt)); df.columns=df.columns.str.strip()
                ibox(f"Loaded <strong>{len(df):,}</strong> rows x <strong>{len(df.columns)}</strong> columns","good")
            except Exception as e: ibox(f"Error: {e}","error")

    if df is not None:
        num_cols=df.select_dtypes(include=np.number).columns.tolist()
        all_cols=df.columns.tolist()
        if not num_cols: ibox("No numeric columns found.","error"); st.stop()

        def detect(kws,pool):
            for k in kws:
                for c in pool:
                    if k.lower() in c.lower(): return c
            return "(none)"

        auto={"date":detect(["year","date","period","month","quarter","time"],all_cols),
              "rev":detect(["revenue","sales","turnover"],num_cols),
              "cost":detect(["cost","expense","cogs"],num_cols),
              "ni":detect(["net income","net profit","pat","profit after"],num_cols),
              "asset":detect(["total asset","asset"],num_cols),
              "liab":detect(["liabilit","debt"],num_cols),
              "eq":detect(["equity","net worth"],num_cols),
              "ca":detect(["current asset"],num_cols),
              "cl":detect(["current liab"],num_cols),
              "sector":detect(["sector","category","segment","industry"],all_cols),
              "units":detect(["units","volume","qty","quantity"],num_cols)}

        def idx_of(v,pool): opts=["(none)"]+pool; return opts.index(v) if v in opts else 0

        with st.expander("Column Mapping (auto-detected)", expanded=True):
            r1,r2,r3=st.columns(3)
            date_col=r1.selectbox("Date/Year",["(none)"]+all_cols,index=idx_of(auto["date"],all_cols))
            rev_col=r2.selectbox("Revenue",["(none)"]+num_cols,index=idx_of(auto["rev"],num_cols))
            cost_col=r3.selectbox("Cost",["(none)"]+num_cols,index=idx_of(auto["cost"],num_cols))
            r4,r5,r6=st.columns(3)
            ni_col=r4.selectbox("Net Income",["(none)"]+num_cols,index=idx_of(auto["ni"],num_cols))
            ast_col=r5.selectbox("Total Assets",["(none)"]+num_cols,index=idx_of(auto["asset"],num_cols))
            lib_col=r6.selectbox("Total Liabilities",["(none)"]+num_cols,index=idx_of(auto["liab"],num_cols))
            r7,r8,r9=st.columns(3)
            eq_col=r7.selectbox("Equity",["(none)"]+num_cols,index=idx_of(auto["eq"],num_cols))
            ca_col=r8.selectbox("Current Assets",["(none)"]+num_cols,index=idx_of(auto["ca"],num_cols))
            cl_col=r9.selectbox("Current Liabilities",["(none)"]+num_cols,index=idx_of(auto["cl"],num_cols))
            r10,r11=st.columns(2)
            sec_col=r10.selectbox("Sector/Category",["(none)"]+all_cols,index=idx_of(auto["sector"],all_cols))
            units_col=r11.selectbox("Units/Qty",["(none)"]+num_cols,index=idx_of(auto["units"],num_cols))

        def C(n): return None if n=="(none)" else n
        rev=C(rev_col); cost=C(cost_col); ni=C(ni_col); ast=C(ast_col)
        lib=C(lib_col); eq=C(eq_col); ca=C(ca_col); cl=C(cl_col)
        dax=C(date_col); sec_c=C(sec_col); units_c=C(units_col)

        def xax(): return df[dax].astype(str).tolist() if dax and dax in df.columns else list(range(len(df)))
        def s(c): return df[c].astype(float)
        def last(c): return float(s(c).iloc[-1]) if c else None

        st.dataframe(df,use_container_width=True,height=180)

        tabs=st.tabs(["KPI","Ratios","Portfolio","Forecast","Risk","Cash Flow","Report",
                      "Benchmark","What-If","Goal Planner","Sectors","Monte Carlo",
                      "Break-Even","Valuation","Period Compare","Heatmap",
                      "FX Impact","Tax","Peer Bench","Insights"])

        with tabs[0]:  # KPI
            sec("Key Performance Indicators \u2014 from your data")
            if not rev: ibox("Map a Revenue column.","info")
            else:
                rs=s(rev); lv=rs.iloc[-1]; prev=rs.iloc[-2] if len(rs)>1 else lv
                grw=(lv-prev)/prev*100 if prev!=0 else 0
                cagr=((rs.iloc[-1]/rs.iloc[0])**(1/max(len(rs)-1,1))-1)*100 if rs.iloc[0]!=0 and len(rs)>1 else 0
                kpis_list=[("Latest Revenue",f"{curr} {lv:,.2f}",f"{'up' if grw>=0 else 'dn'} {abs(grw):.1f}%",grw>=0),
                            ("Total",f"{curr} {rs.sum():,.2f}",f"{len(rs)} periods",True),
                            ("Average",f"{curr} {rs.mean():,.2f}","per period",True),
                            ("CAGR",f"{cagr:.2f}%","annualised",cagr>=0)]
                if cost: gp=lv-s(cost).iloc[-1]; kpis_list.append(("Gross Profit",f"{curr} {gp:,.2f}",f"margin {gp/lv*100:.1f}%",gp>0))
                if ni: nv=s(ni).iloc[-1]; kpis_list.append(("Net Income",f"{curr} {nv:,.2f}",f"margin {nv/lv*100:.1f}%",nv>0))
                if ast: kpis_list.append(("Total Assets",f"{curr} {s(ast).iloc[-1]:,.2f}","",True))
                ck=st.columns(min(len(kpis_list),4))
                for i,(l,v,d,p) in enumerate(kpis_list): kpi(ck[i%4],l,v,d,p)
                fig=go.Figure()
                fig.add_trace(go.Bar(x=xax(),y=rs,name="Revenue",marker_color=C1,opacity=0.85))
                if cost: fig.add_trace(go.Bar(x=xax(),y=s(cost),name="Cost",marker_color=C5,opacity=0.75))
                if ni: fig.add_trace(go.Scatter(x=xax(),y=s(ni),name="Net Income",mode="lines+markers",line=dict(color=C4,width=2.5)))
                fig.update_layout(barmode="group",title="Revenue / Cost / Net Income")
                st.plotly_chart(dark_fig(fig),use_container_width=True)
                if len(df)>1:
                    yoy=rs.pct_change()*100
                    fig3=go.Figure(go.Bar(x=xax(),y=yoy,marker_color=[C4 if v>=0 else C5 for v in yoy.fillna(0)]))
                    fig3.update_layout(title="Period-on-Period Growth %",yaxis_title="%")
                    st.plotly_chart(dark_fig(fig3,280),use_container_width=True)

        with tabs[1]:  # Ratios
            sec("Financial Ratios")
            rv=last(rev); cs_v=last(cost); ns=last(ni); at=last(ast); lb=last(lib); ev=last(eq); cv=last(ca); clv=last(cl)
            ratios={}
            if rv and cs_v and rv!=0: gm=(rv-cs_v)/rv*100; ratios["Gross Margin %"]=(f"{gm:.2f}%","Profitability",">40%",gm>40)
            if ns and rv and rv!=0: nm=ns/rv*100; ratios["Net Margin %"]=(f"{nm:.2f}%","Profitability",">10%",nm>10)
            if ns and at and at!=0: roa=ns/at*100; ratios["ROA %"]=(f"{roa:.2f}%","Profitability",">5%",roa>5)
            if ns and ev and ev!=0: roe=ns/ev*100; ratios["ROE %"]=(f"{roe:.2f}%","Profitability",">15%",roe>15)
            if cv and clv and clv!=0: cr=cv/clv; ratios["Current Ratio"]=(f"{cr:.2f}x","Liquidity","1.5-3x",1.5<=cr<=3)
            if lb and at and at!=0: da=lb/at; ratios["Debt/Assets"]=(f"{da:.2f}x","Leverage","<0.5",da<0.5)
            if lb and ev and ev!=0: de=lb/ev; ratios["Debt/Equity"]=(f"{de:.2f}x","Leverage","<1x",de<1)
            if rv and at and at!=0: ato=rv/at; ratios["Asset Turnover"]=(f"{ato:.2f}x","Efficiency",">1x",ato>1)
            if not ratios: ibox("Map more columns for ratios.","info")
            else:
                for cat in ["Profitability","Liquidity","Leverage","Efficiency"]:
                    items={k:v for k,v in ratios.items() if v[1]==cat}
                    if not items: continue
                    sec(cat); cr2=st.columns(min(len(items),4))
                    for i,(name,(val,_,bench,good)) in enumerate(items.items()):
                        cr2[i%4].markdown(f'<div class="ratio-cell"><div class="ratio-name">{name}</div>'
                                          f'<div class="ratio-val">{"ok" if good else "!"} {val}</div>'
                                          f'<div class="ratio-bench">Benchmark: {bench}</div></div>',unsafe_allow_html=True)
                labels=list(ratios.keys()); scores=[85 if v[3] else 25 for v in ratios.values()]
                fig_r=go.Figure(go.Scatterpolar(r=scores+[scores[0]],theta=labels+[labels[0]],
                                                 fill="toself",fillcolor="rgba(34,211,238,0.12)",line=dict(color=C1,width=2)))
                fig_r.update_layout(polar=dict(bgcolor=CARD_BG,radialaxis=dict(visible=True,range=[0,100],gridcolor=BORDER,color=FONT),
                                                angularaxis=dict(color=FONT)),paper_bgcolor=BG,font=dict(color=FONT),height=400,margin=dict(l=50,r=50,t=44,b=50))
                st.plotly_chart(fig_r,use_container_width=True)


        with tabs[2]:  # Portfolio Builder
            sec("Asset Allocation Builder (manual input)")
            l,r_col=st.columns(2)
            with l:
                na=st.number_input("Number of assets",2,10,5)
                def_names=["Equity","Bonds","Gold","Real Estate","Cash"]; def_w=[40,30,10,15,5]
                def_r=[12.0,7.0,8.0,9.0,4.0]; def_v=[18.0,5.0,12.0,8.0,0.5]
                a_names,a_w,a_r,a_v=[],[],[],[]
                for i in range(na):
                    p1,p2,p3,p4=st.columns([2,1,1,1])
                    nm=p1.text_input("Asset",def_names[i] if i<5 else f"A{i+1}",key=f"pnm{i}")
                    wt=p2.number_input("Wt%",0,100,def_w[i] if i<5 else 10,key=f"pwt{i}")
                    rt=p3.number_input("Ret%",0.0,50.0,def_r[i] if i<5 else 8.0,key=f"prt{i}")
                    vl=p4.number_input("Vol%",0.0,80.0,def_v[i] if i<5 else 10.0,key=f"pvl{i}")
                    a_names.append(nm); a_w.append(wt); a_r.append(rt); a_v.append(vl)
                tw=sum(a_w)
                if tw!=100: ibox(f"Weights sum to {tw}% \u2014 must be 100%","warn")
            with r_col:
                fig_p=go.Figure(go.Pie(labels=a_names,values=a_w,hole=0.52,marker_colors=PIE_COLS[:na]))
                fig_p.update_layout(paper_bgcolor=BG,font=dict(color=FONT),height=300,margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_p,use_container_width=True)
            if tw>0:
                wts_arr=np.array(a_w)/tw; pr=float(np.dot(wts_arr,a_r)); pv=float(np.sqrt(np.dot(wts_arr**2,np.array(a_v)**2)))
                sh=(pr-6.0)/pv if pv!=0 else 0
                m1,m2,m3=st.columns(3)
                kpi(m1,"Expected Return",f"{pr:.2f}%","",pr>=8); kpi(m2,"Portfolio Vol",f"{pv:.1f}%","",pv<20); kpi(m3,"Sharpe",f"{sh:.2f}","",sh>1)
                inv=st.number_input(f"Investment ({curr})",10000,100000000,1000000,50000)
                sip_v=st.number_input(f"Monthly SIP ({curr})",0,1000000,10000,1000)
                fig_proj=go.Figure()
                for sn,rr,col in [("Base",pr,C2),("Optimistic",pr+pv*0.5,C4),("Conservative",pr-pv*0.5,C5)]:
                    yr_list=list(range(11)); vals=[inv]; v_t=float(inv)
                    for _ in range(10): v_t=v_t*(1+rr/100)+sip_v*12; vals.append(v_t)
                    fig_proj.add_trace(go.Scatter(x=yr_list,y=vals,name=sn,mode="lines",line=dict(color=col,width=2.5 if sn=="Base" else 1.5)))
                fig_proj.update_layout(title="10-Year Projection",xaxis_title="Years",yaxis_title=f"Value ({curr})")
                st.plotly_chart(dark_fig(fig_proj),use_container_width=True)

        with tabs[3]:  # Forecast
            if not rev or len(df)<4: ibox("Need Revenue column + at least 4 rows.","info")
            else:
                y_v=s(rev).values; x_v=xax()
                sec("Moving Averages")
                fig_ma=go.Figure()
                fig_ma.add_trace(go.Scatter(x=x_v,y=y_v,name="Actual",line=dict(color=C1,width=2)))
                for w,c in [(3,C3),(5,C4),(10,C7)]:
                    if len(y_v)>=w:
                        ma=pd.Series(y_v).rolling(w,min_periods=1).mean()
                        fig_ma.add_trace(go.Scatter(x=x_v,y=ma,name=f"MA-{w}",line=dict(color=c,width=1.5,dash="dot")))
                fig_ma.update_layout(title="Revenue with Moving Averages")
                st.plotly_chart(dark_fig(fig_ma),use_container_width=True)
                sec("Polynomial Regression Forecast")
                hor=st.slider("Forecast Periods",1,10,5)
                idx_arr=np.arange(len(y_v)).reshape(-1,1)
                poly=PolynomialFeatures(degree=2); Xp=poly.fit_transform(idx_arr)
                mdl_f=LinearRegression().fit(Xp,y_v); fitted=mdl_f.predict(Xp)
                fut_idx=np.arange(len(y_v),len(y_v)+hor).reshape(-1,1)
                preds=mdl_f.predict(poly.transform(fut_idx)); std_err=float(np.std(y_v-fitted))
                fut_x=[f"+{i+1}" for i in range(hor)]
                fig_fc=go.Figure()
                fig_fc.add_trace(go.Scatter(x=list(range(len(y_v))),y=y_v,name="Actual",line=dict(color=C1,width=2)))
                fig_fc.add_trace(go.Scatter(x=fut_x,y=preds,name="Forecast",mode="lines+markers",line=dict(color=C5,width=2,dash="dash")))
                bx=fut_x+fut_x[::-1]; by=list(preds+1.65*std_err)+list((preds-1.65*std_err)[::-1])
                fig_fc.add_trace(go.Scatter(x=bx,y=by,fill="toself",fillcolor="rgba(248,113,113,0.10)",line=dict(color="rgba(0,0,0,0)"),name="90% CI"))
                fig_fc.update_layout(title="Revenue Forecast")
                st.plotly_chart(dark_fig(fig_fc),use_container_width=True)
                fdf=pd.DataFrame({"Period":fut_x,"Forecast":[f"{curr} {p:,.2f}" for p in preds],
                                   "Lower":[f"{curr} {p-1.65*std_err:,.2f}" for p in preds],
                                   "Upper":[f"{curr} {p+1.65*std_err:,.2f}" for p in preds]})
                st.dataframe(safe_style(fdf),use_container_width=True)

        with tabs[4]:  # Risk
            if not rev or len(df)<3: ibox("Need Revenue + at least 3 rows.","info")
            else:
                pct=pd.Series(s(rev).values).pct_change().dropna()*100
                sec("Risk Statistics \u2014 from your data")
                v95=float(pct.quantile(0.05)); cv95=float(pct[pct<=pct.quantile(0.05)].mean()) if (pct<=pct.quantile(0.05)).any() else v95
                risk_d={"Mean Growth":f"{pct.mean():.2f}%","Std Dev":f"{pct.std():.2f}%",
                         "Worst Period":f"{pct.min():.2f}%","Best Period":f"{pct.max():.2f}%",
                         "Skewness":f"{pct.skew():.3f}","Kurtosis":f"{pct.kurtosis():.3f}",
                         "VaR 95%":f"{v95:.2f}%","CVaR 95%":f"{cv95:.2f}%"}
                sc4=st.columns(4)
                for i,(k,v) in enumerate(risk_d.items()): sc4[i%4].markdown(f'<div class="ratio-cell"><div class="ratio-name">{k}</div><div class="ratio-val">{v}</div></div>',unsafe_allow_html=True)
                fig_h=go.Figure(go.Histogram(x=pct,nbinsx=20,marker_color=C2,opacity=0.8))
                fig_h.update_layout(title="Revenue Growth Distribution",xaxis_title="%")
                st.plotly_chart(dark_fig(fig_h,300),use_container_width=True)
                rs2=s(rev); dd=(rs2-rs2.cummax())/rs2.cummax().replace(0,np.nan)*100
                fig_dd=go.Figure(go.Scatter(x=xax(),y=dd,fill="tozeroy",fillcolor="rgba(248,113,113,0.15)",line=dict(color=C5,width=2)))
                fig_dd.update_layout(title="Revenue Drawdown from Peak",yaxis_title="%")
                st.plotly_chart(dark_fig(fig_dd,280),use_container_width=True)
                if len(num_cols)>=2:
                    corr=df[num_cols].astype(float).corr()
                    fig_c2=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",color_continuous_midpoint=0,title="Correlation Matrix")
                    st.plotly_chart(dark_fig(fig_c2,360),use_container_width=True)

        with tabs[5]:  # Cash Flow
            if not rev: ibox("Map Revenue for waterfall.","info")
            else:
                wf_l=["Revenue"]; wf_v=[float(s(rev).iloc[-1])]; wf_m=["absolute"]
                if cost: wf_l+=["Operating Cost"]; wf_v+=[-float(s(cost).iloc[-1])]; wf_m+=["relative"]
                if ni and cost:
                    oe=float(s(rev).iloc[-1])-float(s(cost).iloc[-1])-float(s(ni).iloc[-1])
                    if abs(oe)>1: wf_l+=["Other Exp"]; wf_v+=[-oe]; wf_m+=["relative"]
                if ni: wf_l+=["Net Income"]; wf_v+=[float(s(ni).iloc[-1])]; wf_m+=["total"]
                elif cost:
                    gp=float(s(rev).iloc[-1])-float(s(cost).iloc[-1]); wf_l+=["Gross Profit"]; wf_v+=[gp]; wf_m+=["total"]
                fig_wf=go.Figure(go.Waterfall(measure=wf_m,x=wf_l,y=wf_v,
                    connector=dict(line=dict(color=BORDER,width=1)),
                    increasing=dict(marker_color=C4),decreasing=dict(marker_color=C5),totals=dict(marker_color=C2),
                    text=[f"{curr} {abs(v):,.0f}" for v in wf_v],textposition="outside"))
                fig_wf.update_layout(title="P&L Waterfall (from your data)",showlegend=False)
                st.plotly_chart(dark_fig(fig_wf,400),use_container_width=True)

        with tabs[6]:  # Report
            sec("Auto-Generated Report")
            mapped=sum(1 for x in [rev,cost,ni,ast,lib,eq,ca,cl] if x)
            st.markdown(f"""<div style="background:{CARD_BG};border:1px solid #1e3a5f;border-radius:12px;padding:24px;">
              <h2 style="color:{C1};font-size:1.2rem;">Financial Data Report</h2>
              <p style="color:#475569;font-size:0.8rem;">Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")} | User: {st.session_state.user_name}</p>
              <table style="width:100%;font-size:0.82rem;">
              <tr><td style="color:#64748b;">Rows</td><td style="color:{WHITE};font-weight:600;">{len(df):,}</td>
                  <td style="color:#64748b;">Columns Mapped</td><td style="color:{WHITE};font-weight:600;">{mapped}/8</td></tr>
              </table></div>""",unsafe_allow_html=True)
            desc_df=df[num_cols].astype(float).describe().T.reset_index().rename(columns={"index":"Metric"})
            nd=desc_df.select_dtypes(include="number").columns.tolist()
            st.dataframe(safe_style(desc_df).style.format({c:"{:,.2f}" for c in nd}).background_gradient(cmap="Blues",subset=nd),use_container_width=True)
            st.download_button("Download CSV",data=df.to_csv(index=False).encode("utf-8"),file_name="finiq_export.csv",mime="text/csv")

        with tabs[7]:  # Benchmark
            sec("Benchmark \u2014 compare two of your columns")
            if not rev: ibox("Map a Revenue column first.","info")
            elif len(num_cols)<2: ibox("Need at least 2 numeric columns.","info")
            else:
                b1,b2=st.columns(2)
                pc=b1.selectbox("Primary Series",num_cols,index=0)
                bc=b2.selectbox("Benchmark Series",[c for c in num_cols if c!=pc],index=0)
                ps2=df[pc].astype(float); bs2=df[bc].astype(float)
                psn=(ps2/ps2.iloc[0])*100; bsn=(bs2/bs2.iloc[0])*100
                outperf=float(psn.iloc[-1]-bsn.iloc[-1])
                fig_b=go.Figure()
                fig_b.add_trace(go.Scatter(x=xax(),y=psn,name=pc,line=dict(color=C1,width=2.5)))
                fig_b.add_trace(go.Scatter(x=xax(),y=bsn,name=bc,line=dict(color=C3,width=2,dash="dash")))
                fig_b.add_hline(y=100,line_dash="dot",line_color=BORDER,annotation_text="Base=100")
                fig_b.update_layout(title=f"{pc} vs {bc} (Normalised to 100)")
                st.plotly_chart(dark_fig(fig_b),use_container_width=True)
                ibox(f"<strong>{pc}</strong> {'outperformed' if outperf>=0 else 'underperformed'} <strong>{bc}</strong> by <strong>{abs(outperf):.1f} pts</strong>",
                     "good" if outperf>=0 else "warn")
                spread=ps2-bs2
                fig_sp=go.Figure(go.Bar(x=xax(),y=spread,marker_color=[C4 if v>=0 else C5 for v in spread],opacity=0.85))
                fig_sp.add_hline(y=0,line_color=BORDER)
                fig_sp.update_layout(title=f"Spread ({pc} - {bc})")
                st.plotly_chart(dark_fig(fig_sp,260),use_container_width=True)

        with tabs[8]:  # What-If
            if not rev: ibox("Map Revenue.","info")
            else:
                sec("What-If Scenario Simulator")
                rb=float(s(rev).iloc[-1])
                w1,w2,w3=st.columns(3)
                rc2=w1.slider("Revenue Delta %",-50,100,0,5); cc2=w2.slider("Cost Delta %",-50,100,0,5); tr2=w3.slider("Tax Rate %",0,50,25)
                sr_v=rb*(1+rc2/100); cb_v=(float(s(cost).iloc[-1]) if cost else rb*0.6)*(1+cc2/100)
                sni_v=(sr_v-cb_v)*(1-tr2/100); ni_b=float(s(ni).iloc[-1]) if ni else sni_v; nd=sni_v-ni_b
                s1,s2,s3,s4=st.columns(4)
                kpi(s1,"Sim Revenue",f"{curr} {sr_v:,.0f}","",sr_v>rb); kpi(s2,"Sim EBIT",f"{curr} {sr_v-cb_v:,.0f}","",sr_v-cb_v>0)
                kpi(s3,"Sim Net Inc",f"{curr} {sni_v:,.0f}","after tax",sni_v>0); kpi(s4,"Net Inc Delta",f"{curr} {nd:+,.0f}","vs actual",nd>=0)
                items=[("Revenue +5%",rb*1.05-rb),("Revenue -5%",rb*0.95-rb),
                       ("Cost +5%",-(float(s(cost).iloc[-1])*0.05) if cost else 0),
                       ("Cost -5%",(float(s(cost).iloc[-1])*0.05) if cost else 0)]
                is2=sorted(items,key=lambda x:abs(x[1]),reverse=True)
                fig_tor=go.Figure(go.Bar(x=[v for _,v in is2],y=[n for n,_ in is2],orientation="h",
                                          marker_color=[C4 if v>=0 else C5 for _,v in is2],opacity=0.85))
                fig_tor.add_vline(x=0,line_color=BORDER)
                fig_tor.update_layout(title="Sensitivity Tornado Chart")
                st.plotly_chart(dark_fig(fig_tor,300),use_container_width=True)

        with tabs[9]:  # Goal Planner
            sec("Financial Goal Planner")
            g1,g2=st.columns(2)
            with g1:
                gn=st.text_input("Goal","Retirement"); ga=st.number_input(f"Target ({curr})",100000,10**9,10000000,500000)
                gy=st.slider("Years",1,40,20); cs_v2=st.number_input(f"Current Savings ({curr})",0,10**8,500000,50000)
                mi=st.number_input(f"Monthly SIP ({curr})",0,500000,15000,1000)
                er=st.slider("Return %",1.0,25.0,10.0,0.5); inf3=st.slider("Inflation %",0.0,15.0,6.0,0.5)
            with g2:
                fvs=cs_v2*(1+er/100)**gy
                fvsip=mi*12*(((1+er/100)**gy-1)/(er/100)) if er!=0 else mi*12*gy
                tfv=fvs+fvsip; gr=ga*(1+inf3/100)**gy; sf=max(0,gr-tfv)
                kpi(g2,"Inflation-Adj Target",f"{curr} {gr:,.0f}","",True)
                kpi(g2,"Projected Corpus",f"{curr} {tfv:,.0f}","",tfv>=gr)
                kpi(g2,"Shortfall",f"{curr} {sf:,.0f}","",sf==0)
            yr_a=[]; cr_a=[]; tg_a=[]; vv_t=float(cs_v2)
            for yr in range(gy+1): cr_a.append(vv_t); tg_a.append(ga*(1+inf3/100)**yr); yr_a.append(yr); vv_t=vv_t*(1+er/100)+mi*12
            fig_gl=go.Figure()
            fig_gl.add_trace(go.Scatter(x=yr_a,y=cr_a,name="Corpus",fill="tozeroy",fillcolor="rgba(59,130,246,0.10)",line=dict(color=C2,width=2.5)))
            fig_gl.add_trace(go.Scatter(x=yr_a,y=tg_a,name="Target",line=dict(color=C5,width=2,dash="dash")))
            fig_gl.update_layout(title=f"Goal: {gn}",xaxis_title="Years",yaxis_title=f"{curr}")
            st.plotly_chart(dark_fig(fig_gl),use_container_width=True)
            if sf>0: ibox(f"Need extra SIP of {curr} {sf/(((1+er/100)**gy-1)/(er/100)*12) if er!=0 else sf/(gy*12):,.0f}/month","warn")
            else: ibox(f"On track for {gn}!","good")

        with tabs[10]:  # Sectors
            sec("Sector / Segment Analysis")
            if not sec_c: ibox("Map a Sector/Category column.","info")
            else:
                agg_col=rev if rev else (num_cols[0] if num_cols else None)
                if agg_col:
                    grp=df.groupby(sec_c)[agg_col].sum().sort_values(ascending=False)
                    cp,cb_p=st.columns(2)
                    fig_pie=go.Figure(go.Pie(labels=grp.index.tolist(),values=grp.values,hole=0.5,marker_colors=PIE_COLS[:len(grp)]))
                    fig_pie.update_layout(paper_bgcolor=BG,font=dict(color=FONT),height=340,margin=dict(l=0,r=0,t=40,b=0))
                    cp.plotly_chart(fig_pie,use_container_width=True)
                    fig_sb=go.Figure(go.Bar(x=grp.index.tolist(),y=grp.values,marker_color=PIE_COLS[:len(grp)],opacity=0.85))
                    fig_sb.update_layout(title=f"{agg_col} by {sec_c}",xaxis_tickangle=-30)
                    cb_p.plotly_chart(dark_fig(fig_sb,340),use_container_width=True)
                    if rev and ni:
                        grp2=df.groupby(sec_c)[[rev,ni]].sum().reset_index()
                        fig_m=go.Figure()
                        fig_m.add_trace(go.Bar(x=grp2[sec_c],y=grp2[rev],name="Revenue",marker_color=C1,opacity=0.8))
                        fig_m.add_trace(go.Bar(x=grp2[sec_c],y=grp2[ni],name="Net Income",marker_color=C4,opacity=0.8))
                        fig_m.update_layout(barmode="group",title="Revenue & Net Income by Sector")
                        st.plotly_chart(dark_fig(fig_m),use_container_width=True)
                    st.dataframe(safe_style(grp.reset_index().rename(columns={agg_col:"Total"})),use_container_width=True)


        with tabs[11]:  # Monte Carlo on user data
            sec("Monte Carlo \u2014 on your revenue data")
            if not rev or len(df)<3: ibox("Need Revenue + at least 3 periods.","info")
            else:
                y_rev=s(rev).values; rets_rev=np.diff(y_rev)/y_rev[:-1]
                mu_r=float(rets_rev.mean()); sig_r=float(rets_rev.std())
                mc_c1,mc_c2,mc_c3=st.columns(3)
                n_sim_r=mc_c1.select_slider("Simulations",[500,1000,2000,5000],1000)
                n_per_r=mc_c2.slider("Forward Periods",1,20,5)
                method_r=mc_c3.selectbox("Method",["GBM","Historical Bootstrap"])
                np.random.seed(0); paths_r=np.zeros((n_sim_r,n_per_r+1)); paths_r[:,0]=y_rev[-1]
                for t in range(1,n_per_r+1):
                    if method_r=="GBM":
                        z=np.random.standard_normal(n_sim_r)
                        paths_r[:,t]=paths_r[:,t-1]*np.exp((mu_r-0.5*sig_r**2)+sig_r*z)
                    else:
                        paths_r[:,t]=paths_r[:,t-1]*(1+np.random.choice(rets_rev,n_sim_r))
                fig_mc_r=go.Figure()
                for i in np.random.choice(n_sim_r,min(200,n_sim_r),replace=False):
                    fig_mc_r.add_trace(go.Scatter(x=list(range(n_per_r+1)),y=paths_r[i],mode="lines",
                                                   line=dict(color="rgba(59,130,246,0.03)",width=1),showlegend=False))
                for pv2,col,nm in [(np.percentile(paths_r[:,-1],95),C4,"P95"),
                                    (np.percentile(paths_r[:,-1],50),C7,"Median"),
                                    (np.percentile(paths_r[:,-1],5),C5,"P5")]:
                    idx_r=np.argmin(np.abs(paths_r[:,-1]-pv2))
                    fig_mc_r.add_trace(go.Scatter(x=list(range(n_per_r+1)),y=paths_r[idx_r],mode="lines",
                                                   line=dict(color=col,width=2.5),name=f"{nm}: {curr} {pv2:,.0f}"))
                fig_mc_r.update_layout(title="Revenue Monte Carlo",xaxis_title="Periods Ahead",yaxis_title=f"{curr}")
                st.plotly_chart(dark_fig(fig_mc_r,400),use_container_width=True)
                fv2=paths_r[:,-1]; f1,f2,f3,f4=st.columns(4)
                kpi(f1,"Median",f"{curr} {np.median(fv2):,.0f}","",True)
                kpi(f2,"Best (P95)",f"{curr} {np.percentile(fv2,95):,.0f}","",True)
                kpi(f3,"Worst (P5)",f"{curr} {np.percentile(fv2,5):,.0f}","",False)
                kpi(f4,"Prob > Current",f"{(fv2>y_rev[-1]).mean()*100:.1f}%","",True)

        with tabs[12]:  # Break-Even
            sec("Break-Even Analysis \u2014 from your data")
            if not rev: ibox("Map Revenue.","info")
            else:
                be1,be2=st.columns(2)
                fixed_cost=be1.number_input(f"Fixed Costs ({curr})",0,10**10,
                                             int(float(s(cost).iloc[-1])*0.4) if cost else 500000,50000)
                vc_pct=be2.slider("Variable Cost as % of Revenue",0,100,
                                   int((float(s(cost).iloc[-1])/float(s(rev).iloc[-1])*60)) if cost and rev else 40)
                rev_val=float(s(rev).iloc[-1]); vc_rate=vc_pct/100; cm=1-vc_rate
                if cm>0:
                    be_rev=fixed_cost/cm; rev_range=np.linspace(0,rev_val*2,200)
                    tc=fixed_cost+vc_rate*rev_range
                    fig_be=go.Figure()
                    fig_be.add_trace(go.Scatter(x=rev_range,y=rev_range,name="Revenue",line=dict(color=C4,width=2.5)))
                    fig_be.add_trace(go.Scatter(x=rev_range,y=tc,name="Total Cost",line=dict(color=C5,width=2.5)))
                    fig_be.add_vline(x=be_rev,line_dash="dash",line_color=C7,annotation_text=f"BEP: {curr} {be_rev:,.0f}")
                    fig_be.add_vline(x=rev_val,line_dash="dot",line_color=C1,annotation_text=f"Current: {curr} {rev_val:,.0f}")
                    fig_be.update_layout(title="Break-Even Chart")
                    st.plotly_chart(dark_fig(fig_be,360),use_container_width=True)
                    safety=(rev_val-be_rev)/rev_val*100 if rev_val>0 else 0
                    m1,m2,m3=st.columns(3)
                    kpi(m1,"Break-even Revenue",f"{curr} {be_rev:,.0f}","",True)
                    kpi(m2,"Contribution Margin",f"{cm*100:.1f}%","",cm>0.3)
                    kpi(m3,"Margin of Safety",f"{safety:.1f}%","",safety>15)
                else: ibox("Variable cost >= 100% \u2014 no break-even.","error")

        with tabs[13]:  # Valuation DCF
            sec("DCF Valuation \u2014 from your data")
            if not rev or len(df)<2: ibox("Need Revenue + at least 2 periods.","info")
            else:
                hist_cagr=((s(rev).iloc[-1]/s(rev).iloc[0])**(1/max(len(df)-1,1))-1)*100 if s(rev).iloc[0]!=0 else 5.0
                v1,v2,v3,v4=st.columns(4)
                fcf_margin=v1.slider("FCF Margin %",1,50,int(float(s(ni).iloc[-1])/float(s(rev).iloc[-1])*100) if ni and rev else 15)
                g1_v=v2.slider("Growth Yr 1-5 (%)",-10,50,max(0,int(hist_cagr)),1)
                g2_v=v3.slider("Terminal Growth (%)",0,10,3,1)
                wacc=v4.slider("WACC (%)",5,25,12,1)
                base_fcf=float(s(rev).iloc[-1])*(fcf_margin/100); cf_list=[]
                for yr in range(1,11):
                    g=g1_v/100 if yr<=5 else g2_v/100; cf_list.append(base_fcf*(1+g)**yr)
                pv_cfs=sum(cf/(1+wacc/100)**yr for yr,cf in enumerate(cf_list,1))
                tv=cf_list[-1]*(1+g2_v/100)/(wacc/100-g2_v/100) if wacc!=g2_v else 0
                pv_tv=tv/(1+wacc/100)**10; intrinsic=pv_cfs+pv_tv
                dv1,dv2,dv3,dv4=st.columns(4)
                kpi(dv1,"PV of FCFs",f"{curr} {pv_cfs:,.0f}","",True); kpi(dv2,"PV Terminal Val",f"{curr} {pv_tv:,.0f}","",True)
                kpi(dv3,"Intrinsic Value",f"{curr} {intrinsic:,.0f}","",True); kpi(dv4,"Base FCF",f"{curr} {base_fcf:,.0f}","",True)
                fig_dcf=go.Figure()
                fig_dcf.add_trace(go.Bar(x=[f"Yr{i}" for i in range(1,11)],y=cf_list,name="FCF",marker_color=C2,opacity=0.85))
                pvs_l=[cf/(1+wacc/100)**yr for yr,cf in enumerate(cf_list,1)]
                fig_dcf.add_trace(go.Scatter(x=[f"Yr{i}" for i in range(1,11)],y=pvs_l,name="PV of FCF",mode="lines+markers",line=dict(color=C4,width=2)))
                fig_dcf.update_layout(title="DCF Projected FCFs")
                st.plotly_chart(dark_fig(fig_dcf,320),use_container_width=True)
                ibox(f"Historical CAGR from your data: <strong>{hist_cagr:.1f}%</strong>. Not financial advice.","blue")

        with tabs[14]:  # Period Compare
            sec("Period-over-Period Comparison")
            compare_cols=[c for c in num_cols if df[c].notna().sum()>0]
            sel_cols=st.multiselect("Columns to compare",compare_cols,default=compare_cols[:min(4,len(compare_cols))])
            if sel_cols:
                cdf2=df[([dax] if dax else [])+sel_cols].copy()
                if dax: cdf2=cdf2.set_index(dax)
                pct_chg=cdf2.astype(float).pct_change()*100
                st.markdown("**Absolute Values**")
                abs_d=cdf2.astype(float).reset_index(); nd2=abs_d.select_dtypes(include="number").columns.tolist()
                st.dataframe(safe_style(abs_d).style.format({c:"{:,.2f}" for c in nd2}).background_gradient(cmap="Blues",subset=nd2),use_container_width=True)
                st.markdown("**Period-over-Period Change (%)**")
                pct_d=pct_chg.reset_index(); nd3=pct_d.select_dtypes(include="number").columns.tolist()
                st.dataframe(safe_style(pct_d).style.format({c:"{:+.2f}%" for c in nd3}).background_gradient(cmap="RdYlGn",subset=nd3),use_container_width=True)
                fig_cmp=go.Figure()
                for i,col in enumerate(sel_cols):
                    fig_cmp.add_trace(go.Bar(x=xax(),y=cdf2[col],name=col,marker_color=PIE_COLS[i],opacity=0.8))
                fig_cmp.update_layout(barmode="group",title="Multi-Period Comparison")
                st.plotly_chart(dark_fig(fig_cmp),use_container_width=True)

        with tabs[15]:  # Heatmap
            sec("Data Heatmap \u2014 from your data")
            if len(num_cols)<2: ibox("Need at least 2 numeric columns.","info")
            else:
                hm_cols=st.multiselect("Columns",num_cols,default=num_cols[:min(6,len(num_cols))])
                if hm_cols:
                    hm_data=df[hm_cols].astype(float)
                    hm_norm=hm_data.apply(lambda x:(x-x.min())/(x.max()-x.min()+1e-9)*100,axis=0)
                    fig_hm=px.imshow(hm_norm.T,labels=dict(x="Period",y="Metric",color="Normalised"),
                                      color_continuous_scale="RdYlGn",title="Normalised Heatmap (0-100)",aspect="auto")
                    fig_hm.update_xaxes(ticktext=xax(),tickvals=list(range(len(df))))
                    st.plotly_chart(dark_fig(fig_hm,max(300,len(hm_cols)*50)),use_container_width=True)
                    st.markdown("**Raw Values**")
                    disp=hm_data.reset_index(drop=True); nd4=disp.select_dtypes(include="number").columns.tolist()
                    st.dataframe(safe_style(disp).style.format({c:"{:,.2f}" for c in nd4}).background_gradient(cmap="RdYlGn",axis=0,subset=nd4),use_container_width=True)

        with tabs[16]:  # FX Impact
            sec("FX Currency Impact \u2014 enter your rate")
            if not rev: ibox("Map Revenue.","info")
            else:
                fx1,fx2,fx3=st.columns(3)
                from_c=fx1.selectbox("Source Currency",["USD","INR","EUR","GBP","JPY","AUD","CAD","SGD"],index=0)
                to_c=fx2.selectbox("Target Currency",["INR","USD","EUR","GBP","JPY","AUD","CAD","SGD"],index=0)
                rate=fx3.number_input(f"Rate ({from_c} to {to_c})",min_value=0.0001,value=83.5,step=0.01,format="%.4f")
                ibox(f"Using rate 1 {from_c} = {rate} {to_c}","blue")
                rev_s=s(rev); rev_c=rev_s*rate
                fig_fx=go.Figure()
                fig_fx.add_trace(go.Bar(x=xax(),y=rev_s,name=f"Original ({from_c})",marker_color=C2,opacity=0.8))
                fig_fx.add_trace(go.Bar(x=xax(),y=rev_c,name=f"Converted ({to_c})",marker_color=C4,opacity=0.8))
                fig_fx.update_layout(barmode="group",title=f"Revenue: {from_c} vs {to_c}")
                st.plotly_chart(dark_fig(fig_fx),use_container_width=True)
                rate_range=np.linspace(rate*0.7,rate*1.3,60)
                fig_sens=go.Figure(go.Scatter(x=rate_range,y=[float(rev_s.iloc[-1])*r for r in rate_range],
                                               mode="lines",line=dict(color=C1,width=2),fill="tozeroy",fillcolor="rgba(34,211,238,0.06)"))
                fig_sens.add_vline(x=rate,line_dash="dash",line_color=C7,annotation_text="Your rate")
                fig_sens.update_layout(title="FX Sensitivity",xaxis_title=f"{from_c}/{to_c} Rate",yaxis_title=f"{to_c}")
                st.plotly_chart(dark_fig(fig_sens,280),use_container_width=True)
                fx_t=pd.DataFrame({"Period":xax(),f"Original ({from_c})":rev_s.values,f"Converted ({to_c})":rev_c.values}).round(2)
                st.dataframe(safe_style(fx_t),use_container_width=True)

        with tabs[17]:  # Tax
            sec("Tax Computation \u2014 from your data")
            if not ni: ibox("Map Net Income.","info")
            else:
                tr=st.selectbox("Tax Regime",["India Corporate (25.17%)","India Startup (22%)","US Federal (21%)","UK Corporation (25%)","Custom"])
                tax_rate={"India Corporate (25.17%)":25.17,"India Startup (22%)":22.0,"US Federal (21%)":21.0,"UK Corporation (25%)":25.0}.get(tr,None)
                if tax_rate is None: tax_rate=st.slider("Custom Tax Rate %",0.0,60.0,25.0,0.5)
                ni_s=s(ni); tax_s=ni_s.clip(lower=0)*tax_rate/100; pat_s=ni_s-tax_s
                fig_tax=go.Figure()
                fig_tax.add_trace(go.Bar(x=xax(),y=ni_s,name="Pre-Tax",marker_color=C2,opacity=0.85))
                fig_tax.add_trace(go.Bar(x=xax(),y=tax_s,name="Tax",marker_color=C5,opacity=0.8))
                fig_tax.add_trace(go.Scatter(x=xax(),y=pat_s,name="PAT",mode="lines+markers",line=dict(color=C4,width=2.5)))
                fig_tax.update_layout(barmode="group",title="Income, Tax, and PAT")
                st.plotly_chart(dark_fig(fig_tax),use_container_width=True)
                t1,t2,t3,t4=st.columns(4)
                kpi(t1,"Pre-Tax Total",f"{curr} {ni_s.sum():,.0f}","",True); kpi(t2,"Total Tax",f"{curr} {tax_s.sum():,.0f}","",False)
                kpi(t3,"Total PAT",f"{curr} {pat_s.sum():,.0f}","",True); kpi(t4,"Rate",f"{tax_rate:.1f}%","",True)
                tax_t=pd.DataFrame({"Period":xax(),"Pre-Tax":ni_s.values,"Tax":tax_s.values,"PAT":pat_s.values}).round(2)
                st.dataframe(safe_style(tax_t).style.format({"Pre-Tax":"{:,.2f}","Tax":"{:,.2f}","PAT":"{:,.2f}"}),use_container_width=True)

        with tabs[18]:  # Peer Bench
            sec("Peer Benchmarking \u2014 manual input only (your data)")
            ibox("Enter peer financials from your own research. No external fetches.","blue")
            n_peers=st.number_input("Number of peers (including your company)",2,8,3)
            peer_rows=[]
            for i in range(int(n_peers)):
                st.markdown(f"**{'Your Company' if i==0 else f'Peer {i}'}**")
                pc=st.columns(4)
                name=pc[0].text_input("Company",value="Your Company" if i==0 else f"Peer {i}",key=f"pn_{i}")
                rev_p=pc[1].number_input(f"Revenue ({curr})",value=0.0,key=f"pr_{i}")
                ni_p=pc[2].number_input(f"Net Income ({curr})",value=0.0,key=f"pni_{i}")
                gm_p=pc[3].number_input("Gross Margin %",value=0.0,key=f"pgm_{i}")
                pc2=st.columns(4)
                nm_p=pc2[0].number_input("Net Margin %",value=0.0,key=f"pnm_{i}")
                roe_p=pc2[1].number_input("ROE %",value=0.0,key=f"proe_{i}")
                de_p=pc2[2].number_input("Debt/Equity",value=0.0,key=f"pde_{i}")
                cr_p=pc2[3].number_input("Current Ratio",value=0.0,key=f"pcr_{i}")
                peer_rows.append({"Company":name,"Revenue":rev_p,"Net Income":ni_p,
                                   "Gross Margin %":gm_p,"Net Margin %":nm_p,"ROE %":roe_p,"D/E":de_p,"Curr Ratio":cr_p})
                st.divider()
            if st.button("Generate Peer Analysis"):
                pdf=pd.DataFrame(peer_rows); npc=[c for c in pdf.columns if c!="Company"]
                st.dataframe(safe_style(pdf).style.format({c:"{:,.2f}" for c in npc}).background_gradient(cmap="RdYlGn",axis=0,subset=npc),use_container_width=True)
                for metric in ["Gross Margin %","Net Margin %","ROE %","Curr Ratio"]:
                    vals=pdf[metric].tolist(); lbls=pdf["Company"].tolist()
                    if any(v!=0 for v in vals):
                        colors=[C4 if l==peer_rows[0]["Company"] else PIE_COLS[i+1] for i,l in enumerate(lbls)]
                        fig_pr=go.Figure(go.Bar(x=lbls,y=vals,marker_color=colors,opacity=0.85,
                                                 text=[f"{v:.1f}" for v in vals],textposition="outside"))
                        fig_pr.update_layout(title=f"Peer: {metric}",yaxis_title=metric)
                        st.plotly_chart(dark_fig(fig_pr,280),use_container_width=True)
                avail=[m for m in ["Gross Margin %","Net Margin %","ROE %"] if pdf[m].max()>0]
                if len(avail)>=3:
                    fig_rad=go.Figure()
                    for _,row in pdf.iterrows():
                        vals_r=[float(row[m]) for m in avail]+[float(row[avail[0]])]
                        fig_rad.add_trace(go.Scatterpolar(r=vals_r,theta=avail+[avail[0]],fill="toself",name=row["Company"]))
                    fig_rad.update_layout(polar=dict(bgcolor=CARD_BG,radialaxis=dict(visible=True,gridcolor=BORDER,color=FONT),
                                                      angularaxis=dict(color=FONT)),paper_bgcolor=BG,font=dict(color=FONT),height=380,
                                           legend=dict(bgcolor=CARD_BG,bordercolor=BORDER,borderwidth=1))
                    st.plotly_chart(fig_rad,use_container_width=True)

        with tabs[19]:  # Insights
            sec("Financial Health Insights \u2014 from your data only")
            insights=[]; _sc=[0,0]
            def check(condition,pos_msg,neg_msg,weight=1):
                _sc[1]+=weight
                if condition: _sc[0]+=weight; insights.append(("ok",pos_msg,"good"))
                else: insights.append(("warn",neg_msg,"warn"))
            if rev and len(df)>1:
                rs2=s(rev)
                cagr_i=((rs2.iloc[-1]/rs2.iloc[0])**(1/max(len(rs2)-1,1))-1)*100 if rs2.iloc[0]!=0 else 0
                check(cagr_i>5,f"Revenue CAGR {cagr_i:.1f}% - healthy",f"Revenue CAGR {cagr_i:.1f}% - below 5%")
                check(rs2.is_monotonic_increasing,"Revenue consistently growing","Revenue shows inconsistency or decline",2)
                yoy_l=((rs2.iloc[-1]-rs2.iloc[-2])/abs(rs2.iloc[-2]))*100 if len(rs2)>1 and rs2.iloc[-2]!=0 else 0
                check(yoy_l>0,f"Latest growth: +{yoy_l:.1f}%",f"Latest growth: {yoy_l:.1f}%")
            if rev and cost:
                gm_i=(float(s(rev).iloc[-1])-float(s(cost).iloc[-1]))/float(s(rev).iloc[-1])*100 if float(s(rev).iloc[-1])!=0 else 0
                check(gm_i>40,f"Gross margin {gm_i:.1f}% - above 40%",f"Gross margin {gm_i:.1f}% - below 40%",2)
            if ni and rev:
                nm_i=float(s(ni).iloc[-1])/float(s(rev).iloc[-1])*100 if float(s(rev).iloc[-1])!=0 else 0
                check(nm_i>10,f"Net margin {nm_i:.1f}% - healthy",f"Net margin {nm_i:.1f}% - below 10%")
                check(float(s(ni).iloc[-1])>0,"Company is profitable","Company is at a loss",2)
            if ast and lib:
                solv=float(s(ast).iloc[-1])/float(s(lib).iloc[-1]) if float(s(lib).iloc[-1])!=0 else 999
                check(solv>1.5,f"Solvency {solv:.2f}x - strong",f"Solvency {solv:.2f}x - concern")
            if ca and cl:
                cr_i=float(s(ca).iloc[-1])/float(s(cl).iloc[-1]) if float(s(cl).iloc[-1])!=0 else 999
                check(1.5<=cr_i<=3,f"Current ratio {cr_i:.2f}x - healthy",f"Current ratio {cr_i:.2f}x - outside range")
            score,max_score=_sc[0],_sc[1]
            if not insights: ibox("Map more columns for insights.","info")
            else:
                hp=score/max_score*100 if max_score>0 else 0
                hl="Excellent" if hp>=80 else "Good" if hp>=60 else "Fair" if hp>=40 else "Needs Attention"
                ibox(f"<strong>Health Score: {score}/{max_score} ({hp:.0f}%) - {hl}</strong>",
                     "good" if hp>=70 else "warn" if hp>=40 else "error")
                fig_g=go.Figure(go.Indicator(
                    mode="gauge+number",value=hp,number=dict(suffix="%",font=dict(color=WHITE)),
                    gauge=dict(axis=dict(range=[0,100],tickfont=dict(color=FONT)),
                               bar=dict(color=C4 if hp>=70 else C6 if hp>=40 else C5),
                               bgcolor=CARD_BG,bordercolor=BORDER,
                               steps=[dict(range=[0,40],color="#2b0d0d"),dict(range=[40,70],color="#2b1e0d"),dict(range=[70,100],color="#0d2b1d")])))
                fig_g.update_layout(paper_bgcolor=BG,font=dict(color=FONT),height=260,margin=dict(l=20,r=20,t=60,b=20))
                st.plotly_chart(fig_g,use_container_width=True)
                for icon,msg,kind in insights: ibox(f"{icon} {msg}",kind)
                ibox("Insights are rule-based from your uploaded data only. Consult a financial advisor for decisions.","blue")

# \u2500\u2500 GOAL & WHAT-IF PLANNER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
elif page == "\U0001f3af Goal & What-If Planner":
    st.markdown('<div class="dash-header"><h1>Goal & What-If Planner</h1></div>', unsafe_allow_html=True)
    gpt1,gpt2,gpt3=st.tabs(["Goal Planner","What-If Simulator","SIP Calculator"])
    with gpt1:
        sec("Financial Goal Planner")
        g1,g2=st.columns(2)
        with g1:
            goal_name=st.text_input("Goal","Retirement"); goal_amount=st.number_input(f"Target ({curr})",100000,10**9,10000000,500000)
            goal_years=st.slider("Years",1,40,20); current_save=st.number_input(f"Current Savings ({curr})",0,10**8,500000,50000)
            monthly_inv=st.number_input(f"Monthly SIP ({curr})",0,500000,15000,1000)
            exp_return=st.slider("Expected Return %",1.0,25.0,10.0,0.5); inflation=st.slider("Inflation %",0.0,15.0,6.0,0.5)
        with g2:
            fv_s=current_save*(1+exp_return/100)**goal_years
            fv_sip=monthly_inv*12*(((1+exp_return/100)**goal_years-1)/(exp_return/100)) if exp_return!=0 else monthly_inv*12*goal_years
            tfv=fv_s+fv_sip; gr2=goal_amount*(1+inflation/100)**goal_years; sf2=max(0,gr2-tfv)
            for l,v,p in [("Inflation-Adj Target",f"{curr} {gr2:,.0f}",True),
                           ("Projected Corpus",f"{curr} {tfv:,.0f}",tfv>=gr2),
                           ("Shortfall",f"{curr} {sf2:,.0f}",sf2==0)]:
                kpi(g2,l,v,"",p)
        yr_a3=list(range(goal_years+1)); cr_a3=[]; tg_a3=[]; vv3=float(current_save)
        for yr in yr_a3: cr_a3.append(vv3); tg_a3.append(goal_amount*(1+inflation/100)**yr); vv3=vv3*(1+exp_return/100)+monthly_inv*12
        fig_gl3=go.Figure()
        fig_gl3.add_trace(go.Scatter(x=yr_a3,y=cr_a3,name="Projected",fill="tozeroy",fillcolor="rgba(59,130,246,0.10)",line=dict(color=C2,width=2.5)))
        fig_gl3.add_trace(go.Scatter(x=yr_a3,y=tg_a3,name="Target",line=dict(color=C5,width=2,dash="dash")))
        fig_gl3.update_layout(title=f"Goal: {goal_name}",xaxis_title="Years",yaxis_title=f"{curr}")
        st.plotly_chart(dark_fig(fig_gl3),use_container_width=True)
        if sf2>0: ibox(f"Need extra SIP of {curr} {sf2/(((1+exp_return/100)**goal_years-1)/(exp_return/100)*12) if exp_return!=0 else sf2/(goal_years*12):,.0f}/month","warn")
        else: ibox(f"On track for {goal_name}!","good")
    with gpt2:
        sec("If I Had Invested\u2026 Simulator (live stock data)")
        sym_wi=st.selectbox("Stock",tickers if tickers else ["AAPL"])
        h_wi,_=fetch_ticker(sym_wi,"5y")
        if not h_wi.empty:
            wi_inv=st.number_input(f"Investment ({curr})",1000,10**8,100000,5000)
            invest_date=st.date_input("Investment Date",h_wi.index[0].date())
            h_wi_f=h_wi[h_wi.index.date>=invest_date]
            if not h_wi_f.empty:
                bp=float(h_wi_f["Close"].iloc[0]); cp=float(h_wi_f["Close"].iloc[-1])
                cv=wi_inv/bp*cp; profit=cv-wi_inv; rp=(cv/wi_inv-1)*100
                cagr_wi=((cv/wi_inv)**(365/max((h_wi_f.index[-1]-h_wi_f.index[0]).days,1))-1)*100
                w1,w2,w3,w4=st.columns(4)
                kpi(w1,"Invested",f"{curr} {wi_inv:,.0f}","",True); kpi(w2,"Value",f"{curr} {cv:,.2f}","",True)
                kpi(w3,"P&L",f"{curr} {profit:+,.2f}",f"{rp:+.2f}%",profit>=0); kpi(w4,"CAGR",f"{cagr_wi:.2f}%","",cagr_wi>=0)
                growth=(h_wi_f["Close"].astype(float)/bp)*wi_inv
                fig_wi=go.Figure(go.Scatter(x=h_wi_f.index.astype(str).str[:10],y=growth,fill="tozeroy",fillcolor="rgba(59,130,246,0.10)",line=dict(color=C2,width=2.5)))
                fig_wi.add_hline(y=wi_inv,line_dash="dash",line_color=FONT,annotation_text=f"Invested: {curr} {wi_inv:,.0f}")
                fig_wi.update_layout(title=f"What If You Invested {curr} {wi_inv:,.0f} in {sym_wi}",xaxis_title="Date",yaxis_title=f"{curr}")
                st.plotly_chart(dark_fig(fig_wi,360),use_container_width=True)
    with gpt3:
        sec("SIP Calculator")
        sp1,sp2=st.columns(2)
        sip_amt=sp1.number_input(f"Monthly SIP ({curr})",500,1000000,10000,500); sip_yrs=sp2.slider("Duration (years)",1,40,20)
        sip_ret=sp1.slider("Expected Return %",1.0,30.0,12.0,0.5); sip_inf=sp2.slider("Inflation %",0.0,10.0,6.0,0.5)
        ti=sip_amt*12*sip_yrs; fv_c=sip_amt*12*(((1+sip_ret/100)**sip_yrs-1)/(sip_ret/100)) if sip_ret!=0 else ti
        wc=fv_c-ti; rv=fv_c/(1+sip_inf/100)**sip_yrs
        sc4,sc5,sc6,sc7=st.columns(4)
        kpi(sc4,"Total Invested",f"{curr} {ti:,.0f}","",True); kpi(sc5,"Future Value",f"{curr} {fv_c:,.0f}","",True)
        kpi(sc6,"Wealth Created",f"{curr} {wc:,.0f}","",True); kpi(sc7,"Real Value",f"{curr} {rv:,.0f}","inflation adj.",True)
        yr_a4=[]; iv4=[]; vl4=[]
        for yr in range(1,sip_yrs+1):
            yr_a4.append(yr); iv4.append(sip_amt*12*yr)
            vl4.append(sip_amt*12*(((1+sip_ret/100)**yr-1)/(sip_ret/100)) if sip_ret!=0 else sip_amt*12*yr)
        fig_sip=go.Figure()
        fig_sip.add_trace(go.Bar(x=yr_a4,y=vl4,name="Future Value",marker_color=C2,opacity=0.85))
        fig_sip.add_trace(go.Bar(x=yr_a4,y=iv4,name="Invested",marker_color=C4,opacity=0.6))
        fig_sip.update_layout(barmode="group",title="SIP Growth Year-by-Year",xaxis_title="Year")
        st.plotly_chart(dark_fig(fig_sip,340),use_container_width=True)

# \u2500\u2500 FOOTER \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.markdown(f"""
<div style="text-align:center;padding:24px 0 10px;color:#1e3a5f;font-size:0.7rem;
     font-family:'IBM Plex Mono',monospace;border-top:1px solid #1e2d45;margin-top:32px;">
  FinIQ Pro \u2014 Advanced Financial Intelligence &nbsp;|&nbsp;
  Live data: yfinance + FRED public API + Yahoo RSS &nbsp;|&nbsp;
  Custom upload: 100% your data only &nbsp;|&nbsp;
  @{st.session_state.username} ({st.session_state.role}) &nbsp;|&nbsp; Not financial advice
</div>
""", unsafe_allow_html=True)