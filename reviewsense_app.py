# ============================================================
#  ReviewSense — CLEAN IMPERIAL (TRUE ONYX)
# ============================================================

import time
import re
import string
import os
from collections import Counter

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
import numpy as np

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReviewSense Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (Imperial Gold & Onyx Theme) ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp .page-title, .stApp .page-sub {
    font-family: 'Outfit', sans-serif !important;
}
html, body { color: #fdfcf0; }

/* 🏆 Onyx & Gold Theme */
.stApp {
    background: radial-gradient(circle at 10% 10%, #1a1a1a, #0a0a0a, #000000);
    background-attachment: fixed;
}

.page-title {
    font-size: 3.5rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #d4af37, #f1c40f, #ffea00);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.page-sub { text-align: center; color: #a1a1aa; font-size: 1.1rem; margin-bottom: 30px; }

/* Luxury Cards (Kept for Dashboard) */
.func-card {
    background: rgba(30, 30, 30, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(212, 175, 55, 0.2);
    border-radius: 20px; padding: 25px; text-align: center; height: 100%;
}
.func-title { color: #d4af37; font-weight: 700; margin-bottom: 10px; }
.func-desc { color: #a1a1aa; font-size: 0.9rem; }

.section-head { color: #fff; text-align: center; font-size: 1.8rem; font-weight: 700; margin: 40px 0 20px 0; }

.ms-row { display: flex; align-items: center; padding: 12px; margin-bottom: 8px; border-radius: 10px; background: rgba(20, 20, 20, 0.6); }
.active { border: 1px solid #f1c40f; box-shadow: 0 0 15px rgba(212, 175, 55, 0.2); }
.done { color: #10b981; }

.stMetric { background: rgba(30, 30, 30, 0.8); border-radius: 10px; padding: 10px; border: 1px solid rgba(212, 175, 55, 0.1); }
[data-testid="stMetricValue"] { color: #f1c40f !important; font-weight: 700 !important; }

.realtime-result { 
    background: rgba(212, 175, 55, 0.05); border: 1px solid rgba(212, 175, 55, 0.3); 
    border-radius: 12px; padding: 20px; color: #fdfcf0;
}

/* Primary Gold Button */
.stButton>button {
    background: linear-gradient(135deg, #d4af37, #f1c40f) !important;
    color: #000 !important; border: none !important;
    border-radius: 10px !important; text-transform: uppercase; font-weight: 700 !important;
    letter-spacing: 1px; transition: all 0.3s !important; height: 45px;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(212, 175, 55, 0.4); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  ANALYSIS LOGIC
# ══════════════════════════════════════════════════════════════════════
def clean_text(t):
    t = str(t).lower(); t = re.sub(r"http\S+|\d+", "", t); t = t.translate(str.maketrans("", "", string.punctuation))
    return " ".join([w for w in t.split() if w not in {"is","the","and","to","a","an","of","in","for","on","with"}])

def get_sentiment_hybrid(t):
    ts = str(t).lower(); pol = 0
    try: pol = TextBlob(ts).sentiment.polarity
    except: pol = 0
    pos_k = ["good","great","amazing","excellent","happy","love","best","nice","fantastic","satisfied"]
    neg_k = ["bad","terrible","worst","awful","sad","poor","late","broken","useless","hate"]
    if pol == 0:
        lp = sum(1 for w in pos_k if w in ts); ln = sum(1 for w in neg_k if w in ts)
        if lp > ln: pol = 0.5
        elif ln > lp: pol = -0.5
    if pol > 0.05: return "positive", round(pol, 4)
    elif pol < -0.05: return "negative", round(pol, 4)
    return "neutral", 0.0

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "username" not in st.session_state: st.session_state["username"] = ""
if "page" not in st.session_state: st.session_state["page"] = "landing"
if "uploaded_file" not in st.session_state: st.session_state["uploaded_file"] = None
if "analysis_done" not in st.session_state: st.session_state["analysis_done"] = False

# ══════════════════════════════════════════════════════════════════════
#  COMPONENTS
# ══════════════════════════════════════════════════════════════════════
def topbar():
    c1, c2, c3 = st.columns([5, 1, 4])
    with c1: st.markdown("<h2 style='margin:0; color:#d4af37; font-weight:700;'>🔍 ReviewSense</h2>", unsafe_allow_html=True)
    with c3:
        s1, s2 = st.columns([1.2, 1])
        with s1: st.markdown(f"<div style='background:rgba(212,175,55,0.08); border:1px solid rgba(212,175,55,0.2); border-radius:10px; height:45px; display:flex; align-items:center; justify-content:center; color:#f1c40f; font-weight:600;'>👤 {st.session_state.get('username','user')}</div>", unsafe_allow_html=True)
        with s2:
            if st.button("Logout", key="btn_logout", use_container_width=True):
                st.session_state["logged_in"] = False; st.session_state["page"] = "landing"; st.rerun()

def sidebar_nav():
    if st.sidebar.button("← Back to Upload", key="side_back", use_container_width=True):
        st.session_state["page"] = "upload"; st.rerun()
    st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════
def page_landing():
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        # 🚀 No Card Container
        st.markdown('<div class="page-title">Unlock Hidden Insights <br> In Every Review</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Imperial Intelligence refined for feedback dominance.</div>', unsafe_allow_html=True)
        _, btn_c, _ = st.columns([1, 1.2, 1])
        with btn_c:
            if st.button("🚀  Start Sensing", use_container_width=True, type="primary"):
                st.session_state["page"] = "login"; st.rerun()

def page_login():
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])
    with col:
        # 🛡️ No Card Container
        st.markdown('<h2 style="text-align:center; color:#d4af37; margin-top:0;">Royal Access</h2>', unsafe_allow_html=True)
        with st.form("login"):
            u = st.text_input("Username", placeholder="user")
            p = st.text_input("Password", type="password", placeholder="user123")
            if st.form_submit_button("🔐  Enter Workspace", use_container_width=True):
                if u.lower()=="user" and p=="user123":
                    st.session_state["logged_in"]=True; st.session_state["username"]="user"; st.session_state["page"]="upload"; st.rerun()
                else: st.error("Verification failed.")
        if st.button("← Back"): st.session_state["page"]="landing"; st.rerun()

def page_upload():
    topbar()
    st.markdown('<div class="page-title" style="margin-top:20px; font-size:2.8rem">Why ReviewSense Intelligence?</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Discover the deep signals hidden in customer feedback.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Intelligence Components</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("""<div class="func-card"><div class="func-icon">🧹</div><div class="func-title">Purity Engine</div><div class="func-desc">Automated noise removal for high-performance analysis.</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown("""<div class="func-card"><div class="func-icon">🧠</div><div class="func-title">Sentiment Radar</div><div class="func-desc">Dual-layer polarity detection identifying the emotional tone.</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown("""<div class="func-card"><div class="func-icon">🔑</div><div class="func-title">Theme Extraction</div><div class="func-desc">Converting keywords into strategic roadmaps for growth.</div></div>""", unsafe_allow_html=True)
    
    st.markdown('<div class="section-head">⚡ Realtime Intelligence Sandbox</div>', unsafe_allow_html=True)
    _, sbm, _ = st.columns([1, 2, 1])
    with sbm:
        t_input = st.text_area("Test your feedback here...", placeholder="e.g., Amazing product!", height=80, label_visibility="collapsed")
        if st.button("🧠  Test AI Scan", use_container_width=True):
            if t_input.strip() != "":
                c_txt = clean_text(t_input)
                s_label, s_score = get_sentiment_hybrid(t_input)
                w_counts = Counter(c_txt.split())
                st.markdown(f"""
                    <div class="realtime-result">
                        <p><strong>Refined:</strong> <span style="color:#d4af37;">"{c_txt}"</span></p>
                        <div style="display:flex; justify-content:space-between; text-align:center;">
                            <div><p style="font-size:0.75rem; color:#a1a1aa; margin:0;">TONE</p><h4 style="color:#2ecc71; margin:0;">{s_label.upper()}</h4></div>
                            <div><p style="font-size:0.75rem; color:#a1a1aa; margin:0;">SCORE</p><h4 style="color:#f1c40f; margin:0;">{s_score}</h4></div>
                            <div><p style="font-size:0.75rem; color:#a1a1aa; margin:0;">THEMES</p><h4 style="color:#d4af37; margin:0;">{len(w_counts)}</h4></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-head">Ready for Insights? Upload Feedback.</div>', unsafe_allow_html=True)
    _, m, _ = st.columns([1, 1.8, 1])
    with m:
        up = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"], label_visibility="collapsed")
        if up:
            st.success(f"✅ **{up.name}** is mapped and ready.")
            st.session_state["uploaded_file"]=up
            if st.button("🚀  Run Imperial Pipeline", use_container_width=True, type="primary"):
                st.session_state["analysis_done"]=False; st.session_state["page"]="processing"; st.rerun()
        elif st.button("▶  Run With Demo Knowledge", use_container_width=True):
            st.session_state["uploaded_file"]=None; st.session_state["analysis_done"]=False; st.session_state["page"]="processing"; st.rerun()

def page_processing():
    topbar(); sidebar_nav()
    st.markdown('<div class="page-title" style="font-size:2.5rem">Intelligence Pipeline</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.8, 1])
    with mid:
        ms_p = st.empty(); prog_p = st.empty()
    def update_ui(states, pct, label):
        icons = ["🧹","🧠","🔑","📊"]; titles = ["Detecting","Scanning","Mapping","Visualizing"]
        h = "".join([f"<div class='ms-row {s}'><div class='ms-icon'>{icons[i]}</div><div class='ms-label'><strong>{titles[i]}</strong></div></div>" for i,s in enumerate(states)])
        ms_p.markdown(h, unsafe_allow_html=True)
        prog_p.markdown(f"<div style='background:rgba(255,255,255,0.05); border-radius:10px; height:6px; width:100%; margin:20px 0;'><div style='background:#d4af37; width:{pct}%; height:100%; border-radius:10px;'></div></div><div style='text-align:center; color:#a1a1aa;'>{label}</div>", unsafe_allow_html=True)

    if not st.session_state.get("analysis_done", False):
        update_ui(["active","wait","wait","wait"], 10, "Detecting sensor...")
        up = st.session_state.get("uploaded_file")
        df = (pd.read_excel(up) if up and up.name.endswith(('xlsx','xls')) else pd.read_csv(up)) if up else pd.read_csv("Milestone2_Sentiment_Results.csv")
        f_col = None
        txt_k = ["feedback","review","comment","message","text","opinion"]
        for c in df.columns:
            if c.lower() in txt_k: f_col = c; break
        if not f_col:
            obj_cols = df.select_dtypes(include=['object']).columns
            if len(obj_cols) > 0: f_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
        if not f_col: f_col = df.columns[0]
        update_ui(["active","wait","wait","wait"], 40, f"Processing '{f_col}'...")
        df["clean_feedback"] = df[f_col].astype(str).apply(clean_text)
        update_ui(["done","active","wait","wait"], 65, "Determining Polarity...")
        df[["sentiment","confidence_score"]] = df[f_col].astype(str).apply(lambda x: pd.Series(get_sentiment_hybrid(x)))
        update_ui(["done","done","active","wait"], 85, "Mapping Insights...")
        kdf = pd.DataFrame(Counter(" ".join(df["clean_feedback"]).split()).most_common(50), columns=["keyword","frequency"])
        st.session_state["_df"] = df; st.session_state["_kdf"] = kdf; st.session_state["analysis_done"] = True
    
    update_ui(["done","done","done","done"], 100, "Intelligence Ready!")
    _, c_btn, _ = st.columns([1, 1.2, 1])
    with c_btn:
        if st.button("📊  View Dashboard", use_container_width=True, type="primary"):
            st.session_state["page"] = "dashboard"; st.rerun()

def page_dashboard():
    df = st.session_state.get("_df"); kdf = st.session_state.get("_kdf")
    if df is None: st.error("No data."); st.stop()
    if "date" not in df.columns: df["date"] = pd.Timestamp.now()
    else: df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp.now())
    if "product" not in df.columns: df["product"] = "General"

    sidebar_nav()
    st.sidebar.header("🔍 Global Range")
    s_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
    s_sel = st.sidebar.multiselect("Sentiment", options=list(s_map.values()), default=list(s_map.values()))
    s_keys = [k for k, v in s_map.items() if v in s_sel]
    p_sel = st.sidebar.multiselect("Products", options=sorted(df["product"].unique()), default=sorted(df["product"].unique()))
    c_s1, c_s2 = st.sidebar.columns(2)
    start_d = c_s1.date_input("Start", value=df["date"].min().date())
    end_d = c_s2.date_input("End", value=df["date"].max().date())
    
    f_df = df[(df["sentiment"].isin(s_keys)) & (df["product"].isin(p_sel)) & (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
    topbar()
    st.markdown('<h3 style="text-align:center; color:#f1c40f; margin-top:0;">📊 Imperial Intelligence Dashboard</h3>', unsafe_allow_html=True)
    if f_df.empty: st.warning("⚠️ No matching data."); st.stop()

    m1, m2, m3, m4 = st.columns(4)
    total = len(f_df); m1.metric("Market Volume", total)
    for col, s, color in zip([m2, m3, m4], ["positive", "negative", "neutral"], ["#d4af37", "#a1a1aa", "#ffffff"]):
        cnt = len(f_df[f_df["sentiment"] == s]); pct = (cnt/total*100) if total>0 else 0
        col.metric(s_map[s], f"{pct:.1f}%", delta=f"{cnt} items")

    st.markdown("---")
    fig1, ax1 = plt.subplots(figsize=(10, 4.2))
    plt.style.use('dark_background'); fig1.patch.set_facecolor('#000000'); ax1.set_facecolor('#000000')
    counts = f_df["sentiment"].value_counts()
    ax1.bar([s_map[s] for s in ["positive","negative","neutral"]], [counts.get(s, 0) for s in ["positive","negative","neutral"]], color=["#d4af37", "#a1a1aa", "#444444"])
    st.pyplot(fig1)

    st.markdown("### 📱 Competitive Intelligence")
    cp1, cp2 = st.columns([1, 1]); 
    with cp1:
        ps = f_df.groupby("product")["sentiment"].value_counts().unstack(fill_value=0)
        for s in ["positive", "negative", "neutral"]:
            if s not in ps.columns: ps[s] = 0
        ps["Total"] = ps.sum(axis=1); ps["Positive %"] = (ps["positive"] / ps["Total"] * 100).round(1)
        st.dataframe(ps.sort_values("Positive %", ascending=False).rename(columns=s_map), use_container_width=True)
    with cp2:
        fig_hm, ax_hm = plt.subplots(); sns.heatmap(ps[["positive","negative","neutral"]], annot=True, fmt="d", cmap="YlOrBr", ax=ax_hm); st.pyplot(fig_hm)

    st.markdown("### 📈 Performance Benchmarks")
    t1, t2 = st.columns(2)
    with t1:
        if len(f_df["date"].unique()) >= 1:
            f_df["month"] = f_df["date"].dt.to_period("M").astype(str)
            trend = f_df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
            for s in ["positive", "negative", "neutral"]:
                if s not in trend.columns: trend[s] = 0
            fig_t, ax_t = plt.subplots(); fig_t.patch.set_facecolor('#000000'); ax_t.set_facecolor('#000000')
            cols_t = {"positive": "#d4af37", "negative": "#ffffff", "neutral": "#444444"}
            for sk in ["positive", "negative", "neutral"]: ax_t.plot(trend.index.astype(str), trend[sk], marker='o', label=s_map[sk], color=cols_t[sk], linewidth=2.5)
            ax_t.legend(); st.pyplot(fig_t)
        else: st.info("Historical data required.")
    with t2:
        fig_r, ax_r = plt.subplots(); ax_r.set_facecolor('#000000'); ax_r.hist(f_df["confidence_score"], bins=20, color="#d4af37", edgecolor="#000000"); st.pyplot(fig_r)

    st.markdown("### 🔑 Semantic Themes")
    k1, k2 = st.columns(2)
    with k1: fig_kw, ax_kw = plt.subplots(); sns.barplot(data=kdf.head(10), x="frequency", y="keyword", palette="YlOrBr", ax=ax_kw); st.pyplot(fig_kw)
    with k2:
        wc = WordCloud(width=600, height=400, background_color="#000000", colormap="YlOrBr").generate_from_frequencies(dict(zip(kdf["keyword"], kdf["frequency"])))
        fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off"); st.pyplot(fig_wc)

    with st.expander("💼 Advanced Data Export"):
        st.dataframe(f_df.head(20), use_container_width=True)
        st.download_button("📩 Download Results (CSV)", f_df.to_csv(index=False).encode("utf-8"), "ReviewSense_Final_Analysis.csv", use_container_width=True)

# ── Router ──────────────────────────────────────────────────────────
p = st.session_state["page"]; 
if p == "landing": page_landing()
elif p == "login": page_login()
elif p == "upload": page_upload()
elif p == "processing": page_processing()
elif p == "dashboard": page_dashboard()
