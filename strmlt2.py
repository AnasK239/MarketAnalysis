import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upwork Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0a0e1a 0%, #0f1422 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #131a2e 0%, #1a2342 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 40px rgba(59, 130, 246, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 800;
        background: linear-gradient(90deg, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 12px 0 6px 0;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #3b82f6, #8b5cf6, transparent) 1;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }
    .section-sub {
        font-size: 14px;
        color: #64748b;
        margin-top: 0px;
        margin-bottom: 20px;
        font-weight: 400;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 40%, #1e1b4b 70%, #0f1117 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 48px rgba(59, 130, 246, 0.1);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #f1f5f9;
        font-size: 36px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero .accent {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #94a3b8;
        font-size: 16px;
        margin: 10px 0 0 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
        line-height: 1.6;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.15)) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
        color: #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(59, 130, 246, 0.1);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }

    /* Dividers */
    hr {
        border-color: rgba(59, 130, 246, 0.1);
        margin: 32px 0;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, #131a2e 0%, #1a2342 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.1);
    }
    .stat-card .number {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.05));
        border-left: 3px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
    }
    .insight-box strong { color: #e2e8f0; }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 13px;
        padding: 32px 0 16px 0;
        border-top: 1px solid rgba(59, 130, 246, 0.1);
        margin-top: 48px;
    }
    .footer a { color: #3b82f6; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Layout Template ────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 26, 46, 0.8)",
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        title=dict(font=dict(size=18, color="#e2e8f0")),
        xaxis=dict(
            gridcolor="rgba(59, 130, 246, 0.08)",
            zerolinecolor="rgba(59, 130, 246, 0.1)",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(59, 130, 246, 0.08)",
            zerolinecolor="rgba(59, 130, 246, 0.1)",
            tickfont=dict(color="#64748b"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#3b82f6",
            font=dict(color="#e2e8f0", size=13),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.2)",
            font=dict(color="#94a3b8"),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
)

COLORS = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "success": "#10b981",
    "orange": "#f97316",
    "gradient_blue": ["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"],
    "gradient_warm": ["#7c2d12", "#c2410c", "#f97316", "#fb923c", "#fdba74"],
}

EXP_COLORS = {
    "Expert": "#ef4444",
    "Intermediate": "#f59e0b",
    "Entry Level": "#3b82f6",
}

# ─── Data Loading & Parsing ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)

    def parse_cost_budget(cost_str):
        if pd.isna(cost_str):
            return None, None, None, None, None
        s = str(cost_str).lower().replace(",", "")
        is_hourly = "hourly" in s
        cleaned = re.sub(r"[^0-9.\-]", " ", s).strip()
        parts = [float(p) for p in re.split(r"[\s\-]+", cleaned) if p]
        if not parts:
            return None, None, None, None, is_hourly
        if is_hourly:
            if len(parts) >= 2:
                return parts[0], parts[1], (parts[0] + parts[1]) / 2, None, True
            return parts[0], parts[0], parts[0], None, True
        return None, None, None, parts[0], False

    def parse_estimated_time(time_str):
        if pd.isna(time_str):
            return None, None, None
        time_str = str(time_str).lower()
        months, hpw = None, None
        m = re.search(r"(\d+)(?: to (\d+))?\s+month", time_str)
        if m:
            months = (int(m.group(1)) + int(m.group(2))) / 2 if m.group(2) else float(m.group(1))
        if "30+" in time_str:
            hpw = 35.0
        elif "less than 30" in time_str:
            hpw = 25.0
        elif "10" in time_str:
            hpw = 10.0
        total = months * 4.33 * hpw if months and hpw else None
        return months, hpw, total

    def parse_proposals_avg(val):
        if pd.isna(val):
            return None
        nums = re.findall(r"\d+", str(val))
        if not nums:
            return None
        return sum(float(n) for n in nums) / len(nums)

    def extract_hours(date_str):
        if pd.isna(date_str):
            return None
        s = str(date_str).lower().strip()
        fixed = {
            "yesterday": 24, "last week": 168, "last month": 720,
            "last quarter": 2160, "last year": 8760, "just now": 0,
        }
        for k, v in fixed.items():
            if k in s:
                return float(v)
        patterns = [
            (r"(\d+)\s+minutes?", 1 / 60),
            (r"(?:an|(\d+))\s+hours?", 1),
            (r"(?:a|(\d+))\s+days?", 24),
            (r"(?:a|(\d+))\s+weeks?", 168),
            (r"(?:a|(\d+))\s+months?", 720),
            (r"(?:a|(\d+))\s+quarters?", 2160),
            (r"(?:a|(\d+))\s+years?", 8760),
            (r"(\d+)\s+seconds?", 1 / 3600),
        ]
        for pat, mul in patterns:
            mobj = re.search(pat, s)
            if mobj:
                val = float(mobj.group(1)) if mobj.group(1) else 1.0
                return val * mul
        return None

    temp = df["Cost Hourly / Budget"].apply(lambda x: pd.Series(parse_cost_budget(x)))
    df[["min_hourly_cost", "max_hourly_cost", "average_hourly_cost", "fixed_budget", "is_hourly"]] = temp
    df[["estimated_months", "hours_per_week", "total_estimated_hours"]] = df["Estimated Time"].apply(
        lambda x: pd.Series(parse_estimated_time(x))
    )
    df["average_no_proposals"] = df["Proposals"].apply(parse_proposals_avg)
    df["elapsed_time"] = df["Date"].apply(extract_hours)
    df.rename(columns={"Unnamed: 9": "location"}, inplace=True)
    df["location"] = df["location"].fillna("Global / Not Specified")
    df.drop(columns=["Cost Hourly / Budget", "Estimated Time", "Proposals", "Job URL"],
            inplace=True, errors="ignore")

    critical = ["Title", "Date", "Category", "is_hourly", "elapsed_time", "Experience Level"]
    df.dropna(subset=critical, inplace=True)
    df["Skills and Expertise"] = df["Skills and Expertise"].fillna("Not Specified")
    df["average_no_proposals"] = df["average_no_proposals"].fillna(0)

    for col in ["min_hourly_cost", "max_hourly_cost", "average_hourly_cost", "fixed_budget"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper, upper, df[col])

    return df


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Market Intel")
    uploaded = st.file_uploader("Upload `upwork_jobs_All.csv`", type="csv")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    sections = [
        "📌 Overview",
        "💰 Lucrative Niches",
        "🌟 Entry-Level Spots",
        "⚡ Competition Velocity",
        "🔥 In-Demand Skills",
        "📊 Supply vs Demand",
        "🤖 Market Archetypes",
        "🛠️ Tech Stack Intel",
        "📝 Keyword Analysis",
        "🎯 Sweet Spot Skills",
    ]
    selected = st.radio("Jump to section", sections, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<p style='color:#475569; font-size:12px; text-align:center;'>"
        "Built with ❤️ using Streamlit & Plotly</p>",
        unsafe_allow_html=True,
    )

# ─── Load Data ─────────────────────────────────────────────────────────────────
if uploaded:
    import io
    df = load_data(io.StringIO(uploaded.read().decode("utf-8")))
else:
    try:
        df = load_data("upwork_jobs_All.csv")
    except FileNotFoundError:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin: 60px auto;
            max-width: 600px;
        ">
            <div style="font-size: 48px; margin-bottom: 16px;">📂</div>
            <h3 style="color: #f59e0b; margin: 0 0 8px 0;">No Data Found</h3>
            <p style="color: #94a3b8; margin: 0;">
                Please upload <code style="color: #f59e0b;">upwork_jobs_All.csv</code>
                using the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()


# ─── Pre-compute shared datasets ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def precompute(_df):
    df = _df.copy()

    valid_s = df[df["Skills and Expertise"] != "Not Specified"].copy()
    valid_s["Individual_Skill"] = valid_s["Skills and Expertise"].str.split(",")
    skills_exploded = valid_s.explode("Individual_Skill")
    skills_exploded["Individual_Skill"] = skills_exploded["Individual_Skill"].str.strip()
    skills_exploded = skills_exploded[skills_exploded["Individual_Skill"] != ""]

    hourly_df = df[df["is_hourly"] == True].copy()
    fixed_df = df[df["is_hourly"] == False].copy()

    tech_cats = [
        "database management & administration", "ERP / CRM Software",
        "Information Security & Compliance", "Network & System Administration",
        "DevOps & Solution Architecture", "AI Apps & Integration",
        "Desktop Application Development ", "Game Design & Development",
        "Mobile Development", "Other - Software Development",
        "Product Management & Scrum", "QA Testing", "Scripts & Utilities",
        "Web & Mobile Design", "Web Development", "Data Analysis & Testing",
        "Data Extraction/ETL", "Data Mining & Management",
        "AI & Machine Learning", "Virtual Assistance",
        "Data Entry & Transcription Services",
    ]
    tech_hourly = df[df["Category"].isin(tech_cats) & (df["is_hourly"] == True)].dropna(
        subset=["Skills and Expertise", "average_hourly_cost"]
    ).copy()

    hourly_niches = hourly_df.groupby(["Category", "Experience Level"]).agg(
        median_hourly_rate=("average_hourly_cost", "median"),
        job_count=("average_hourly_cost", "count"),
    ).reset_index()
    hourly_niches = hourly_niches[hourly_niches["job_count"] >= 30]
    top_hourly_niches = hourly_niches.sort_values("median_hourly_rate", ascending=False)

    fixed_niches = fixed_df.groupby(["Category", "Experience Level"]).agg(
        median_fixed_budget=("fixed_budget", "median"),
        job_count=("fixed_budget", "count"),
    ).reset_index()
    fixed_niches = fixed_niches[fixed_niches["job_count"] >= 30]
    top_fixed_niches = fixed_niches.sort_values("median_fixed_budget", ascending=False)

    entry_df = df[df["Experience Level"] == "Entry Level"].copy()
    h_entry = entry_df[entry_df["is_hourly"] == True]
    hourly_cat = h_entry.groupby("Category").agg(
        total_demand=("average_hourly_cost", "count"),
        avg_competition=("average_no_proposals", "median"),
        median_hourly_rate=("average_hourly_cost", "median"),
    ).reset_index()
    hourly_sweet = hourly_cat[hourly_cat["total_demand"] >= 20].sort_values("avg_competition")

    f_entry = entry_df[entry_df["is_hourly"] == False]
    fixed_cat = f_entry.groupby("Category").agg(
        total_demand=("fixed_budget", "count"),
        avg_competition=("average_no_proposals", "median"),
        median_fixed_budget=("fixed_budget", "median"),
    ).reset_index()
    fixed_sweet = fixed_cat[fixed_cat["total_demand"] >= 20].sort_values("avg_competition")

    hourly_sweet["Opportunity_Score"] = hourly_sweet["median_hourly_rate"] / (hourly_sweet["avg_competition"] + 1)
    fixed_sweet["Opportunity_Score"] = fixed_sweet["median_fixed_budget"] / (fixed_sweet["avg_competition"] + 1)

    return {
        "skills_exploded": skills_exploded,
        "hourly_df": hourly_df,
        "fixed_df": fixed_df,
        "tech_hourly": tech_hourly,
        "top_hourly_niches": top_hourly_niches,
        "top_fixed_niches": top_fixed_niches,
        "hourly_sweet": hourly_sweet,
        "fixed_sweet": fixed_sweet,
    }


data = precompute(df)
skills_exploded = data["skills_exploded"]
hourly_df = data["hourly_df"]
fixed_df = data["fixed_df"]
tech_hourly = data["tech_hourly"]
top_hourly_niches = data["top_hourly_niches"]
top_fixed_niches = data["top_fixed_niches"]
hourly_sweet = data["hourly_sweet"]
fixed_sweet = data["fixed_sweet"]


# ─── Plotly Helper ─────────────────────────────────────────────────────────────
def style_fig(fig, height=500):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 18, 32, 0.6)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        title_font=dict(size=18, color="#e2e8f0"),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="rgba(59, 130, 246, 0.5)",
            font=dict(color="#e2e8f0", size=13, family="Inter"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(59, 130, 246, 0.2)",
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(59, 130, 246, 0.06)",
            zerolinecolor="rgba(59, 130, 246, 0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(59, 130, 246, 0.06)",
            zerolinecolor="rgba(59, 130, 246, 0.1)",
        ),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>📊 Upwork <span class="accent">Market Intelligence</span></h1>
  <p>Data-driven insights into job demand, pay rates, competition, and skill opportunities
  across the Upwork platform. Explore interactive charts to find your competitive edge.</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Overview KPIs
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📌 Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">High-level snapshot of the dataset</div>', unsafe_allow_html=True)

# Custom metric cards with HTML
median_pay = hourly_df["average_hourly_cost"].median()
avg_proposals = df["average_no_proposals"].mean()

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="number">{len(df):,}</div>
        <div class="label">Total Jobs</div>
    </div>
    <div class="stat-card">
        <div class="number">{len(hourly_df):,}</div>
        <div class="label">Hourly Jobs</div>
    </div>
    <div class="stat-card">
        <div class="number">{len(fixed_df):,}</div>
        <div class="label">Fixed Jobs</div>
    </div>
    <div class="stat-card">
        <div class="number">{df['Category'].nunique()}</div>
        <div class="label">Categories</div>
    </div>
    <div class="stat-card">
        <div class="number">${median_pay:.0f}/hr</div>
        <div class="label">Median Hourly Pay</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    exp_counts = df["Experience Level"].value_counts().reset_index()
    exp_counts.columns = ["Level", "Count"]
    exp_counts["Color"] = exp_counts["Level"].map(EXP_COLORS)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=exp_counts["Level"],
        y=exp_counts["Count"],
        marker=dict(
            color=exp_counts["Color"],
            line=dict(width=0),
            opacity=0.9,
        ),
        text=exp_counts["Count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13, family="Inter"),
        hovertemplate="<b>%{x}</b><br>Jobs: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Jobs by Experience Level",
        yaxis_title="Jobs Posted",
        xaxis_title="",
        showlegend=False,
    )
    style_fig(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    top_cats = df["Category"].value_counts().head(10).reset_index()
    top_cats.columns = ["Category", "Count"]
    top_cats = top_cats.sort_values("Count", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_cats["Category"],
        x=top_cats["Count"],
        orientation="h",
        marker=dict(
            color=np.linspace(0.3, 1, len(top_cats)),
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
            line=dict(width=0),
            opacity=0.9,
        ),
        text=top_cats["Count"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Jobs: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Top 10 Categories by Volume",
        xaxis_title="Total Jobs",
        yaxis_title="",
    )
    style_fig(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Lucrative Niches
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💰 Top 15 Most Lucrative Niches</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Highest median rates by Category × Experience Level (min 30 jobs)</div>', unsafe_allow_html=True)

tab_h, tab_f = st.tabs(["💵 Hourly Contracts", "📋 Fixed-Price Contracts"])

with tab_h:
    top_h15 = top_hourly_niches.head(15).copy()
    top_h15["label"] = top_h15["Category"] + " (" + top_h15["Experience Level"] + ")"
    top_h15 = top_h15.sort_values("median_hourly_rate", ascending=True)
    top_h15["Color"] = top_h15["Experience Level"].map(EXP_COLORS)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_h15["label"],
        x=top_h15["median_hourly_rate"],
        orientation="h",
        marker=dict(color=top_h15["Color"], opacity=0.9, line=dict(width=0)),
        text=top_h15.apply(lambda r: f"${r['median_hourly_rate']:.0f}/hr  ({r['job_count']:.0f} jobs)", axis=1),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10, family="Inter"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median Rate: $%{x:.0f}/hr<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top 15 Hourly Niches by Median Rate",
        xaxis_title="Median Hourly Rate ($/hr)",
        yaxis_title="",
        xaxis=dict(range=[0, top_h15["median_hourly_rate"].max() * 1.35]),
    )
    style_fig(fig, height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <strong>Expert-level</strong> niches dominate the top — '
        "but some <strong>Intermediate</strong> categories also command premium rates.</div>",
        unsafe_allow_html=True,
    )

with tab_f:
    top_f15 = top_fixed_niches.head(15).copy()
    top_f15["label"] = top_f15["Category"] + " (" + top_f15["Experience Level"] + ")"
    top_f15 = top_f15.sort_values("median_fixed_budget", ascending=True)
    top_f15["Color"] = top_f15["Experience Level"].map(EXP_COLORS)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_f15["label"],
        x=top_f15["median_fixed_budget"],
        orientation="h",
        marker=dict(color=top_f15["Color"], opacity=0.9, line=dict(width=0)),
        text=top_f15.apply(lambda r: f"${r['median_fixed_budget']:.0f}  ({r['job_count']:.0f} jobs)", axis=1),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10, family="Inter"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Median Budget: $%{x:.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top 15 Fixed-Price Niches by Median Budget",
        xaxis_title="Median Fixed Budget ($)",
        yaxis_title="",
        xaxis=dict(range=[0, top_f15["median_fixed_budget"].max() * 1.35]),
    )
    style_fig(fig, height=550)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Entry-Level Sweet Spots
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🌟 Entry-Level Sweet Spots & Opportunity Score</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Categories with lowest competition and highest opportunity score for beginners</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📉 Lowest Competition", "🏆 Opportunity Score (Pay ÷ Competition)"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        plot_data = hourly_sweet.head(10).sort_values("avg_competition", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_data["Category"],
            x=plot_data["avg_competition"],
            orientation="h",
            marker=dict(
                color=plot_data["avg_competition"],
                colorscale=[[0, "#3b82f6"], [1, "#1e3a5f"]],
                opacity=0.9,
            ),
            text=plot_data.apply(
                lambda r: f"  {r['avg_competition']:.0f} props · ${r['median_hourly_rate']:.0f}/hr", axis=1
            ),
            textposition="outside",
            textfont=dict(color="#4ade80", size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median Proposals: %{x:.0f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Hourly — Lowest Competition",
            xaxis_title="Median Proposals (Lower = Less Competition)",
            yaxis_title="",
            xaxis=dict(range=[0, plot_data["avg_competition"].max() * 1.5]),
        )
        style_fig(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        plot_data = fixed_sweet.head(10).sort_values("avg_competition", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_data["Category"],
            x=plot_data["avg_competition"],
            orientation="h",
            marker=dict(
                color=plot_data["avg_competition"],
                colorscale=[[0, "#f97316"], [1, "#7c2d12"]],
                opacity=0.9,
            ),
            text=plot_data.apply(
                lambda r: f"  {r['avg_competition']:.0f} props · ${r['median_fixed_budget']:.0f}", axis=1
            ),
            textposition="outside",
            textfont=dict(color="#4ade80", size=10),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median Proposals: %{x:.0f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Fixed-Price — Lowest Competition",
            xaxis_title="Median Proposals (Lower = Less Competition)",
            yaxis_title="",
            xaxis=dict(range=[0, plot_data["avg_competition"].max() * 1.5]),
        )
        style_fig(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        best_h = hourly_sweet.sort_values("Opportunity_Score", ascending=False).head(10)
        best_h = best_h.sort_values("Opportunity_Score", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=best_h["Category"],
            x=best_h["Opportunity_Score"],
            orientation="h",
            marker=dict(
                color=best_h["Opportunity_Score"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
                opacity=0.9,
            ),
            text=best_h.apply(lambda r: f"  ${r['median_hourly_rate']:.0f}/hr", axis=1),
            textposition="outside",
            textfont=dict(color="#4ade80", size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Opportunity Score: %{x:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Hourly — Opportunity Score",
            xaxis_title="Opportunity Score (Higher = Better)",
            yaxis_title="",
        )
        style_fig(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        best_f = fixed_sweet.sort_values("Opportunity_Score", ascending=False).head(10)
        best_f = best_f.sort_values("Opportunity_Score", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=best_f["Category"],
            x=best_f["Opportunity_Score"],
            orientation="h",
            marker=dict(
                color=best_f["Opportunity_Score"],
                colorscale=[[0, "#7c2d12"], [0.5, "#f97316"], [1, "#fdba74"]],
                opacity=0.9,
            ),
            text=best_f.apply(lambda r: f"  ${r['median_fixed_budget']:.0f}", axis=1),
            textposition="outside",
            textfont=dict(color="#4ade80", size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Opportunity Score: %{x:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Fixed-Price — Opportunity Score",
            xaxis_title="Opportunity Score (Higher = Better)",
            yaxis_title="",
        )
        style_fig(fig, height=450)
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="insight-box">💡 <strong>Opportunity Score = Pay ÷ (Competition + 1)</strong>. '
    "Higher scores indicate niches where you'll earn more with fewer competitors.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Competition Velocity
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚡ Competition Velocity</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">How fast do proposals accumulate after a job is posted?</div>', unsafe_allow_html=True)

time_bins = [0, 2, 12, 24, 72, 168, float("inf")]
time_labels = ["Lightning (0-2h)", "Fast (2-12h)", "Same Day (12-24h)",
               "Few Days (1-3d)", "A Week (3-7d)", "Stale (7d+)"]
df["Time_Live_Bracket"] = pd.cut(df["elapsed_time"], bins=time_bins, labels=time_labels)
vel = df.groupby("Time_Live_Bracket", observed=True).agg(
    total_jobs=("elapsed_time", "count"),
    median_proposals=("average_no_proposals", "median"),
).reset_index()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Bar(
        x=vel["Time_Live_Bracket"],
        y=vel["total_jobs"],
        name="Jobs Posted",
        marker=dict(
            color=vel["total_jobs"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
            opacity=0.85,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Jobs: %{y:,}<extra></extra>",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=vel["Time_Live_Bracket"],
        y=vel["median_proposals"],
        name="Median Proposals",
        mode="lines+markers+text",
        line=dict(color="#f97316", width=3),
        marker=dict(size=12, color="#f97316", line=dict(color="#0a0e1a", width=2)),
        text=vel["median_proposals"].apply(lambda x: f"{x:.0f}"),
        textposition="top center",
        textfont=dict(color="#f97316", size=12, family="Inter"),
        hovertemplate="<b>%{x}</b><br>Median Proposals: %{y:.0f}<extra></extra>",
    ),
    secondary_y=True,
)

fig.update_layout(
    title="Competition Velocity: How Fast Do Proposals Arrive?",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_yaxes(title_text="Total Jobs Posted", secondary_y=False)
fig.update_yaxes(title_text="Median Proposals", secondary_y=True,
                 range=[0, vel["median_proposals"].max() * 1.5])
style_fig(fig, height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="insight-box">⚡ <strong>Speed matters!</strong> Jobs attract the most proposals '
    "within the first 24 hours. Apply early to beat the competition.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — In-Demand Skills
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔥 Top 20 Most In-Demand Skills</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Volume of jobs requiring each skill with median rate & proposals</div>', unsafe_allow_html=True)

tab_sh, tab_sf = st.tabs(["💵 Hourly Skills", "📋 Fixed-Price Skills"])

with tab_sh:
    hourly_skills = skills_exploded[skills_exploded["is_hourly"] == True]
    hourly_ss = hourly_skills.groupby("Individual_Skill").agg(
        total_demand=("average_hourly_cost", "count"),
        median_hourly_rate=("average_hourly_cost", "median"),
        median_proposals=("average_no_proposals", "median"),
    ).reset_index()
    top_hourly_s = hourly_ss[hourly_ss["total_demand"] >= 50].sort_values(
        "total_demand", ascending=False
    ).head(20)
    top_hourly_s = top_hourly_s.sort_values("total_demand", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_hourly_s["Individual_Skill"],
        x=top_hourly_s["total_demand"],
        orientation="h",
        marker=dict(
            color=top_hourly_s["median_hourly_rate"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
            colorbar=dict(title="$/hr", tickfont=dict(color="#94a3b8")),
            opacity=0.9,
        ),
        text=top_hourly_s.apply(
            lambda r: f"  ${r['median_hourly_rate']:.0f}/hr · {r['median_proposals']:.0f} props", axis=1
        ),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Demand: %{x:,} jobs<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top 20 In-Demand Skills — Hourly",
        xaxis_title="Total Jobs (Demand)",
        yaxis_title="",
        xaxis=dict(range=[0, top_hourly_s["total_demand"].max() * 1.35]),
    )
    style_fig(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab_sf:
    fixed_skills = skills_exploded[skills_exploded["is_hourly"] == False]
    fixed_ss = fixed_skills.groupby("Individual_Skill").agg(
        total_demand=("fixed_budget", "count"),
        median_fixed_budget=("fixed_budget", "median"),
        median_proposals=("average_no_proposals", "median"),
    ).reset_index()
    top_fixed_s = fixed_ss[fixed_ss["total_demand"] >= 50].sort_values(
        "total_demand", ascending=False
    ).head(20)
    top_fixed_s = top_fixed_s.sort_values("total_demand", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_fixed_s["Individual_Skill"],
        x=top_fixed_s["total_demand"],
        orientation="h",
        marker=dict(
            color=top_fixed_s["median_fixed_budget"],
            colorscale=[[0, "#7c2d12"], [0.5, "#f97316"], [1, "#fdba74"]],
            colorbar=dict(title="Budget $", tickfont=dict(color="#94a3b8")),
            opacity=0.9,
        ),
        text=top_fixed_s.apply(
            lambda r: f"  ${r['median_fixed_budget']:.0f} · {r['median_proposals']:.0f} props", axis=1
        ),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Demand: %{x:,} jobs<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top 20 In-Demand Skills — Fixed-Price",
        xaxis_title="Total Jobs (Demand)",
        yaxis_title="",
        xaxis=dict(range=[0, top_fixed_s["total_demand"].max() * 1.35]),
    )
    style_fig(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Supply vs Demand by Pay Bracket
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Supply vs. Demand by Pay Bracket</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Job volume (demand) vs. median proposals (supply) across pay tiers</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    h_jobs = hourly_df.copy()
    h_bins = [0, 15, 30, 50, 100, float("inf")]
    h_labels = ["$0-15/hr", "$16-30/hr", "$31-50/hr", "$51-100/hr", "$100+/hr"]
    h_jobs["Pay_Bracket"] = pd.cut(h_jobs["average_hourly_cost"], bins=h_bins, labels=h_labels)
    h_sd = h_jobs.groupby("Pay_Bracket", observed=True).agg(
        total_jobs_posted=("average_hourly_cost", "count"),
        median_proposals=("average_no_proposals", "median"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=h_sd["Pay_Bracket"], y=h_sd["total_jobs_posted"],
            name="Jobs (Demand)",
            marker=dict(color="#3b82f6", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Jobs: %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=h_sd["Pay_Bracket"], y=h_sd["median_proposals"],
            name="Proposals (Supply)",
            mode="lines+markers+text",
            line=dict(color="#e2e8f0", width=2.5),
            marker=dict(size=10, color="#e2e8f0", line=dict(color="#0a0e1a", width=2)),
            text=h_sd["median_proposals"].apply(lambda x: f"{x:.0f}"),
            textposition="top center",
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{x}</b><br>Median Proposals: %{y:.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Hourly Contracts",
                      legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="Jobs Posted", secondary_y=False)
    fig.update_yaxes(title_text="Median Proposals", secondary_y=True,
                     range=[0, h_sd["median_proposals"].max() * 1.5])
    style_fig(fig, height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    f_jobs = fixed_df.copy()
    f_bins = [0, 50, 150, 500, 1500, float("inf")]
    f_labels = ["$0-50", "$51-150", "$151-500", "$501-1500", "$1500+"]
    f_jobs["Budget_Bracket"] = pd.cut(f_jobs["fixed_budget"], bins=f_bins, labels=f_labels)
    f_sd = f_jobs.groupby("Budget_Bracket", observed=True).agg(
        total_jobs_posted=("fixed_budget", "count"),
        median_proposals=("average_no_proposals", "median"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=f_sd["Budget_Bracket"], y=f_sd["total_jobs_posted"],
            name="Jobs (Demand)",
            marker=dict(color="#f97316", opacity=0.85),
            hovertemplate="<b>%{x}</b><br>Jobs: %{y:,}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=f_sd["Budget_Bracket"], y=f_sd["median_proposals"],
            name="Proposals (Supply)",
            mode="lines+markers+text",
            line=dict(color="#e2e8f0", width=2.5),
            marker=dict(size=10, color="#e2e8f0", line=dict(color="#0a0e1a", width=2)),
            text=f_sd["median_proposals"].apply(lambda x: f"{x:.0f}"),
            textposition="top center",
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{x}</b><br>Median Proposals: %{y:.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Fixed-Price Contracts",
                      legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="Jobs Posted", secondary_y=False)
    fig.update_yaxes(title_text="Median Proposals", secondary_y=True,
                     range=[0, f_sd["median_proposals"].max() * 1.5])
    style_fig(fig, height=450)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ML Market Archetypes
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🤖 ML Market Archetypes</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">K-Means clustering reveals 3 distinct job types in Tech on Upwork</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Running K-Means clustering…")
def run_kmeans(_tech_hourly):
    cluster_df = _tech_hourly.dropna(
        subset=["average_hourly_cost", "average_no_proposals", "elapsed_time"]
    ).copy()
    features = cluster_df[["average_hourly_cost", "average_no_proposals", "elapsed_time"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["Market_Archetype"] = km.fit_predict(scaled)
    ca = (
        cluster_df.groupby("Market_Archetype")
        .agg(
            job_count=("average_hourly_cost", "count"),
            avg_pay=("average_hourly_cost", "median"),
            avg_competition=("average_no_proposals", "median"),
            avg_hours_live=("elapsed_time", "median"),
        )
        .round(2)
    )
    return ca


ca = run_kmeans(tech_hourly)
ca = ca.sort_values("avg_pay")
labels_ = ["Budget (Low Pay)", "Mid-Market (Medium Pay)", "Premium (High Pay)"]
colors_k = [COLORS["primary"], COLORS["warning"], COLORS["danger"]]
ca["Label"] = labels_

col1, col2, col3 = st.columns(3)

metrics_list = [
    ("avg_pay", "Median Hourly Rate", "$", "/hr"),
    ("avg_competition", "Median Proposals", "", ""),
    ("avg_hours_live", "Median Hours Live", "", " hrs"),
]

for col, (metric_col, title, prefix, suffix) in zip([col1, col2, col3], metrics_list):
    with col:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ca["Label"],
            y=ca[metric_col],
            marker=dict(
                color=colors_k,
                opacity=0.9,
                line=dict(width=0),
            ),
            text=ca.apply(
                lambda r: f"{prefix}{r[metric_col]:.0f}{suffix}<br><span style='font-size:10px; color:#94a3b8'>"
                          f"{r['job_count']:,.0f} jobs</span>",
                axis=1,
            ),
            textposition="outside",
            textfont=dict(size=13, color="#e2e8f0"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{title}: {prefix}%{{y:.0f}}{suffix}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            yaxis_title="",
            xaxis_title="",
            showlegend=False,
            yaxis=dict(range=[0, ca[metric_col].max() * 1.35]),
        )
        style_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="insight-box">🤖 <strong>K-Means clustering</strong> segments tech jobs into 3 archetypes. '
    "Premium jobs have highest pay but also attract more competition and stay live longer.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Tech Stack Intelligence
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🛠️ Tech Stack Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Best skill combinations and financial ceiling jump from Entry → Expert</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Crunching tech stack combinations…")
def compute_tech_stacks(_tech_hourly):
    th = _tech_hourly.copy()
    th["Job_ID"] = th.index
    sk = th.copy()
    sk["Skill"] = sk["Skills and Expertise"].str.split(",")
    sk = sk.explode("Skill")
    sk["Skill"] = sk["Skill"].str.strip()
    sk = sk[(sk["Skill"] != "") & (sk["Skill"] != "Not Specified")]

    pairs = pd.merge(
        sk[["Job_ID", "Skill", "average_hourly_cost", "average_no_proposals"]],
        sk[["Job_ID", "Skill"]],
        on="Job_ID",
    )
    unique_pairs = pairs[pairs["Skill_x"] < pairs["Skill_y"]].copy()
    stack_a = (
        unique_pairs.groupby(["Skill_x", "Skill_y"])
        .agg(
            total_jobs=("average_hourly_cost", "count"),
            median_hourly_rate=("average_hourly_cost", "median"),
            median_proposals=("average_no_proposals", "median"),
        )
        .reset_index()
    )
    viable = stack_a[stack_a["total_jobs"] >= 30].copy()
    viable["Stack_Opportunity_Score"] = viable["median_hourly_rate"] / (viable["median_proposals"] + 1)
    top_stacks = viable.sort_values("Stack_Opportunity_Score", ascending=False).head(15)
    top_stacks["Stack"] = top_stacks["Skill_x"] + " + " + top_stacks["Skill_y"]

    sk["Experience Level"] = sk["Experience Level"].astype(str).str.strip().str.title()
    cg = (
        sk.groupby(["Skill", "Experience Level"])
        .agg(
            median_pay=("average_hourly_cost", "median"),
            job_count=("average_hourly_cost", "count"),
        )
        .reset_index()
    )
    pg = cg.pivot(index="Skill", columns="Experience Level", values=["median_pay", "job_count"])
    pg.columns = ["_".join(c).strip() for c in pg.columns]
    ec = "job_count_Entry Level"
    xc = "job_count_Expert"
    top_scaling = None
    if ec in pg.columns and xc in pg.columns:
        vg = pg[(pg[ec] >= 10) & (pg[xc] >= 10)].copy()
        vg["Financial_Ceiling_Jump"] = vg["median_pay_Expert"] - vg["median_pay_Entry Level"]
        vg["Percentage_Increase"] = vg["Financial_Ceiling_Jump"] / vg["median_pay_Entry Level"] * 100
        top_scaling = vg.sort_values("Financial_Ceiling_Jump", ascending=False).head(15).reset_index()
    return top_stacks, top_scaling


top_stacks, top_scaling = compute_tech_stacks(tech_hourly)

tab_stacks, tab_ceiling = st.tabs(["🔗 Best Skill Combos", "📈 Financial Ceiling Jump"])

with tab_stacks:
    plot_s = top_stacks.sort_values("Stack_Opportunity_Score", ascending=True).copy()

    fig = go.Figure()

    # Color by opportunity score — higher = more green-blue
    fig.add_trace(go.Bar(
        y=plot_s["Stack"],
        x=plot_s["Stack_Opportunity_Score"],
        orientation="h",
        marker=dict(
            color=plot_s["Stack_Opportunity_Score"],
            colorscale=[
                [0, "#1e3a5f"],
                [0.3, "#2563eb"],
                [0.6, "#3b82f6"],
                [1, "#06b6d4"],
            ],
            colorbar=dict(
                title=dict(text="Score", font=dict(color="#94a3b8")),
                tickfont=dict(color="#94a3b8"),
            ),
            opacity=0.9,
            line=dict(width=0),
        ),
        text=plot_s.apply(
            lambda r: f"  ${r['median_hourly_rate']:.0f}/hr · {r['total_jobs']:.0f} jobs · "
                      f"{r['median_proposals']:.0f} props",
            axis=1,
        ),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10, family="Inter"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Opportunity Score: %{x:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Top 15 Tech Stack Combinations by Opportunity Score",
        xaxis_title="Stack Opportunity Score (Higher = Better)",
        yaxis_title="",
        xaxis=dict(range=[0, plot_s["Stack_Opportunity_Score"].max() * 1.4]),
    )
    style_fig(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">🔗 <strong>Skill combos</strong> are scored by '
        "<code>Median Rate ÷ (Median Proposals + 1)</code>. Stacks at the top offer the best "
        "pay-to-competition ratio.</div>",
        unsafe_allow_html=True,
    )

with tab_ceiling:
    if top_scaling is not None and not top_scaling.empty:
        ps = top_scaling.sort_values("Financial_Ceiling_Jump", ascending=True).copy()

        fig = go.Figure()

        # Expert pay — full bar
        fig.add_trace(go.Bar(
            y=ps["Skill"],
            x=ps["median_pay_Expert"],
            orientation="h",
            name="Expert Pay",
            marker=dict(color="#3b82f6", opacity=0.85, line=dict(width=0)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Expert Pay: $%{x:.0f}/hr<br>"
                "<extra></extra>"
            ),
        ))

        # Entry pay — overlaid shorter bar
        fig.add_trace(go.Bar(
            y=ps["Skill"],
            x=ps["median_pay_Entry Level"],
            orientation="h",
            name="Entry-Level Pay",
            marker=dict(color="#93c5fd", opacity=0.9, line=dict(width=0)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Entry Pay: $%{x:.0f}/hr<br>"
                "<extra></extra>"
            ),
        ))

        # Annotations showing the jump
        for i, row in ps.iterrows():
            idx = list(ps["Skill"]).index(row["Skill"])
            fig.add_annotation(
                x=row["median_pay_Expert"] + 1,
                y=idx,
                text=f"+${row['Financial_Ceiling_Jump']:.0f} ({row['Percentage_Increase']:.0f}%)",
                showarrow=False,
                font=dict(color="#ef4444", size=11, family="Inter"),
                xanchor="left",
                yanchor="middle",
            )

        fig.update_layout(
            title="Top 15 Skills by Financial Ceiling Jump (Entry → Expert)",
            xaxis_title="Median Hourly Rate ($/hr)",
            yaxis_title="",
            barmode="overlay",
            xaxis=dict(range=[0, ps["median_pay_Expert"].max() * 1.45]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        style_fig(fig, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box">📈 <strong>Financial Ceiling Jump</strong> shows how much more '
            "Experts earn vs. Entry-Level freelancers for the same skill. "
            "Skills with the biggest jumps have the highest long-term earning potential.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("⚠️ Not enough data to compute the Entry → Expert promotion delta.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Keyword Pay Analysis
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📝 Highest-Paying Title Keywords</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Which words in job titles signal premium pay in Tech?</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def keyword_analysis(_tech_hourly):
    nlp = _tech_hourly.dropna(subset=["Title", "average_hourly_cost"]).copy()
    nlp["Clean_Title"] = nlp["Title"].str.lower().apply(lambda x: re.sub(r"[^a-z\s]", "", x))
    nlp["Word"] = nlp["Clean_Title"].str.split()
    words = nlp.explode("Word")
    stop = {
        "and", "for", "to", "a", "in", "of", "with", "the", "is", "on",
        "looking", "needed", "expert", "developer", "engineer", "an", "be",
        "are", "we", "our", "you", "your",
    }
    words = words[~words["Word"].isin(stop)]
    kv = (
        words.groupby("Word")
        .agg(
            total_mentions=("average_hourly_cost", "count"),
            median_hourly_rate=("average_hourly_cost", "median"),
        )
        .reset_index()
    )
    return (
        kv[kv["total_mentions"] >= 20]
        .sort_values("median_hourly_rate", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


pk = keyword_analysis(tech_hourly)
plot_pk = pk.sort_values("median_hourly_rate", ascending=True)

fig = go.Figure()

fig.add_trace(go.Bar(
    y=plot_pk["Word"],
    x=plot_pk["median_hourly_rate"],
    orientation="h",
    marker=dict(
        color=plot_pk["median_hourly_rate"],
        colorscale=[
            [0, "#14532d"],
            [0.3, "#16a34a"],
            [0.6, "#4ade80"],
            [1, "#bbf7d0"],
        ],
        colorbar=dict(
            title=dict(text="$/hr", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        opacity=0.9,
        line=dict(width=0),
    ),
    text=plot_pk.apply(
        lambda r: f"  ${r['median_hourly_rate']:.0f}/hr — {r['total_mentions']:.0f} mentions",
        axis=1,
    ),
    textposition="outside",
    textfont=dict(color="#4ade80", size=11, family="Inter"),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Median Rate: $%{x:.0f}/hr<br>"
        "<extra></extra>"
    ),
))

fig.update_layout(
    title="Highest-Paying Keywords in Tech Job Titles",
    xaxis_title="Median Hourly Rate ($/hr)",
    yaxis_title="",
    xaxis=dict(range=[0, plot_pk["median_hourly_rate"].max() * 1.35]),
)
style_fig(fig, height=550)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="insight-box">📝 <strong>Pro tip:</strong> Including these high-value keywords in your '
    "Upwork profile and proposals may help you appear in better-paying search results.</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Sweet Spot Skills (\$151–\$500)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 Underserved Skills in the \$151–\$500 Sweet Spot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Low-competition skills with high demand in the best fixed-budget tier</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def sweet_spot_skills(_fixed_df):
    ss = _fixed_df[(_fixed_df["fixed_budget"] >= 151) & (_fixed_df["fixed_budget"] <= 500)].copy()
    sk = ss.dropna(subset=["Skills and Expertise"]).copy()
    sk["Individual Skill"] = sk["Skills and Expertise"].str.split(",")
    sk = sk.explode("Individual Skill")
    sk["Individual Skill"] = sk["Individual Skill"].str.strip()
    sd = (
        sk.groupby(["Category", "Individual Skill"])
        .agg(
            demand_job_count=("fixed_budget", "count"),
            supply_median_proposals=("average_no_proposals", "median"),
        )
        .reset_index()
    )
    under = (
        sd[sd["demand_job_count"] >= 50]
        .sort_values(["supply_median_proposals", "demand_job_count"], ascending=[True, False])
        .head(20)
        .reset_index(drop=True)
    )
    under["Label"] = under["Individual Skill"] + " (" + under["Category"] + ")"
    return under


under = sweet_spot_skills(fixed_df)

# Scatter plot for a more insightful interactive view
fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=under["supply_median_proposals"],
    y=under["demand_job_count"],
    mode="markers+text",
    marker=dict(
        size=under["demand_job_count"] / under["demand_job_count"].max() * 50 + 12,
        color=under["supply_median_proposals"],
        colorscale=[
            [0, "#10b981"],
            [0.5, "#f59e0b"],
            [1, "#ef4444"],
        ],
        colorbar=dict(
            title=dict(text="Proposals", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
        ),
        opacity=0.85,
        line=dict(color="rgba(255,255,255,0.15)", width=1),
    ),
    text=under["Individual Skill"],
    textposition="top center",
    textfont=dict(color="#e2e8f0", size=10, family="Inter"),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "Category: %{customdata[0]}<br>"
        "Competition: %{x:.0f} proposals<br>"
        "Demand: %{y} jobs<br>"
        "<extra></extra>"
    ),
    customdata=under[["Category"]].values,
))

fig_scatter.update_layout(
    title="Sweet Spot Map — Lower Left = Best Opportunity",
    xaxis_title="Median Proposals (Competition) — Lower is Better →",
    yaxis_title="Job Count (Demand) — Higher is Better →",
    xaxis=dict(range=[
        under["supply_median_proposals"].min() - 2,
        under["supply_median_proposals"].max() + 5,
    ]),
)

# Add a "sweet spot" annotation box
fig_scatter.add_shape(
    type="rect",
    x0=under["supply_median_proposals"].min() - 1,
    y0=under["demand_job_count"].median(),
    x1=under["supply_median_proposals"].median(),
    y1=under["demand_job_count"].max() + 10,
    line=dict(color="rgba(16, 185, 129, 0.4)", width=2, dash="dot"),
    fillcolor="rgba(16, 185, 129, 0.04)",
)
fig_scatter.add_annotation(
    x=under["supply_median_proposals"].min(),
    y=under["demand_job_count"].max() + 5,
    text="🎯 Sweet Spot Zone",
    showarrow=False,
    font=dict(color="#10b981", size=13, family="Inter"),
    xanchor="left",
)

style_fig(fig_scatter, height=550)
st.plotly_chart(fig_scatter, use_container_width=True)

# Also show the bar chart view in an expander
with st.expander("📊 View as Bar Chart", expanded=False):
    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        y=under["Label"],
        x=under["supply_median_proposals"],
        orientation="h",
        marker=dict(
            color=under["supply_median_proposals"],
            colorscale=[
                [0, "#10b981"],
                [0.4, "#f59e0b"],
                [1, "#ef4444"],
            ],
            opacity=0.9,
            line=dict(width=0),
        ),
        text=under.apply(
            lambda r: f"  {r['demand_job_count']:.0f} jobs", axis=1
        ),
        textposition="outside",
        textfont=dict(color="#4ade80", size=10, family="Inter"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Competition: %{x:.0f} median proposals<br>"
            "<extra></extra>"
        ),
    ))

    fig_bar.update_layout(
        title="Underserved Skills — Ranked by Competition (Lower = Better)",
        xaxis_title="Median Proposals (Competition)",
        yaxis_title="",
        xaxis=dict(range=[0, under["supply_median_proposals"].max() + 10]),
        yaxis=dict(autorange="reversed"),
    )
    style_fig(fig_bar, height=700)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown(
    '<div class="insight-box">🎯 <strong>The \$151–\$500 range</strong> is the sweet spot for fixed-price '
    "contracts — high enough to be worthwhile, low enough for clients to hire without lengthy vetting. "
    "Skills in the <strong>green zone</strong> (low proposals, high demand) are your best entry points.</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# BONUS — Quick Data Explorer
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔍 Quick Data Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Filter and explore the raw dataset interactively</div>',
    unsafe_allow_html=True,
)

with st.expander("🔎 Open Data Explorer", expanded=False):
    explorer_cols = st.columns([2, 2, 1])

    with explorer_cols[0]:
        selected_cats = st.multiselect(
            "Filter by Category",
            options=sorted(df["Category"].unique()),
            default=[],
            placeholder="All categories",
        )

    with explorer_cols[1]:
        selected_exp = st.multiselect(
            "Filter by Experience Level",
            options=sorted(df["Experience Level"].unique()),
            default=[],
            placeholder="All levels",
        )

    with explorer_cols[2]:
        contract_type = st.selectbox(
            "Contract Type",
            options=["All", "Hourly", "Fixed"],
        )

    # Apply filters
    filtered = df.copy()
    if selected_cats:
        filtered = filtered[filtered["Category"].isin(selected_cats)]
    if selected_exp:
        filtered = filtered[filtered["Experience Level"].isin(selected_exp)]
    if contract_type == "Hourly":
        filtered = filtered[filtered["is_hourly"] == True]
    elif contract_type == "Fixed":
        filtered = filtered[filtered["is_hourly"] == False]

    # Summary stats
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="number">{len(filtered):,}</div>
            <div class="label">Matching Jobs</div>
        </div>
        <div class="stat-card">
            <div class="number">${filtered['average_hourly_cost'].median():.0f}/hr</div>
            <div class="label">Median Hourly</div>
        </div>
        <div class="stat-card">
            <div class="number">${filtered['fixed_budget'].median():.0f}</div>
            <div class="label">Median Budget</div>
        </div>
        <div class="stat-card">
            <div class="number">{filtered['average_no_proposals'].median():.0f}</div>
            <div class="label">Median Proposals</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Distribution chart
    dist_tab1, dist_tab2 = st.tabs(["💵 Pay Distribution", "📊 Proposals Distribution"])

    with dist_tab1:
        if contract_type != "Fixed":
            hourly_filtered = filtered[filtered["is_hourly"] == True]["average_hourly_cost"].dropna()
            if len(hourly_filtered) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=hourly_filtered,
                    nbinsx=40,
                    marker=dict(
                        color="#3b82f6",
                        opacity=0.8,
                        line=dict(color="rgba(59, 130, 246, 0.3)", width=1),
                    ),
                    hovertemplate="Rate: $%{x:.0f}/hr<br>Count: %{y}<extra></extra>",
                ))
                fig.update_layout(
                    title="Hourly Rate Distribution",
                    xaxis_title="Hourly Rate ($/hr)",
                    yaxis_title="Number of Jobs",
                    bargap=0.05,
                )
                style_fig(fig, height=400)
                st.plotly_chart(fig, use_container_width=True)

        if contract_type != "Hourly":
            fixed_filtered = filtered[filtered["is_hourly"] == False]["fixed_budget"].dropna()
            if len(fixed_filtered) > 0:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=fixed_filtered,
                    nbinsx=40,
                    marker=dict(
                        color="#f97316",
                        opacity=0.8,
                        line=dict(color="rgba(249, 115, 22, 0.3)", width=1),
                    ),
                    hovertemplate="Budget: $%{x:.0f}<br>Count: %{y}<extra></extra>",
                ))
                fig.update_layout(
                    title="Fixed Budget Distribution",
                    xaxis_title="Fixed Budget ($)",
                    yaxis_title="Number of Jobs",
                    bargap=0.05,
                )
                style_fig(fig, height=400)
                st.plotly_chart(fig, use_container_width=True)

    with dist_tab2:
        props = filtered["average_no_proposals"].dropna()
        if len(props) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=props,
                nbinsx=30,
                marker=dict(
                    color="#8b5cf6",
                    opacity=0.8,
                    line=dict(color="rgba(139, 92, 246, 0.3)", width=1),
                ),
                hovertemplate="Proposals: %{x:.0f}<br>Count: %{y}<extra></extra>",
            ))
            fig.update_layout(
                title="Proposals Distribution",
                xaxis_title="Number of Proposals",
                yaxis_title="Number of Jobs",
                bargap=0.05,
            )
            style_fig(fig, height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Show filtered data table
    show_cols = [
        "Title", "Category", "Experience Level", "average_hourly_cost",
        "fixed_budget", "average_no_proposals", "Skills and Expertise",
    ]
    available_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols].head(100),
        use_container_width=True,
        height=400,
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div style="margin-bottom: 12px;">
        <span style="font-size: 24px;">📊</span>
    </div>
    <div style="font-size: 14px; color: #64748b; font-weight: 600; margin-bottom: 4px;">
        Upwork Market Intelligence Dashboard
    </div>
    <div style="font-size: 12px; color: #475569;">
        Powered by Python · Streamlit · Plotly · scikit-learn
    </div>
    <div style="font-size: 11px; color: #334155; margin-top: 8px;">
        Interactive charts support zoom, pan, hover, and export.
        Click the 📷 icon on any chart to download as PNG.
    </div>
</div>
""", unsafe_allow_html=True)