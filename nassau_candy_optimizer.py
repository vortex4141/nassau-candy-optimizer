import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Nassau Candy — Factory Optimizer", page_icon="🍬")

# ── CSS Theme — Artisan Chocolate ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Warm dark background ── */
.stApp { background: #1a0c04; min-height: 100vh; }

/* ── Hero banner ── */
.hero-banner {
    background: #231007;
    border: 1px solid #3d1e08;
    border-left: 4px solid #c8941a;
    border-radius: 12px;
    padding: 36px 44px 32px;
    margin-bottom: 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.45);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0d9c0;
    margin: 0 0 6px 0;
    line-height: 1.2;
    letter-spacing: -0.3px;
}
.hero-subtitle {
    color: #9a7a5a;
    font-size: 0.92rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.2px;
}
.hero-tag {
    display: inline-block;
    background: #2e1508;
    border: 1px solid #5a3010;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.70rem;
    color: #c8941a;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #231007 !important;
    border: 1px solid #3d1e08 !important;
    border-radius: 10px !important;
    padding: 22px 24px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.30) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: #c8941a !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.40) !important;
}
[data-testid="metric-container"] label {
    color: #9a7a5a !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0d9c0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #f0d9c0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #c8941a !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e0c04 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid #3d1e08 !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    color: #7a5a3a !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 9px 18px !important;
    transition: background 0.18s ease, color 0.18s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #2e1508 !important;
    color: #d4b896 !important;
}
.stTabs [aria-selected="true"] {
    background: #c8941a !important;
    color: #fff8f0 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    border-color: #a07010 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a0c04 !important;
    border-right: 1px solid #3d1e08 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c8941a !important;
    font-weight: 600 !important;
}

/* ── Headings ── */
h1, h2, h3 {
    color: #f0d9c0 !important;
    font-weight: 700 !important;
}
h2 {
    border-bottom: 2px solid #3d1e08 !important;
    padding-bottom: 8px !important;
}
h2::after { display: none !important; }
h3 { color: #d4b896 !important; font-size: 1.05rem !important; }

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: #3d1e08 !important;
    margin: 28px 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #1e0c04 !important;
    border-radius: 10px !important;
    border: 1px solid #3d1e08 !important;
}

/* ── Info / alert boxes ── */
[data-testid="stNotification"] {
    background: #231007 !important;
    border: 1px solid #5a3010 !important;
    border-left: 3px solid #c8941a !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #c8941a !important;
    color: #1a0c04 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 26px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    transition: background 0.18s ease, box-shadow 0.18s ease !important;
}
.stDownloadButton > button:hover {
    background: #a07010 !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4) !important;
}

/* ── Multiselect / tags ── */
[data-baseweb="tag"] {
    background: #3d1e08 !important;
    border: 1px solid #5a3010 !important;
    border-radius: 6px !important;
    color: #f0d9c0 !important;
}

/* ── Inputs ── */
[data-baseweb="input"], [data-baseweb="select"] {
    background: #231007 !important;
    border-color: #3d1e08 !important;
    color: #f0d9c0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1a0c04; }
::-webkit-scrollbar-thumb { background: #5a3010; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c8941a; }

/* ── Body text ── */
p, .stMarkdown p { color: #c4a882 !important; line-height: 1.65 !important; }
label { color: #9a7a5a !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-tag">🍫 Nassau Candy Distributor</div>
  <p class="hero-title">Factory Reallocation & Shipping Optimizer</p>
  <p class="hero-subtitle">Predictive modeling · Scenario simulation · ML-powered factory reassignment recommendations</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176, -81.088371),
    "Sugar Shack":       (48.11914,  -96.18115),
    "Secret Factory":    (41.446333, -90.565487),
    "The Other Factory": (35.1175,   -89.971107),
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# Region centroids matching actual dataset values
REGION_COORDS = {
    # Actual regions in Nassau dataset
    "Interior":  (39.5000,  -98.3500),   # US Interior / Central
    "Atlantic":  (37.0902,  -76.3452),   # US Atlantic Coast
    "Gulf":      (29.5000,  -91.0000),   # Gulf Coast
    "Pacific":   (37.7749, -122.4194),   # US Pacific Coast
    # Common fallbacks
    "West":      (37.7749, -119.4194),
    "East":      (40.7128,  -74.0060),
    "South":     (29.7604,  -95.3698),
    "Central":   (41.8781,  -87.6298),
    "North":     (44.9778,  -93.2650),
    "Midwest":   (41.8781,  -87.6298),
    "Northeast": (42.3601,  -71.0589),
    "Southeast": (33.7490,  -84.3880),
    "Southwest": (33.4484, -112.0740),
    "Northwest": (47.6062, -122.3321),
}

ALL_FACTORIES = list(FACTORY_COORDS.keys())
SHIP_MODES    = ["First Class", "Second Class", "Standard Class", "Same Day"]

# ── Haversine distance ────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path):
    if path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=False, errors='coerce')
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=False, errors='coerce')
    # Use pre-cleaned Lead Time if present, else compute
    if 'Lead Time' not in df.columns:
        df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    df = df[df['Lead Time'] >= 0]

    # Factory assignment — try exact match first, then case-insensitive
    df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY)
    if df['Factory'].isna().all():
        pf_lower = {k.lower(): v for k, v in PRODUCT_FACTORY.items()}
        df['Factory'] = df['Product Name'].str.strip().str.lower().map(pf_lower)

    # Keep unmatched rows too — assign factory from Division as fallback
    division_factory = {
        'Chocolate': "Wicked Choccy's",
        'Sugar':     'Sugar Shack',
        'Other':     'Secret Factory',
    }
    df['Factory'] = df['Factory'].fillna(df.get('Division', pd.Series(dtype=str)).map(division_factory))
    df = df.dropna(subset=['Region', 'Ship Mode'])

    # Standardise region casing for coord lookup
    df['Region_Key'] = df['Region'].str.strip().str.title()

    # Factory → Region distance
    def dist_row(row):
        fcoord = FACTORY_COORDS.get(row['Factory'])
        rcoord = REGION_COORDS.get(row['Region_Key'])
        if fcoord and rcoord:
            return haversine(fcoord[0], fcoord[1], rcoord[0], rcoord[1])
        return np.nan
    df['Distance_Miles'] = df.apply(dist_row, axis=1)

    # Financials
    df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')
    df['Sales']        = pd.to_numeric(df['Sales'],        errors='coerce')
    df['Cost']         = pd.to_numeric(df['Cost'],         errors='coerce')
    df['Units']        = pd.to_numeric(df['Units'],        errors='coerce')
    df['Profit_Margin'] = df['Gross Profit'] / df['Sales'].replace(0, np.nan)

    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    from sklearn.impute import SimpleImputer

    # Only use columns that actually exist in the dataframe
    cat_cols  = [c for c in ['Product Name', 'Factory', 'Region', 'Ship Mode', 'Division'] if c in df.columns]
    num_cols  = [c for c in ['Distance_Miles', 'Units', 'Cost'] if c in df.columns]
    feat_cols = cat_cols + num_cols

    work = df[feat_cols + ['Lead Time']].copy()
    work = work.dropna(subset=['Lead Time'])
    for col in cat_cols:
        work[col] = work[col].fillna('Unknown').astype(str)

    X = work[feat_cols].copy()
    y = work['Lead Time'].copy()

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Fill numeric cols: all-NaN columns (e.g. Distance_Miles when no region match) get 0
    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isna().all():
                X[col] = 0.0
            else:
                X[col] = X[col].fillna(X[col].median())

    # Drop any remaining all-NaN columns so shape stays consistent
    X = X.dropna(axis=1, how='all')
    feat_cols = X.columns.tolist()   # truth comes from X, not the original list

    imputer = SimpleImputer(strategy='median')
    X_arr   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_arr, columns=feat_cols, index=X.index)

    # Adaptive test size — ensure at least 5 test samples
    n = len(X)
    test_sz = min(max(5 / n, 0.2), 0.4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model_defs = {
        'Linear Regression':   LinearRegression(),
        'Random Forest':       RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    results, trained = {}, {}
    for name, mdl in model_defs.items():
        if name == 'Linear Regression':
            mdl.fit(X_tr_s, y_train)
            preds = mdl.predict(X_te_s)
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)

        results[name] = {
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            'MAE':  round(mean_absolute_error(y_test, preds),         4),
            'R2':   round(r2_score(y_test, preds),                    4),
        }
        trained[name] = mdl

    best = max(results, key=lambda k: results[k]['R2'])

    # Feature importance from best non-linear model
    best_nonlinear = 'Random Forest' if results['Random Forest']['R2'] >= results['Gradient Boosting']['R2'] else 'Gradient Boosting'
    feat_imp = pd.DataFrame({
        'Feature':    feat_cols,
        'Importance': trained[best_nonlinear].feature_importances_
    }).sort_values('Importance', ascending=False)

    return trained, results, best, scaler, le_dict, feat_cols, X_test, y_test, feat_imp

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cluster_routes(df, n_clusters=4):
    cluster_df = df[['Distance_Miles', 'Lead Time', 'Profit_Margin', 'Cost']].dropna().copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(cluster_df)
    km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['Cluster'] = km.fit_predict(X_s)
    df_merged = df.copy()
    df_merged.loc[cluster_df.index, 'Route_Cluster'] = cluster_df['Cluster']
    cluster_profiles = cluster_df.groupby('Cluster').mean().round(2)
    return df_merged, cluster_profiles

# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def simulate(product, region, ship_mode, df, trained, best, scaler, le_dict, feat_cols, alpha):
    current_factory = PRODUCT_FACTORY.get(product, ALL_FACTORIES[0])
    div  = df[df['Product Name'] == product]['Division'].mode()
    div  = div.iloc[0] if len(div) > 0 else 'Sugar'
    avg_cost  = df[df['Product Name'] == product]['Cost'].mean()
    avg_units = df[df['Product Name'] == product]['Units'].mean()
    if pd.isna(avg_cost):  avg_cost  = df['Cost'].mean()
    if pd.isna(avg_units): avg_units = df['Units'].mean()

    rows = []
    for factory in ALL_FACTORIES:
        fcoord = FACTORY_COORDS.get(factory)
        rcoord = REGION_COORDS.get(region.strip().title())
        dist   = haversine(fcoord[0], fcoord[1], rcoord[0], rcoord[1]) if (fcoord and rcoord) else df['Distance_Miles'].mean()

        row = pd.DataFrame({'Product Name': [product], 'Factory': [factory],
                            'Region': [region], 'Ship Mode': [ship_mode],
                            'Division': [div], 'Distance_Miles': [dist],
                            'Units': [avg_units], 'Cost': [avg_cost]})
        for col in ['Product Name', 'Factory', 'Region', 'Ship Mode', 'Division']:
            le = le_dict[col]
            try:
                row[col] = le.transform(row[col].astype(str))
            except ValueError:
                row[col] = 0
        row = row.fillna(0)[feat_cols]
        mdl = trained[best]
        pred = mdl.predict(scaler.transform(row))[0] if best == 'Linear Regression' else mdl.predict(row)[0]
        pred = max(0, round(pred, 2))

        hist = df[(df['Product Name'] == product) & (df['Factory'] == factory)]
        pm   = hist['Profit_Margin'].mean() * 100 if len(hist) > 0 and not pd.isna(hist['Profit_Margin'].mean()) else df['Profit_Margin'].mean() * 100

        rows.append({'Factory': factory, 'Is_Current': factory == current_factory,
                     'Pred_Lead_Time': pred, 'Distance_Miles': round(dist, 1),
                     'Profit_Margin_Pct': round(pm if not pd.isna(pm) else 0, 2)})

    sim = pd.DataFrame(rows)
    cur_lt = sim[sim['Is_Current']]['Pred_Lead_Time'].values
    cur_lt = float(cur_lt[0]) if len(cur_lt) > 0 else sim['Pred_Lead_Time'].mean()
    sim['Lead_Time_Delta']  = (cur_lt - sim['Pred_Lead_Time']).round(2)
    sim['LT_Reduction_Pct'] = ((cur_lt - sim['Pred_Lead_Time']) / max(cur_lt, 0.01) * 100).round(2)

    lt_min, lt_max = sim['Pred_Lead_Time'].min(), sim['Pred_Lead_Time'].max()
    pm_min, pm_max = sim['Profit_Margin_Pct'].min(), sim['Profit_Margin_Pct'].max()
    lt_score = 1 - (sim['Pred_Lead_Time'] - lt_min) / max(lt_max - lt_min, 0.01)
    pm_score = (sim['Profit_Margin_Pct'] - pm_min) / max(pm_max - pm_min, 0.01)
    sim['Composite_Score'] = ((1 - alpha) * lt_score + alpha * pm_score * 100).round(2)
    sim['Confidence_Score'] = (100 - (sim['Pred_Lead_Time'].std() / max(sim['Pred_Lead_Time'].mean(), 0.01) * 100)).clip(0, 100).round(1)

    return sim.sort_values('Composite_Score', ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# BULK RECOMMENDATIONS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_all_recommendations(_df, _trained, best, _scaler, _le_dict, feat_cols, alpha, region, ship_mode):
    recs = []
    for product, current_factory in PRODUCT_FACTORY.items():
        try:
            sim = simulate(product, region, ship_mode, _df, _trained, best, _scaler, _le_dict, feat_cols, alpha)
            best_row = sim.iloc[0]
            cur_row  = sim[sim['Is_Current']].iloc[0] if sim['Is_Current'].any() else sim.iloc[-1]
            recs.append({
                'Product':               product,
                'Division':              _df[_df['Product Name'] == product]['Division'].mode().iloc[0] if len(_df[_df['Product Name'] == product]) > 0 else 'N/A',
                'Current Factory':       current_factory,
                'Recommended Factory':   best_row['Factory'],
                'Change Needed':         best_row['Factory'] != current_factory,
                'Current Lead Time':     cur_row['Pred_Lead_Time'],
                'Optimal Lead Time':     best_row['Pred_Lead_Time'],
                'Lead Time Reduction %': best_row['LT_Reduction_Pct'],
                'Profit Margin %':       best_row['Profit_Margin_Pct'],
                'Composite Score':       best_row['Composite_Score'],
            })
        except Exception:
            continue
    return pd.DataFrame(recs)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = r"C:\Users\MSI INDI\Desktop\nssau project\nassau_cleaned.xlsx"

try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ Dataset not found at `{DATA_PATH}`. Please place your Excel file in the same folder and update the `DATA_PATH` variable.")
    st.stop()

trained, model_results, best_model, scaler, le_dict, feat_cols, X_test, y_test, feat_imp = train_models(raw_df)
clustered_df, cluster_profiles = cluster_routes(raw_df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Control Panel")

all_regions   = sorted(raw_df['Region'].dropna().unique().tolist())
all_products  = sorted(PRODUCT_FACTORY.keys())
all_divisions = sorted(raw_df['Division'].dropna().unique().tolist())
all_ship      = sorted(raw_df['Ship Mode'].dropna().unique().tolist()) or SHIP_MODES

sel_region   = st.sidebar.selectbox("Target Region", all_regions)
sel_product  = st.sidebar.selectbox("Product", all_products)
sel_ship     = st.sidebar.selectbox("Ship Mode", all_ship)
sel_division = st.sidebar.multiselect("Division Filter", all_divisions, default=all_divisions)
alpha        = st.sidebar.slider("Optimisation Priority", 0.0, 1.0, 0.4,
                                 help="0 = Prioritise Speed  |  1 = Prioritise Profit")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(raw_df):,} orders")
st.sidebar.markdown(f"**Best Model:** {best_model}")
st.sidebar.markdown(f"**R² Score:** {model_results[best_model]['R2']:.3f}")

# Filter
df = raw_df[raw_df['Division'].isin(sel_division)] if sel_division else raw_df

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & KPIs",
    "🤖 Model Performance",
    "🏭 Factory Simulator",
    "🔄 What-If Scenario",
    "🏆 Recommendations",
    "⚠️ Risk & Impact",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW & KPIs
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Operational KPIs")

    avg_lt        = df['Lead Time'].mean()
    avg_lt_region = df[df['Region'] == sel_region]['Lead Time'].mean() if sel_region in df['Region'].values else avg_lt
    total_sales   = df['Sales'].sum()
    total_profit  = df['Gross Profit'].sum()
    overall_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    rec_coverage  = len(PRODUCT_FACTORY) / len(PRODUCT_FACTORY) * 100  # 100% product coverage

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Lead Time (Days)",         f"{avg_lt:.1f}",             help="Mean shipping lead time across all orders.")
    c2.metric(f"Lead Time — {sel_region}",    f"{avg_lt_region:.1f}",      help="Avg lead time for selected region.")
    c3.metric("Overall Profit Margin",         f"{overall_margin:.1f}%",    help="Gross Profit ÷ Total Sales.")
    c4.metric("Recommendation Coverage",       f"{rec_coverage:.0f}%",      help="% of products covered by the simulation engine.")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        # Lead time by factory
        lt_factory = df.groupby('Factory')['Lead Time'].mean().reset_index().sort_values('Lead Time')
        fig1 = px.bar(lt_factory, x='Factory', y='Lead Time', color='Lead Time',
                      color_continuous_scale='RdYlGn_r', title="Avg Lead Time by Factory",
                      template='plotly_dark', text=lt_factory['Lead Time'].map('{:.1f}d'.format))
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                           font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                           xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Days'),
                           showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        fig1.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Profit margin by division
        pm_div = df.groupby('Division')['Profit_Margin'].mean().reset_index()
        pm_div['Profit_Margin'] = (pm_div['Profit_Margin'] * 100).round(2)
        fig2 = px.pie(pm_div, names='Division', values='Profit_Margin',
                      title="Profit Margin by Division", template='plotly_dark',
                      color_discrete_sequence=['#d4a017', '#7b3f00', '#f5c842', '#a0522d', '#c8860a'], hole=0.45)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'),
                           title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                           legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        # Lead time distribution by ship mode
        fig3 = px.box(df, x='Ship Mode', y='Lead Time', color='Ship Mode',
                      title="Lead Time Distribution by Ship Mode",
                      color_discrete_sequence=['#d4a017','#7b3f00','#f5c842','#a0522d'],
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                           font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                           showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                           xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # Orders by region
        region_vol = df.groupby('Region').agg(Orders=('Order ID', 'count'),
                                               Avg_LT=('Lead Time', 'mean')).reset_index()
        fig4 = px.scatter(region_vol, x='Orders', y='Avg_LT', text='Region', size='Orders',
                          color='Avg_LT', color_continuous_scale='RdYlGn_r',
                          title="Region: Volume vs Avg Lead Time", template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                           font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                           xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                           margin=dict(l=10, r=10, t=50, b=10))
        fig4.update_traces(textfont=dict(color='white'))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.subheader("Factory Performance Summary")
    factory_summary = df.groupby('Factory').agg(
        Orders=('Order ID', 'count'),
        Avg_Lead_Time=('Lead Time', 'mean'),
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Gross Profit', 'sum'),
        Avg_Profit_Margin=('Profit_Margin', 'mean'),
    ).round(2).reset_index()
    factory_summary['Avg_Profit_Margin'] = (factory_summary['Avg_Profit_Margin'] * 100).round(2)
    st.dataframe(factory_summary, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Predictive Model Evaluation")
    st.markdown(f"**Target Variable:** Shipping Lead Time (days)  |  **Best Model:** `{best_model}`")

    # Model comparison table
    perf_df = pd.DataFrame(model_results).T.reset_index().rename(columns={'index': 'Model'})
    perf_df['Selected'] = perf_df['Model'].apply(lambda m: '✅ Best' if m == best_model else '')

    c1, c2, c3 = st.columns(3)
    for i, (_, row) in enumerate(perf_df.iterrows()):
        col = [c1, c2, c3][i]
        col.metric(row['Model'], f"R² {row['R2']:.3f}",
                   delta=f"RMSE {row['RMSE']:.3f}  |  MAE {row['MAE']:.3f}")

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        # RMSE comparison
        fig_rmse = px.bar(perf_df, x='Model', y='RMSE', color='Model',
                          color_discrete_sequence=['#d4a017', '#7b3f00', '#f5c842'],
                          text=perf_df['RMSE'].map('{:.3f}'.format),
                          title="RMSE Comparison (lower = better)", template='plotly_dark')
        fig_rmse.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                               font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                               showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                               xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                               yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        fig_rmse.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_m2:
        # R² comparison
        fig_r2 = px.bar(perf_df, x='Model', y='R2', color='Model',
                        color_discrete_sequence=['#d4a017', '#7b3f00', '#f5c842'],
                        text=perf_df['R2'].map('{:.3f}'.format),
                        title="R² Score Comparison (higher = better)", template='plotly_dark')
        fig_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                             font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                             showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                             xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                             yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        fig_r2.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    col_m3, col_m4 = st.columns(2)

    with col_m3:
        # Feature importance
        fig_fi = px.bar(feat_imp.head(8), x='Importance', y='Feature', orientation='h',
                        title="Feature Importance (Best Tree Model)",
                        color='Importance', color_continuous_scale='Plasma', template='plotly_dark')
        fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                             font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                             xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                             yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                             margin=dict(l=10, r=10, t=50, b=10))
        fig_fi.update_traces(marker_line_width=0)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_m4:
        # Route clustering
        st.markdown("#### Route Performance Clusters")
        st.info("""
        **Cluster Interpretation:**
        Routes are grouped by Distance, Lead Time, Profit Margin, and Cost.
        — Low lead time + high margin = **Efficient routes**
        — High lead time + low margin = **Problem routes** needing reallocation
        """)
        cluster_display = cluster_profiles.copy()
        cluster_display.index = [f"Cluster {i}" for i in cluster_display.index]
        st.dataframe(cluster_display.rename(columns={
            'Distance_Miles': 'Avg Distance (mi)',
            'Lead Time':      'Avg Lead Time (d)',
            'Profit_Margin':  'Avg Profit Margin',
            'Cost':           'Avg Cost'
        }), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FACTORY OPTIMIZATION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🏭 Factory Optimization Simulator")
    st.markdown(f"Simulating **{sel_product}** → Region: **{sel_region}** | Ship Mode: **{sel_ship}** | Priority α: **{alpha}**")

    sim_result = simulate(sel_product, sel_region, sel_ship, df, trained, best_model, scaler, le_dict, feat_cols, alpha)

    current_lt  = sim_result[sim_result['Is_Current']]['Pred_Lead_Time'].values
    current_lt  = float(current_lt[0]) if len(current_lt) > 0 else 0
    best_factory = sim_result.iloc[0]['Factory']
    best_lt      = sim_result.iloc[0]['Pred_Lead_Time']
    improvement  = sim_result.iloc[0]['LT_Reduction_Pct']

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Factory",        PRODUCT_FACTORY.get(sel_product, "N/A"))
    k2.metric("Current Pred Lead Time", f"{current_lt:.1f} days")
    k3.metric("Optimal Factory",        best_factory)
    k4.metric("Lead Time Improvement",  f"{improvement:+.1f}%",
              delta=f"{best_lt:.1f} days optimal")

    st.markdown("---")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        colors = ['#f5c842' if r['Is_Current'] else '#d4a017' for _, r in sim_result.iterrows()]
        fig_sim = px.bar(sim_result, x='Factory', y='Pred_Lead_Time',
                         color='Composite_Score', color_continuous_scale='RdYlGn',
                         text=sim_result['Pred_Lead_Time'].map('{:.1f}d'.format),
                         title=f"Predicted Lead Time per Factory — {sel_product}",
                         template='plotly_dark')
        fig_sim.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                              font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                              xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                              yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Predicted Lead Time (days)'),
                              margin=dict(l=10, r=10, t=50, b=10))
        fig_sim.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_sim, use_container_width=True)

    with col_s2:
        fig_comp = px.scatter(sim_result, x='Pred_Lead_Time', y='Profit_Margin_Pct',
                              size='Composite_Score', color='Factory', text='Factory',
                              title="Lead Time vs Profit Margin Trade-off",
                              template='plotly_dark',
                              color_discrete_sequence=['#d4a017','#7b3f00','#f5c842','#a0522d','#c8860a'])
        fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                               font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                               xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Predicted Lead Time (days)'),
                               yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Profit Margin (%)'),
                               legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=10, r=10, t=50, b=10))
        fig_comp.update_traces(textfont=dict(color='white', size=9))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Factory Comparison Table")
    display_sim = sim_result.copy()
    display_sim['Status'] = display_sim['Is_Current'].map({True: '📍 Current', False: '🔄 Alternative'})
    st.dataframe(display_sim[['Factory', 'Status', 'Pred_Lead_Time', 'Lead_Time_Delta',
                               'LT_Reduction_Pct', 'Distance_Miles',
                               'Profit_Margin_Pct', 'Composite_Score']].rename(columns={
        'Pred_Lead_Time':    'Pred Lead Time (d)',
        'Lead_Time_Delta':   'LT Delta (d)',
        'LT_Reduction_Pct':  'LT Reduction %',
        'Distance_Miles':    'Distance (mi)',
        'Profit_Margin_Pct': 'Profit Margin %',
        'Composite_Score':   'Composite Score',
    }), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — WHAT-IF SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔄 What-If Scenario Analysis")
    st.markdown("Compare current factory assignment against the model's recommended alternative.")

    current_factory_wif = PRODUCT_FACTORY.get(sel_product)
    sim_wif = simulate(sel_product, sel_region, sel_ship, df, trained, best_model, scaler, le_dict, feat_cols, alpha)
    recommended_wif = sim_wif.iloc[0]
    current_wif     = sim_wif[sim_wif['Is_Current']].iloc[0] if sim_wif['Is_Current'].any() else sim_wif.iloc[-1]

    # Scenario cards
    col_cur, col_vs, col_rec = st.columns([5, 1, 5])

    with col_cur:
        st.markdown(f"""
        <div style="background:rgba(255,140,0,0.08);border:1px solid rgba(255,140,0,0.3);
                    border-radius:16px;padding:24px;text-align:center;">
            <p style="color:#d4a017;font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0">
                CURRENT ASSIGNMENT</p>
            <p style="color:#ffffff;font-size:1.8rem;font-weight:800;margin:8px 0">{current_factory_wif}</p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Lead Time: <b style="color:#fff">{current_wif['Pred_Lead_Time']:.1f} days</b></p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Distance: <b style="color:#fff">{current_wif['Distance_Miles']:.0f} mi</b></p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Profit Margin: <b style="color:#fff">{current_wif['Profit_Margin_Pct']:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_vs:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:160px;'>"
                    "<p style='color:#d4a017;font-size:1.5rem;font-weight:800'>→</p></div>", unsafe_allow_html=True)

    with col_rec:
        change_flag = recommended_wif['Factory'] != current_factory_wif
        border_col  = '#f5c842' if change_flag else '#a08060'
        label       = 'RECOMMENDED REASSIGNMENT' if change_flag else 'NO CHANGE NEEDED'
        st.markdown(f"""
        <div style="background:rgba(0,255,136,0.08);border:1px solid {border_col};
                    border-radius:16px;padding:24px;text-align:center;">
            <p style="color:{border_col};font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0">
                {label}</p>
            <p style="color:#ffffff;font-size:1.8rem;font-weight:800;margin:8px 0">{recommended_wif['Factory']}</p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Lead Time: <b style="color:#fff">{recommended_wif['Pred_Lead_Time']:.1f} days</b></p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Distance: <b style="color:#fff">{recommended_wif['Distance_Miles']:.0f} mi</b></p>
            <p style="color:#a0aec0;font-size:0.9rem;margin:4px 0">Profit Margin: <b style="color:#fff">{recommended_wif['Profit_Margin_Pct']:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Waterfall chart — improvement breakdown
    delta_lt   = current_wif['Pred_Lead_Time'] - recommended_wif['Pred_Lead_Time']
    delta_dist = current_wif['Distance_Miles'] - recommended_wif['Distance_Miles']
    delta_pm   = recommended_wif['Profit_Margin_Pct'] - current_wif['Profit_Margin_Pct']

    fig_wf = go.Figure(go.Waterfall(
        orientation='v',
        measure=['absolute', 'relative', 'total'],
        x=['Current Lead Time', 'Reduction from Reallocation', 'Optimal Lead Time'],
        y=[current_wif['Pred_Lead_Time'], -delta_lt, recommended_wif['Pred_Lead_Time']],
        connector=dict(line=dict(color='rgba(255,255,255,0.2)')),
        decreasing=dict(marker=dict(color='#c8860a')),
        increasing=dict(marker=dict(color='#7b3f00')),
        totals=dict(marker=dict(color='#d4a017')),
        text=[f"{current_wif['Pred_Lead_Time']:.1f}d",
              f"{-delta_lt:+.1f}d",
              f"{recommended_wif['Pred_Lead_Time']:.1f}d"],
        textposition='outside',
    ))
    fig_wf.update_layout(
        title=dict(text='Lead Time Improvement Waterfall', font=dict(color='#ffffff', size=15), x=0.01),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
        font=dict(color='#a0aec0'), template='plotly_dark',
        yaxis=dict(title='Days', gridcolor='rgba(255,255,255,0.05)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")

    # Historical performance at recommended factory
    hist_data = df[
        (df['Product Name'] == sel_product) |
        (df['Factory'].isin([current_factory_wif, recommended_wif['Factory']]))
    ].copy()

    if len(hist_data) > 0:
        st.subheader("Historical Lead Time — Current vs Recommended Factory")
        hist_compare = hist_data[
            hist_data['Factory'].isin([current_factory_wif, recommended_wif['Factory']])
        ].groupby(['Factory', pd.Grouper(key='Order Date', freq='ME')])['Lead Time'].mean().reset_index()
        fig_hist = px.line(hist_compare, x='Order Date', y='Lead Time', color='Factory',
                           color_discrete_map={current_factory_wif: '#d4a017',
                                               recommended_wif['Factory']: '#f5c842'},
                           title="Monthly Avg Lead Time Trend", template='plotly_dark')
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                               font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                               xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                               yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Avg Lead Time (days)'),
                               legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=10, r=10, t=50, b=10))
        fig_hist.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — RECOMMENDATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🏆 Factory Reassignment Recommendations")
    st.markdown(f"Bulk simulation across all products  |  Region: **{sel_region}**  |  Ship Mode: **{sel_ship}**  |  α = **{alpha}**")

    with st.spinner("Running simulation across all products..."):
        all_recs = generate_all_recommendations(
            df, trained, best_model, scaler, le_dict, feat_cols, alpha, sel_region, sel_ship
        )

    changes_needed = all_recs[all_recs['Change Needed']]
    no_change      = all_recs[~all_recs['Change Needed']]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Products Analysed",      len(all_recs))
    r2.metric("Reassignments Suggested", len(changes_needed))
    r3.metric("Already Optimal",         len(no_change))
    r4.metric("Avg Lead Time Reduction", f"{all_recs['Lead Time Reduction %'].mean():.1f}%")

    st.markdown("---")

    # Top recommendations
    st.subheader("Top Reassignment Recommendations")
    top_recs = changes_needed.sort_values('Composite Score', ascending=False).head(10)

    if len(top_recs) > 0:
        fig_rec = px.bar(top_recs, x='Product', y='Lead Time Reduction %',
                         color='Composite Score', color_continuous_scale='RdYlGn',
                         text=top_recs['Lead Time Reduction %'].map('{:.1f}%'.format),
                         title="Top Products by Lead Time Reduction Potential",
                         template='plotly_dark')
        fig_rec.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                              font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=15), x=0.01),
                              xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=-30),
                              yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='LT Reduction (%)'),
                              margin=dict(l=10, r=10, t=50, b=80))
        fig_rec.update_traces(marker_line_width=0, textfont=dict(color='white'))
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.success("All products are already assigned to their optimal factories for the selected region and ship mode.")

    st.markdown("---")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("Reassignment Recommendations")
        if len(changes_needed) > 0:
            st.dataframe(changes_needed[[
                'Product', 'Division', 'Current Factory', 'Recommended Factory',
                'Current Lead Time', 'Optimal Lead Time', 'Lead Time Reduction %',
                'Profit Margin %', 'Composite Score'
            ]].sort_values('Composite Score', ascending=False), use_container_width=True)
        else:
            st.info("No reassignments recommended.")

    with col_r2:
        st.subheader("Already Optimal Assignments")
        if len(no_change) > 0:
            st.dataframe(no_change[[
                'Product', 'Current Factory', 'Optimal Lead Time', 'Profit Margin %', 'Composite Score'
            ]], use_container_width=True)
        else:
            st.info("No products are currently optimally assigned.")

    st.markdown("---")
    csv_recs = all_recs.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Full Recommendations Report", data=csv_recs,
                       file_name="nassau_factory_recommendations.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — RISK & IMPACT PANEL
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("⚠️ Risk & Impact Assessment")

    # High-risk routes: high lead time AND low profit margin
    lt_thresh  = df['Lead Time'].quantile(0.75)
    pm_thresh  = df['Profit_Margin'].quantile(0.25)
    high_risk  = df[(df['Lead Time'] >= lt_thresh) & (df['Profit_Margin'] <= pm_thresh)]

    w1, w2, w3 = st.columns(3)
    w1.metric("High-Risk Orders",      f"{len(high_risk):,}", help="High lead time + low profit margin.")
    w2.metric("High-Risk Revenue",     f"${high_risk['Sales'].sum():,.0f}")
    w3.metric("High-Risk Lost Profit", f"${high_risk['Gross Profit'].sum():,.0f}")

    st.markdown("---")

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        # Risk heatmap: factory × region
        risk_heat = df.groupby(['Factory', 'Region'])['Lead Time'].mean().reset_index()
        risk_pivot = risk_heat.pivot(index='Factory', columns='Region', values='Lead Time').fillna(0)
        fig_rh = px.imshow(risk_pivot, text_auto='.1f', color_continuous_scale='RdYlGn_r',
                           title="Avg Lead Time Heatmap: Factory × Region",
                           template='plotly_dark', aspect='auto')
        fig_rh.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'),
                             title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                             margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_rh, use_container_width=True)

    with col_w2:
        # Profit sensitivity scatter
        fig_ps = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                            x='Lead Time', y='Profit_Margin',
                            color='Factory', opacity=0.5,
                            color_discrete_sequence=['#d4a017','#7b3f00','#f5c842','#a0522d','#c8860a'],
                            title="Lead Time vs Profit Margin (Sensitivity Cloud)",
                            template='plotly_dark')
        fig_ps.add_hline(y=float(pm_thresh), line_dash='dot', line_color='#c8860a',
                         annotation_text='Low Margin Threshold', annotation_font_color='#c8860a')
        fig_ps.add_vline(x=float(lt_thresh), line_dash='dot', line_color='#c8860a',
                         annotation_text='High LT Threshold', annotation_font_color='#c8860a')
        fig_ps.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)',
                             font=dict(color='#a0aec0'), title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                             xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Lead Time (days)'),
                             yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Profit Margin'),
                             legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_ps, use_container_width=True)

    st.markdown("---")

    # Congested region–product combinations
    st.subheader("Congested Region–Product Combinations")
    congested = df.groupby(['Region', 'Product Name']).agg(
        Avg_Lead_Time=('Lead Time', 'mean'),
        Orders=('Order ID', 'count'),
        Avg_Profit_Margin=('Profit_Margin', 'mean')
    ).reset_index()
    congested['Risk_Score'] = (congested['Avg_Lead_Time'] / congested['Avg_Lead_Time'].max() * 50 +
                               (1 - congested['Avg_Profit_Margin'].fillna(0).clip(0, 1)) * 50).round(1)
    congested = congested.sort_values('Risk_Score', ascending=False)

    fig_cong = px.treemap(congested.head(30),
                          path=['Region', 'Product Name'], values='Orders',
                          color='Risk_Score', color_continuous_scale='RdYlGn_r',
                          title="Top 30 Congested Region–Product Pairs (by Risk Score)",
                          template='plotly_dark')
    fig_cong.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#a0aec0'),
                           title=dict(font=dict(color='#ffffff', size=14), x=0.01),
                           margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_cong, use_container_width=True)

    st.markdown("---")

    # Profit impact alerts
    st.subheader("Profit Impact Alerts — Reassignment Risk")
    if 'all_recs' in dir() and len(all_recs) > 0:
        risk_recs = all_recs[all_recs['Change Needed'] & (all_recs['Profit Margin %'] < 10)].copy()
        if len(risk_recs) > 0:
            st.warning(f"⚠️ {len(risk_recs)} products have recommended reassignments with **profit margin below 10%** — validate before execution.")
            st.dataframe(risk_recs[['Product', 'Current Factory', 'Recommended Factory',
                                     'Lead Time Reduction %', 'Profit Margin %', 'Composite Score']], use_container_width=True)
        else:
            st.success("✅ All recommended reassignments maintain healthy profit margins (≥10%).")

    csv_risk = high_risk[['Order ID', 'Product Name', 'Factory', 'Region',
                           'Lead Time', 'Sales', 'Gross Profit', 'Ship Mode']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export High-Risk Orders", data=csv_risk,
                       file_name="nassau_high_risk_orders.csv", mime="text/csv")
