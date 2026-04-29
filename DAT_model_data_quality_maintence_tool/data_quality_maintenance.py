"""
╔══════════════════════════════════════════════════════════════╗
║        DATA QUALITY MAINTENANCE TOOL — Streamlit             ║
║   Tab 1: Data Quality Analyzer  |  Tab 2: Data Preparation   ║
╚══════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install streamlit pandas numpy plotly scipy openpyxl xlrd scikit-learn

Run:
    streamlit run data_quality_maintenance.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import io
import json
import re
from datetime import datetime

from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler,
    RobustScaler, MaxAbsScaler
)

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Quality Maintenance Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0b0f1a; color: #e8eef8; }

  /* Header */
  .dqa-header {
    background: linear-gradient(135deg, #0f1525 0%, #131929 100%);
    border: 1px solid #243050;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .dqa-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 0%, rgba(79,142,247,.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .dqa-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #e8eef8;
    margin: 0 0 6px 0;
  }
  .dqa-header p { color: #6b7fa3; font-size: 14px; margin: 0; }

  /* KPI cards */
  .kpi-card {
    background: #1a2235;
    border: 1px solid #243050;
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
    min-width: 130px;
  }
  .kpi-val { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 700; line-height: 1.1; }
  .kpi-lbl { font-size: 11px; color: #6b7fa3; margin-top: 4px; }
  .kpi-badge {
    display: inline-block;
    margin-top: 6px;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
  }
  .good { color: #22d3a5; }
  .warn { color: #f5a623; }
  .bad  { color: #f05a5a; }
  .badge-good { background: rgba(34,211,165,.12); color: #22d3a5; }
  .badge-warn { background: rgba(245,166,35,.12);  color: #f5a623; }
  .badge-bad  { background: rgba(240,90,90,.12);   color: #f05a5a; }

  /* Section headers */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #6b7fa3;
    margin: 24px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #243050;
  }

  /* Issue cards */
  .issue-crit { border-left: 3px solid #f05a5a; background: rgba(240,90,90,.05); }
  .issue-warn { border-left: 3px solid #f5a623; background: rgba(245,166,35,.05); }
  .issue-info { border-left: 3px solid #4f8ef7; background: rgba(79,142,247,.05); }
  .issue-ok   { border-left: 3px solid #22d3a5; background: rgba(34,211,165,.05); }
  .issue-item {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid #243050;
  }
  .issue-title { font-weight: 500; color: #e8eef8; font-size: 14px; margin-bottom: 4px; }
  .issue-desc  { color: #6b7fa3; font-size: 12px; line-height: 1.6; }
  .issue-fix {
    display: inline-block;
    margin-top: 8px;
    font-size: 11px;
    color: #4f8ef7;
    font-family: 'DM Mono', monospace;
    background: rgba(79,142,247,.1);
    padding: 3px 10px;
    border-radius: 4px;
  }

  /* Reco cards */
  .reco-card {
    background: #1a2235;
    border: 1px solid #243050;
    border-radius: 12px;
    padding: 18px;
    height: 100%;
  }
  .reco-num  { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 700; color: #243050; line-height: 1; margin-bottom: 10px; }
  .reco-title{ font-weight: 500; font-size: 14px; color: #e8eef8; margin-bottom: 6px; }
  .reco-desc { font-size: 12px; color: #6b7fa3; line-height: 1.65; }
  .reco-tag  {
    display: inline-block;
    margin-top: 10px;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    background: rgba(79,142,247,.1);
    color: #4f8ef7;
  }

  /* Grade badge */
  .grade-A { background: rgba(34,211,165,.15); color: #22d3a5; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .grade-B { background: rgba(245,166,35,.15);  color: #f5a623; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .grade-C { background: rgba(240,90,90,.15);   color: #f05a5a; padding: 4px 14px; border-radius: 20px; font-size: 12px; font-weight: 600; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #131929; border-radius: 10px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; color: #6b7fa3; border-radius: 7px; font-size: 13px; padding: 8px 20px; }
  .stTabs [aria-selected="true"] { background: #1a2235 !important; color: #4f8ef7 !important; }

  /* Metric */
  [data-testid="stMetric"] { background: #1a2235; border: 1px solid #243050; border-radius: 12px; padding: 16px; }
  [data-testid="stMetricLabel"] { color: #6b7fa3 !important; font-size: 11px !important; }
  [data-testid="stMetricValue"] { color: #e8eef8 !important; font-family: 'Syne', sans-serif !important; }

  /* Dataframe */
  .stDataFrame { background: #131929 !important; }
  thead tr th { background: #1a2235 !important; color: #6b7fa3 !important; }

  /* Upload */
  [data-testid="stFileUploader"] > div { background: #131929; border: 1.5px dashed #243050; border-radius: 14px; padding: 20px; }

  div[data-testid="column"] > div { background: transparent !important; }

  /* Buttons */
  .stButton > button {
    background: #1a2235;
    border: 1px solid #243050;
    color: #e8eef8;
    border-radius: 8px;
    font-size: 13px;
  }
  .stButton > button:hover {
    background: #243050;
    border-color: #4f8ef7;
    color: #4f8ef7;
  }

  /* Select boxes */
  .stSelectbox > div > div { background: #131929 !important; border-color: #243050 !important; }
  .stMultiSelect > div > div { background: #131929 !important; border-color: #243050 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SHARED: LOAD DATA
# ═══════════════════════════════════════════════════════════════

MISSING_MARKERS = ['', 'null', 'NULL', 'none', 'None', 'NONE', 'na', 'NA', 'N/A',
                   'n/a', 'NaN', 'nan', '#N/A', '-', '--', '?', 'missing', 'MISSING']

def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, na_values=MISSING_MARKERS, keep_default_na=True)
    elif name.endswith('.tsv') or name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, sep='\t', na_values=MISSING_MARKERS, keep_default_na=True)
    elif name.endswith('.xlsx') or name.endswith('.xls'):
        df = pd.read_excel(uploaded_file, na_values=MISSING_MARKERS)
    elif name.endswith('.json'):
        content = json.load(uploaded_file)
        df = pd.DataFrame(content) if isinstance(content, list) else pd.json_normalize(content)
    else:
        df = pd.read_csv(uploaded_file, na_values=MISSING_MARKERS)
    return df


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def infer_col_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'empty'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if series.dtype == object:
        sample = non_null.head(20).astype(str)
        dt_hits = sample.apply(lambda v: bool(re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', v)))
        if dt_hits.mean() > 0.7:
            return 'datetime'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    coerced = pd.to_numeric(non_null, errors='coerce')
    if coerced.notna().mean() > 0.85:
        return 'numeric'
    uniq_ratio = non_null.nunique() / len(non_null)
    if uniq_ratio < 0.40 and non_null.nunique() <= 50:
        return 'categorical'
    return 'text'


def detect_outliers_iqr(series: pd.Series):
    nums = pd.to_numeric(series, errors='coerce').dropna()
    if len(nums) < 4:
        return pd.Series([], dtype=float), 0, None, None
    q1, q3 = nums.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series([], dtype=float), 0, None, None
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = nums[(nums < lo) | (nums > hi)]
    return outliers, len(outliers), lo, hi


def detect_outliers_zscore(series: pd.Series, threshold=3.0):
    nums = pd.to_numeric(series, errors='coerce').dropna()
    if len(nums) < 4:
        return 0
    z = np.abs(stats.zscore(nums))
    return int((z > threshold).sum())


def type_inconsistency(series: pd.Series, inferred_type: str) -> int:
    if inferred_type not in ('numeric', 'categorical', 'text'):
        return 0
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return 0
    num_mask = non_null.apply(lambda v: bool(re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$', v.strip())))
    num_count = num_mask.sum()
    str_count = (~num_mask).sum()
    if num_count > 0 and str_count > 0:
        minority = min(num_count, str_count)
        majority = max(num_count, str_count)
        if minority / majority > 0.05:
            return int(minority)
    return 0


def analyze_dataset(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    col_stats = {}

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        pct_missing = round(missing / n_rows * 100, 2)
        non_null = series.dropna()
        inferred = infer_col_type(series)
        unique_count = int(non_null.nunique())
        pct_unique = round(unique_count / n_rows * 100, 2) if n_rows > 0 else 0

        outlier_vals, outlier_count, out_lo, out_hi = detect_outliers_iqr(series)
        zscore_outliers = detect_outliers_zscore(series)
        type_issues = type_inconsistency(series, inferred)

        top_values = {}
        if inferred in ('categorical', 'text') and unique_count > 0:
            vc = non_null.astype(str).value_counts().head(8)
            top_values = vc.to_dict()

        num_stats = {}
        if inferred == 'numeric':
            nums = pd.to_numeric(series, errors='coerce').dropna()
            if len(nums) > 0:
                num_stats = {
                    'mean': round(float(nums.mean()), 4),
                    'median': round(float(nums.median()), 4),
                    'std': round(float(nums.std()), 4),
                    'min': round(float(nums.min()), 4),
                    'max': round(float(nums.max()), 4),
                    'skewness': round(float(nums.skew()), 4),
                }

        col_stats[col] = {
            'type': inferred,
            'missing': missing,
            'pct_missing': pct_missing,
            'unique': unique_count,
            'pct_unique': pct_unique,
            'outliers_iqr': outlier_count,
            'outliers_zscore': zscore_outliers,
            'out_lo': out_lo,
            'out_hi': out_hi,
            'type_issues': type_issues,
            'top_values': top_values,
            'num_stats': num_stats,
        }

    dup_mask = df.duplicated(keep='first')
    dup_count = int(dup_mask.sum())
    dup_indices = df.index[dup_mask].tolist()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    num_cols = [c for c in df.columns if col_stats[c]['type'] == 'numeric']
    high_corr_pairs = []
    if len(num_cols) >= 2:
        num_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
        corr = num_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack()
        strong = pairs[pairs > 0.95]
        high_corr_pairs = [(a, b, round(v, 3)) for (a, b), v in strong.items()][:5]

    total_cells = n_rows * n_cols if n_cols > 0 else 1
    total_missing = sum(s['missing'] for s in col_stats.values())
    total_outliers = sum(s['outliers_iqr'] for s in col_stats.values())
    total_type_issues = sum(s['type_issues'] for s in col_stats.values())

    miss_pen  = min(40, (total_missing / total_cells) * 100 * 2)
    dup_pen   = min(20, (dup_count / n_rows) * 100 * 1.5) if n_rows > 0 else 0
    out_pen   = min(20, (total_outliers / total_cells) * 100 * 3)
    type_pen  = min(20, (total_type_issues / total_cells) * 100 * 3)
    score = max(0, round(100 - miss_pen - dup_pen - out_pen - type_pen))

    return {
        'n_rows': n_rows, 'n_cols': n_cols, 'col_stats': col_stats,
        'dup_count': dup_count, 'dup_indices': dup_indices, 'dup_mask': dup_mask,
        'constant_cols': constant_cols, 'high_corr_pairs': high_corr_pairs,
        'total_missing': total_missing, 'total_outliers': total_outliers,
        'total_type_issues': total_type_issues, 'score': score, 'total_cells': total_cells,
    }


def get_faulty_rows(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    col_stats = result['col_stats']
    issues_per_row = {}

    for col in df.columns:
        s = col_stats[col]
        for idx in df.index[df[col].isna()]:
            issues_per_row.setdefault(idx, []).append(f"Missing @ {col}")
        if s['type'] == 'numeric' and s['out_lo'] is not None:
            nums = pd.to_numeric(df[col], errors='coerce')
            for idx in nums[(nums < s['out_lo']) | (nums > s['out_hi'])].index:
                issues_per_row.setdefault(idx, []).append(f"Outlier @ {col}")
        if s['type_issues'] > 0:
            non_null = df[col].dropna().astype(str)
            num_mask = non_null.apply(lambda v: bool(re.match(r'^-?\d+(\.\d+)?$', v.strip())))
            minority_type = 'numeric' if num_mask.sum() < (~num_mask).sum() else 'text'
            bad_idx = non_null[num_mask].index if minority_type == 'numeric' else non_null[~num_mask].index
            for idx in bad_idx:
                issues_per_row.setdefault(idx, []).append(f"Type issue @ {col}")

    for idx in result['dup_indices']:
        issues_per_row.setdefault(idx, []).append("Duplicate row")

    if not issues_per_row:
        return pd.DataFrame()

    faulty_idx = list(issues_per_row.keys())[:200]
    faulty_df = df.loc[faulty_idx].copy()
    faulty_df.insert(0, '⚠ Issues', [' | '.join(issues_per_row[i]) for i in faulty_idx])
    faulty_df.index.name = 'Row #'
    return faulty_df


# ─── CHART BUILDERS ───────────────────────────────────────────

DARK_BG  = 'rgba(0,0,0,0)'
GRID_CLR = 'rgba(36,48,80,0.6)'
TEXT_CLR = '#6b7fa3'
ACCENT   = '#4f8ef7'

def chart_layout(title=''):
    return dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(family='DM Sans', color=TEXT_CLR, size=11),
        title=dict(text=title, font=dict(family='Syne', size=13, color='#e8eef8'), x=0),
        margin=dict(l=10, r=10, t=35, b=10),
    )


def build_score_gauge(score: int) -> go.Figure:
    color = '#22d3a5' if score >= 75 else '#f5a623' if score >= 50 else '#f05a5a'
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=score,
        number=dict(suffix='%', font=dict(family='Syne', size=42, color='#e8eef8')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_CLR, tickfont=dict(size=10), nticks=6),
            bar=dict(color=color, thickness=0.28),
            bgcolor='#243050', borderwidth=0,
            steps=[
                dict(range=[0, 50],  color='rgba(240,90,90,.08)'),
                dict(range=[50, 75], color='rgba(245,166,35,.08)'),
                dict(range=[75,100], color='rgba(34,211,165,.08)'),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(**chart_layout(), height=220, showlegend=False)
    return fig


def build_missing_bar(col_stats: dict) -> go.Figure:
    cols_with_missing = {c: s for c, s in col_stats.items() if s['missing'] > 0}
    if not cols_with_missing:
        fig = go.Figure()
        fig.add_annotation(text='No missing values ✓', x=0.5, y=0.5,
                           showarrow=False, font=dict(color='#22d3a5', size=16))
        fig.update_layout(**chart_layout('Missing Values per Column'), height=280, showlegend=False)
        return fig
    sorted_cols = sorted(cols_with_missing.items(), key=lambda x: -x[1]['pct_missing'])
    labels = [c for c, _ in sorted_cols]
    values = [s['pct_missing'] for _, s in sorted_cols]
    colors = ['#f05a5a' if v > 50 else '#f5a623' if v > 20 else ACCENT for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, marker_line_width=0,
        text=[f'{v}%' for v in values], textposition='outside',
        textfont=dict(size=10, color='#e8eef8'),
    ))
    fig.update_layout(
        **chart_layout('Missing Values per Column (%)'),
        height=300, showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_CLR, range=[0, min(max(values) * 1.25, 115)], title='% Missing'),
    )
    return fig


def build_health_bar(col_stats: dict) -> go.Figure:
    cols = list(col_stats.keys())[:20]
    values = [round(100 - col_stats[c]['pct_missing'], 1) for c in cols]
    colors = ['#22d3a5' if v >= 90 else '#f5a623' if v >= 70 else '#f05a5a' for v in values]
    fig = go.Figure(go.Bar(
        x=cols, y=values, marker_color=colors, marker_line_width=0,
        text=[f'{v}%' for v in values], textposition='outside',
        textfont=dict(size=9, color='#e8eef8'),
    ))
    fig.update_layout(
        **chart_layout('Column Completeness (%)'),
        height=300, showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_CLR, range=[0, 115], title='% Complete'),
    )
    return fig


def build_issue_donut(result: dict) -> go.Figure:
    labels = ['Missing', 'Duplicates', 'Outliers', 'Type Issues']
    values = [result['total_missing'], result['dup_count'],
              result['total_outliers'], result['total_type_issues']]
    colors = ['#f05a5a', '#a78bfa', '#f5a623', ACCENT]
    total = sum(values)
    if total == 0:
        fig = go.Figure()
        fig.add_annotation(text='No issues found ✓', x=0.5, y=0.5,
                           showarrow=False, font=dict(color='#22d3a5', size=16))
        fig.update_layout(**chart_layout('Issue Breakdown'), height=280, showlegend=False)
        return fig
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#131929', width=2)),
        hole=0.62, textinfo='label+percent', textfont=dict(size=10),
        hovertemplate='%{label}: %{value}<extra></extra>',
    ))
    fig.add_annotation(text=f'<b>{total}</b><br>issues', x=0.5, y=0.5,
                       font=dict(size=13, color='#e8eef8', family='Syne'), showarrow=False)
    fig.update_layout(**chart_layout('Issue Breakdown'), height=280, showlegend=True,
                      legend=dict(orientation='h', y=-0.12, font=dict(size=10)))
    return fig


def build_type_pie(col_stats: dict) -> go.Figure:
    from collections import Counter
    counts = Counter(s['type'] for s in col_stats.values())
    labels, values = zip(*counts.items()) if counts else ([], [])
    colors_map = {'numeric':'#4f8ef7','categorical':'#a78bfa','datetime':'#22d3a5','text':'#f5a623','empty':'#f05a5a'}
    colors = [colors_map.get(l, '#6b7fa3') for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color='#131929', width=2)),
        hole=0.55, textinfo='label+percent', textfont=dict(size=10),
    ))
    fig.update_layout(**chart_layout('Column Type Distribution'), height=280, showlegend=True,
                      legend=dict(orientation='h', y=-0.12, font=dict(size=10)))
    return fig


def build_outlier_bar(col_stats: dict) -> go.Figure:
    out_cols = {c: s['outliers_iqr'] for c, s in col_stats.items()
                if s['type'] == 'numeric' and s['outliers_iqr'] > 0}
    if not out_cols:
        fig = go.Figure()
        fig.add_annotation(text='No outliers detected ✓', x=0.5, y=0.5,
                           showarrow=False, font=dict(color='#22d3a5', size=16))
        fig.update_layout(**chart_layout('Outlier Count per Column'), height=260, showlegend=False)
        return fig
    sorted_oc = sorted(out_cols.items(), key=lambda x: -x[1])
    labels, values = zip(*sorted_oc)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color='#f5a623', marker_line_width=0,
        text=values, textposition='outside', textfont=dict(size=10, color='#e8eef8'),
    ))
    fig.update_layout(
        **chart_layout('Outlier Count per Column (IQR)'),
        height=260, showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-30),
        yaxis=dict(gridcolor=GRID_CLR, title='Outlier count'),
    )
    return fig


def build_heatmap(df: pd.DataFrame, col_stats: dict) -> go.Figure:
    num_cols = [c for c in df.columns if col_stats[c]['type'] == 'numeric'][:15]
    if len(num_cols) < 2:
        return None
    num_df = df[num_cols].apply(pd.to_numeric, errors='coerce')
    corr = num_df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,'#4f8ef7'],[0.5,'#131929'],[1,'#f05a5a']],
        zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate='%{text}', textfont=dict(size=9),
        hovertemplate='%{x} × %{y}: %{z}<extra></extra>',
        colorbar=dict(thickness=10, tickfont=dict(size=9, color=TEXT_CLR)),
    ))
    fig.update_layout(
        **chart_layout('Correlation Heatmap (Numeric Columns)'),
        height=max(300, len(num_cols) * 30), showlegend=False,
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def build_distribution_hist(df: pd.DataFrame, col: str) -> go.Figure:
    nums = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(nums) == 0:
        return None
    fig = go.Figure(go.Histogram(x=nums, nbinsx=30, marker_color=ACCENT, marker_line_width=0, opacity=0.85))
    q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    for bound, label in [(lo, 'IQR low'), (hi, 'IQR high')]:
        fig.add_vline(x=bound, line_dash='dash', line_color='#f05a5a',
                      annotation_text=label, annotation_font=dict(size=9, color='#f05a5a'))
    fig.update_layout(
        **chart_layout(f'Distribution: {col}'),
        height=220, showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=GRID_CLR, title='Count'),
        bargap=0.05,
    )
    return fig


def build_recommendations(result: dict, col_stats: dict) -> list:
    recos = []
    n = 1

    if result['total_missing'] > 0:
        high_miss = [c for c, s in col_stats.items() if s['pct_missing'] > 50]
        med_miss  = [c for c, s in col_stats.items() if 0 < s['pct_missing'] <= 50]
        desc = ''
        if high_miss:
            desc += f"Drop columns with >50% missing: {', '.join(high_miss[:3])}. "
        if med_miss:
            desc += f"Impute: use <b>median</b> for numeric, <b>mode</b> for categorical, or <b>KNNImputer</b> for complex patterns."
        recos.append({'num': f'{n:02d}', 'title': 'Handle Missing Values',
                      'desc': desc or 'Apply appropriate imputation strategies.',
                      'tag': 'sklearn.impute / pandas.fillna'})
        n += 1

    if result['dup_count'] > 0:
        recos.append({'num': f'{n:02d}', 'title': 'Remove Duplicate Rows',
                      'desc': f"{result['dup_count']} exact duplicates found. Use df.drop_duplicates(). Also consider fuzzy deduplication on key identifier columns.",
                      'tag': 'pandas.drop_duplicates'})
        n += 1

    if result['total_outliers'] > 0:
        recos.append({'num': f'{n:02d}', 'title': 'Treat Statistical Outliers',
                      'desc': 'Cap outliers using IQR Winsorization (robust) or Z-score filtering (|z|>3). For tree-based ML models outliers are less critical; for linear models use RobustScaler.',
                      'tag': 'scipy.stats.mstats.winsorize / RobustScaler'})
        n += 1

    if result['total_type_issues'] > 0:
        recos.append({'num': f'{n:02d}', 'title': 'Fix Type Inconsistencies',
                      'desc': 'Columns mixing numeric and text values indicate dirty data entry. Use pd.to_numeric(errors="coerce") to coerce, then review flagged rows manually.',
                      'tag': 'pd.to_numeric(errors="coerce")'})
        n += 1

    if result['constant_cols']:
        recos.append({'num': f'{n:02d}', 'title': 'Drop Constant Columns',
                      'desc': f"Columns with zero variance carry no information: {', '.join(result['constant_cols'][:5])}. Drop them before modeling.",
                      'tag': 'VarianceThreshold(threshold=0)'})
        n += 1

    if result['high_corr_pairs']:
        pairs_str = ', '.join([f'{a}↔{b} ({v})' for a, b, v in result['high_corr_pairs'][:3]])
        recos.append({'num': f'{n:02d}', 'title': 'Handle High Correlation',
                      'desc': f'Highly correlated pairs found: {pairs_str}. Consider dropping one of each pair to reduce multicollinearity, or apply PCA.',
                      'tag': 'sklearn.decomposition.PCA'})
        n += 1

    recos.append({'num': f'{n:02d}', 'title': 'Normalize & Encode Features',
                  'desc': 'Scale numeric features with StandardScaler or MinMaxScaler. Encode categoricals with OneHotEncoder (low cardinality) or TargetEncoder (high cardinality).',
                  'tag': 'sklearn.preprocessing'})
    n += 1

    recos.append({'num': f'{n:02d}', 'title': 'Validate Domain Rules',
                  'desc': 'Run business-logic checks: date ranges, value bounds, referential integrity. Automate ongoing validation with Great Expectations or custom assertion scripts.',
                  'tag': 'great_expectations / pandera'})
    return recos


def render_issue(icon, title, desc, fix, severity='warn'):
    st.markdown(f"""
    <div class="issue-item issue-{severity}">
      <div style="font-size:18px;margin-bottom:4px">{icon}</div>
      <div class="issue-title">{title}</div>
      <div class="issue-desc">{desc}</div>
      <span class="issue-fix">Fix: {fix}</span>
    </div>""", unsafe_allow_html=True)


def score_color(score):
    return '#22d3a5' if score >= 75 else '#f5a623' if score >= 50 else '#f05a5a'


def score_grade(score):
    if score >= 80:   return 'A', 'Good', 'grade-A'
    elif score >= 55: return 'B', 'Fair', 'grade-B'
    else:             return 'C', 'Poor', 'grade-C'


# ═══════════════════════════════════════════════════════════════
#  TAB 1 RENDER
# ═══════════════════════════════════════════════════════════════

def render_analyzer_tab(uploaded):
    if not uploaded:
        st.info("👆 Upload a dataset above to begin the quality analysis.")
        st.markdown("""
        <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap">
          <span style="background:#1a2235;border:1px solid #243050;border-radius:6px;padding:4px 12px;font-size:12px;font-family:'DM Mono',monospace;color:#4f8ef7">CSV</span>
          <span style="background:#1a2235;border:1px solid #243050;border-radius:6px;padding:4px 12px;font-size:12px;font-family:'DM Mono',monospace;color:#4f8ef7">TSV</span>
          <span style="background:#1a2235;border:1px solid #243050;border-radius:6px;padding:4px 12px;font-size:12px;font-family:'DM Mono',monospace;color:#4f8ef7">XLSX</span>
          <span style="background:#1a2235;border:1px solid #243050;border-radius:6px;padding:4px 12px;font-size:12px;font-family:'DM Mono',monospace;color:#4f8ef7">JSON</span>
        </div>""", unsafe_allow_html=True)
        return

    with st.spinner("Analyzing your dataset..."):
        try:
            uploaded.seek(0)
            df = load_data(uploaded)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            return

        if df.empty:
            st.warning("The file appears to be empty.")
            return

        result = analyze_dataset(df)

    col_stats = result['col_stats']
    score = result['score']
    grade, grade_label, grade_cls = score_grade(score)

    st.markdown(f"""
    <div style="font-size:12px;color:#6b7fa3;font-family:'DM Mono',monospace;margin-bottom:16px">
      📁 <b style="color:#e8eef8">{uploaded.name}</b> &nbsp;·&nbsp;
      {result['n_rows']:,} rows &nbsp;×&nbsp; {result['n_cols']} columns &nbsp;·&nbsp;
      {result['total_cells']:,} cells total &nbsp;·&nbsp;
      Analyzed at {datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

    # Score + KPIs
    st.markdown('<div class="section-title">Overall Quality Score</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([1, 2.8])

    with sc1:
        st.plotly_chart(build_score_gauge(score), use_container_width=True, config={'displayModeBar': False})
        st.markdown(f'<div style="text-align:center"><span class="{grade_cls}">Grade {grade} — {grade_label}</span></div>', unsafe_allow_html=True)

    with sc2:
        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)
        with k1: st.metric("Total Rows", f"{result['n_rows']:,}")
        with k2:
            miss_pct = round(result['total_missing'] / result['total_cells'] * 100, 1)
            st.metric("Missing Values", f"{result['total_missing']:,}", f"{miss_pct}% of cells")
        with k3:
            st.metric("Duplicate Rows", f"{result['dup_count']:,}",
                      f"{round(result['dup_count']/result['n_rows']*100,1)}% of data" if result['n_rows'] else "0%")
        with k4: st.metric("Outliers (IQR)", f"{result['total_outliers']:,}")
        with k5: st.metric("Type Issues", f"{result['total_type_issues']:,}")
        with k6:
            high_miss = len([c for c, s in col_stats.items() if s['pct_missing'] > 50])
            st.metric("High-miss Cols (>50%)", str(high_miss))

    # Column Profile
    st.markdown('<div class="section-title">Column Profile</div>', unsafe_allow_html=True)
    profile_rows = []
    for col, s in col_stats.items():
        status = "✓ Clean" if s['missing'] == 0 and s['outliers_iqr'] == 0 and s['type_issues'] == 0 else "⚠ Issues"
        profile_rows.append({
            'Column': col, 'Type': s['type'],
            '# Missing': s['missing'], '% Missing': f"{s['pct_missing']}%",
            'Unique': s['unique'], '% Unique': f"{s['pct_unique']}%",
            'Outliers (IQR)': s['outliers_iqr'], 'Type Issues': s['type_issues'], 'Status': status,
        })
    profile_df = pd.DataFrame(profile_rows)

    def style_status(val):
        if '✓' in str(val): return 'color: #22d3a5; font-weight:500'
        if '⚠' in str(val): return 'color: #f5a623; font-weight:500'
        return ''
    def style_missing(val):
        v = float(str(val).replace('%',''))
        if v > 50: return 'color: #f05a5a'
        if v > 20: return 'color: #f5a623'
        if v > 0:  return 'color: #4f8ef7'
        return 'color: #22d3a5'

    styled = (profile_df.style
        .applymap(style_status, subset=['Status'])
        .applymap(style_missing, subset=['% Missing']))
    st.dataframe(styled, use_container_width=True, height=min(400, 40 + len(profile_rows)*35))

    # Charts
    st.markdown('<div class="section-title">Visual Analytics</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1: st.plotly_chart(build_issue_donut(result), use_container_width=True, config={'displayModeBar': False})
    with ch2: st.plotly_chart(build_type_pie(col_stats), use_container_width=True, config={'displayModeBar': False})

    ch3, ch4 = st.columns(2)
    with ch3: st.plotly_chart(build_missing_bar(col_stats), use_container_width=True, config={'displayModeBar': False})
    with ch4: st.plotly_chart(build_health_bar(col_stats), use_container_width=True, config={'displayModeBar': False})

    st.plotly_chart(build_outlier_bar(col_stats), use_container_width=True, config={'displayModeBar': False})

    heatmap = build_heatmap(df, col_stats)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True, config={'displayModeBar': False})

    # Distribution Explorer
    num_cols = [c for c in df.columns if col_stats[c]['type'] == 'numeric']
    if num_cols:
        st.markdown('<div class="section-title">Distribution Explorer</div>', unsafe_allow_html=True)
        selected = st.selectbox("Select a numeric column to inspect its distribution:", num_cols, key="dist_sel")
        if selected:
            dist_fig = build_distribution_hist(df, selected)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})
            s = col_stats[selected]
            if s['num_stats']:
                ns = s['num_stats']
                d1,d2,d3,d4,d5,d6 = st.columns(6)
                for col_el, label, val in zip(
                    [d1,d2,d3,d4,d5,d6],
                    ['Mean','Median','Std Dev','Min','Max','Skewness'],
                    [ns.get('mean'),ns.get('median'),ns.get('std'),ns.get('min'),ns.get('max'),ns.get('skewness')]
                ):
                    with col_el: st.metric(label, val)

    # Categorical Distribution
    cat_cols = [c for c in df.columns if col_stats[c]['type'] == 'categorical']
    if cat_cols:
        st.markdown('<div class="section-title">Categorical Value Distribution</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Select a categorical column:", cat_cols, key="cat_sel")
        if cat_sel and col_stats[cat_sel]['top_values']:
            tv = col_stats[cat_sel]['top_values']
            labels_tv = list(tv.keys())
            values_tv = list(tv.values())
            total_tv = sum(values_tv)
            fig_tv = go.Figure(go.Bar(
                x=values_tv, y=labels_tv, orientation='h',
                marker_color=ACCENT, marker_line_width=0,
                text=[f'{v} ({round(v/total_tv*100,1)}%)' for v in values_tv],
                textposition='outside', textfont=dict(size=10, color='#e8eef8'),
            ))
            fig_tv.update_layout(
                **chart_layout(f'Top values in "{cat_sel}"'),
                height=max(200, len(labels_tv) * 32 + 60), showlegend=False,
                xaxis=dict(showgrid=False, title='Count'),
                yaxis=dict(showgrid=False, autorange='reversed'),
            )
            st.plotly_chart(fig_tv, use_container_width=True, config={'displayModeBar': False})

    # Issues
    st.markdown('<div class="section-title">Issues Found</div>', unsafe_allow_html=True)
    issues_found = False

    if result['total_missing'] > 0:
        issues_found = True
        worst = sorted(col_stats.items(), key=lambda x: -x[1]['pct_missing'])
        worst_str = ', '.join([f"{c} ({s['pct_missing']}%)" for c, s in worst[:3] if s['missing'] > 0])
        affected = sum(1 for s in col_stats.values() if s['missing'] > 0)
        render_issue('⚠️', f"{result['total_missing']:,} missing values across {affected} columns",
                     f"Most affected: {worst_str}",
                     "Impute with median/mode or KNNImputer; drop columns >50% missing", 'crit')

    if result['dup_count'] > 0:
        issues_found = True
        render_issue('⬦', f"{result['dup_count']:,} duplicate rows ({round(result['dup_count']/result['n_rows']*100,1)}% of data)",
                     "Exact duplicate rows detected. May skew model training and statistical analysis.",
                     "df.drop_duplicates(inplace=True)", 'warn')

    if result['total_outliers'] > 0:
        issues_found = True
        out_cols_str = ', '.join([f"{c}({s['outliers_iqr']})" for c, s in col_stats.items() if s['outliers_iqr'] > 0][:5])
        render_issue('◎', f"{result['total_outliers']:,} statistical outliers detected (IQR method)",
                     f"Found in: {out_cols_str}",
                     "Use IQR capping, Winsorization, or Z-score filtering (|z|>3)", 'warn')

    if result['total_type_issues'] > 0:
        issues_found = True
        type_cols = [c for c, s in col_stats.items() if s['type_issues'] > 0]
        render_issue('⚡', f"Type inconsistency in {len(type_cols)} column(s): {', '.join(type_cols[:3])}",
                     "Columns contain mixed numeric and text values — indicates dirty data entry or encoding issues.",
                     'pd.to_numeric(errors="coerce") + manual review', 'crit')

    if result['constant_cols']:
        issues_found = True
        render_issue('■', f"{len(result['constant_cols'])} constant column(s) with zero variance",
                     f"Columns: {', '.join(result['constant_cols'][:5])} — carry no information for ML.",
                     "df.drop(columns=[...]) or VarianceThreshold(threshold=0)", 'info')

    if result['high_corr_pairs']:
        issues_found = True
        pairs_str = ', '.join([f"{a}↔{b}({v})" for a, b, v in result['high_corr_pairs'][:3]])
        render_issue('◈', f"{len(result['high_corr_pairs'])} highly correlated column pair(s) (r>0.95)",
                     f"Pairs: {pairs_str}. Multicollinearity can hurt linear models.",
                     "Drop one of each pair or apply PCA", 'info')

    if not issues_found:
        render_issue('✓', 'No critical issues found',
                     'Dataset appears clean based on automated checks.',
                     'Still recommend manual domain-specific validation', 'ok')

    # Faulty Rows
    st.markdown('<div class="section-title">Faulty Rows Sample</div>', unsafe_allow_html=True)
    faulty_df = get_faulty_rows(df, result)

    if faulty_df.empty:
        st.success("✓ No faulty rows detected.")
    else:
        total_faulty = len(faulty_df)
        st.markdown(f'<span style="color:#6b7fa3;font-size:12px">Showing up to 50 of {total_faulty} faulty rows.</span>', unsafe_allow_html=True)

        def highlight_issues(val):
            v = str(val)
            if 'Missing' in v:   return 'background-color:rgba(240,90,90,.12);color:#f05a5a'
            if 'Outlier' in v:   return 'background-color:rgba(245,166,35,.12);color:#f5a623'
            if 'Duplicate' in v: return 'background-color:rgba(167,139,250,.12);color:#a78bfa'
            if 'Type' in v:      return 'background-color:rgba(79,142,247,.12);color:#4f8ef7'
            return ''

        show_df = faulty_df.head(50)
        styled_faulty = show_df.style.applymap(highlight_issues, subset=['⚠ Issues'])
        st.dataframe(styled_faulty, use_container_width=True, height=min(400, 50 + len(show_df)*35))

    # Recommendations
    st.markdown('<div class="section-title">Preprocessing Recommendations</div>', unsafe_allow_html=True)
    recos = build_recommendations(result, col_stats)
    r_cols = st.columns(3)
    for i, reco in enumerate(recos):
        with r_cols[i % 3]:
            st.markdown(f"""
            <div class="reco-card" style="margin-bottom:14px">
              <div class="reco-num">{reco['num']}</div>
              <div class="reco-title">{reco['title']}</div>
              <div class="reco-desc">{reco['desc']}</div>
              <span class="reco-tag">{reco['tag']}</span>
            </div>""", unsafe_allow_html=True)

    # Export
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        profile_csv = profile_df.to_csv(index=False)
        st.download_button("⬇ Download Column Profile (CSV)", data=profile_csv,
                           file_name=f"quality_profile_{uploaded.name}.csv", mime='text/csv')
    with ex2:
        if not faulty_df.empty:
            faulty_csv = faulty_df.to_csv()
            st.download_button("⬇ Download Faulty Rows (CSV)", data=faulty_csv,
                               file_name=f"faulty_rows_{uploaded.name}.csv", mime='text/csv')
        else:
            st.button("⬇ No faulty rows to export", disabled=True, key="no_faulty")
    with ex3:
        summary = {
            'file': uploaded.name,
            'analyzed_at': datetime.now().isoformat(),
            'quality_score': score, 'grade': grade,
            'n_rows': result['n_rows'], 'n_cols': result['n_cols'],
            'total_missing': result['total_missing'],
            'duplicate_rows': result['dup_count'],
            'total_outliers': result['total_outliers'],
            'type_issues': result['total_type_issues'],
            'constant_cols': result['constant_cols'],
            'high_corr_pairs': result['high_corr_pairs'],
            'column_stats': {
                c: {k: v for k, v in s.items() if k not in ('top_values','num_stats')}
                for c, s in col_stats.items()
            }
        }
        st.download_button("⬇ Download Summary (JSON)", data=json.dumps(summary, indent=2),
                           file_name=f"quality_summary_{uploaded.name}.json", mime='application/json')


# ═══════════════════════════════════════════════════════════════
#  TAB 2 RENDER — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def render_preparation_tab(uploaded):
    if not uploaded:
        st.info("👆 Upload a dataset above to begin data preparation.")
        return

    # Initialize session state for this file (keyed by name + size to detect re-uploads)
    file_key = f"prep_df_{uploaded.name}_{uploaded.size}"
    if file_key not in st.session_state:
        try:
            uploaded.seek(0)
            st.session_state[file_key] = load_data(uploaded)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            return

    df = st.session_state[file_key]

    # Current Data Preview
    st.markdown('<div class="section-title">Current Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=250)

    # Data Info
    st.markdown('<div class="section-title">Data Information</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Missing": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%',
        "Unique": df.nunique()
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("---")

    # ── STEP 1: Remove Duplicates ──────────────────────────────
    st.markdown('<div class="section-title">Step 1 — Remove Duplicate Rows</div>', unsafe_allow_html=True)
    dup_count = int(df.duplicated().sum())
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"""
        <div style="background:#1a2235;border:1px solid #243050;border-radius:10px;padding:14px 18px;">
          <span style="color:#6b7fa3;font-size:13px">Duplicate rows found: </span>
          <span style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;color:{'#f05a5a' if dup_count > 0 else '#22d3a5'}">
            {dup_count}
          </span>
        </div>""", unsafe_allow_html=True)
    with col_b:
        if st.button("🗑 Remove Duplicates", key="dup_btn", disabled=(dup_count == 0)):
            before = len(df)
            df = df.drop_duplicates()
            st.session_state[file_key] = df
            st.success(f"✅ Removed {before - len(df)} duplicate rows")
            st.rerun()

    st.markdown("---")

    # ── STEP 2: Missing Value Handling ──────────────────────────
    st.markdown('<div class="section-title">Step 2 — Missing Value Handling</div>', unsafe_allow_html=True)
    cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]

    if not cols_with_missing:
        st.success("✅ No missing values found in the dataset.")
    else:
        mv_map = {}
        mv_cols = st.columns(min(3, len(cols_with_missing)))
        for i, col in enumerate(cols_with_missing):
            with mv_cols[i % 3]:
                mv_map[col] = st.selectbox(
                    f"{col} ({df[col].isnull().sum()} missing)",
                    ["None", "Mean", "Median", "Mode", "Drop Column"],
                    key=f"mv_{col}"
                )
        if st.button("✅ Apply Missing Value Handling", key="mv_btn"):
            for col, method in mv_map.items():
                if method == "Mean" and df[col].dtype != 'object':
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "Median" and df[col].dtype != 'object':
                    df[col] = df[col].fillna(df[col].median())
                elif method == "Mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif method == "Drop Column":
                    df = df.drop(columns=[col])
            st.session_state[file_key] = df
            st.success("✅ Missing values handled")
            st.rerun()

    st.markdown("---")

    # ── STEP 3: Categorical Cleaning ────────────────────────────
    st.markdown('<div class="section-title">Step 3 — Categorical Cleaning</div>', unsafe_allow_html=True)
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if not obj_cols:
        st.info("No text/categorical columns found.")
    else:
        cat_cols = st.multiselect(
            "Select columns to clean (lowercase + strip whitespace):",
            obj_cols, key="clean_cols"
        )
        if st.button("✅ Apply Cleaning", key="clean_btn"):
            for col in cat_cols:
                df[col] = df[col].astype(str).str.lower().str.strip()
            st.session_state[file_key] = df
            st.success(f"✅ Cleaned {len(cat_cols)} column(s)")
            st.rerun()

    st.markdown("---")

    # ── STEP 4: Encoding ────────────────────────────────────────
    st.markdown('<div class="section-title">Step 4 — Encoding</div>', unsafe_allow_html=True)
    obj_cols_enc = df.select_dtypes(include='object').columns.tolist()
    if not obj_cols_enc:
        st.info("No categorical columns to encode.")
    else:
        enc_cols = st.multiselect("Select columns to encode:", obj_cols_enc, key="enc_cols")
        enc_method = st.selectbox("Encoding Method:", ["Label Encoding", "One Hot Encoding"], key="enc_method")
        if st.button("✅ Apply Encoding", key="enc_btn"):
            if enc_method == "Label Encoding":
                le = LabelEncoder()
                for col in enc_cols:
                    df[col] = le.fit_transform(df[col].astype(str))
            elif enc_method == "One Hot Encoding":
                df = pd.get_dummies(df, columns=enc_cols)
            st.session_state[file_key] = df
            st.success(f"✅ {enc_method} applied to {len(enc_cols)} column(s)")
            st.rerun()

    st.markdown("---")

    # ── STEP 5: Outlier Handling ─────────────────────────────────
    st.markdown('<div class="section-title">Step 5 — Outlier Handling</div>', unsafe_allow_html=True)
    num_cols_out = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if not num_cols_out:
        st.info("No numeric columns found.")
    else:
        out_cols = st.multiselect("Select numeric columns:", num_cols_out, key="out_cols")
        out_method = st.selectbox("Method:", ["Cap (Winsorize)", "Remove Rows"], key="out_method")
        if st.button("✅ Apply Outlier Handling", key="out_btn"):
            for col in out_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if out_method == "Remove Rows":
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                else:
                    df[col] = df[col].clip(lower, upper)
            st.session_state[file_key] = df
            st.success(f"✅ Outliers handled in {len(out_cols)} column(s)")
            st.rerun()

    st.markdown("---")

    # ── STEP 6: Scaling ──────────────────────────────────────────
    st.markdown('<div class="section-title">Step 6 — Feature Scaling</div>', unsafe_allow_html=True)
    numeric_cols_all = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if not numeric_cols_all:
        st.info("No numeric columns to scale.")
    else:
        scale_map = {}
        st.markdown('<span style="color:#6b7fa3;font-size:12px">Select a scaling method for each numeric column:</span>', unsafe_allow_html=True)

        # Layout in 3 columns for scale selectors
        sc_cols_ui = st.columns(3)
        for i, col in enumerate(numeric_cols_all):
            with sc_cols_ui[i % 3]:
                scale_map[col] = st.selectbox(
                    col,
                    ["None", "Standard (Z-score)", "MinMax (0-1)", "Robust (IQR)", "MaxAbs"],
                    key=f"scale_{col}"
                )

        if st.button("✅ Apply Scaling", key="scale_btn"):
            scaler_map = {
                "Standard (Z-score)": StandardScaler(),
                "MinMax (0-1)": MinMaxScaler(),
                "Robust (IQR)": RobustScaler(),
                "MaxAbs": MaxAbsScaler(),
            }
            applied = 0
            for col, method in scale_map.items():
                if method != "None" and method in scaler_map:
                    scaler = scaler_map[method]
                    df[[col]] = scaler.fit_transform(df[[col]])
                    applied += 1
            st.session_state[file_key] = df
            st.success(f"✅ Scaling applied to {applied} column(s)")
            st.rerun()

    st.markdown("---")

    # ── Result Preview + Actions ─────────────────────────────────
    st.markdown('<div class="section-title">Processed Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=250)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1: st.metric("Rows", f"{df.shape[0]:,}")
    with col_info2: st.metric("Columns", f"{df.shape[1]}")
    with col_info3: st.metric("Missing Cells", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    # ── Reset + Download ─────────────────────────────────────────
    action1, action2 = st.columns(2)
    with action1:
        if st.button("🔄 Reset to Original Data", key="reset_btn"):
            try:
                uploaded.seek(0)
                st.session_state[file_key] = load_data(uploaded)
                st.success("✅ Data reset to original")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

    with action2:
        st.download_button(
            "⬇️ Download Processed CSV",
            data=df.to_csv(index=False),
            file_name=f"processed_{uploaded.name.replace('.xlsx','').replace('.json','')}.csv",
            mime="text/csv",
            key="download_btn"
        )


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # App header
    st.markdown("""
    <div class="dqa-header">
      <h1>🧠 Data Quality Maintenance Tool</h1>
      <p>
        Upload your dataset once — then audit its quality and preprocess it in one unified workspace.
        &nbsp;·&nbsp; Supports CSV, TSV, Excel (.xlsx/.xls), and JSON
      </p>
    </div>""", unsafe_allow_html=True)

    # Shared file uploader at the top
    uploaded = st.file_uploader(
        "📂 Upload your dataset to get started",
        type=['csv', 'tsv', 'txt', 'xlsx', 'xls', 'json'],
        key="shared_upload"
    )

    st.markdown("---")

    # Two tabs
    tab1, tab2 = st.tabs(["📊  Data Quality Analyzer", "🛠️  Data Preparation"])

    with tab1:
        render_analyzer_tab(uploaded)

    with tab2:
        render_preparation_tab(uploaded)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:32px;padding-top:20px;border-top:1px solid #243050;
                font-size:11px;color:#6b7fa3;font-family:'DM Mono',monospace">
      Data Quality Maintenance Tool &nbsp;·&nbsp; Built with Streamlit + Plotly + pandas + scikit-learn + scipy
    </div>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()