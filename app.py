"""
Dashboard GB — Revenue by asset in Great Britain / Ingresos por activo en Gran Bretaña
Reads the parquet files published by gb_publish.py at gs://miguel-energia-gb-dashboard/gb/

Local:  streamlit run app.py        (uses your gcloud credentials if there is no secrets.toml)
Cloud:  Streamlit Community Cloud with [gcp_service_account] in Secrets
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Corporate network with SSL inspection (local only): use the Windows certificate store if truststore is present
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

st.set_page_config(page_title="GB asset revenue", page_icon="⚡", layout="wide")

DEFAULT_BUCKET = "miguel-energia-gb-dashboard"
PREFIX = "gb"

# ──────────────────────────────────────────────────────────────────────────────
# Language
# ──────────────────────────────────────────────────────────────────────────────
LANGS = {"English": "en", "Español": "es"}

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

T = {
    "app_title": {"en": "Revenue by asset in Great Britain", "es": "Ingresos por activo en Gran Bretaña"},
    "app_sub": {
        "en": "{a} to {b}. Five revenue sources per unit: wholesale, balancing, frequency, reserve and capacity.",
        "es": "{a} a {b}. Cinco fuentes de ingreso por unidad: mercado, balance, frecuencia, reservas y capacidad.",
    },
    "sidebar_period": {"en": "Period", "es": "Periodo"},
    "sidebar_months": {"en": "Months", "es": "Meses"},
    "sidebar_caption_data": {
        "en": "Data from {a} to {b}. Published {t} UTC.",
        "es": "Datos del {a} al {b}. Publicado {t} UTC.",
    },
    "sidebar_caption_method": {
        "en": "Wholesale is an estimate (physical programme × market price). Balancing and NESO services are "
        "published data. Capacity is the obligation × auction price, without indexation.",
        "es": "Mercado es una estimación (programa físico × precio de mercado). Balance y servicios de NESO son "
        "datos publicados. Capacidad es la obligación × precio de subasta, sin indexación.",
    },
    "tab_assets": {"en": "Econergy assets", "es": "Activos Econergy"},
    "tab_bess": {"en": "Battery comparison", "es": "Comparativa de baterías"},
    "tab_units": {"en": "Search a unit", "es": "Buscar unidad"},
    "tab_market": {"en": "Market", "es": "Mercado"},
    "layer_wholesale": {"en": "Wholesale", "es": "Mercado"},
    "layer_balancing": {"en": "Balancing", "es": "Balance"},
    "layer_frequency": {"en": "Frequency", "es": "Frecuencia"},
    "layer_reserve": {"en": "Reserve", "es": "Reservas"},
    "layer_capacity": {"en": "Capacity", "es": "Capacidad"},
    "total": {"en": "Total", "es": "Total"},
    "asset_select": {"en": "Asset", "es": "Activo"},
    "asset_none": {
        "en": "No asset in the asset map has data for this period. Widen the period or check gb_asset_map.csv.",
        "es": "Ningún activo de la tabla de activos tiene datos en este periodo. Amplía el periodo o revisa gb_asset_map.csv.",
    },
    "per_mw_month": {"en": "per MW per month", "es": "por MW al mes"},
    "fact_period_revenue": {"en": "Revenue for the period", "es": "Ingreso del periodo"},
    "fact_rank": {"en": "Rank", "es": "Posición"},
    "fact_rank_val": {
        "en": "{r} of {n} batteries of 10 MW or more",
        "es": "{r} de {n} baterías de 10 MW o más",
    },
    "fact_spv": {"en": "SPV", "es": "Sociedad"},
    "fact_units": {"en": "Elexon / NESO unit", "es": "Unidad Elexon / NESO"},
    "fact_lead_party": {"en": "Elexon lead party", "es": "Lead party Elexon"},
    "fact_eac_participant": {"en": "NESO services operator", "es": "Operador en servicios NESO"},
    "fact_capacity": {"en": "Capacity", "es": "Capacidad"},
    "fact_cm_contracts": {"en": "Capacity Market contracts", "es": "Contratos de capacidad"},
    "toggle_per_mw": {"en": "Show per MW", "es": "Ver por MW"},
    "axis_gbp_per_mw": {"en": "£ per MW", "es": "£ por MW"},
    "axis_gbp": {"en": "£", "es": "£"},
    "day_by_day": {"en": "Day by day", "es": "Día a día"},
    "daily_missing": {"en": "No daily data available: {e}", "es": "No hay datos diarios disponibles: {e}"},
    "day_operation": {"en": "Operation on a day", "es": "Operación en un día"},
    "day_slider": {"en": "Day", "es": "Día"},
    "chart_energy_hh": {"en": "Energy per half hour (MWh)", "es": "Energía por media hora (MWh)"},
    "chart_neso_mw": {"en": "NESO services awarded (MW)", "es": "Servicios de NESO adjudicados (MW)"},
    "series_programme": {"en": "Programme (+export / −import)", "es": "Programa (+descarga / −carga)"},
    "series_bm_up": {"en": "Balancing: up", "es": "Balance: subir"},
    "series_bm_down": {"en": "Balancing: down", "es": "Balance: bajar"},
    "axis_sp": {"en": "Settlement period", "es": "Periodo de liquidación"},
    "day_summary": {
        "en": "Revenue for the day: {t}. Average imbalance price: £{p}/MWh.",
        "es": "Ingreso del día: {t}. Precio de desvío medio: £{p}/MWh.",
    },
    "sp_missing": {"en": "No half-hourly detail available: {e}", "es": "No hay detalle por periodo disponible: {e}"},
    "portfolio_title": {"en": "Portfolio in Great Britain", "es": "Cartera en Gran Bretaña"},
    "col_project": {"en": "Project", "es": "Proyecto"},
    "col_spv": {"en": "SPV", "es": "Sociedad"},
    "col_status": {"en": "Status", "es": "Estado"},
    "col_cm": {"en": "Capacity contracts", "es": "Contratos de capacidad"},
    "col_neso_unit": {"en": "NESO unit", "es": "Unidad NESO"},
    "col_elexon_unit": {"en": "Elexon unit", "es": "Unidad Elexon"},
    "col_notes": {"en": "Notes", "es": "Notas"},
    "filter_min_mw": {"en": "Minimum capacity (MW)", "es": "Capacidad mínima (MW)"},
    "filter_top_n": {"en": "Show top", "es": "Mostrar las mejores"},
    "filter_only_bess": {"en": "Batteries only", "es": "Solo baterías"},
    "chart_gbp_per_mw_month": {"en": "Revenue per MW per month ({n} units)", "es": "Ingreso por MW al mes ({n} unidades)"},
    "caption_bold": {
        "en": "Assets from the asset map are shown in bold. If not among the top, they are appended at the end.",
        "es": "En negrita, activos de la tabla de activos. Si no están entre las mejores, se añaden al final.",
    },
    "col_unit": {"en": "Unit", "es": "Unidad"},
    "col_lead_party": {"en": "Lead party", "es": "Lead party"},
    "col_mw": {"en": "MW", "es": "MW"},
    "col_kgbp_mw_month": {"en": "£k/MW/month", "es": "£k/MW/mes"},
    "col_total_gbp": {"en": "Total £", "es": "Total £"},
    "download_comparison": {"en": "Download comparison (CSV)", "es": "Descargar comparativa (CSV)"},
    "search_placeholder": {"en": "e.g. Pillswood, Statkraft, T_SGRWO", "es": "p. ej. Pillswood, Statkraft, T_SGRWO"},
    "search_label": {"en": "Search by name, owner or code", "es": "Buscar por nombre, titular o código"},
    "search_none": {
        "en": "No unit matches the search for this period. Try part of the name or code.",
        "es": "Ninguna unidad coincide con la búsqueda en este periodo. Prueba con parte del nombre o del código.",
    },
    "unit_select": {"en": "Unit", "es": "Unidad"},
    "metric_revenue": {"en": "Revenue for the period", "es": "Ingreso del periodo"},
    "metric_kgbp_mw": {"en": "£ thousand per MW per month", "es": "£ miles por MW al mes"},
    "metric_export": {"en": "Exported (MWh)", "es": "Exportado (MWh)"},
    "metric_import": {"en": "Imported (MWh)", "es": "Importado (MWh)"},
    "unit_caption": {
        "en": "Lead party: {lp}. Technology: {ft}. Capacity: {mw} MW.",
        "es": "Lead party: {lp}. Tecnología: {ft}. Capacidad: {mw} MW.",
    },
    "download_months": {"en": "Download months (CSV)", "es": "Descargar meses (CSV)"},
    "market_prices": {"en": "Daily prices", "es": "Precios diarios"},
    "market_band": {"en": "Imbalance: daily range", "es": "Desvío: rango del día"},
    "market_avg": {"en": "Imbalance: average", "es": "Desvío: media"},
    "market_mid": {"en": "Wholesale (MID)", "es": "Mercado (MID)"},
    "axis_gbp_mwh": {"en": "£/MWh", "es": "£/MWh"},
    "market_payments": {"en": "Daily payments by the system operator", "es": "Pagos diarios del operador"},
    "series_freq_reserve": {"en": "Frequency and reserve", "es": "Frecuencia y reservas"},
    "market_caption": {
        "en": "Elexon-registered units only. Aggregated units that only sell services to NESO are not included.",
        "es": "Solo unidades registradas en Elexon. Las unidades agregadas que solo venden servicios a NESO no están incluidas.",
    },
    "eac_price_title": {"en": "NESO service prices", "es": "Precio de los servicios de NESO"},
    "eac_products": {"en": "Products", "es": "Productos"},
    "axis_gbp_mw_h": {"en": "£ per MW and hour", "es": "£ por MW y hora"},
    "eac_caption": {
        "en": "Weighted average price by awarded MW. D = dynamic (C containment, M moderation, R regulation; "
        "H raise, L lower). BR, QR, SR = balancing, quick and slow reserve (P raise, N lower).",
        "es": "Precio medio ponderado por MW adjudicados. D = dinámicos (C contención, M moderación, R "
        "regulación; H subir, L bajar). BR, QR, SR = reservas de balance, rápida y lenta (P subir, N bajar).",
    },
    "market_missing": {"en": "No market data available: {e}", "es": "No hay datos de mercado disponibles: {e}"},
    "load_error": {
        "en": "Could not read the dashboard data. Check that gb_publish.py has run and that the credentials "
        "have access to the bucket. Detail: {e}",
        "es": "No se han podido leer los datos del dashboard. Comprueba que gb_publish.py se ha ejecutado y que "
        "las credenciales tienen acceso al bucket. Detalle: {e}",
    },
}


def tr(key: str, **kwargs) -> str:
    s = T.get(key, {}).get(st.session_state["lang"], key)
    return s.format(**kwargs) if kwargs else s


MESES = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "es": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Visual identity: each revenue layer keeps a fixed colour across the whole app
# ──────────────────────────────────────────────────────────────────────────────
INK = "#14212E"
MUTED = "#5E6B78"
RULE = "#D3DAE1"
PAGE = "#EEF2F5"


def layers():
    return [
        ("WHOLESALE_GBP", tr("layer_wholesale"), "#6F8FAF"),
        ("BM_GBP", tr("layer_balancing"), "#E3A21A"),
        ("RESPONSE_GBP", tr("layer_frequency"), "#1F9E89"),
        ("RESERVE_GBP", tr("layer_reserve"), "#6C5BB5"),
        ("CAPACITY_GBP", tr("layer_capacity"), "#B5424F"),
    ]


LAYER_COLS = ["WHOLESALE_GBP", "BM_GBP", "RESPONSE_GBP", "RESERVE_GBP", "CAPACITY_GBP"]

st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stDataFrame, label, input, button {{
  font-family: "IBM Plex Sans", system-ui, sans-serif;
}}
.stApp {{ background: {PAGE}; }}
h1, h2, h3, .gb-figure {{
  font-family: "Archivo", "IBM Plex Sans", sans-serif; font-stretch: 87.5%; color: {INK};
  letter-spacing: -0.01em;
}}
h1 {{ font-weight: 780; font-size: 2.1rem; line-height: 1.1; margin-bottom: 0.2rem; }}
h2 {{ font-weight: 700; font-size: 1.45rem; }}
h3 {{ font-weight: 650; font-size: 1.1rem; }}
.gb-sub {{ color: {MUTED}; font-size: 0.95rem; max-width: 72ch; margin-bottom: 1rem; }}
.gb-figure {{ font-size: 3.4rem; font-weight: 800; line-height: 1; }}
.gb-figure-unit {{ font-family: "Archivo"; font-stretch: 87.5%; font-size: 1.1rem; color: {MUTED}; margin-left: .35rem; }}
.gb-facts {{ color: {INK}; font-size: 0.92rem; line-height: 1.6; }}
.gb-facts b {{ font-weight: 600; }}
.gb-legend span {{ display: inline-block; margin-right: 1rem; font-size: .88rem; color: {INK}; }}
.gb-legend i {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: "Archivo"; font-stretch: 87.5%; font-weight: 600; font-size: 1.02rem; padding: .4rem 0; }}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def _secrets() -> dict:
    try:
        return {k: st.secrets[k] for k in st.secrets}
    except Exception:  # noqa: BLE001
        return {}


@st.cache_resource(show_spinner=False)
def gcs_bucket():
    from google.cloud import storage

    sec = _secrets()
    if "gcp_service_account" in sec:
        from google.oauth2 import service_account

        info = dict(sec["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(info)
        client = storage.Client(project=info.get("project_id"), credentials=creds)
    else:
        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "miguel-energia-gb"))
    return client.bucket(sec.get("gb_dashboard_bucket", DEFAULT_BUCKET))


@st.cache_data(ttl=3600, show_spinner="Loading data…")
def load(name: str) -> pd.DataFrame:
    data = gcs_bucket().blob(f"{PREFIX}/{name}.parquet").download_as_bytes()
    df = pd.read_parquet(io.BytesIO(data))
    for col in ("MONTH", "DATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_meta() -> dict:
    return json.loads(gcs_bucket().blob(f"{PREFIX}/gb_meta.json").download_as_text())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def fmt_num(x: float, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if st.session_state["lang"] == "es":
        s = f"{x:,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.{dec}f}"


def fmt_gbp(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    a = abs(x)
    if a >= 1e6:
        return f"£{fmt_num(x / 1e6, 2)} M"
    if a >= 1e3:
        return f"£{fmt_num(x / 1e3, 0)} k"
    return f"£{fmt_num(x, 0)}"


def month_label(ts: pd.Timestamp) -> str:
    names = MESES[st.session_state["lang"]]
    return f"{names[ts.month - 1]} {ts.year}"


def val(x, default: str = "–") -> str:
    return default if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else str(x)


def capacity_mw(row) -> float:
    gen = row.get("GENERATION_CAPACITY_MW") or 0
    dem = abs(row.get("DEMAND_CAPACITY_MW") or 0)
    cap = gen if gen and gen > 0 else dem
    return float(cap) if cap and cap > 0 else np.nan


def unit_label(row) -> str:
    code = row.get("BM_UNIT")
    name = row.get("BM_UNIT_NAME")
    if isinstance(row.get("PROJECT"), str) and row.get("PROJECT"):
        return f"{row['PROJECT']}, {val(row.get('OWNER_GROUP'), '')} ({code})"
    if isinstance(name, str) and name and name != code:
        return f"{name} ({code})"
    return str(code)


def base_layout(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        title=dict(text=title, font=dict(family="Archivo, sans-serif", size=16, color=INK)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode="relative",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE)
    fig.update_yaxes(gridcolor=RULE, zerolinecolor=MUTED, zerolinewidth=1)
    return fig


def layer_stack(df: pd.DataFrame, x: str, height: int = 380, per_mw: float | None = None, x_is_month: bool = False,
                title: str | None = None) -> go.Figure:
    fig = go.Figure()
    xs = df[x].map(month_label) if x_is_month else df[x]
    div = per_mw if per_mw and per_mw > 0 else 1.0
    for col, label, color in layers():
        if col in df.columns:
            fig.add_bar(x=xs, y=df[col].fillna(0) / div, name=label, marker_color=color,
                        hovertemplate=f"{label}: %{{y:,.0f}}<extra></extra>")
    if "TOTAL_GBP" in df.columns:
        fig.add_scatter(x=xs, y=df["TOTAL_GBP"].fillna(0) / div, name=tr("total"), mode="markers",
                        marker=dict(symbol="line-ew", size=18, line=dict(width=3, color=INK)),
                        hovertemplate=f"{tr('total')}: %{{y:,.0f}}<extra></extra>")
    return base_layout(fig, height, title)


def legend_html() -> str:
    return '<div class="gb-legend">' + "".join(f'<span><i style="background:{c}"></i>{l}</span>' for _, l, c in layers()) + "</div>"


# ──────────────────────────────────────────────────────────────────────────────
# Load common data
# ──────────────────────────────────────────────────────────────────────────────
try:
    meta = load_meta()
    units = load("gb_units")
    monthly = load("gb_monthly_bmu")
    assets = load("gb_asset_map")
except Exception as e:  # noqa: BLE001
    st.error(tr("load_error", e=e))
    st.stop()

units["CAP_MW"] = units.apply(capacity_mw, axis=1)
units["LABEL"] = units.apply(unit_label, axis=1)
unit_info = units.set_index("BM_UNIT")

months = sorted(monthly["MONTH"].dropna().unique())
with st.sidebar:
    lang_name = st.radio("Language / Idioma", list(LANGS.keys()),
                         index=list(LANGS.values()).index(st.session_state["lang"]), horizontal=True)
    st.session_state["lang"] = LANGS[lang_name]

    st.markdown(f"### {tr('sidebar_period')}")
    if len(months) > 1:
        m_from, m_to = st.select_slider(
            tr("sidebar_months"), options=months, value=(months[0], months[-1]),
            format_func=lambda m: month_label(pd.Timestamp(m)), label_visibility="collapsed",
        )
    else:
        m_from = m_to = months[0]
        st.write(month_label(pd.Timestamp(m_from)))
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption(tr("sidebar_caption_data", a=meta.get("min_date"), b=meta.get("max_date"),
               t=meta.get("published_utc", "")[:16].replace("T", " ")))
    st.caption(tr("sidebar_caption_method"))

m_from, m_to = pd.Timestamp(m_from), pd.Timestamp(m_to)
mon = monthly[(monthly["MONTH"] >= m_from) & (monthly["MONTH"] <= m_to)].copy()


def period_by_unit(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("BM_UNIT", as_index=False)[LAYER_COLS + ["TOTAL_GBP", "EXPORT_MWH", "IMPORT_MWH", "DAYS"]].sum()
    agg = agg.merge(units[["BM_UNIT", "LABEL", "CAP_MW", "LEAD_PARTY_NAME", "IS_BATTERY", "PROJECT", "OWNER_GROUP",
                           "BM_UNIT_NAME", "NG_BM_UNIT", "FUEL_TYPE"]], on="BM_UNIT", how="left")
    months_eq = agg["DAYS"].replace(0, np.nan) / 30.4
    agg["GBP_K_PER_MW_MONTH"] = agg["TOTAL_GBP"] / agg["CAP_MW"] / months_eq / 1000
    for col in LAYER_COLS:
        agg[f"{col}_PER_MW_MONTH"] = agg[col] / agg["CAP_MW"] / months_eq / 1000
    return agg


per_unit = period_by_unit(mon)

st.markdown(f"# {tr('app_title')}")
st.markdown(f'<div class="gb-sub">{tr("app_sub", a=month_label(m_from), b=month_label(m_to))}</div>',
           unsafe_allow_html=True)

tab_assets, tab_bess, tab_units, tab_market = st.tabs(
    [tr("tab_assets"), tr("tab_bess"), tr("tab_units"), tr("tab_market")]
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Econergy assets
# ──────────────────────────────────────────────────────────────────────────────
with tab_assets:
    projects = (
        assets.groupby("PROJECT", as_index=False)
        .agg(SPV=("SPV", "first"), STATUS=("STATUS", "first"), NG_BM_UNIT=("NG_BM_UNIT", "first"),
             BM_UNIT=("BM_UNIT", "first"), CMU=("CMU_ID", lambda s: ", ".join(sorted(set(s.dropna())))),
             NOTES=("NOTES", "first"))
    )
    live = projects[projects["BM_UNIT"].isin(per_unit["BM_UNIT"])]

    if live.empty:
        st.info(tr("asset_none"))
    else:
        project = st.selectbox(tr("asset_select"), live["PROJECT"].tolist(), label_visibility="collapsed")
        prow = live[live["PROJECT"] == project].iloc[0]
        bmu = prow["BM_UNIT"]
        urow = per_unit[per_unit["BM_UNIT"] == bmu].iloc[0]
        uinfo = unit_info.loc[bmu] if bmu in unit_info.index else pd.Series(dtype=object)

        peers = per_unit[(per_unit["IS_BATTERY"] == True) & (per_unit["CAP_MW"] >= 10) & per_unit["GBP_K_PER_MW_MONTH"].notna()]  # noqa: E712
        rank = int((peers["GBP_K_PER_MW_MONTH"] > urow["GBP_K_PER_MW_MONTH"]).sum()) + 1 if len(peers) else None

        left, right = st.columns([1, 1.6], gap="large")
        with left:
            st.markdown(f"## {project}")
            st.markdown(
                f'<div class="gb-figure">£{fmt_num(urow["GBP_K_PER_MW_MONTH"], 1)} k'
                f'<span class="gb-figure-unit">{tr("per_mw_month")}</span></div>',
                unsafe_allow_html=True,
            )
            facts = [
                f"<b>{tr('fact_period_revenue')}:</b> {fmt_gbp(urow['TOTAL_GBP'])}",
                f"<b>{tr('fact_rank')}:</b> {tr('fact_rank_val', r=rank, n=len(peers))}" if rank else None,
                f"<b>{tr('fact_spv')}:</b> {val(prow['SPV'])}",
                f"<b>{tr('fact_units')}:</b> {bmu} / {val(prow['NG_BM_UNIT'])}",
                f"<b>{tr('fact_lead_party')}:</b> {val(uinfo.get('LEAD_PARTY_NAME'))}",
                f"<b>{tr('fact_eac_participant')}:</b> {val(uinfo.get('EAC_PARTICIPANT'))}",
                f"<b>{tr('fact_capacity')}:</b> {fmt_num(urow['CAP_MW'], 1)} MW",
                f"<b>{tr('fact_cm_contracts')}:</b> {val(prow['CMU'])}",
            ]
            st.markdown('<div class="gb-facts">' + "<br>".join(f for f in facts if f) + "</div>", unsafe_allow_html=True)

        with right:
            per_mw = st.toggle(tr("toggle_per_mw"), value=False, key="asset_per_mw")
            asset_m = mon[mon["BM_UNIT"] == bmu].sort_values("MONTH")
            fig = layer_stack(asset_m, "MONTH", height=360, per_mw=urow["CAP_MW"] if per_mw else None, x_is_month=True)
            fig.update_yaxes(title_text=tr("axis_gbp_per_mw") if per_mw else tr("axis_gbp"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {tr('day_by_day')}")
        try:
            daily = load("gb_daily_bmu")
            asset_d = daily[(daily["BM_UNIT"] == bmu) & (daily["DATE"] >= m_from) & (daily["DATE"] < m_to + pd.offsets.MonthBegin(1))]
            st.plotly_chart(layer_stack(asset_d.sort_values("DATE"), "DATE", height=320), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(tr("daily_missing", e=e))

        st.markdown(f"### {tr('day_operation')}")
        try:
            sp = load("gb_sp_assets")
            sp_a = sp[sp["BM_UNIT"] == bmu]
            days = sorted(sp_a["DATE"].dt.date.unique())
            if days:
                day = st.select_slider(tr("day_slider"), options=days, value=days[-1], format_func=lambda d: d.strftime("%d/%m/%Y"))
                one = sp_a[sp_a["DATE"].dt.date == day].sort_values("SP")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                    subplot_titles=(tr("chart_energy_hh"), tr("chart_neso_mw")))
                fig.add_bar(x=one["SP"], y=one["PN_MWH"], name=tr("series_programme"), marker_color="#6F8FAF", row=1, col=1)
                fig.add_bar(x=one["SP"], y=one["BM_OFFER_MWH"], name=tr("series_bm_up"), marker_color="#E3A21A", row=1, col=1)
                fig.add_bar(x=one["SP"], y=one["BM_BID_MWH"], name=tr("series_bm_down"), marker_color="#B98300", row=1, col=1)
                fig.add_scatter(x=one["SP"], y=one["RESPONSE_MW"], name=tr("layer_frequency"), mode="lines",
                                line=dict(color="#1F9E89", width=2, shape="hv"), row=2, col=1)
                fig.add_scatter(x=one["SP"], y=one["RESERVE_MW"], name=tr("layer_reserve"), mode="lines",
                                line=dict(color="#6C5BB5", width=2, shape="hv"), row=2, col=1)
                base_layout(fig, height=520)
                fig.update_xaxes(title_text=tr("axis_sp"), row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(tr("day_summary", t=fmt_gbp(one["TOTAL_GBP"].sum()),
                             p=fmt_num(one["SYSTEM_PRICE_GBP_MWH"].mean(), 1)))
        except Exception as e:  # noqa: BLE001
            st.warning(tr("sp_missing", e=e))

    st.markdown(f"### {tr('portfolio_title')}")
    st.dataframe(
        projects[["PROJECT", "SPV", "STATUS", "CMU", "NG_BM_UNIT", "BM_UNIT", "NOTES"]].rename(columns={
            "PROJECT": tr("col_project"), "SPV": tr("col_spv"), "STATUS": tr("col_status"), "CMU": tr("col_cm"),
            "NG_BM_UNIT": tr("col_neso_unit"), "BM_UNIT": tr("col_elexon_unit"), "NOTES": tr("col_notes")}),
        hide_index=True, use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2) Battery comparison
# ──────────────────────────────────────────────────────────────────────────────
with tab_bess:
    c1, c2, c3 = st.columns([1, 1, 2])
    min_mw = c1.number_input(tr("filter_min_mw"), min_value=0, max_value=500, value=10, step=5)
    top_n = c2.number_input(tr("filter_top_n"), min_value=5, max_value=100, value=25, step=5)
    only_bess = c3.toggle(tr("filter_only_bess"), value=True)

    pool = per_unit[(per_unit["CAP_MW"] >= min_mw) & per_unit["GBP_K_PER_MW_MONTH"].notna()]
    if only_bess:
        pool = pool[pool["IS_BATTERY"] == True]  # noqa: E712
    pool = pool.sort_values("GBP_K_PER_MW_MONTH", ascending=False)
    top = pool.head(int(top_n))
    econ = pool[pool["PROJECT"].notna() & ~pool["BM_UNIT"].isin(top["BM_UNIT"])]
    show = pd.concat([top, econ]).iloc[::-1]

    st.markdown(f"### {tr('chart_gbp_per_mw_month', n=len(pool))}")
    fig = go.Figure()
    ylabels = [f"<b>{l}</b>" if p else l for l, p in zip(show["LABEL"], show["PROJECT"].notna())]
    unit_word = "month" if st.session_state["lang"] == "en" else "mes"
    for col, label, color in layers():
        fig.add_bar(y=ylabels, x=show[f"{col}_PER_MW_MONTH"].fillna(0), name=label, orientation="h", marker_color=color,
                    hovertemplate=f"{label}: £%{{x:,.2f}} k/MW/{unit_word}<extra></extra>")
    base_layout(fig, height=max(360, 22 * len(show) + 80))
    fig.update_layout(hovermode="y unified")
    axis_x_label = "£ thousand per MW per month" if st.session_state["lang"] == "en" else "£ miles por MW al mes"
    fig.update_xaxes(title_text=axis_x_label, gridcolor=RULE, showgrid=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(tr("caption_bold"))

    table = pool[["LABEL", "LEAD_PARTY_NAME", "CAP_MW", "GBP_K_PER_MW_MONTH", "TOTAL_GBP"] + LAYER_COLS].rename(columns={
        "LABEL": tr("col_unit"), "LEAD_PARTY_NAME": tr("col_lead_party"), "CAP_MW": tr("col_mw"),
        "GBP_K_PER_MW_MONTH": tr("col_kgbp_mw_month"), "TOTAL_GBP": tr("col_total_gbp"),
        **{c: l + " £" for c, l, _ in layers()}})
    st.dataframe(table, hide_index=True, use_container_width=True,
                column_config={tr("col_mw"): st.column_config.NumberColumn(format="%.1f"),
                              tr("col_kgbp_mw_month"): st.column_config.NumberColumn(format="%.2f"),
                              **{c: st.column_config.NumberColumn(format="%.0f") for c in table.columns if c.endswith("£")}})
    st.download_button(tr("download_comparison"), table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="gb_battery_comparison.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Search a unit
# ──────────────────────────────────────────────────────────────────────────────
with tab_units:
    q = st.text_input(tr("search_label"), placeholder=tr("search_placeholder"))
    cand = per_unit
    if q:
        ql = q.lower()
        mask = (per_unit["LABEL"].str.lower().str.contains(ql, na=False)
                | per_unit["LEAD_PARTY_NAME"].str.lower().str.contains(ql, na=False)
                | per_unit["BM_UNIT"].str.lower().str.contains(ql, na=False)
                | per_unit["NG_BM_UNIT"].fillna("").str.lower().str.contains(ql, na=False))
        cand = per_unit[mask]
    cand = cand.sort_values("TOTAL_GBP", ascending=False).head(300)

    if cand.empty:
        st.info(tr("search_none"))
    else:
        pick = st.selectbox(tr("unit_select"), cand["BM_UNIT"].tolist(),
                            format_func=lambda b: cand.set_index("BM_UNIT").loc[b, "LABEL"])
        row = cand[cand["BM_UNIT"] == pick].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(tr("metric_revenue"), fmt_gbp(row["TOTAL_GBP"]))
        k2.metric(tr("metric_kgbp_mw"), fmt_num(row["GBP_K_PER_MW_MONTH"], 2))
        k3.metric(tr("metric_export"), fmt_num(row["EXPORT_MWH"]))
        k4.metric(tr("metric_import"), fmt_num(row["IMPORT_MWH"]))
        st.caption(tr("unit_caption", lp=val(row["LEAD_PARTY_NAME"]), ft=val(row["FUEL_TYPE"]), mw=fmt_num(row["CAP_MW"], 1)))

        um = mon[mon["BM_UNIT"] == pick].sort_values("MONTH")
        st.plotly_chart(layer_stack(um, "MONTH", height=340, x_is_month=True), use_container_width=True)

        try:
            daily = load("gb_daily_bmu")
            ud = daily[(daily["BM_UNIT"] == pick) & (daily["DATE"] >= m_from) & (daily["DATE"] < m_to + pd.offsets.MonthBegin(1))]
            if not ud.empty:
                st.plotly_chart(layer_stack(ud.sort_values("DATE"), "DATE", height=300), use_container_width=True)
        except Exception:  # noqa: BLE001
            pass

        export = um.assign(MONTH_LABEL=um["MONTH"].map(month_label))[
            ["MONTH_LABEL", "EXPORT_MWH", "IMPORT_MWH", "BM_OFFER_MWH", "BM_BID_MWH", "TOTAL_GBP"] + LAYER_COLS]
        st.download_button(tr("download_months"), export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"gb_{pick}_monthly.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Market
# ──────────────────────────────────────────────────────────────────────────────
with tab_market:
    try:
        mkt = load("gb_market_daily").sort_values("DATE")
        mkt = mkt[(mkt["DATE"] >= m_from) & (mkt["DATE"] < m_to + pd.offsets.MonthBegin(1))]

        st.markdown(f"### {tr('market_prices')}")
        fig = go.Figure()
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_MAX"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_MIN"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(227,162,26,0.18)", name=tr("market_band"))
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_AVG"], mode="lines", line=dict(color="#E3A21A", width=2), name=tr("market_avg"))
        fig.add_scatter(x=mkt["DATE"], y=mkt["MID_GBP_MWH"], mode="lines", line=dict(color=INK, width=2), name=tr("market_mid"))
        base_layout(fig, height=360)
        fig.update_yaxes(title_text=tr("axis_gbp_mwh"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {tr('market_payments')}")
        fig = go.Figure()
        fig.add_bar(x=mkt["DATE"], y=mkt["BM_TOTAL_GBP"], name=tr("layer_balancing"), marker_color="#E3A21A")
        fig.add_bar(x=mkt["DATE"], y=mkt["NESO_SERVICES_GBP"], name=tr("series_freq_reserve"), marker_color="#1F9E89")
        base_layout(fig, height=320)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text=tr("axis_gbp"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(tr("market_caption"))

        eac = load("gb_eac_prices_daily")
        eac = eac[(eac["DATE"] >= m_from) & (eac["DATE"] < m_to + pd.offsets.MonthBegin(1))]
        st.markdown(f"### {tr('eac_price_title')}")
        products = sorted(eac["PRODUCT"].dropna().unique())
        default = [p for p in ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"] if p in products]
        chosen = st.multiselect(tr("eac_products"), products, default=default or products[:4])
        fig = go.Figure()
        for p in chosen:
            s = eac[eac["PRODUCT"] == p].sort_values("DATE")
            fig.add_scatter(x=s["DATE"], y=s["PRICE_GBP_MW_H"], mode="lines", name=p)
        base_layout(fig, height=340)
        fig.update_yaxes(title_text=tr("axis_gbp_mw_h"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(tr("eac_caption"))
    except Exception as e:  # noqa: BLE001
        st.warning(tr("market_missing", e=e))
