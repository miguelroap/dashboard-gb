"""
Dashboard GB — Ingresos por activo en Gran Bretaña
Lee los parquet publicados por gb_publish.py en gs://miguel-energia-gb-dashboard/gb/

Local:   streamlit run app.py        (usa tus credenciales de gcloud si no hay secrets.toml)
Cloud:   Streamlit Community Cloud con [gcp_service_account] en Secrets
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

# Red corporativa con inspección SSL (solo local): usar certificados de Windows si está truststore
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

st.set_page_config(page_title="Ingresos por activo en Gran Bretaña", page_icon="⚡", layout="wide")

DEFAULT_BUCKET = "miguel-energia-gb-dashboard"
PREFIX = "gb"

# ──────────────────────────────────────────────────────────────────────────────
# Identidad visual: cada capa de ingreso tiene un color fijo en toda la app
# ──────────────────────────────────────────────────────────────────────────────
INK = "#14212E"
MUTED = "#5E6B78"
RULE = "#D3DAE1"
PAGE = "#EEF2F5"
LAYERS = [
    ("WHOLESALE_GBP", "Mercado", "#6F8FAF"),
    ("BM_GBP", "Balance", "#E3A21A"),
    ("RESPONSE_GBP", "Frecuencia", "#1F9E89"),
    ("RESERVE_GBP", "Reservas", "#6C5BB5"),
    ("CAPACITY_GBP", "Capacidad", "#B5424F"),
]
LAYER_COLS = [c for c, _, _ in LAYERS]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

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
# Datos
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


@st.cache_data(ttl=3600, show_spinner="Cargando datos…")
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
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def fmt_num(x: float, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


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
    return f"{MESES[ts.month - 1]} {ts.year}"


def capacity_mw(row) -> float:
    gen = row.get("GENERATION_CAPACITY_MW") or 0
    dem = abs(row.get("DEMAND_CAPACITY_MW") or 0)
    cap = gen if gen and gen > 0 else dem
    return float(cap) if cap and cap > 0 else np.nan


def val(x, default: str = "–") -> str:
    return default if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else str(x)


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
    for col, label, color in LAYERS:
        if col in df.columns:
            fig.add_bar(x=xs, y=df[col].fillna(0) / div, name=label, marker_color=color,
                        hovertemplate=f"{label}: %{{y:,.0f}}<extra></extra>")
    if "TOTAL_GBP" in df.columns:
        fig.add_scatter(x=xs, y=df["TOTAL_GBP"].fillna(0) / div, name="Total", mode="markers",
                        marker=dict(symbol="line-ew", size=18, line=dict(width=3, color=INK)),
                        hovertemplate="Total: %{y:,.0f}<extra></extra>")
    return base_layout(fig, height, title)


def legend_html() -> str:
    return '<div class="gb-legend">' + "".join(f'<span><i style="background:{c}"></i>{l}</span>' for _, l, c in LAYERS) + "</div>"


# ──────────────────────────────────────────────────────────────────────────────
# Carga y filtros comunes
# ──────────────────────────────────────────────────────────────────────────────
try:
    meta = load_meta()
    units = load("gb_units")
    monthly = load("gb_monthly_bmu")
    assets = load("gb_asset_map")
except Exception as e:  # noqa: BLE001
    st.error(
        "No se han podido leer los datos del dashboard. Comprueba que gb_publish.py se ha ejecutado y que las "
        f"credenciales tienen acceso al bucket. Detalle: {e}"
    )
    st.stop()

units["CAP_MW"] = units.apply(capacity_mw, axis=1)
units["LABEL"] = units.apply(unit_label, axis=1)
unit_info = units.set_index("BM_UNIT")

months = sorted(monthly["MONTH"].dropna().unique())
with st.sidebar:
    st.markdown("### Periodo")
    if len(months) > 1:
        m_from, m_to = st.select_slider(
            "Meses", options=months, value=(months[0], months[-1]), format_func=lambda m: month_label(pd.Timestamp(m)),
            label_visibility="collapsed",
        )
    else:
        m_from = m_to = months[0]
        st.write(month_label(pd.Timestamp(m_from)))
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption(
        f"Datos del {meta.get('min_date')} al {meta.get('max_date')}. Publicado {meta.get('published_utc', '')[:16].replace('T', ' ')} UTC."
    )
    st.caption(
        "Mercado es una estimación (programa físico × precio de mercado). Balance y servicios de NESO son datos "
        "publicados. Capacidad es la obligación × precio de subasta, sin indexación."
    )

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

st.markdown("# Ingresos por activo en Gran Bretaña")
st.markdown(
    f'<div class="gb-sub">{month_label(m_from)} a {month_label(m_to)}. Cinco fuentes de ingreso por unidad: '
    "mercado, balance, frecuencia, reservas y capacidad.</div>",
    unsafe_allow_html=True,
)

tab_assets, tab_bess, tab_units, tab_market = st.tabs(["Activos Econergy", "Comparativa de baterías", "Buscar unidad", "Mercado"])

# ──────────────────────────────────────────────────────────────────────────────
# 1) Activos Econergy
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
        st.info("Ningún activo de la tabla de activos tiene datos en este periodo. Amplía el periodo o revisa gb_asset_map.csv.")
    else:
        project = st.selectbox("Activo", live["PROJECT"].tolist(), label_visibility="collapsed")
        prow = live[live["PROJECT"] == project].iloc[0]
        bmu = prow["BM_UNIT"]
        urow = per_unit[per_unit["BM_UNIT"] == bmu].iloc[0]
        uinfo = unit_info.loc[bmu] if bmu in unit_info.index else pd.Series(dtype=object)

        # Posición frente al resto de baterías
        peers = per_unit[(per_unit["IS_BATTERY"] == True) & (per_unit["CAP_MW"] >= 10) & per_unit["GBP_K_PER_MW_MONTH"].notna()]  # noqa: E712
        rank = int((peers["GBP_K_PER_MW_MONTH"] > urow["GBP_K_PER_MW_MONTH"]).sum()) + 1 if len(peers) else None

        left, right = st.columns([1, 1.6], gap="large")
        with left:
            st.markdown(f"## {project}")
            st.markdown(
                f'<div class="gb-figure">£{fmt_num(urow["GBP_K_PER_MW_MONTH"], 1)} k'
                f'<span class="gb-figure-unit">por MW al mes</span></div>',
                unsafe_allow_html=True,
            )
            facts = [
                f"<b>Ingreso del periodo:</b> {fmt_gbp(urow['TOTAL_GBP'])}",
                f"<b>Posición:</b> {rank} de {len(peers)} baterías de 10 MW o más" if rank else None,
                f"<b>Sociedad:</b> {val(prow['SPV'])}",
                f"<b>Unidad Elexon / NESO:</b> {bmu} / {val(prow['NG_BM_UNIT'])}",
                f"<b>Lead party Elexon:</b> {val(uinfo.get('LEAD_PARTY_NAME'))}",
                f"<b>Operador en servicios NESO:</b> {val(uinfo.get('EAC_PARTICIPANT'))}",
                f"<b>Capacidad:</b> {fmt_num(urow['CAP_MW'], 1)} MW",
                f"<b>Contratos de capacidad:</b> {val(prow['CMU'])}",
            ]
            st.markdown('<div class="gb-facts">' + "<br>".join(f for f in facts if f) + "</div>", unsafe_allow_html=True)

        with right:
            per_mw = st.toggle("Ver por MW", value=False, key="asset_per_mw")
            asset_m = mon[mon["BM_UNIT"] == bmu].sort_values("MONTH")
            fig = layer_stack(asset_m, "MONTH", height=360, per_mw=urow["CAP_MW"] if per_mw else None, x_is_month=True)
            fig.update_yaxes(title_text="£ por MW" if per_mw else "£")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Día a día")
        try:
            daily = load("gb_daily_bmu")
            asset_d = daily[(daily["BM_UNIT"] == bmu) & (daily["DATE"] >= m_from) & (daily["DATE"] < m_to + pd.offsets.MonthBegin(1))]
            st.plotly_chart(layer_stack(asset_d.sort_values("DATE"), "DATE", height=320), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"No hay datos diarios disponibles: {e}")

        st.markdown("### Operación en un día")
        try:
            sp = load("gb_sp_assets")
            sp_a = sp[sp["BM_UNIT"] == bmu]
            days = sorted(sp_a["DATE"].dt.date.unique())
            if days:
                day = st.select_slider("Día", options=days, value=days[-1], format_func=lambda d: d.strftime("%d/%m/%Y"))
                one = sp_a[sp_a["DATE"].dt.date == day].sort_values("SP")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                                    subplot_titles=("Energía por media hora (MWh)", "Servicios de NESO adjudicados (MW)"))
                fig.add_bar(x=one["SP"], y=one["PN_MWH"], name="Programa (+descarga / −carga)", marker_color="#6F8FAF", row=1, col=1)
                fig.add_bar(x=one["SP"], y=one["BM_OFFER_MWH"], name="Balance: subir", marker_color="#E3A21A", row=1, col=1)
                fig.add_bar(x=one["SP"], y=one["BM_BID_MWH"], name="Balance: bajar", marker_color="#B98300", row=1, col=1)
                fig.add_scatter(x=one["SP"], y=one["RESPONSE_MW"], name="Frecuencia", mode="lines", line=dict(color="#1F9E89", width=2, shape="hv"), row=2, col=1)
                fig.add_scatter(x=one["SP"], y=one["RESERVE_MW"], name="Reservas", mode="lines", line=dict(color="#6C5BB5", width=2, shape="hv"), row=2, col=1)
                base_layout(fig, height=520)
                fig.update_xaxes(title_text="Periodo de liquidación", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Ingreso del día: {fmt_gbp(one['TOTAL_GBP'].sum())}. "
                           f"Precio de desvío medio: £{fmt_num(one['SYSTEM_PRICE_GBP_MWH'].mean(), 1)}/MWh.")
        except Exception as e:  # noqa: BLE001
            st.warning(f"No hay detalle por periodo disponible: {e}")

    st.markdown("### Cartera en Gran Bretaña")
    st.dataframe(
        projects[["PROJECT", "SPV", "STATUS", "CMU", "NG_BM_UNIT", "BM_UNIT", "NOTES"]].rename(columns={
            "PROJECT": "Proyecto", "SPV": "Sociedad", "STATUS": "Estado", "CMU": "Contratos de capacidad",
            "NG_BM_UNIT": "Unidad NESO", "BM_UNIT": "Unidad Elexon", "NOTES": "Notas"}),
        hide_index=True, use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2) Comparativa de baterías
# ──────────────────────────────────────────────────────────────────────────────
with tab_bess:
    c1, c2, c3 = st.columns([1, 1, 2])
    min_mw = c1.number_input("Capacidad mínima (MW)", min_value=0, max_value=500, value=10, step=5)
    top_n = c2.number_input("Mostrar las mejores", min_value=5, max_value=100, value=25, step=5)
    only_bess = c3.toggle("Solo baterías", value=True)

    pool = per_unit[(per_unit["CAP_MW"] >= min_mw) & per_unit["GBP_K_PER_MW_MONTH"].notna()]
    if only_bess:
        pool = pool[pool["IS_BATTERY"] == True]  # noqa: E712
    pool = pool.sort_values("GBP_K_PER_MW_MONTH", ascending=False)
    top = pool.head(int(top_n))
    econ = pool[pool["PROJECT"].notna() & ~pool["BM_UNIT"].isin(top["BM_UNIT"])]
    show = pd.concat([top, econ]).iloc[::-1]

    st.markdown(f"### Ingreso por MW al mes ({len(pool)} unidades)")
    fig = go.Figure()
    ylabels = [f"<b>{l}</b>" if p else l for l, p in zip(show["LABEL"], show["PROJECT"].notna())]
    for col, label, color in LAYERS:
        fig.add_bar(y=ylabels, x=show[f"{col}_PER_MW_MONTH"].fillna(0), name=label, orientation="h", marker_color=color,
                    hovertemplate=f"{label}: £%{{x:,.2f}} k/MW/mes<extra></extra>")
    base_layout(fig, height=max(360, 22 * len(show) + 80))
    fig.update_layout(hovermode="y unified")
    fig.update_xaxes(title_text="£ miles por MW al mes", gridcolor=RULE, showgrid=True)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("En negrita, activos de la tabla de activos. Si no están entre las mejores, se añaden al final.")

    table = pool[["LABEL", "LEAD_PARTY_NAME", "CAP_MW", "GBP_K_PER_MW_MONTH", "TOTAL_GBP"] + LAYER_COLS].rename(columns={
        "LABEL": "Unidad", "LEAD_PARTY_NAME": "Lead party", "CAP_MW": "MW", "GBP_K_PER_MW_MONTH": "£k/MW/mes",
        "TOTAL_GBP": "Total £", **{c: l + " £" for c, l, _ in LAYERS}})
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"MW": st.column_config.NumberColumn(format="%.1f"),
                                "£k/MW/mes": st.column_config.NumberColumn(format="%.2f"),
                                **{c: st.column_config.NumberColumn(format="%.0f") for c in table.columns if c.endswith("£")}})
    st.download_button("Descargar comparativa (CSV)", table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="gb_comparativa_baterias.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Buscar unidad
# ──────────────────────────────────────────────────────────────────────────────
with tab_units:
    q = st.text_input("Buscar por nombre, titular o código", placeholder="p. ej. Pillswood, Statkraft, T_SGRWO")
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
        st.info("Ninguna unidad coincide con la búsqueda en este periodo. Prueba con parte del nombre o del código.")
    else:
        pick = st.selectbox("Unidad", cand["BM_UNIT"].tolist(), format_func=lambda b: cand.set_index("BM_UNIT").loc[b, "LABEL"])
        row = cand[cand["BM_UNIT"] == pick].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ingreso del periodo", fmt_gbp(row["TOTAL_GBP"]))
        k2.metric("£ miles por MW al mes", fmt_num(row["GBP_K_PER_MW_MONTH"], 2))
        k3.metric("Exportado (MWh)", fmt_num(row["EXPORT_MWH"]))
        k4.metric("Importado (MWh)", fmt_num(row["IMPORT_MWH"]))
        st.caption(f"Lead party: {val(row['LEAD_PARTY_NAME'])}. Tecnología: {val(row['FUEL_TYPE'])}. Capacidad: {fmt_num(row['CAP_MW'], 1)} MW.")

        um = mon[mon["BM_UNIT"] == pick].sort_values("MONTH")
        st.plotly_chart(layer_stack(um, "MONTH", height=340, x_is_month=True), use_container_width=True)

        try:
            daily = load("gb_daily_bmu")
            ud = daily[(daily["BM_UNIT"] == pick) & (daily["DATE"] >= m_from) & (daily["DATE"] < m_to + pd.offsets.MonthBegin(1))]
            if not ud.empty:
                st.plotly_chart(layer_stack(ud.sort_values("DATE"), "DATE", height=300), use_container_width=True)
        except Exception:  # noqa: BLE001
            pass

        export = um.assign(MES=um["MONTH"].map(month_label))[["MES", "EXPORT_MWH", "IMPORT_MWH", "BM_OFFER_MWH", "BM_BID_MWH", "TOTAL_GBP"] + LAYER_COLS]
        st.download_button("Descargar meses (CSV)", export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"gb_{pick}_mensual.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Mercado
# ──────────────────────────────────────────────────────────────────────────────
with tab_market:
    try:
        mkt = load("gb_market_daily").sort_values("DATE")
        mkt = mkt[(mkt["DATE"] >= m_from) & (mkt["DATE"] < m_to + pd.offsets.MonthBegin(1))]

        st.markdown("### Precios diarios")
        fig = go.Figure()
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_MAX"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_MIN"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(227,162,26,0.18)", name="Desvío: rango del día")
        fig.add_scatter(x=mkt["DATE"], y=mkt["SYSTEM_PRICE_AVG"], mode="lines", line=dict(color="#E3A21A", width=2), name="Desvío: media")
        fig.add_scatter(x=mkt["DATE"], y=mkt["MID_GBP_MWH"], mode="lines", line=dict(color=INK, width=2), name="Mercado (MID)")
        base_layout(fig, height=360)
        fig.update_yaxes(title_text="£/MWh")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Pagos diarios del operador")
        fig = go.Figure()
        fig.add_bar(x=mkt["DATE"], y=mkt["BM_TOTAL_GBP"], name="Balance", marker_color="#E3A21A")
        fig.add_bar(x=mkt["DATE"], y=mkt["NESO_SERVICES_GBP"], name="Frecuencia y reservas", marker_color="#1F9E89")
        base_layout(fig, height=320)
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="£")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Solo unidades registradas en Elexon. Las unidades agregadas que solo venden servicios a NESO no están incluidas.")

        eac = load("gb_eac_prices_daily")
        eac = eac[(eac["DATE"] >= m_from) & (eac["DATE"] < m_to + pd.offsets.MonthBegin(1))]
        st.markdown("### Precio de los servicios de NESO")
        products = sorted(eac["PRODUCT"].dropna().unique())
        default = [p for p in ["DCH", "DCL", "DMH", "DML", "DRH", "DRL"] if p in products]
        chosen = st.multiselect("Productos", products, default=default or products[:4])
        fig = go.Figure()
        for p in chosen:
            s = eac[eac["PRODUCT"] == p].sort_values("DATE")
            fig.add_scatter(x=s["DATE"], y=s["PRICE_GBP_MW_H"], mode="lines", name=p)
        base_layout(fig, height=340)
        fig.update_yaxes(title_text="£ por MW y hora")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Precio medio ponderado por MW adjudicados. D = dinámicos (C contención, M moderación, R regulación; H subir, L bajar). "
                   "BR, QR, SR = reservas de balance, rápida y lenta (P subir, N bajar).")
    except Exception as e:  # noqa: BLE001
        st.warning(f"No hay datos de mercado disponibles: {e}")
