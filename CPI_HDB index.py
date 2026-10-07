"""
Singapore birth rate vs. housing and consumer prices: exploratory dashboard.

Run with:   streamlit run app.py
Data:       sg_birth_hdb_cpi.csv (same folder as this file), or upload your own
            CSV with the same column names from the sidebar.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "sg_birth_hdb_cpi.csv"

YEAR = "Year"
CPI = "CPI_HDB_CPI_Consumer_Price_Index_for_Singapore"
HDB = "CPI_HDB_HDB_Resale_Price_Index"
REAL = "Real_HDB_Resale_Price_Index"  # derived: HDB index / CPI * 100
TFR = "BirthRate_Total_Fertility_Rate_TFR_Per_Female"

PRICE_LABELS = {
    HDB: "HDB Resale Price Index",
    CPI: "Consumer Price Index (CPI)",
    REAL: "Real HDB price (HDB index ÷ CPI)",
}
PRICE_SHORT = {HDB: "HDB resale prices", CPI: "CPI", REAL: "Real HDB prices"}

AGE_COLS = {
    "BirthRate_15_19_Years_Per_Thousand_Females": "15–19",
    "BirthRate_20_24_Years_Per_Thousand_Females": "20–24",
    "BirthRate_25_29_Years_Per_Thousand_Females": "25–29",
    "BirthRate_30_34_Years_Per_Thousand_Females": "30–34",
    "BirthRate_35_39_Years_Per_Thousand_Females": "35–39",
    "BirthRate_40_44_Years_Per_Thousand_Females": "40–44",
    "BirthRate_45_49_Years_Per_Thousand_Females": "45–49",
}
ETHNIC_COLS = {
    "BirthRate_Chinese_Per_Female": "Chinese",
    "BirthRate_Malays_Per_Female": "Malays",
    "BirthRate_Indians_Per_Female": "Indians",
}

BIRTH_LABELS = {
    TFR: "Total fertility rate (births per female)",
    "BirthRate_Crude_Birth_Rate_Per_Thousand_Residents": "Crude birth rate (per 1,000 residents)",
    "BirthRate_Total_Live_Births_Number": "Total live births",
    "BirthRate_Resident_Live_Births_Number": "Resident live births",
    "BirthRate_Citizen_Live_Births_Number": "Citizen live births",
    "BirthRate_Gross_Reproduction_Rate_Per_Female": "Gross reproduction rate (per female)",
    "BirthRate_Net_Reproduction_Rate_Per_Female": "Net reproduction rate (per female)",
}
BIRTH_LABELS.update({c: f"Births per 1,000 females aged {a}" for c, a in AGE_COLS.items()})
BIRTH_LABELS.update({c: f"Fertility rate, {e} (per female)" for c, e in ETHNIC_COLS.items()})

# One fixed colour per measure, so a series keeps its colour on every chart.
C_BIRTH = "#2a78d6"  # blue
C_HDB = "#eb6834"    # orange
C_CPI = "#1baf7a"    # aqua
C_MUTED = "#8a8a86"
PRICE_COLOR = {HDB: C_HDB, CPI: C_CPI, REAL: C_HDB}

LEVELS = "Levels"
YOY = "Year-on-year % change"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def birth_label(col: str) -> str:
    """Readable name for a birth-rate column (falls back to the raw name)."""
    return BIRTH_LABELS.get(col, col.replace("BirthRate_", "").replace("_", " "))


def fmt(v: float) -> str:
    """Format a value: thousands separator for counts, two decimals for rates."""
    if pd.isna(v):
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def pct_change(first: float, last: float) -> float:
    return (last / first - 1) * 100 if first else np.nan


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


@st.cache_data
def load_data(source) -> pd.DataFrame:
    """Read the CSV, keep numeric columns, sort by year, add the real HDB index."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in (YEAR, CPI, HDB) if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[YEAR]).sort_values(YEAR).reset_index(drop=True)
    df[YEAR] = df[YEAR].astype(int)
    df[REAL] = df[HDB] / df[CPI] * 100
    return df


def pair_series(df: pd.DataFrame, x_col: str, y_col: str, mode: str, lag: int) -> pd.DataFrame:
    """Return Year / x / y, optionally as year-on-year % change, with x lagged.

    lag = 1 pairs the birth measure in year t with the price indicator in t-1.
    """
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    if mode == YOY:
        x = x.pct_change() * 100
        y = y.pct_change() * 100
    out = pd.DataFrame({YEAR: df[YEAR].values, "x": x.shift(lag).values, "y": y.values})
    return out.dropna().reset_index(drop=True)


def corr_stats(x, y):
    """Pearson r, p-value, n and least-squares line. None if it cannot be computed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return None
    r, p = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {"r": float(r), "p": float(p), "n": int(len(x)),
            "slope": float(slope), "intercept": float(intercept)}


def strength(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


def base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        margin=dict(l=10, r=20, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def end_labels(values, label: str):
    """Text list that labels only the last point of a line."""
    return [""] * (len(values) - 1) + [label]


def line_trace(x, y, name: str, color: str, label: str = None, value_fmt: str = ",.2f") -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name,
        mode="lines+markers+text" if label else "lines+markers",
        text=end_labels(y, label) if label else None,
        textposition="middle right",
        cliponaxis=False,
        line=dict(color=color, width=2),
        marker=dict(size=8, color=color),
        hovertemplate=name + ": %{y:" + value_fmt + "}<extra></extra>",
    )


def _chart_kwargs() -> dict:
    """Full-width charts on both newer and older Streamlit versions."""
    try:
        params = inspect.signature(st.plotly_chart).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, **CHART_KW)


# --------------------------------------------------------------------------
# Page setup and data
# --------------------------------------------------------------------------
st.set_page_config(page_title="Singapore births vs. prices", layout="wide")
CHART_KW = _chart_kwargs()

st.title("Do rising prices go with fewer births in Singapore?")
st.caption(
    "Explore how birth-rate measures move against the HDB Resale Price Index and the "
    "Consumer Price Index. Use the sidebar to change the measure, years, and test."
)

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader(
        "Use your own CSV (optional)", type="csv",
        help="Needs the columns Year, the HDB index, the CPI, and BirthRate_* columns "
             "named as in the bundled file.",
    )

try:
    data = load_data(upload if upload is not None else str(DATA_FILE))
except FileNotFoundError:
    st.error(f"Could not find {DATA_FILE.name}. Keep it in the same folder as app.py, "
             "or upload a CSV from the sidebar.")
    st.stop()
except Exception as exc:  # bad upload: show the reason rather than a traceback
    st.error(f"Could not read the data: {exc}")
    st.stop()

birth_cols = [c for c in BIRTH_LABELS if c in data.columns]
birth_cols += [c for c in data.columns if c.startswith("BirthRate_") and c not in birth_cols]
if not birth_cols:
    st.error("No BirthRate_* columns found in the data.")
    st.stop()

with st.sidebar:
    st.header("Explore")
    y_min, y_max = int(data[YEAR].min()), int(data[YEAR].max())
    if y_min < y_max:
        year_from, year_to = st.slider("Years", y_min, y_max, (y_min, y_max))
    else:
        year_from, year_to = y_min, y_max

    birth_col = st.selectbox("Birth-rate measure", birth_cols, format_func=birth_label)
    price_col = st.radio("Price indicator", list(PRICE_LABELS), format_func=PRICE_LABELS.get)

    st.header("Relationship test")
    mode = st.radio(
        "Compare", [LEVELS, YOY],
        help="Levels compares the values themselves. Year-on-year % change removes the "
             "shared trend and is the stricter test.",
    )
    lag = st.slider(
        "Price lead (years)", 0, 3, 0,
        help="1 = compare births in a year with prices one year earlier.",
    )
    st.caption("Annual data. With so few years, treat every result as indicative only.")

df = data[(data[YEAR] >= year_from) & (data[YEAR] <= year_to)].reset_index(drop=True)
if len(df) < 3:
    st.warning("Select at least three years to draw the charts.")
    st.stop()

b_name = birth_label(birth_col)
p_name = PRICE_LABELS[price_col]
p_short = PRICE_SHORT[price_col]
first, last = df.iloc[0], df.iloc[-1]
y0, y1 = int(first[YEAR]), int(last[YEAR])

# --------------------------------------------------------------------------
# Headline numbers
# --------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{b_name}, {y1}", fmt(last[birth_col]),
          f"{pct_change(first[birth_col], last[birth_col]):+.1f}% since {y0}")
k2.metric(f"HDB Resale Price Index, {y1}", fmt(last[HDB]),
          f"{pct_change(first[HDB], last[HDB]):+.1f}% since {y0}", delta_color="off")
k3.metric(f"Consumer Price Index, {y1}", fmt(last[CPI]),
          f"{pct_change(first[CPI], last[CPI]):+.1f}% since {y0}", delta_color="off")
k4.metric(f"Real HDB price (HDB ÷ CPI), {y1}", fmt(last[REAL]),
          f"{pct_change(first[REAL], last[REAL]):+.1f}% since {y0}", delta_color="off")

tab_trend, tab_rel, tab_groups, tab_data = st.tabs(
    ["Trends", "Prices vs. births", "Who is having fewer babies?", "Data"]
)

# --------------------------------------------------------------------------
# Tab 1: trends
# --------------------------------------------------------------------------
with tab_trend:
    left, right = st.columns(2)

    with left:
        value_fmt = ",.0f" if df[birth_col].abs().max() >= 1000 else ",.2f"
        fig = go.Figure(line_trace(df[YEAR], df[birth_col], b_name, C_BIRTH, value_fmt=value_fmt))
        base_layout(fig, b_name)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(dtick=1)
        show(fig)

    with right:
        fig = go.Figure()
        fig.add_trace(line_trace(df[YEAR], df[HDB], "HDB Resale Price Index", C_HDB, "HDB"))
        fig.add_trace(line_trace(df[YEAR], df[CPI], "Consumer Price Index", C_CPI, "CPI"))
        base_layout(fig, "Price indices")
        fig.update_xaxes(dtick=1, range=[y0 - 0.3, y1 + 0.9])
        show(fig)

    # Everything rebased to 100 in the first selected year: one axis, one unit.
    st.subheader(f"Change since {y0}, all rebased to 100")
    fig = go.Figure()
    for col, name, color, short in [
        (birth_col, b_name, C_BIRTH, "Births"),
        (HDB, "HDB Resale Price Index", C_HDB, "HDB"),
        (CPI, "Consumer Price Index", C_CPI, "CPI"),
    ]:
        rebased = df[col] / first[col] * 100
        fig.add_trace(line_trace(df[YEAR], rebased, name, color, short, ".1f"))
    fig.add_hline(y=100, line_dash="dot", line_color=C_MUTED, line_width=1)
    base_layout(fig, f"Index, {y0} = 100", height=420)
    fig.update_xaxes(dtick=1, range=[y0 - 0.3, y1 + 0.9])
    show(fig)
    st.markdown(
        f"From **{y0}** to **{y1}**: {b_name.lower()} changed by "
        f"**{pct_change(first[birth_col], last[birth_col]):+.1f}%**, HDB resale prices by "
        f"**{pct_change(first[HDB], last[HDB]):+.1f}%**, and consumer prices by "
        f"**{pct_change(first[CPI], last[CPI]):+.1f}%**."
    )

# --------------------------------------------------------------------------
# Tab 2: relationship between the price indicator and the birth measure
# --------------------------------------------------------------------------
with tab_rel:
    pairs = pair_series(df, price_col, birth_col, mode, lag)
    res = corr_stats(pairs["x"], pairs["y"]) if len(pairs) else None

    unit = " (YoY % change)" if mode == YOY else ""
    lag_txt = f", {lag} year(s) earlier" if lag else ""
    x_title = f"{p_name}{unit}{lag_txt}"
    y_title = f"{b_name}{unit}"

    if res is None:
        st.warning("Not enough overlapping years for this combination. "
                   "Widen the year range or reduce the price lead.")
    else:
        chart_col, stat_col = st.columns([3, 2])

        with chart_col:
            xs = np.linspace(pairs["x"].min(), pairs["x"].max(), 50)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=res["slope"] * xs + res["intercept"], mode="lines",
                name="Linear fit", line=dict(color=C_MUTED, width=2, dash="dash"),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=pairs["x"], y=pairs["y"], mode="markers+text", name="Year",
                text=pairs[YEAR].astype(str), textposition="top center",
                marker=dict(size=11, color=C_BIRTH, line=dict(width=1, color="white")),
                hovertemplate="<b>%{text}</b><br>" + p_short + ": %{x:,.2f}<br>"
                              + "Births: %{y:,.2f}<extra></extra>",
            ))
            base_layout(fig, "Each dot is one year", height=440)
            fig.update_layout(hovermode="closest", showlegend=False,
                              xaxis_title=x_title, yaxis_title=y_title)
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
            show(fig)

        with stat_col:
            m1, m2 = st.columns(2)
            m1.metric("Correlation (r)", f"{res['r']:+.2f}")
            m2.metric("R²", f"{res['r'] ** 2:.2f}")
            m3, m4 = st.columns(2)
            m3.metric("p-value", fmt_p(res["p"]))
            m4.metric("Years compared", res["n"])

            direction = "lower" if res["r"] < 0 else "higher"
            significant = res["p"] < 0.05
            if mode == LEVELS:
                st.markdown(
                    f"**Reading:** years with a higher {p_name} had a **{direction}** "
                    f"{b_name.lower()}. The association is **{strength(res['r'])}** and "
                    f"{'statistically significant' if significant else 'not statistically significant'} "
                    f"at the 5% level."
                )
                st.info(
                    "Both series trend steadily over time, so a level correlation is easy "
                    "to inflate. Switch **Compare** to *Year-on-year % change* to check "
                    "whether years with faster price rises also saw sharper falls in births."
                )
            else:
                st.markdown(
                    f"**Reading:** in years when {p_short} grew faster, "
                    f"{b_name.lower()} grew **{'slower' if res['r'] < 0 else 'faster'}**. "
                    f"The association is **{strength(res['r'])}** and "
                    f"{'statistically significant' if significant else 'not statistically significant'} "
                    f"at the 5% level."
                )
                st.info(
                    "This is the stricter test: it removes the shared trend and asks "
                    "whether the year-to-year swings move together."
                )
            st.caption("Correlation does not show causation, and a handful of annual "
                       "observations cannot rule out chance or other drivers.")

    # Correlation at each price lead, for the current measure and comparison mode.
    st.subheader("Does it matter how far prices lead births?")
    lag_rows = []
    for k in range(0, 4):
        pk = pair_series(df, price_col, birth_col, mode, k)
        rk = corr_stats(pk["x"], pk["y"]) if len(pk) else None
        if rk:
            lag_rows.append({"lag": k, "r": rk["r"], "p": rk["p"], "n": rk["n"]})
    if lag_rows:
        lag_df = pd.DataFrame(lag_rows)
        fig = go.Figure(go.Bar(
            x=[f"{k} yr" for k in lag_df["lag"]], y=lag_df["r"],
            marker_color=C_BIRTH,
            marker_opacity=[1.0 if k == lag else 0.45 for k in lag_df["lag"]],
            text=[f"{r:+.2f}" for r in lag_df["r"]], textposition="outside",
            customdata=np.column_stack([lag_df["n"], [fmt_p(p) for p in lag_df["p"]]]),
            hovertemplate="Price lead %{x}<br>r = %{y:+.2f}<br>years compared: "
                          "%{customdata[0]}<br>p = %{customdata[1]}<extra></extra>",
        ))
        base_layout(fig, f"Correlation between {p_short} and {b_name.lower()}, "
                         f"by price lead ({mode.lower()})", height=340)
        fig.update_layout(hovermode="closest", showlegend=False,
                          xaxis_title="Years by which prices lead births",
                          yaxis_title="Correlation (r)")
        fig.update_yaxes(range=[-1.15, 1.15], zeroline=True,
                         zerolinecolor=C_MUTED, zerolinewidth=1)
        show(fig)
        st.caption("The highlighted bar is the price lead chosen in the sidebar. "
                   "Each extra year of lead drops one observation.")
    else:
        st.warning("Not enough years to compare different price leads.")

# --------------------------------------------------------------------------
# Tab 3: breakdown by age of mother or ethnic group
# --------------------------------------------------------------------------
with tab_groups:
    options = {}
    if any(c in df.columns for c in AGE_COLS):
        options["Age of mother"] = {c: a for c, a in AGE_COLS.items() if c in df.columns}
    if any(c in df.columns for c in ETHNIC_COLS):
        options["Ethnic group"] = {c: e for c, e in ETHNIC_COLS.items() if c in df.columns}

    if not options:
        st.info("The data has no age-group or ethnic-group birth-rate columns.")
    else:
        pick_col, show_col = st.columns(2)
        by = pick_col.radio("Break down by", list(options), horizontal=True)
        change_as = show_col.radio("Show change as", ["Absolute change", "% change"],
                                   horizontal=True)
        absolute = change_as == "Absolute change"
        groups = options[by]
        unit_txt = "births per 1,000 females" if by == "Age of mother" else "births per female"

        rows = []
        for col, label in groups.items():
            pk = pair_series(df, price_col, col, mode, lag)
            rk = corr_stats(pk["x"], pk["y"]) if len(pk) else None
            rows.append({
                "Group": label,
                f"{y0}": first[col],
                f"{y1}": last[col],
                "Change (%)": pct_change(first[col], last[col]),
                "Change": last[col] - first[col],
                "r": rk["r"] if rk else np.nan,
            })
        g = pd.DataFrame(rows)

        left, right = st.columns(2)
        with left:
            change_col = "Change" if absolute else "Change (%)"
            y_axis = f"Change in {unit_txt}" if absolute else "Change (%)"
            bar_text = ([f"{v:+.2f}" for v in g["Change"]] if absolute
                        else [f"{v:+.0f}%" for v in g["Change (%)"]])
            fig = go.Figure(go.Bar(
                x=g["Group"], y=g[change_col], marker_color=C_BIRTH,
                text=bar_text, textposition="outside",
                customdata=np.column_stack([g[f"{y0}"], g[f"{y1}"], g["Change"], g["Change (%)"]]),
                hovertemplate="<b>%{x}</b><br>" + f"{y0}: " + "%{customdata[0]:,.2f}<br>"
                              + f"{y1}: " + "%{customdata[1]:,.2f}<br>"
                              + "Change: %{customdata[2]:+.2f} (%{customdata[3]:+.1f}%)"
                              + "<extra></extra>",
            ))
            base_layout(fig, f"Change in fertility rate, {y0} to {y1}")
            fig.update_layout(hovermode="closest", showlegend=False, yaxis_title=y_axis)
            fig.update_yaxes(zeroline=True, zerolinecolor=C_MUTED, zerolinewidth=1)
            show(fig)

        with right:
            fig = go.Figure(go.Bar(
                x=g["Group"], y=g["r"], marker_color=PRICE_COLOR[price_col],
                text=["n/a" if pd.isna(v) else f"{v:+.2f}" for v in g["r"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>r = %{y:+.2f}<extra></extra>",
            ))
            base_layout(fig, f"Correlation with {p_short} ({mode.lower()}"
                             + (f", lead {lag} yr" if lag else "") + ")")
            fig.update_layout(hovermode="closest", showlegend=False,
                              yaxis_title="Correlation (r)")
            fig.update_yaxes(range=[-1.15, 1.15], zeroline=True,
                             zerolinecolor=C_MUTED, zerolinewidth=1)
            show(fig)

        valid = g.dropna(subset=["Change"])
        if len(valid):
            def describe(row) -> str:
                return (f"**{row['Group']}** ({row['Change']:+.2f} {unit_txt}, "
                        f"{row['Change (%)']:+.1f}%)")
            worst = valid.loc[valid["Change"].idxmin()]
            best = valid.loc[valid["Change"].idxmax()]
            msg = f"Largest fall from {y0} to {y1}: {describe(worst)}. "
            if best["Change"] > 0:
                msg += f"Fertility **rose** for {describe(best)}."
            else:
                msg += f"Smallest fall: {describe(best)}."
            st.markdown(msg)
        if by == "Age of mother":
            st.caption("Rates for the youngest and oldest age groups are very small, so their "
                       "percentage changes swing widely on tiny absolute differences.")

# --------------------------------------------------------------------------
# Tab 4: data
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("Correlation of every birth measure with the chosen price indicator")
    rows = []
    for col in birth_cols:
        pk = pair_series(df, price_col, col, mode, lag)
        rk = corr_stats(pk["x"], pk["y"]) if len(pk) else None
        if rk:
            rows.append({"Birth measure": birth_label(col), "r": round(rk["r"], 2),
                         "p-value": fmt_p(rk["p"]), "Years compared": rk["n"]})
    if rows:
        st.caption(f"{p_name}, {mode.lower()}" + (f", price lead {lag} yr" if lag else ""))
        st.dataframe(pd.DataFrame(rows).sort_values("r").reset_index(drop=True),
                     hide_index=True)

    st.subheader(f"Data used, {y0} to {y1}")
    table = df.round({REAL: 2}).rename(
        columns={**{c: birth_label(c) for c in birth_cols}, **PRICE_LABELS})
    st.dataframe(table, hide_index=True)
    st.download_button(
        "Download this table as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"sg_birth_prices_{y0}_{y1}.csv",
        mime="text/csv",
    )
