from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

INDICATOR_PATTERN = re.compile(r"^i\d+$", flags=re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VALUES_CANDIDATES = [
    PROJECT_ROOT / "valeur_externe.csv",
    PROJECT_ROOT / "valeurs_externes.csv",
    PROJECT_ROOT.parent / "valeur_externe.csv",
    PROJECT_ROOT.parent / "valeurs_externes.csv",
]

BOUND_METHODS = ["data", "p01_p99", "p05_p95", "iqr_clip"]
BOUND_LABELS = {
    "data": "Min/Max observes",
    "p01_p99": "Percentiles 1% / 99%",
    "p05_p95": "Percentiles 5% / 95%",
    "iqr_clip": "IQR clip (Q1-1.5IQR, Q3+1.5IQR)",
}

DIRECTION_ASC = "Croissant (plus grand = meilleur)"
DIRECTION_DESC = "Decroissant (plus petit = meilleur)"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def normalize_col_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def parse_k_grid(raw_text: str) -> tuple[list[float], list[str]]:
    tokens = [tok for tok in re.split(r"[\s,;]+", str(raw_text).strip()) if tok]
    values: list[float] = []
    bad_tokens: list[str] = []

    for token in tokens:
        try:
            values.append(float(token.replace(",", ".")))
        except ValueError:
            bad_tokens.append(token)

    return sorted(set(values)), bad_tokens


@st.cache_data(show_spinner=False)
def load_values_dataset(values_path: str) -> tuple[pd.DataFrame, list[str]]:
    path = Path(values_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]

    indicator_cols = [
        col
        for col in df.columns
        if INDICATOR_PATTERN.match(str(col).strip())
    ]

    if not indicator_cols:
        numeric_cols = []
        for col in df.columns:
            parsed = pd.to_numeric(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                numeric_cols.append(col)
        indicator_cols = numeric_cols

    if not indicator_cols:
        raise ValueError(
            "Aucune colonne exploitable trouvee. Verifiez le fichier valeur_externe.csv"
        )

    for col in indicator_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, sorted(indicator_cols, key=lambda x: str(x).lower())


def compute_candidate_bounds(raw_series: pd.Series) -> dict | None:
    series = pd.to_numeric(raw_series, errors="coerce").dropna()
    if series.empty:
        return None

    data_min = float(series.min())
    data_max = float(series.max())

    if np.isclose(data_min, data_max):
        return {
            "data": (data_min, data_max),
            "p01_p99": (data_min, data_max),
            "p05_p95": (data_min, data_max),
            "iqr_clip": (data_min, data_max),
            "q25": data_min,
            "q50": data_min,
            "q75": data_min,
            "outlier_rate": 0.0,
            "auto_method": "data",
        }

    q01, q05, q25, q50, q75, q95, q99 = (
        series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .astype(float)
        .tolist()
    )

    iqr = max(q75 - q25, 0.0)
    iqr_min = max(data_min, q25 - 1.5 * iqr)
    iqr_max = min(data_max, q75 + 1.5 * iqr)

    if np.isclose(iqr_min, iqr_max):
        iqr_min, iqr_max = data_min, data_max

    outlier_rate = float(((series < iqr_min) | (series > iqr_max)).mean())

    if outlier_rate > 0.12:
        auto_method = "p05_p95"
    elif outlier_rate > 0.05:
        auto_method = "p01_p99"
    else:
        auto_method = "data"

    return {
        "data": (data_min, data_max),
        "p01_p99": (float(q01), float(q99)),
        "p05_p95": (float(q05), float(q95)),
        "iqr_clip": (float(iqr_min), float(iqr_max)),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "outlier_rate": outlier_rate,
        "auto_method": auto_method,
    }


def normalize_0_1(
    values: pd.Series | np.ndarray,
    x_min: float,
    x_max: float,
    direction: str,
) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    vals = parsed if isinstance(parsed, pd.Series) else pd.Series(parsed)
    out = pd.Series(np.nan, index=vals.index, dtype=float)

    if pd.isna(x_min) or pd.isna(x_max) or np.isclose(x_min, x_max):
        return out

    low = min(float(x_min), float(x_max))
    high = max(float(x_min), float(x_max))

    if direction == DIRECTION_ASC:
        out = (vals - low) / (high - low)
    else:
        out = (high - vals) / (high - low)

    return out.clip(0.0, 1.0)


def score_0_100(
    values: pd.Series | np.ndarray,
    x_min: float,
    x_max: float,
    k: float,
    direction: str,
) -> pd.Series:
    scaled = normalize_0_1(values, x_min=x_min, x_max=x_max, direction=direction)
    if np.isclose(k, 0.0):
        score = scaled
    else:
        denominator = np.expm1(k)
        if np.isclose(denominator, 0.0):
            score = scaled
        else:
            score = np.expm1(k * scaled) / denominator
    return (100.0 * score).clip(0.0, 100.0)


def summarize_series(series: pd.Series, label: str) -> dict:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return {
            "serie": label,
            "n": 0,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "mediane": np.nan,
            "mean": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "std": np.nan,
        }

    return {
        "serie": label,
        "n": int(clean.size),
        "min": round(float(clean.min()), 6),
        "p01": round(float(clean.quantile(0.01)), 6),
        "p05": round(float(clean.quantile(0.05)), 6),
        "mediane": round(float(clean.quantile(0.50)), 6),
        "mean": round(float(clean.mean()), 6),
        "p95": round(float(clean.quantile(0.95)), 6),
        "p99": round(float(clean.quantile(0.99)), 6),
        "max": round(float(clean.max()), 6),
        "std": round(float(clean.std(ddof=0)), 6),
    }


def compute_k_diagnostics(
    raw_series: pd.Series,
    x_min: float,
    x_max: float,
    k_grid: list[float],
    direction: str,
    target_std: float,
    extreme_low: float,
    extreme_high: float,
) -> pd.DataFrame:
    raw = pd.to_numeric(raw_series, errors="coerce").dropna()
    rows = []

    for k in sorted(set(float(v) for v in k_grid if float(v) > 0.0)):
        score_vals = score_0_100(raw, x_min=x_min, x_max=x_max, k=k, direction=direction)
        valid = pd.Series(score_vals).dropna()
        if valid.empty:
            continue

        mean_score = float(valid.mean())
        std_score = float(valid.std(ddof=0))
        low_rate = float((valid <= extreme_low).mean())
        high_rate = float((valid >= extreme_high).mean())
        extreme_rate = float(((valid <= extreme_low) | (valid >= extreme_high)).mean())

        center_penalty = min(abs(mean_score - 50.0) / 50.0, 1.0)
        spread_penalty = min(abs(std_score - target_std) / max(target_std, 1e-9), 1.0)
        extreme_penalty = min(extreme_rate, 1.0)

        raw_balance = 1.0 - (
            0.45 * center_penalty
            + 0.35 * spread_penalty
            + 0.20 * extreme_penalty
        )
        equilibre = max(0.0, min(1.0, raw_balance))

        rows.append(
            {
                "k": float(k),
                "mean_score": mean_score,
                "std_score": std_score,
                "low_rate": low_rate,
                "high_rate": high_rate,
                "extreme_rate": extreme_rate,
                "equilibre_score": 100.0 * equilibre,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "k",
                "mean_score",
                "std_score",
                "low_rate",
                "high_rate",
                "extreme_rate",
                "equilibre_score",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["equilibre_score", "k"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_candidate_table(
    candidates: dict,
    selected_method: str,
    x_min: float,
    x_max: float,
) -> pd.DataFrame:
    rows = []

    for method in BOUND_METHODS:
        bound_min, bound_max = candidates[method]
        rows.append(
            {
                "methode": method,
                "libelle": BOUND_LABELS[method],
                "x_min": round(float(bound_min), 6),
                "x_max": round(float(bound_max), 6),
                "retenue": "oui" if method == selected_method else "",
            }
        )

    rows.append(
        {
            "methode": "applique",
            "libelle": "Bornes effectivement appliquees",
            "x_min": round(float(x_min), 6),
            "x_max": round(float(x_max), 6),
            "retenue": "oui",
        }
    )

    return pd.DataFrame(rows)


def build_indicator_figure(
    raw_series: pd.Series,
    candidates: dict,
    x_min: float,
    x_max: float,
    k_selected: float,
    compare_k: list[float],
    bins: int,
    direction: str,
):
    raw = pd.to_numeric(raw_series, errors="coerce").dropna()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    data_min, data_max = candidates["data"]
    p01, p99 = candidates["p01_p99"]
    p05, p95 = candidates["p05_p95"]
    iqr_min, iqr_max = candidates["iqr_clip"]

    axes[0, 0].hist(raw, bins=bins, color="#7da0ca", alpha=0.85, edgecolor="white")
    axes[0, 0].axvline(data_min, color="gray", linestyle="-", linewidth=1, label="data min")
    axes[0, 0].axvline(data_max, color="gray", linestyle="-", linewidth=1, label="data max")
    axes[0, 0].axvline(p01, color="#2ca02c", linestyle="--", linewidth=1.3, label="p01")
    axes[0, 0].axvline(p99, color="#2ca02c", linestyle="--", linewidth=1.3, label="p99")
    axes[0, 0].axvline(p05, color="#ff7f0e", linestyle="--", linewidth=1.8, label="p05")
    axes[0, 0].axvline(p95, color="#ff7f0e", linestyle="--", linewidth=1.8, label="p95")
    axes[0, 0].axvline(iqr_min, color="#9467bd", linestyle=":", linewidth=1.5, label="iqr min")
    axes[0, 0].axvline(iqr_max, color="#9467bd", linestyle=":", linewidth=1.5, label="iqr max")
    axes[0, 0].axvline(x_min, color="red", linestyle="-", linewidth=2.3, label="x_min retenu")
    axes[0, 0].axvline(x_max, color="red", linestyle="-", linewidth=2.3, label="x_max retenu")
    axes[0, 0].set_title("Histogramme + bornes candidates")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("Frequence")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].boxplot(
        raw.values,
        vert=False,
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#d9f0d3", "edgecolor": "#4d9221"},
        medianprops={"color": "#005a32", "linewidth": 2.0},
    )
    axes[0, 1].axvline(x_min, color="red", linestyle="-", linewidth=2.2, label="x_min")
    axes[0, 1].axvline(x_max, color="red", linestyle="-", linewidth=2.2, label="x_max")
    axes[0, 1].set_title("Boxplot + bornes retenues")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_yticks([])
    axes[0, 1].legend(fontsize=8)

    x_sorted = np.sort(raw.values)
    y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    axes[1, 0].plot(x_sorted, y_cdf, color="#1f77b4", linewidth=2)
    axes[1, 0].axvline(x_min, color="red", linestyle="-", linewidth=2)
    axes[1, 0].axvline(x_max, color="red", linestyle="-", linewidth=2)
    axes[1, 0].set_title("CDF brute")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Part cumulee")
    axes[1, 0].set_ylim(0, 1)

    x_low = float(raw.min())
    x_high = float(raw.max())
    if np.isclose(x_low, x_high):
        x_low -= 1.0
        x_high += 1.0
    x_grid = np.linspace(x_low, x_high, 400)

    k_values = sorted(set([float(k) for k in compare_k if float(k) > 0.0] + [float(k_selected)]))
    for k_value in k_values:
        y_grid = score_0_100(
            pd.Series(x_grid),
            x_min=x_min,
            x_max=x_max,
            k=float(k_value),
            direction=direction,
        )
        is_selected = np.isclose(k_value, k_selected)
        axes[1, 1].plot(
            x_grid,
            y_grid.values,
            linewidth=2.6 if is_selected else 1.6,
            alpha=1.0 if is_selected else 0.8,
            label=f"k={k_value:g}" + (" (choisi)" if is_selected else ""),
        )

    axes[1, 1].set_title("Impact de k sur y(x)")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y(x) sur 0-100")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Direction={direction} | x_min={x_min:.4g} | x_max={x_max:.4g} | k={k_selected:g}",
        fontsize=12,
    )
    return fig


def build_k_diagnostic_figure(df_k: pd.DataFrame, best_k: float):
    by_k = df_k.sort_values("k").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)

    axes[0].plot(by_k["k"], by_k["equilibre_score"], marker="o", color="#3182bd", linewidth=2)
    axes[0].axvline(best_k, color="red", linestyle="--", linewidth=1.5)
    axes[0].scatter([best_k], [float(df_k.iloc[0]["equilibre_score"])], color="red", zorder=3)
    axes[0].set_title("Score d equilibre selon k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Equilibre / 100")
    axes[0].set_ylim(0, 100)

    axes[1].plot(by_k["k"], by_k["mean_score"], marker="o", color="#31a354", label="mean score")
    axes[1].plot(by_k["k"], by_k["std_score"], marker="o", color="#756bb1", label="std score")
    axes[1].plot(
        by_k["k"],
        100.0 * by_k["extreme_rate"],
        marker="o",
        color="#e6550d",
        label="extreme rate (%)",
    )
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Mean / std / extremes selon k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Valeur")
    axes[1].legend(fontsize=8)

    return fig


def build_score_distribution_figure(
    scores: pd.Series,
    bins: int,
    extreme_low: float,
    extreme_high: float,
):
    series = pd.Series(scores).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)

    hist_bins = max(10, int(bins / 2))
    axes[0].hist(series, bins=hist_bins, color="#74c476", alpha=0.85, edgecolor="white")
    axes[0].axvline(extreme_low, color="#e6550d", linestyle="--", linewidth=1.5, label="extreme bas")
    axes[0].axvline(extreme_high, color="#e6550d", linestyle="--", linewidth=1.5, label="extreme haut")
    axes[0].set_title("Distribution des scores (k choisi)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequence")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(
        series.values,
        vert=False,
        widths=0.35,
        patch_artist=True,
        boxprops={"facecolor": "#c6dbef", "edgecolor": "#08519c"},
        medianprops={"color": "#08306b", "linewidth": 2.0},
    )
    axes[1].axvline(extreme_low, color="#e6550d", linestyle="--", linewidth=1.5)
    axes[1].axvline(extreme_high, color="#e6550d", linestyle="--", linewidth=1.5)
    axes[1].set_title("Boxplot des scores (k choisi)")
    axes[1].set_xlabel("Score")
    axes[1].set_yticks([])

    return fig


@st.cache_data(show_spinner=False)
def compute_global_summary(df_values: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    rows = []
    for ind in indicator_cols:
        series = pd.to_numeric(df_values[ind], errors="coerce").dropna()
        if series.empty:
            continue

        candidates = compute_candidate_bounds(series)
        if candidates is None:
            continue

        rows.append(
            {
                "indicator": ind,
                "n": int(series.size),
                "min": float(series.min()),
                "p05": float(series.quantile(0.05)),
                "mediane": float(series.quantile(0.50)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "outlier_rate_iqr": float(candidates["outlier_rate"]),
                "auto_method": str(candidates["auto_method"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "indicator",
                "n",
                "min",
                "p05",
                "mediane",
                "p95",
                "max",
                "outlier_rate_iqr",
                "auto_method",
            ]
        )

    return pd.DataFrame(rows).sort_values("indicator").reset_index(drop=True)


st.set_page_config(page_title="Diagnostic x_min / x_max / k", layout="wide")
st.title("Diagnostic interactif x_min, x_max et k")
st.caption(
    "Choisissez un indicateur dans valeur_externe.csv puis ajustez les bornes et la courbure k pour calibrer y(x)."
)

st.markdown("Formule utilisee (k > 0):")
st.latex(r"u(x)=\mathrm{clip}\left(\frac{x-x_{\min}}{x_{\max}-x_{\min}},0,1\right)")
st.latex(r"u_{dec}(x)=\mathrm{clip}\left(\frac{x_{\max}-x}{x_{\max}-x_{\min}},0,1\right)")
st.latex(r"y(x)=100\times\frac{e^{k\,u(x)}-1}{e^k-1}")

with st.sidebar:
    st.header("1) Donnees")
    values_path = st.text_input(
        "Chemin valeur_externe.csv",
        value=str(first_existing(DEFAULT_VALUES_CANDIDATES)),
    )

    st.header("2) Reglages globaux")
    direction = st.selectbox(
        "Sens de la performance",
        options=[DIRECTION_ASC, DIRECTION_DESC],
        index=0,
    )
    bins = int(st.slider("Bins histogrammes", min_value=20, max_value=120, value=45, step=5))

    k_selected = float(
        st.number_input("k applique (k > 0)", min_value=0.01, max_value=50.0, value=2.0, step=0.1)
    )
    k_grid_text = st.text_input(
        "Grille de k (diagnostic)",
        value="0.1,0.2,0.5,1,1.5,2,3,5,8,12",
    )
    compare_k_text = st.text_input(
        "k compares sur la courbe",
        value="0.3,1,2,5",
    )

    target_std = float(
        st.slider("Cible ecart-type du score", min_value=5.0, max_value=35.0, value=18.0, step=0.5)
    )
    extreme_low = float(
        st.slider("Seuil extreme bas", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    )
    extreme_high = float(
        st.slider("Seuil extreme haut", min_value=70.0, max_value=100.0, value=95.0, step=0.5)
    )

if extreme_low >= extreme_high:
    st.error("Le seuil extreme bas doit etre strictement inferieur au seuil extreme haut.")
    st.stop()

k_grid_values, bad_k_tokens = parse_k_grid(k_grid_text)
removed_non_positive_k = [k for k in k_grid_values if k <= 0.0]
k_grid = [k for k in k_grid_values if k > 0.0]
k_grid = sorted(set(k_grid + [k_selected]))

compare_k_values, bad_compare_tokens = parse_k_grid(compare_k_text)
removed_non_positive_compare = [k for k in compare_k_values if k <= 0.0]
compare_k = [k for k in compare_k_values if k > 0.0]
compare_k = sorted(set(compare_k + [k_selected]))

if bad_k_tokens:
    st.sidebar.warning("Valeurs k ignorees: " + ", ".join(sorted(set(bad_k_tokens))))
if bad_compare_tokens:
    st.sidebar.warning("Valeurs compare_k ignorees: " + ", ".join(sorted(set(bad_compare_tokens))))
if removed_non_positive_k:
    st.sidebar.warning("k <= 0 ignores dans la grille de diagnostic.")
if removed_non_positive_compare:
    st.sidebar.warning("k <= 0 ignores dans les courbes comparees.")

if not k_grid:
    st.error("La grille de k est vide. Ajoutez au moins une valeur strictement positive.")
    st.stop()

try:
    with st.spinner("Chargement des donnees..."):
        df_values, indicator_cols = load_values_dataset(values_path)
except Exception as exc:
    st.error(f"Erreur de chargement: {exc}")
    st.stop()

if not indicator_cols:
    st.error("Aucun indicateur disponible.")
    st.stop()

default_index = 0
if "i005" in indicator_cols:
    default_index = indicator_cols.index("i005")

with st.sidebar:
    st.header("3) Choix indicateur")
    selected_indicator = st.selectbox(
        "Indicateur",
        options=indicator_cols,
        index=default_index,
    )

raw_series = pd.to_numeric(df_values[selected_indicator], errors="coerce").dropna()
if raw_series.empty:
    st.error("L indicateur choisi ne contient pas de valeurs numeriques exploitables.")
    st.stop()

candidates = compute_candidate_bounds(raw_series)
if candidates is None:
    st.error("Impossible de calculer des bornes candidates sur cet indicateur.")
    st.stop()

data_min, data_max = candidates["data"]
if np.isclose(data_min, data_max):
    st.warning("Toutes les valeurs de cet indicateur sont identiques. Le diagnostic est impossible.")
    st.stop()

auto_method = str(candidates["auto_method"])
source_options = ["auto"] + BOUND_METHODS
source_labels = {
    "auto": f"Auto ({BOUND_LABELS[auto_method]})",
    "data": BOUND_LABELS["data"],
    "p01_p99": BOUND_LABELS["p01_p99"],
    "p05_p95": BOUND_LABELS["p05_p95"],
    "iqr_clip": BOUND_LABELS["iqr_clip"],
}

with st.sidebar:
    st.header("4) Bornes x_min / x_max")
    selected_source = st.selectbox(
        "Source de bornes",
        options=source_options,
        format_func=lambda key: source_labels[key],
        index=0,
    )

    if selected_source == "auto":
        base_method = auto_method
    else:
        base_method = selected_source

    default_min, default_max = candidates[base_method]
    x_min, x_max = st.slider(
        "Ajuster x_min et x_max",
        min_value=float(data_min),
        max_value=float(data_max),
        value=(float(default_min), float(default_max)),
    )

    st.caption(
        f"Auto detecte: {BOUND_LABELS[auto_method]} | outliers IQR: {100.0 * candidates['outlier_rate']:.2f}%"
    )

if np.isclose(x_min, x_max):
    st.error("x_min et x_max ne doivent pas etre egaux.")
    st.stop()

method_display = base_method
if (not np.isclose(x_min, default_min)) or (not np.isclose(x_max, default_max)):
    method_display = f"{base_method} + ajustement manuel"

df_k_diag = compute_k_diagnostics(
    raw_series=raw_series,
    x_min=float(x_min),
    x_max=float(x_max),
    k_grid=k_grid,
    direction=direction,
    target_std=target_std,
    extreme_low=extreme_low,
    extreme_high=extreme_high,
)

if df_k_diag.empty:
    st.error("Impossible de calculer le diagnostic k avec les parametres fournis.")
    st.stop()

best_row = df_k_diag.iloc[0]
best_k = float(best_row["k"])

score_selected = score_0_100(
    raw_series,
    x_min=float(x_min),
    x_max=float(x_max),
    k=float(k_selected),
    direction=direction,
)
score_best = score_0_100(
    raw_series,
    x_min=float(x_min),
    x_max=float(x_max),
    k=float(best_k),
    direction=direction,
)

st.subheader(f"Indicateur: {selected_indicator}")

metric_cols = st.columns(6)
metric_cols[0].metric("Nb valeurs", f"{int(raw_series.size)}")
metric_cols[1].metric("x_min", f"{float(x_min):.4g}")
metric_cols[2].metric("x_max", f"{float(x_max):.4g}")
metric_cols[3].metric("k choisi", f"{float(k_selected):g}")
metric_cols[4].metric("k recommande", f"{best_k:g}")
metric_cols[5].metric("Equilibre best", f"{float(best_row['equilibre_score']):.2f} / 100")

st.write(
    "Methode active:",
    method_display,
    "| Mean score (k choisi):",
    f"{float(pd.Series(score_selected).dropna().mean()):.2f}",
    "| Std score (k choisi):",
    f"{float(pd.Series(score_selected).dropna().std(ddof=0)):.2f}",
)

fig_main = build_indicator_figure(
    raw_series=raw_series,
    candidates=candidates,
    x_min=float(x_min),
    x_max=float(x_max),
    k_selected=float(k_selected),
    compare_k=compare_k,
    bins=bins,
    direction=direction,
)
st.pyplot(fig_main, use_container_width=True)
plt.close(fig_main)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Bornes candidates")
    st.dataframe(
        build_candidate_table(
            candidates=candidates,
            selected_method=base_method,
            x_min=float(x_min),
            x_max=float(x_max),
        ),
        use_container_width=True,
    )

with right_col:
    st.markdown("### Resume statistique")
    stats_df = pd.DataFrame(
        [
            summarize_series(raw_series, "x brut"),
            summarize_series(score_selected, f"score(k={k_selected:g})"),
            summarize_series(score_best, f"score(k={best_k:g})"),
        ]
    )
    st.dataframe(stats_df, use_container_width=True)

st.markdown("### Diagnostic de la courbure k")
fig_k = build_k_diagnostic_figure(df_k_diag, best_k=best_k)
st.pyplot(fig_k, use_container_width=True)
plt.close(fig_k)

st.dataframe(
    df_k_diag[["k", "equilibre_score", "mean_score", "std_score", "low_rate", "high_rate", "extreme_rate"]],
    use_container_width=True,
)

st.markdown("### Distribution des scores (k choisi)")
fig_scores = build_score_distribution_figure(
    scores=score_selected,
    bins=bins,
    extreme_low=extreme_low,
    extreme_high=extreme_high,
)
st.pyplot(fig_scores, use_container_width=True)
plt.close(fig_scores)

with st.expander("Vue globale des indicateurs"):
    global_summary = compute_global_summary(df_values, indicator_cols)
    st.dataframe(global_summary, use_container_width=True)
    st.download_button(
        label="Telecharger resume indicateurs (CSV)",
        data=global_summary.to_csv(index=False).encode("utf-8"),
        file_name="resume_indicateurs.csv",
        mime="text/csv",
    )

export_df = df_values.copy()
export_df[f"score_{selected_indicator}_k_{k_selected:g}"] = score_0_100(
    df_values[selected_indicator],
    x_min=float(x_min),
    x_max=float(x_max),
    k=float(k_selected),
    direction=direction,
)
export_df[f"score_{selected_indicator}_k_best_{best_k:g}"] = score_0_100(
    df_values[selected_indicator],
    x_min=float(x_min),
    x_max=float(x_max),
    k=float(best_k),
    direction=direction,
)

st.download_button(
    label="Telecharger CSV avec scores calcules",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"scores_{selected_indicator}.csv",
    mime="text/csv",
)