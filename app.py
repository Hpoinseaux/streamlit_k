from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

INDICATOR_PATTERN = re.compile(r"^i\d+$", flags=re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VALUES_CANDIDATES = [
    PROJECT_ROOT / "valeur_externe.csv",
    PROJECT_ROOT / "streamlit_diag" / "source" / "valeur_externe.csv",
]
DEFAULT_PARAMS_CANDIDATES = [
    PROJECT_ROOT / "streamlit_diag" / "source" / "Parametres_indicateurs.csv",
]


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def normalize_col_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def find_col(columns, expected_keys: list[str]) -> str | None:
    norm_map = {normalize_col_name(col): col for col in columns}
    for key in expected_keys:
        found = norm_map.get(normalize_col_name(key))
        if found:
            return found
    return None


def parse_float_fr(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if text.lower() in {"", "nan", "none", "/"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


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


def parse_manual_bounds(raw_text: str) -> tuple[dict[str, tuple[float, float]], list[str]]:
    bounds: dict[str, tuple[float, float]] = {}
    errors: list[str] = []

    lines = str(raw_text).splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            key, value_part = [chunk.strip() for chunk in line.split(":", 1)]
        elif "=" in line:
            key, value_part = [chunk.strip() for chunk in line.split("=", 1)]
        else:
            errors.append(f"Ligne {line_no}: separateur ':' ou '=' manquant.")
            continue

        indicator = key.lower()
        if not INDICATOR_PATTERN.match(indicator):
            errors.append(f"Ligne {line_no}: indicateur invalide '{key}'.")
            continue

        numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", value_part)
        if len(numbers) < 2:
            errors.append(f"Ligne {line_no}: deux bornes numeriques attendues.")
            continue

        x_min = parse_float_fr(numbers[0])
        x_max = parse_float_fr(numbers[1])
        if np.isnan(x_min) or np.isnan(x_max) or np.isclose(x_min, x_max):
            errors.append(f"Ligne {line_no}: bornes invalides pour {indicator}.")
            continue

        bounds[indicator] = (float(x_min), float(x_max))

    return bounds, errors


@st.cache_data(show_spinner=False)
def load_values_dataset(values_path: str) -> tuple[pd.DataFrame, list[str]]:
    path = Path(values_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, low_memory=False)
    indicator_cols = [
        col
        for col in df.columns
        if INDICATOR_PATTERN.match(str(col).strip())
    ]

    if not indicator_cols:
        raise ValueError("Aucune colonne indicateur iXXX trouvee dans valeur_externe.csv")

    df[indicator_cols] = df[indicator_cols].apply(pd.to_numeric, errors="coerce")
    return df, indicator_cols


@st.cache_data(show_spinner=False)
def load_params_dataset(params_path: str) -> pd.DataFrame:
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def build_bounds_reference(
    df_values: pd.DataFrame,
    indicator_cols: list[str],
    df_params_raw: pd.DataFrame,
) -> pd.DataFrame:
    bounds_data = []
    for ind in indicator_cols:
        series = pd.to_numeric(df_values[ind], errors="coerce").dropna()
        bounds_data.append(
            {
                "indicator": ind.lower(),
                "x_min_data": float(series.min()) if not series.empty else np.nan,
                "x_max_data": float(series.max()) if not series.empty else np.nan,
            }
        )
    df_bounds_data = pd.DataFrame(bounds_data)

    id_col = find_col(df_params_raw.columns, ["idindicateurs", "idindicateur"])
    label_col = find_col(df_params_raw.columns, ["libelleindicateurs", "libelle"])
    score0_col = find_col(df_params_raw.columns, ["valeurbornescore0"])
    score100_col = find_col(df_params_raw.columns, ["valeurbornescore100"])

    if not all([id_col, score0_col, score100_col]):
        raise ValueError("Colonnes de bornes introuvables dans Parametres_indicateurs.csv")

    df_bounds_param = df_params_raw[[id_col, score0_col, score100_col]].copy()
    df_bounds_param = df_bounds_param.rename(
        columns={
            id_col: "indicator",
            score0_col: "x_min_param",
            score100_col: "x_max_param",
        }
    )
    df_bounds_param["indicator"] = (
        df_bounds_param["indicator"].astype(str).str.strip().str.lower()
    )
    df_bounds_param["x_min_param"] = df_bounds_param["x_min_param"].apply(parse_float_fr)
    df_bounds_param["x_max_param"] = df_bounds_param["x_max_param"].apply(parse_float_fr)
    df_bounds_param = df_bounds_param.drop_duplicates(subset=["indicator"], keep="first")

    if label_col is not None:
        df_labels = df_params_raw[[id_col, label_col]].copy().rename(
            columns={id_col: "indicator", label_col: "label"}
        )
        df_labels["indicator"] = df_labels["indicator"].astype(str).str.strip().str.lower()
        df_labels = df_labels.drop_duplicates(subset=["indicator"], keep="first")
    else:
        df_labels = pd.DataFrame(
            {
                "indicator": [col.lower() for col in indicator_cols],
                "label": [col.lower() for col in indicator_cols],
            }
        )

    df_bounds = df_bounds_data.merge(df_bounds_param, on="indicator", how="left")
    df_bounds = df_bounds.merge(df_labels, on="indicator", how="left")
    df_bounds["x_min"] = df_bounds["x_min_param"].combine_first(df_bounds["x_min_data"])
    df_bounds["x_max"] = df_bounds["x_max_param"].combine_first(df_bounds["x_max_data"])

    return df_bounds


def compute_candidate_bounds(raw_series: pd.Series):
    series = pd.to_numeric(raw_series, errors="coerce").dropna()
    if series.empty:
        return None

    data_min = float(series.min())
    data_max = float(series.max())

    q01, q05, q25, q50, q75, q95, q99 = series.quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).astype(float).tolist()

    iqr = q75 - q25
    iqr_min = max(data_min, q25 - 1.5 * iqr)
    iqr_max = min(data_max, q75 + 1.5 * iqr)

    if np.isclose(iqr_min, iqr_max):
        iqr_min, iqr_max = data_min, data_max

    return {
        "data": (data_min, data_max),
        "p01_p99": (q01, q99),
        "p05_p95": (q05, q95),
        "iqr_clip": (float(iqr_min), float(iqr_max)),
        "q01": q01,
        "q05": q05,
        "q50": q50,
        "q95": q95,
        "q99": q99,
    }


def choose_bounds_auto(
    raw_series: pd.Series,
    candidates: dict,
    moderate_outlier_rate: float,
    high_outlier_rate: float,
):
    series = pd.to_numeric(raw_series, errors="coerce").dropna()
    if series.empty:
        return None

    iqr_min, iqr_max = candidates["iqr_clip"]
    iqr_outlier_rate = float(((series < iqr_min) | (series > iqr_max)).mean())

    if iqr_outlier_rate > high_outlier_rate:
        method = "p05_p95"
    elif iqr_outlier_rate > moderate_outlier_rate:
        method = "p01_p99"
    else:
        method = "data"

    x_min, x_max = candidates[method]
    if np.isclose(x_min, x_max):
        method = "data"
        x_min, x_max = candidates["data"]

    return method, float(x_min), float(x_max), round(iqr_outlier_rate, 4)


def normalize_0_1(values, x_min, x_max):
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype=float)

    if pd.isna(x_min) or pd.isna(x_max) or np.isclose(x_min, x_max):
        return out

    if x_max > x_min:
        out = (vals - x_min) / (x_max - x_min)
    else:
        out = (x_min - vals) / (x_min - x_max)

    return out.clip(0.0, 1.0)


def score_0_100(values, x_min, x_max, k):
    scaled = normalize_0_1(values, x_min=x_min, x_max=x_max)
    if np.isclose(k, 0.0):
        score = scaled
    else:
        score = np.expm1(k * scaled) / np.expm1(k)
    return (100.0 * score).clip(0.0, 100.0)


def equilibrium_metrics(
    score_series,
    target_std: float,
    extreme_low: float,
    extreme_high: float,
):
    series = pd.Series(score_series).dropna()
    if series.empty:
        return None

    mean_score = float(series.mean())
    std_score = float(series.std(ddof=0))
    extreme_rate = float(((series <= extreme_low) | (series >= extreme_high)).mean())

    center_penalty = min(abs(mean_score - 50.0) / 50.0, 1.0)
    spread_penalty = min(abs(std_score - target_std) / max(target_std, 1e-9), 1.0)
    extreme_penalty = min(extreme_rate, 1.0)

    raw_score = 1.0 - (
        0.45 * center_penalty
        + 0.35 * spread_penalty
        + 0.20 * extreme_penalty
    )
    equilibrium = max(0.0, min(1.0, raw_score))

    return {
        "mean_score": round(mean_score, 3),
        "std_score": round(std_score, 3),
        "extreme_rate": round(extreme_rate, 4),
        "equilibre_score": round(100.0 * equilibrium, 3),
    }


def compute_recommendations(
    df_values: pd.DataFrame,
    indicator_cols: list[str],
    df_bounds: pd.DataFrame,
    k_grid: list[float],
    manual_bounds: dict[str, tuple[float, float]],
    target_std: float,
    extreme_low: float,
    extreme_high: float,
    moderate_outlier_rate: float,
    high_outlier_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recommendation_rows = []
    k_details_all = []

    for ind in indicator_cols:
        ind_lower = ind.lower()
        raw = pd.to_numeric(df_values[ind], errors="coerce").dropna()
        if raw.empty:
            continue

        row = df_bounds.loc[df_bounds["indicator"] == ind_lower]
        if row.empty:
            continue

        label_value = row["label"].iloc[0] if "label" in row.columns else ind_lower
        if pd.isna(label_value):
            label_value = ind_lower

        candidates = compute_candidate_bounds(raw)
        if candidates is None:
            continue

        auto_choice = choose_bounds_auto(
            raw,
            candidates,
            moderate_outlier_rate=moderate_outlier_rate,
            high_outlier_rate=high_outlier_rate,
        )
        if auto_choice is None:
            continue

        auto_method, auto_x_min, auto_x_max, iqr_outlier_rate = auto_choice

        if ind_lower in manual_bounds:
            x_min, x_max = manual_bounds[ind_lower]
            bounds_method = "manual"
        else:
            x_min, x_max = auto_x_min, auto_x_max
            bounds_method = auto_method

        if pd.isna(x_min) or pd.isna(x_max) or np.isclose(x_min, x_max):
            continue

        k_rows = []
        for k in k_grid:
            score_vals = score_0_100(raw, x_min=x_min, x_max=x_max, k=float(k))
            metrics = equilibrium_metrics(
                score_vals,
                target_std=target_std,
                extreme_low=extreme_low,
                extreme_high=extreme_high,
            )
            if metrics is None:
                continue
            k_rows.append({"k": float(k), **metrics})

        if not k_rows:
            continue

        k_df = (
            pd.DataFrame(k_rows)
            .sort_values("equilibre_score", ascending=False)
            .reset_index(drop=True)
        )
        best = k_df.iloc[0]

        recommendation_rows.append(
            {
                "indicator": ind_lower,
                "label": str(label_value),
                "n": int(raw.size),
                "bounds_method": bounds_method,
                "x_min": round(float(x_min), 6),
                "x_max": round(float(x_max), 6),
                "iqr_outlier_rate": iqr_outlier_rate,
                "data_min": round(candidates["data"][0], 6),
                "data_max": round(candidates["data"][1], 6),
                "p01": round(candidates["q01"], 6),
                "p05": round(candidates["q05"], 6),
                "median_raw": round(candidates["q50"], 6),
                "p95": round(candidates["q95"], 6),
                "p99": round(candidates["q99"], 6),
                "iqr_clip_min": round(candidates["iqr_clip"][0], 6),
                "iqr_clip_max": round(candidates["iqr_clip"][1], 6),
                "best_k": float(best["k"]),
                "equilibre_score": float(best["equilibre_score"]),
                "mean_score": float(best["mean_score"]),
                "std_score": float(best["std_score"]),
                "extreme_rate": float(best["extreme_rate"]),
            }
        )

        k_df["indicator"] = ind_lower
        k_df["label"] = str(label_value)
        k_df["x_min"] = float(x_min)
        k_df["x_max"] = float(x_max)
        k_df["bounds_method"] = bounds_method
        k_details_all.append(k_df)

    df_reco = pd.DataFrame(recommendation_rows)
    if not df_reco.empty:
        df_reco = df_reco.sort_values("indicator").reset_index(drop=True)

    if k_details_all:
        df_k_details = pd.concat(k_details_all, ignore_index=True)
    else:
        df_k_details = pd.DataFrame(
            columns=[
                "k",
                "mean_score",
                "std_score",
                "extreme_rate",
                "equilibre_score",
                "indicator",
                "label",
                "x_min",
                "x_max",
                "bounds_method",
            ]
        )

    return df_reco, df_k_details


def build_diagnostic_figure(
    raw: pd.Series,
    reco_row: pd.Series,
    k_curve: pd.DataFrame,
    compare_k: list[float],
    bins: int,
):
    x_min = float(reco_row["x_min"])
    x_max = float(reco_row["x_max"])
    best_k = float(reco_row["best_k"])
    indicator = str(reco_row["indicator"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].hist(raw, bins=bins, color="#7da0ca", alpha=0.85, edgecolor="white")
    axes[0, 0].axvline(float(reco_row["data_min"]), color="gray", linestyle="-", linewidth=1, label="data_min")
    axes[0, 0].axvline(float(reco_row["data_max"]), color="gray", linestyle="-", linewidth=1, label="data_max")
    axes[0, 0].axvline(float(reco_row["p01"]), color="#2ca02c", linestyle="--", linewidth=1.5, label="p01")
    axes[0, 0].axvline(float(reco_row["p99"]), color="#2ca02c", linestyle="--", linewidth=1.5, label="p99")
    axes[0, 0].axvline(float(reco_row["p05"]), color="#ff7f0e", linestyle="--", linewidth=1.8, label="p05")
    axes[0, 0].axvline(float(reco_row["p95"]), color="#ff7f0e", linestyle="--", linewidth=1.8, label="p95")
    axes[0, 0].axvline(x_min, color="red", linestyle="-", linewidth=2, label="x_min retenu")
    axes[0, 0].axvline(x_max, color="red", linestyle="-", linewidth=2, label="x_max retenu")
    axes[0, 0].set_title(f"{indicator} | Distribution brute et bornes")
    axes[0, 0].set_xlabel("Valeur brute")
    axes[0, 0].set_ylabel("Frequence")
    axes[0, 0].legend(fontsize=8, ncol=2)

    x_sorted = np.sort(raw.values)
    y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    axes[0, 1].plot(x_sorted, y_cdf, color="#1f77b4", linewidth=2)
    axes[0, 1].axvline(x_min, color="red", linestyle="-", linewidth=2)
    axes[0, 1].axvline(x_max, color="red", linestyle="-", linewidth=2)
    axes[0, 1].set_title("CDF brute")
    axes[0, 1].set_xlabel("Valeur brute")
    axes[0, 1].set_ylabel("Part cumulee")
    axes[0, 1].set_ylim(0, 1)

    x_grid = np.linspace(float(raw.min()), float(raw.max()), 400)
    k_to_plot = sorted(set(compare_k + [best_k]))
    for k_value in k_to_plot:
        y_grid = score_0_100(pd.Series(x_grid), x_min=x_min, x_max=x_max, k=k_value)
        is_best = np.isclose(k_value, best_k)
        linewidth = 2.8 if is_best else 1.8
        alpha = 1.0 if is_best else 0.8
        label = f"k={k_value:g}" + (" (best)" if is_best else "")
        axes[1, 0].plot(x_grid, y_grid.values, linewidth=linewidth, alpha=alpha, label=label)

    axes[1, 0].set_title("Impact de k sur la courbe de scoring")
    axes[1, 0].set_xlabel("Valeur brute")
    axes[1, 0].set_ylabel("Score (0-100)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].legend(fontsize=8)

    bars = axes[1, 1].bar(
        k_curve["k"].astype(str),
        k_curve["equilibre_score"],
        color="#9ecae1",
        edgecolor="#4a708b",
    )
    for idx, k_value in enumerate(k_curve["k"]):
        if np.isclose(float(k_value), best_k):
            bars[idx].set_color("#d62728")

    for bar in bars:
        height = float(bar.get_height())
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.8,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    axes[1, 1].set_title("Qualite globale par k")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Equilibre / 100")
    axes[1, 1].set_ylim(0, 100)

    fig.suptitle(
        f"{indicator} | methode={reco_row['bounds_method']} | k recommande={best_k:g}",
        fontsize=13,
    )

    return fig


def build_best_score_figure(best_scores: pd.Series, bins: int):
    series = pd.Series(best_scores).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)

    hist_bins = max(10, int(bins / 2))
    axes[0].hist(series, bins=hist_bins, color="#74c476", alpha=0.85, edgecolor="white")

    q05, q50, q95 = series.quantile([0.05, 0.50, 0.95]).tolist()
    axes[0].axvline(q05, color="#ff7f0e", linestyle="--", linewidth=1.5, label="p05")
    axes[0].axvline(q50, color="#1f77b4", linestyle="-", linewidth=1.8, label="mediane")
    axes[0].axvline(q95, color="#ff7f0e", linestyle="--", linewidth=1.5, label="p95")
    axes[0].set_title("Distribution des scores (k recommande)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequence")
    axes[0].legend(fontsize=8)

    sorted_scores = np.sort(series.values)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1].plot(sorted_scores, cdf, color="#2c7fb8", linewidth=2)
    axes[1].set_title("CDF des scores (k recommande)")
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Part cumulee")
    axes[1].set_ylim(0, 1)

    return fig


def summarize_series(series: pd.Series, label: str) -> dict:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return {
            "series": label,
            "n": 0,
            "min": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "std": np.nan,
        }

    return {
        "series": label,
        "n": int(clean.size),
        "min": round(float(clean.min()), 6),
        "p05": round(float(clean.quantile(0.05)), 6),
        "median": round(float(clean.quantile(0.50)), 6),
        "mean": round(float(clean.mean()), 6),
        "p95": round(float(clean.quantile(0.95)), 6),
        "max": round(float(clean.max()), 6),
        "std": round(float(clean.std(ddof=0)), 6),
    }


def build_candidate_table(reco_row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "methode": "data",
                "x_min": float(reco_row["data_min"]),
                "x_max": float(reco_row["data_max"]),
                "description": "min/max observes",
            },
            {
                "methode": "p01_p99",
                "x_min": float(reco_row["p01"]),
                "x_max": float(reco_row["p99"]),
                "description": "percentiles 1% et 99%",
            },
            {
                "methode": "p05_p95",
                "x_min": float(reco_row["p05"]),
                "x_max": float(reco_row["p95"]),
                "description": "percentiles 5% et 95%",
            },
            {
                "methode": "iqr_clip",
                "x_min": float(reco_row["iqr_clip_min"]),
                "x_max": float(reco_row["iqr_clip_max"]),
                "description": "clipping IQR (1.5 * IQR)",
            },
            {
                "methode": "retenue",
                "x_min": float(reco_row["x_min"]),
                "x_max": float(reco_row["x_max"]),
                "description": str(reco_row["bounds_method"]),
            },
        ]
    )


st.set_page_config(page_title="Diag360 - bornes et k", layout="wide")
st.title("Diag360 - Diagnostic bornes et k")
st.caption(
    "Choisissez un indicateur pour analyser les bornes candidates, le k recommande et les courbes de scoring."
)

with st.sidebar:
    st.header("Configuration")

    values_path = st.text_input(
        "Chemin valeur_externe.csv",
        value=str(first_existing(DEFAULT_VALUES_CANDIDATES)),
    )
    params_path = st.text_input(
        "Chemin Parametres_indicateurs.csv",
        value=str(first_existing(DEFAULT_PARAMS_CANDIDATES)),
    )

    k_grid_text = st.text_input(
        "Grille de k (separee par virgules)",
        value="-6,-4,-3,-2,-1,0,1,2,3,4,6",
    )
    compare_k_text = st.text_input(
        "k affiches sur la courbe (separes par virgules)",
        value="-3,-1,0,1,3",
    )

    top_k = int(
        st.number_input("Nombre de lignes Top k", min_value=3, max_value=50, value=10, step=1)
    )
    bins = int(
        st.slider("Bins histogrammes", min_value=20, max_value=120, value=40, step=5)
    )

    target_std = float(
        st.slider(
            "Cible ecart-type score",
            min_value=8.0,
            max_value=35.0,
            value=18.0,
            step=0.5,
        )
    )
    extreme_low = float(
        st.slider("Seuil extreme bas", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    )
    extreme_high = float(
        st.slider("Seuil extreme haut", min_value=80.0, max_value=100.0, value=95.0, step=0.5)
    )

    moderate_outlier_rate = float(
        st.slider(
            "Seuil outliers IQR moyen",
            min_value=0.00,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
        )
    )
    high_outlier_rate = float(
        st.slider(
            "Seuil outliers IQR fort",
            min_value=0.01,
            max_value=0.40,
            value=0.12,
            step=0.01,
            format="%.2f",
        )
    )

    manual_bounds_text = st.text_area(
        "Bornes manuelles (optionnel)",
        value="",
        help="Format: i005: 5, 85 (une ligne par indicateur)",
        height=120,
    )

k_grid, bad_k_tokens = parse_k_grid(k_grid_text)
compare_k, bad_compare_tokens = parse_k_grid(compare_k_text)
manual_bounds, manual_errors = parse_manual_bounds(manual_bounds_text)

if bad_k_tokens:
    st.sidebar.warning("Valeurs k ignorees: " + ", ".join(sorted(set(bad_k_tokens))))
if bad_compare_tokens:
    st.sidebar.warning("Valeurs compare_k ignorees: " + ", ".join(sorted(set(bad_compare_tokens))))
if manual_errors:
    st.sidebar.warning("Erreurs bornes manuelles:\n- " + "\n- ".join(manual_errors))

if not k_grid:
    st.error("La grille de k est vide. Renseignez au moins une valeur numerique.")
    st.stop()

if not compare_k:
    compare_k = [-3.0, -1.0, 0.0, 1.0, 3.0]

if extreme_low >= extreme_high:
    st.error("Le seuil extreme bas doit etre inferieur au seuil extreme haut.")
    st.stop()

if moderate_outlier_rate >= high_outlier_rate:
    st.error("Le seuil outliers moyen doit etre inferieur au seuil outliers fort.")
    st.stop()

try:
    with st.spinner("Chargement des donnees et calcul des recommandations..."):
        df_values, indicator_cols = load_values_dataset(values_path)
        df_params_raw = load_params_dataset(params_path)
        df_bounds = build_bounds_reference(df_values, indicator_cols, df_params_raw)
        df_reco_bounds_k, df_k_details = compute_recommendations(
            df_values=df_values,
            indicator_cols=indicator_cols,
            df_bounds=df_bounds,
            k_grid=k_grid,
            manual_bounds=manual_bounds,
            target_std=target_std,
            extreme_low=extreme_low,
            extreme_high=extreme_high,
            moderate_outlier_rate=moderate_outlier_rate,
            high_outlier_rate=high_outlier_rate,
        )
except Exception as exc:
    st.error(f"Erreur de chargement ou de calcul: {exc}")
    st.stop()

if df_reco_bounds_k.empty:
    st.warning("Aucune recommandation produite avec les parametres actuels.")
    st.stop()

reco_display = df_reco_bounds_k.copy()
reco_display["display"] = reco_display.apply(
    lambda row: f"{row['indicator']} - {row['label']}" if str(row.get("label", "")).strip() else row["indicator"],
    axis=1,
)

if "i005" in set(reco_display["indicator"]):
    default_index = int(reco_display.index[reco_display["indicator"] == "i005"][0])
else:
    default_index = 0

selected_display = st.selectbox(
    "Choisir un indicateur",
    options=reco_display["display"].tolist(),
    index=default_index,
)
selected_indicator = reco_display.loc[
    reco_display["display"] == selected_display, "indicator"
].iloc[0]

reco_row = df_reco_bounds_k[df_reco_bounds_k["indicator"] == selected_indicator].iloc[0]
raw_series = pd.to_numeric(df_values[selected_indicator], errors="coerce").dropna()

k_detail = (
    df_k_details[df_k_details["indicator"] == selected_indicator]
    .sort_values("equilibre_score", ascending=False)
    .reset_index(drop=True)
)
k_curve = k_detail.sort_values("k").reset_index(drop=True)

st.subheader(f"Detail indicateur: {selected_indicator}")

metric_cols = st.columns(5)
metric_cols[0].metric("k recommande", f"{float(reco_row['best_k']):g}")
metric_cols[1].metric("Equilibre", f"{float(reco_row['equilibre_score']):.2f} / 100")
metric_cols[2].metric("Mean score", f"{float(reco_row['mean_score']):.2f}")
metric_cols[3].metric("Std score", f"{float(reco_row['std_score']):.2f}")
metric_cols[4].metric("Extreme rate", f"{100.0 * float(reco_row['extreme_rate']):.2f}%")

st.write(
    "Methode de bornes:",
    str(reco_row["bounds_method"]),
    "| x_min:",
    float(reco_row["x_min"]),
    "| x_max:",
    float(reco_row["x_max"]),
    "| iqr outlier rate:",
    f"{100.0 * float(reco_row['iqr_outlier_rate']):.2f}%",
)

fig_main = build_diagnostic_figure(
    raw=raw_series,
    reco_row=reco_row,
    k_curve=k_curve,
    compare_k=compare_k,
    bins=bins,
)
st.pyplot(fig_main, use_container_width=True)
plt.close(fig_main)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Top k (detail)")
    st.dataframe(
        k_detail.head(top_k)[["k", "equilibre_score", "mean_score", "std_score", "extreme_rate"]],
        use_container_width=True,
    )

with right_col:
    st.markdown("### Bornes candidates")
    st.dataframe(build_candidate_table(reco_row), use_container_width=True)

best_scores = score_0_100(
    raw_series,
    x_min=float(reco_row["x_min"]),
    x_max=float(reco_row["x_max"]),
    k=float(reco_row["best_k"]),
)

st.markdown("### Distribution du score avec k recommande")
fig_score = build_best_score_figure(best_scores=best_scores, bins=bins)
st.pyplot(fig_score, use_container_width=True)
plt.close(fig_score)

stats_df = pd.DataFrame(
    [
        summarize_series(raw_series, "raw_values"),
        summarize_series(best_scores, "score_best_k"),
    ]
)
st.dataframe(stats_df, use_container_width=True)

with st.expander("Voir tableau global des recommandations"):
    st.dataframe(df_reco_bounds_k, use_container_width=True)

    st.download_button(
        label="Telecharger recommendations (CSV)",
        data=df_reco_bounds_k.to_csv(index=False).encode("utf-8"),
        file_name="recommandations_bornes_k.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Telecharger details k (CSV)",
        data=df_k_details.to_csv(index=False).encode("utf-8"),
        file_name="details_k_par_indicateur.csv",
        mime="text/csv",
    )
