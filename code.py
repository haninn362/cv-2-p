# ============================================================
# Streamlit App: Classification + Optimisation + Forecasting
# ============================================================
# This app combines:
#   - Demand Classification (p & CV² → Syntetos & Boylan)
#   - Optimisation (n*, Qr*, Qw*)
#   - Forecasting with SBA, Croston, SES (grid search + best params)
#   - Comparison Tables (Mean Holding & CT)
#
# Excel reading is unified:
#   Column 0 = Date, Column 1 = Stock, Column 2 = Consumption
# ============================================================

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple
from scipy.stats import nbinom

# ========================== Constants ==========================
ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF       # ≈ 0.757576
CV2_CUTOFF = 0.49

LEAD_TIME_UNI = 1
LEAD_TIME_SUPPLIER_UNI = 3
SERVICE_LEVEL_UNI = 0.95
NB_SIM_UNI = 800
RNG_SEED_UNI = 42

ALPHAS_UNI = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
WINDOW_RATIOS_UNI = [0.6, 0.7, 0.8]
RECALC_INTERVALS_UNI = [1, 2, 5, 10, 15, 20]

DISPLAY_COLUMNS_UNI = [
    "date", "code", "interval", "real_demand", "stock_on_hand_running",
    "stock_after_interval", "can_cover_interval", "order_policy",
    "reorder_point_usine", "lead_time_usine_days", "lead_time_supplier_days",
    "reorder_point_fournisseur", "stock_status", "rop_usine_minus_real_running"
]

# ========================== Streamlit setup ==========================
st.set_page_config(page_title="Classification + Forecasting App", layout="wide")

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { display: none; }
      .block-container { padding-top: 120px; }
      .fixed-header {
        position: fixed; top: 0; left: 0; right: 0;
        z-index: 10000;
        background: var(--background-color, #ffffff);
        border-bottom: 1px solid rgba(49,51,63,.14);
        box-shadow: 0 2px 10px rgba(0,0,0,.05);
      }
      .fixed-inner { padding: .55rem .9rem .8rem; max-width: 1200px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================== Helpers ==========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _read_timeseries_by_position(excel_path: str, sheet_name: str):
    """Read timeseries: Col A=date, Col B=stock, Col C=consumption"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.normalize()
    stock = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0).astype(float)
    cons  = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).astype(float)

    ts_cons  = pd.DataFrame({"d": dates, "q": cons}).dropna(subset=["d"]).set_index("d")["q"].sort_index()
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna(subset=["d"]).set_index("d")["s"].sort_index()

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = pd.date_range(min_date, max_date, freq="D")

    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    return cons_daily, stock_daily

# ========================== Classification ==========================
def choose_method(p: float, cv2: float) -> Tuple[str, str]:
    if pd.isna(p) or pd.isna(cv2): return "Données insuffisantes", ""
    if p <= 0: return "Aucune demande", ""
    if p >= P_CUTOFF and cv2 <= CV2_CUTOFF: return "Régulier", "SES"
    if p >= P_CUTOFF and cv2 > CV2_CUTOFF:  return "Erratique", "SES"
    if p < P_CUTOFF and cv2 <= CV2_CUTOFF:  return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"

def compute_everything(df: pd.DataFrame):
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    combined_rows, per_product_vals, max_len = [], {}, 0
    for _, row in df.iterrows():
        produit = str(row.iloc[0])
        numeric = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0).values
        nz = numeric != 0
        vals = numeric[nz].tolist()

        arr_dates = parsed_dates[nz]
        if vals and arr_dates.notna().all():
            inter = pd.Series(arr_dates).diff().dropna().dt.days.tolist()
            inter_arrivals = [1] + inter
        else:
            inter_arrivals = []

        max_len = max(max_len, len(vals), len(inter_arrivals))
        combined_rows.append((produit, vals, inter_arrivals))
        per_product_vals[produit] = vals

    final_rows = []
    for produit, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([produit, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Produit", "Type"] + list(range(max_len)))

    stats_rows = []
    for produit, vals in per_product_vals.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            moyenne = s.mean()
            ecart = s.std(ddof=1)
            cv2 = (ecart / moyenne) ** 2 if moyenne != 0 else np.nan
        else:
            moyenne = ecart = cv2 = np.nan
        stats_rows.append([produit, moyenne, ecart, cv2])

    stats_df = (
        pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "écart-type", "CV^2"])
        .set_index("Produit").sort_index()
    )

    counts_rows = []
    for produit, vals in per_product_vals.items():
        n_freq = len(vals)
        p = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([produit, n_periods, n_freq, p])

    counts_df = (
        pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquences", "p"])
        .set_index("Produit").sort_index()
    )

    methods_df = stats_df.join(counts_df, how="outer")
    cats = methods_df.apply(lambda r: choose_method(r["p"], r["CV^2"]), axis=1, result_type="expand")
    methods_df["Catégorie"] = cats[0]
    methods_df["Méthode suggérée"] = cats[1]
    methods_df = methods_df[["CV^2", "p", "Catégorie", "Méthode suggérée"]]
    return combined_df, stats_df, counts_df, methods_df

def make_plot(methods_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = methods_df["p"].clip(lower=0, upper=1)
    y = methods_df["CV^2"]
    ax.scatter(x, y)
    for label, xi, yi in zip(methods_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(f"{label} (p={xi:.3f}, CV²={yi:.3f})", (xi, yi),
                        textcoords="offset points", xytext=(5, 5))
    ax.axvline(P_CUTOFF, linestyle="--")
    ax.axhline(CV2_CUTOFF, linestyle="--")
    ax.set_xlabel("p (part des périodes non nulles)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("CV²")
    ax.set_title("Classification (p vs CV²) — Syntetos & Boylan")
    fig.tight_layout()
    return fig


# ========================== Optimisation ==========================
def _get_excel_bytes(file_like) -> bytes:
    if file_like is None: return b""
    if hasattr(file_like, "getvalue"):
        try: return file_like.getvalue()
        except Exception: pass
    try:
        data = file_like.read()
        return data
    finally:
        try: file_like.seek(0)
        except Exception: pass

def _find_first_col(df: pd.DataFrame, starts_with: str = None, contains: str = None):
    for c in df.columns:
        cn = _norm(c)
        if starts_with and cn.startswith(starts_with): return c
        if contains and contains in cn: return c
    return None

def compute_qr_qw_from_workbook(file_like, conso_sheet_hint: str = "consommation depots externe",
                                time_series_prefix: str = "time seri"):
    info_msgs, warn_msgs = [], []
    if file_like is None:
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    data_bytes = _get_excel_bytes(file_like)
    if not data_bytes:
        warn_msgs.append("Classeur d’optimisation vide ou illisible.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    xls = pd.ExcelFile(io.BytesIO(data_bytes))

    sheet_names_norm = {_norm(s): s for s in xls.sheet_names}
    conso_sheet = sheet_names_norm.get(_norm(conso_sheet_hint))
    if not conso_sheet:
        cands = [s for s in xls.sheet_names if _norm(conso_sheet_hint) in _norm(s)]
        if cands: conso_sheet = cands[0]
    if not conso_sheet:
        warn_msgs.append("Feuille 'consommation depots externe' introuvable.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    df_conso = pd.read_excel(io.BytesIO(data_bytes), sheet_name=conso_sheet)

    code_col = next((c for c in df_conso.columns if "code produit" in _norm(c)), None) or "Code Produit"
    qty_col = None
    for c in df_conso.columns:
        nc = _norm(c)
        if nc in ("quantite stial", "quantité stial"): qty_col = c; break
    if qty_col is None:
        for c in df_conso.columns:
            nc = _norm(c)
            if "quantite stial" in nc or "quantité stial" in nc: qty_col = c; break
    if qty_col is None:
        for key in ["quantite", "quantité", "qte"]:
            cand = next((c for c in df_conso.columns if key in _norm(c)), None)
            if cand: qty_col = cand; break

    if code_col is None or qty_col is None:
        warn_msgs.append("Colonnes 'Code Produit' et/ou 'Quantite STIAL' introuvables.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    conso_series = df_conso.groupby(code_col, dropna=False)[qty_col].sum(numeric_only=True)
    info_msgs.append(f"Feuille de consommation : '{conso_sheet}' (lignes : {len(df_conso)})")
    info_msgs.append(f"Colonne quantité utilisée : '{qty_col}'")

    ts_sheets = [s for s in xls.sheet_names if _norm(s).startswith(_norm(time_series_prefix))]
    if not ts_sheets:
        warn_msgs.append("Aucune feuille 'time serie*' trouvée (ex. 'time serie EM0400').")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    rows = []
    for sheet in ts_sheets:
        try:
            df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet)
            code_produit = sheet.split()[-1]

            cr_col = _find_first_col(df, starts_with="cr")
            cw_col = _find_first_col(df, starts_with="cw")
            aw_col = _find_first_col(df, starts_with="aw")
            ar_col = _find_first_col(df, starts_with="ar")
            if not all([cr_col, cw_col, aw_col, ar_col]):
                warn_msgs.append(f"[{sheet}] Paramètres CR/CW/AW/AR manquants — ignoré.")
                continue

            C_r = pd.to_numeric(df[cr_col].iloc[0], errors="coerce")
            C_w = pd.to_numeric(df[cw_col].iloc[0], errors="coerce")
            A_w = pd.to_numeric(df[aw_col].iloc[0], errors="coerce")
            A_r = pd.to_numeric(df[ar_col].iloc[0], errors="coerce")
            if any(pd.isna(v) for v in [C_r, C_w, A_w, A_r]) or any(v == 0 for v in [C_w, A_r]):
                warn_msgs.append(f"[{sheet}] Valeurs de paramètres invalides — ignoré.")
                continue

            n = (A_w * C_r) / (A_r * C_w)
            n = 1 if n < 1 else round(n)
            n1, n2 = int(n), int(n) + 1
            F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
            F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
            n_star = n1 if F_n1 <= F_n2 else n2

            D = conso_series.get(code_produit, 0)
            tau = 1
            denom = (n_star * C_w + C_r * tau)
            if denom <= 0:
                warn_msgs.append(f"[{sheet}] Dénominateur non positif pour Q* — ignoré.")
                continue

            if D is None or D <= 0:
                warn_msgs.append(f"[{sheet}] Demande non positive D={D} → Q*=0.")
                Q_r_star = 0.0
            else:
                Q_r_star = ((2 * (A_r + A_w / n_star) * D) / denom) ** 0.5

            Q_w_star = n_star * Q_r_star
            rows.append({
                "Code Produit": str(code_produit),
                "n*": int(n_star),
                "Qr*": round(float(Q_r_star), 2),
                "Qw*": round(float(Q_w_star), 2),
            })
        except Exception as e:
            warn_msgs.append(f"[{sheet}] Échec : {e}")

    result_df = pd.DataFrame(rows).sort_values("Code Produit") if rows else pd.DataFrame(
        columns=["Code Produit", "n*", "Qr*", "Qw*"]
    )
    return result_df, info_msgs, warn_msgs

# ========================== Forecasting — SBA ==========================
def _croston_or_sba_forecast_array(x, alpha: float, variant: str = "sba"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": float("inf")}
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    first = nz_idx[0]
    z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else:
        p = len(x) / len(nz_idx)
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    if variant.lower() == "sba":
        f *= (1 - alpha / 2.0)
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}



def rolling_sba_with_rops_single_run(
    excel_path: str,
    product_code: str,
    alpha: float,
    window_ratio: float,
    interval: int,
    lead_time: int,
    lead_time_supplier: int,
    service_level: float,
    nb_sim: int,
    rng_seed: int,
    variant: str = "sba",
):
    xls = pd.ExcelFile(excel_path)
    sheet = None
    for s in xls.sheet_names:
        if str(product_code).lower() in s.lower():
            sheet = s
            break
    if not sheet:
        return pd.DataFrame()

    df = pd.read_excel(excel_path, sheet_name=sheet)
    if len(df.columns) < 3:
        return pd.DataFrame()
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    stock = pd.to_numeric(df.iloc[:, 1], errors="coerce").astype(float)
    cons = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).astype(float)

    ts_cons  = pd.DataFrame({"d": dates, "q": cons}).dropna(subset=["d"]).sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna(subset=["d"]).sort_values("d").set_index("d")["s"]

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = pd.date_range(min_date, max_date, freq="D")

    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2: return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows = []
    rop_carry_running, stock_after_interval = 0.0, 0.0

    def _interval_sum_next_days(daily, start_idx: int, interval: int) -> float:
        s = start_idx + 1; e = s + int(max(0, interval))
        return float(pd.Series(daily).iloc[s:e].sum())

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]; test_date = cons_daily.index[i]
            fc = _croston_or_sba_forecast_array(train, alpha=alpha, variant=variant)
            f = float(fc["forecast_per_period"])
            sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            if not np.isfinite(sigma_period): sigma_period = 0.0

            real_demand = _interval_sum_next_days(cons_daily, i, interval)
            stock_on_hand_running = _interval_sum_next_days(stock_daily, i, interval)
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand
            next_real_demand = _interval_sum_next_days(cons_daily, i + interval, interval) if (i + interval) < len(vals) else 0.0
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))

            rop_carry_running += float(ROP_u - real_demand)
            stock_status = "holding" if stock_after_interval > 0 else "rupture"
            rows.append({
                "date": test_date.date(),
                "code": product_code,
                "interval": int(interval),
                "real_demand": float(real_demand),
                "stock_on_hand_running": float(stock_on_hand_running),
                "stock_after_interval": float(stock_after_interval),
                "can_cover_interval": can_cover_interval,
                "order_policy": order_policy,
                "reorder_point_usine": float(ROP_u),
                "lead_time_usine_days": int(lead_time),
                "lead_time_supplier_days": int(lead_time_supplier),
                "reorder_point_fournisseur": float(ROP_f),
                "stock_status": stock_status,
                "rop_usine_minus_real_running": float(rop_carry_running),
            })
    return pd.DataFrame(rows)

def compute_metrics_forecast(df_run: pd.DataFrame):
    if df_run.empty or "real_demand" not in df_run or "reorder_point_usine" not in df_run:
        return np.nan, np.nan, np.nan, np.nan
    est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, np.nan)
    e = df_run["real_demand"] - est
    ME = e.mean(); absME = e.abs().mean(); MSE = (e**2).mean()
    RMSE = float(np.sqrt(MSE)) if np.isfinite(MSE) else np.nan
    return ME, absME, MSE, RMSE

# ========================== Forecasting — Croston ==========================
def _croston_forecast_array(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": float("inf")}
    nz_idx = [i for i, v in enumerate(x) if v > 0]; first = nz_idx[0]; z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else: p = len(x) / len(nz_idx)
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}

def rolling_croston_with_rops_single_run(
    excel_path: str, product_code: str,
    alpha: float, window_ratio: float, interval: int,
    lead_time: int, lead_time_supplier: int,
    service_level: float, nb_sim: int, rng_seed: int
):
    return rolling_sba_with_rops_single_run(
        excel_path, product_code, alpha, window_ratio, interval,
        lead_time, lead_time_supplier, service_level, nb_sim, rng_seed,
        variant="croston"
    )



# ========================== Forecasting — SES ==========================
def _ses_forecast_array(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return {"forecast_per_period": 0.0}
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return {"forecast_per_period": float(l)}

def rolling_ses_with_rops_single_run(
    excel_path: str,
    product_code: str,
    alpha: float,
    window_ratio: float,
    interval: int,
    lead_time: int,
    lead_time_supplier: int,
    service_level: float,
    nb_sim: int,
    rng_seed: int,
):
    xls = pd.ExcelFile(excel_path)
    sheet = None
    for s in xls.sheet_names:
        if str(product_code).lower() in s.lower():
            sheet = s
            break
    if not sheet:
        return pd.DataFrame()

    df = pd.read_excel(excel_path, sheet_name=sheet)
    if len(df.columns) < 3:
        return pd.DataFrame()

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    stock = pd.to_numeric(df.iloc[:, 1], errors="coerce").astype(float)
    cons = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0.0).astype(float)

    ts_cons = pd.DataFrame({"d": dates, "q": cons}).dropna(subset=["d"]).sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna(subset=["d"]).sort_values("d").set_index("d")["s"]

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = pd.date_range(min_date, max_date, freq="D")

    cons_daily = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)

    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows = []
    rop_carry_running, stock_after_interval = 0.0, 0.0

    def _interval_sum_next_days(daily, start_idx: int, interval: int) -> float:
        s = start_idx + 1
        e = s + int(max(0, interval))
        return float(pd.Series(daily).iloc[s:e].sum())

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]

            fc = _ses_forecast_array(train, alpha=alpha)
            f = float(fc["forecast_per_period"])

            sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            if not np.isfinite(sigma_period):
                sigma_period = 0.0

            real_demand = _interval_sum_next_days(cons_daily, i, interval)
            stock_on_hand_running = _interval_sum_next_days(stock_daily, i, interval)
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand

            next_real_demand = (
                _interval_sum_next_days(cons_daily, i + interval, interval)
                if (i + interval) < len(vals)
                else 0.0
            )
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(
                np.percentile(
                    nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng),
                    100 * service_level,
                )
            )

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(
                np.percentile(
                    nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng),
                    100 * service_level,
                )
            )

            rop_carry_running += float(ROP_u - real_demand)
            stock_status = "holding" if stock_after_interval > 0 else "rupture"

            rows.append(
                {
                    "date": test_date.date(),
                    "code": product_code,
                    "interval": int(interval),
                    "real_demand": float(real_demand),
                    "stock_on_hand_running": float(stock_on_hand_running),
                    "stock_after_interval": float(stock_after_interval),
                    "can_cover_interval": can_cover_interval,
                    "order_policy": order_policy,
                    "reorder_point_usine": float(ROP_u),
                    "lead_time_usine_days": int(lead_time),
                    "lead_time_supplier_days": int(lead_time_supplier),
                    "reorder_point_fournisseur": float(ROP_f),
                    "stock_status": stock_status,
                    "rop_usine_minus_real_running": float(rop_carry_running),
                }
            )

    return pd.DataFrame(rows)


# ========================== Comparison Tables ==========================
def compute_ct(D, Qw, Qr, A_w=50, A_R=70, pi=1.0, tau=1.0, T_w=1.0, C_w=5.0, C_R=8.0):
    Iw_prime, Ir_prime = Qw / 2.0, Qr / 2.0
    Cw_prime, Cr_prime = C_w, C_R - C_w
    return (
        A_w * (D / Qw if Qw > 0 else 0)
        + pi * T_w * Cw_prime * Iw_prime
        + A_R * (D / Qr if Qr > 0 else 0)
        + tau * Cr_prime * Ir_prime
    )


def build_comparison_tables(df_best_sba, df_best_croston, df_best_ses, excel_path):
    SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]
    records_holding, records_ct = [], []

    for method_name, df_best, runner in [
        ("SBA", df_best_sba, rolling_sba_with_rops_single_run),
        ("Croston", df_best_croston, rolling_croston_with_rops_single_run),
        ("SES", df_best_ses, rolling_ses_with_rops_single_run),
    ]:
        for _, r in df_best.iterrows():
            code = r["code"]
            alpha, w, itv = r["alpha"], r["window_ratio"], r["recalc_interval"]

            for sl in SERVICE_LEVELS:
                df_run = runner(
                    excel_path=excel_path,
                    product_code=code,
                    alpha=float(alpha),
                    window_ratio=float(w),
                    interval=int(itv),
                    lead_time=LEAD_TIME_UNI,
                    lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
                    service_level=sl,
                    nb_sim=NB_SIM_UNI,
                    rng_seed=RNG_SEED_UNI,
                    **({"variant": "sba"} if method_name == "SBA" else {}),
                )

                if df_run.empty:
                    mean_holding_val, CT_val = np.nan, np.nan
                else:
                    mean_holding_val = df_run.loc[
                        df_run["stock_status"] == "holding",
                        "rop_usine_minus_real_running",
                    ].mean()
                    D = df_run["real_demand"].sum()
                    Qw = df_run["reorder_point_fournisseur"].mean()
                    Qr = df_run["reorder_point_usine"].mean()
                    CT_val = compute_ct(D, Qw, Qr)

                records_holding.append(
                    {
                        "product": code,
                        "method": method_name,
                        "service_level": f"{int(sl*100)}%",
                        "Mean_Holding": mean_holding_val,
                    }
                )
                records_ct.append(
                    {
                        "product": code,
                        "method": method_name,
                        "service_level": f"{int(sl*100)}%",
                        "CT": CT_val,
                    }
                )

    df_holding = pd.DataFrame(records_holding)
    df_ct = pd.DataFrame(records_ct)

    table_holding = df_holding.pivot_table(
        index="product", columns=["method", "service_level"], values="Mean_Holding"
    )
    table_ct = df_ct.pivot_table(
        index="product", columns=["method", "service_level"], values="CT"
    )
    return table_holding, table_ct



# ========================== Streamlit UI for Forecasting ==========================
st.header("🔮 Prévisions — SBA / Croston / SES")

excel_path_input = st.file_uploader(
    "Classeur Excel pour prévisions (ex. PFE_HANIN.xlsx)",
    type=["xlsx", "xls"],
    key="forecast_excel",
    help="Le fichier doit contenir les onglets 'time serie *' pour chaque produit."
)

if excel_path_input:
    # Sauvegarde du fichier uploadé en mémoire
    excel_bytes = excel_path_input.read()
    excel_path = io.BytesIO(excel_bytes)

    if st.button("▶ Lancer les prévisions (SBA, Croston, SES)"):
        with st.spinner("Calcul des meilleures combinaisons..."):
            # SBA
            df_best_sba = _grid_and_final_sba(excel_path)
            # Croston
            df_best_croston = _grid_and_final_croston(excel_path)
            # SES
            df_best_ses = _grid_and_final_ses(excel_path)

            st.success("Prévisions calculées avec succès ✅")

            # Affichage
            st.subheader("📈 Résultats SBA")
            st.dataframe(df_best_sba, use_container_width=True)

            st.subheader("📈 Résultats Croston")
            st.dataframe(df_best_croston, use_container_width=True)

            st.subheader("📈 Résultats SES")
            st.dataframe(df_best_ses, use_container_width=True)

            # Build comparison tables
            table_holding, table_ct = build_comparison_tables(
                df_best_sba, df_best_croston, df_best_ses, excel_path
            )

            st.subheader("📊 Tableau Comparatif — Mean Holding")
            st.dataframe(table_holding, use_container_width=True)

            st.subheader("💰 Tableau Comparatif — CT")
            st.dataframe(table_ct, use_container_width=True)

            # Téléchargements
            st.download_button(
                "⬇ Télécharger Tableau Mean Holding (CSV)",
                data=table_holding.to_csv().encode("utf-8"),
                file_name="comparison_mean_holding.csv",
                mime="text/csv"
            )
            st.download_button(
                "⬇ Télécharger Tableau CT (CSV)",
                data=table_ct.to_csv().encode("utf-8"),
                file_name="comparison_ct.csv",
                mime="text/csv"
            )

else:
    st.info("Veuillez téléverser un fichier Excel contenant les feuilles 'time serie *'.")

# ========================== END APP ==========================
st.markdown("---")
st.caption("Application intégrée : Classification + Optimisation + Prévision (SBA, Croston, SES) + Comparaison CT/Holding")
