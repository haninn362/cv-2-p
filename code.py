# app_by_position_debug.py
import io
import re
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ===== SciPy NB (primary path) + safe fallback
try:
    from scipy.stats import nbinom
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# --------------------- Syntetos & Boylan thresholds ---------------------
ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF       # ≈ 0.757576
CV2_CUTOFF = 0.49

st.set_page_config(page_title="Classification de la demande — p & CV²", layout="wide")

# ======================== Styles (fixed bar) ========================
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] { display: none; }
      .block-container { padding-top: 120px; }
      @media (max-width: 880px) { .block-container { padding-top: 140px; } }
      .fixed-header {
        position: fixed; top: 0; left: 0; right: 0;
        z-index: 10000;
        background: var(--background-color, #ffffff);
        border-bottom: 1px solid rgba(49,51,63,.14);
        box-shadow: 0 2px 10px rgba(0,0,0,.05);
      }
      .fixed-inner { padding: .55rem .9rem .8rem; max-width: 1200px; margin: 0 auto; }
      :root { --ctrl-h: 46px; }
      .controls-holder { position: relative; height: var(--ctrl-h); }
      .controls-right {
        position: absolute; right: 0; top: 0; display: flex; gap: .75rem; align-items: center;
      }
      .controls-right .control { width: 280px; max-width: 320px; }
      .fixed-header .stFileUploader { width: 100%; }
      .fixed-header .stFileUploader > div > div {
        height: var(--ctrl-h); display:flex; align-items:center;
        border-radius: 999px !important; border: 1px solid rgba(49,51,63,.25) !important;
        padding: .15rem .9rem !important; background: rgba(0,0,0,0.02) !important;
      }
      .fixed-header .stFileUploader label, .fixed-header .stFileUploader small { display:none; }
      .fixed-header .stButton>button {
        height: var(--ctrl-h); width: 100%; border-radius: 999px; font-weight: 700; padding: .45rem 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================== Helpers / Excel I/O =======================
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

# ============================ Classification logic ============================
def choose_method(p: float, cv2: float) -> Tuple[str, str]:
    if pd.isna(p) or pd.isna(cv2): return "Données insuffisantes", ""
    if p <= 0: return "Aucune demande", ""
    if p >= P_CUTOFF and cv2 <= CV2_CUTOFF: return "Régulier", "SES"
    if p >= P_CUTOFF and cv2 > CV2_CUTOFF:  return "Erratique", "SES"
    if p < P_CUTOFF and cv2 <= CV2_CUTOFF:  return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"

def compute_everything(df: pd.DataFrame):
    # --- product codes are in col0, demand values from col1..end ---
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


# ======================== Optimisation (n*, Qr*, Qw*) =======================
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

    # --- detect consumption sheet ---
    sheet_names_norm = {s.lower().strip(): s for s in xls.sheet_names}
    conso_sheet = sheet_names_norm.get(conso_sheet_hint.lower().strip())
    if not conso_sheet:
        cands = [s for s in xls.sheet_names if conso_sheet_hint.lower() in s.lower()]
        if cands: conso_sheet = cands[0]
    if not conso_sheet:
        warn_msgs.append("Feuille 'consommation depots externe' introuvable.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    df_conso = pd.read_excel(io.BytesIO(data_bytes), sheet_name=conso_sheet, header=0)

    # === Debug Preview ===
    st.subheader("🔍 Aperçu — Feuille consommation")
    st.caption(f"Feuille détectée : {conso_sheet}")
    st.dataframe(df_conso.head(), use_container_width=True)

    # --- use fixed column positions: col0=Code Produit, col1=Quantité STIAL ---
    try:
        df_conso = df_conso.iloc[:, [0, 1]]
        df_conso.columns = ["Code Produit", "Quantite STIAL"]
    except Exception:
        warn_msgs.append("Impossible de forcer les colonnes [0,1] pour consommation.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    conso_series = df_conso.groupby("Code Produit", dropna=False)["Quantite STIAL"].sum(numeric_only=True)
    info_msgs.append(f"Feuille consommation : '{conso_sheet}' (lignes : {len(df_conso)})")
    info_msgs.append("Colonnes utilisées : 0=Code Produit, 1=Quantité STIAL")

    # --- detect time series sheets ---
    ts_sheets = [s for s in xls.sheet_names if s.lower().startswith(time_series_prefix.lower())]
    if not ts_sheets:
        warn_msgs.append("Aucune feuille 'time serie *' trouvée.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    rows = []
    for sheet in ts_sheets:
        try:
            df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet, header=0)

            # === Debug Preview ===
            st.subheader(f"🔍 Aperçu — {sheet}")
            st.dataframe(df.head(), use_container_width=True)

            code_produit = sheet.split()[-1]

            # --- parameters always at fixed positions: col0=CR, col1=CW, col2=AW, col3=AR ---
            if df.shape[1] < 4:
                warn_msgs.append(f"[{sheet}] Moins de 4 colonnes — ignoré.")
                continue

            C_r = pd.to_numeric(df.iloc[0, 0], errors="coerce")
            C_w = pd.to_numeric(df.iloc[0, 1], errors="coerce")
            A_w = pd.to_numeric(df.iloc[0, 2], errors="coerce")
            A_r = pd.to_numeric(df.iloc[0, 3], errors="coerce")

            if any(pd.isna(v) for v in [C_r, C_w, A_w, A_r]) or any(v == 0 for v in [C_w, A_r]):
                warn_msgs.append(f"[{sheet}] Valeurs invalides (CR,CW,AW,AR) — ignoré.")
                continue

            # --- compute n* ---
            n = (A_w * C_r) / (A_r * C_w)
            n = 1 if n < 1 else round(n)
            n1, n2 = int(n), int(n) + 1
            F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
            F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
            n_star = n1 if F_n1 <= F_n2 else n2

            # --- compute Qr*, Qw* ---
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



# ============================== Forecasting / ROP tools (POSitional) ==============================
# Reads strictly by position: A=date, B=stock/receipts, C=consumption.
# Adds light Streamlit previews to help diagnose empty results.

def _fc_parse_num_locale_positional(series) -> pd.Series:
    s = pd.Series(series)
    v1 = pd.to_numeric(s, errors="coerce")
    if v1.notna().mean() >= 0.60:
        return v1.fillna(0.0).astype(float)
    # try French locale (space/nbsp separators, commas)
    s2 = (s.astype(str)
            .str.replace("\u00A0", "", regex=False)  # nbsp
            .str.replace(" ", "", regex=False))
    s2b = (s2.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    v2 = pd.to_numeric(s2b, errors="coerce")
    return v2.fillna(0.0).astype(float)

def _fc_daily_B_and_C_positional(xls_bytes: bytes, sheet_name: str):
    # Strict positional reader: A=date, B=stock_on_hand/receipts, C=consommation
    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=sheet_name, header=0)

    # --- Debug Preview ---
    st.caption(f"Lecture (positional) de la feuille: **{sheet_name}**")
    st.dataframe(df.head(8), use_container_width=True)

    if df.shape[1] < 3:
        raise ValueError(f"Feuille '{sheet_name}' doit avoir au moins 3 colonnes (A,B,C).")

    date_col  = df.columns[0]
    stock_col = df.columns[1]
    cons_col  = df.columns[2]

    # parse
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    stock = _fc_parse_num_locale_positional(df[stock_col])
    cons  = _fc_parse_num_locale_positional(df[cons_col])

    g = (
        pd.DataFrame({"date": dates, "b": stock, "c": cons})
        .dropna(subset=["date"])
        .groupby("date", as_index=True)[["b", "c"]].sum(numeric_only=True)
        .sort_index()
    )
    if g.empty:
        # show a small warning box with head of raw columns to help
        st.warning(f"Feuille '{sheet_name}': aucune donnée exploitable après parsing "
                   f"(A=date, B=stock/receipts, C=consommation). Vérifiez formats & valeurs non nulles.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    stock_daily = g["b"].reindex(full_idx, fill_value=0.0); stock_daily.index.name = "date"
    cons_daily  = g["c"].reindex(full_idx, fill_value=0.0); cons_daily.index.name  = "date"

    # extra quick preview
    st.caption("📆 Séries quotidiennes (échantillon)")
    st.dataframe(
        pd.DataFrame({
            "stock_daily": stock_daily.head(10).values,
            "cons_daily":  cons_daily.head(10).values
        }, index=stock_daily.head(10).index),
        use_container_width=True
    )

    return stock_daily, cons_daily

def _nb_quantile_fallback(mean_, var_, service_level, nb_sim, rng):
    mean_ = float(max(mean_, 0.0))
    var_  = float(max(var_, mean_ + 1e-9))
    if mean_ == 0.0:
        return 0.0
    sl = float(np.clip(service_level, 1e-6, 1 - 1e-6))
    p = float(np.clip(mean_ / var_, 1e-12, 1 - 1e-12))
    r = float(mean_**2 / (var_ - mean_)) if var_ > mean_ else 1e6
    if not np.isfinite(r) or r <= 0:
        return 0.0
    # Gamma–Poisson mixture
    lam = rng.gamma(shape=r, scale=(1.0 - p) / p, size=int(max(1, nb_sim)))
    y = rng.poisson(lam)
    return float(np.percentile(y, 100.0 * sl))

def _nbinom_quantile_scipy(mean_, var_, service_level, nb_sim, rng):
    mean_ = float(max(mean_, 0.0))
    var_  = float(max(var_, mean_ + 1e-9))
    if mean_ == 0.0:
        return 0.0
    p = float(np.clip(mean_ / var_, 1e-12, 1 - 1e-12))
    r = float(mean_**2 / (var_ - mean_)) if var_ > mean_ else 1e6
    sample = nbinom.rvs(r, p, size=int(max(1, nb_sim)), random_state=rng)
    return float(np.percentile(sample, 100.0 * float(service_level)))

def _fc_ses_forecast_positional(x, alpha: float):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": 1.0}
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    f = float(l)
    return {"forecast_per_period": f, "z_t": f, "p_t": 1.0}

def _fc_croston_or_sba_forecast_positional(x, alpha: float, variant: str = "sba"):
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

def _fc_forecast_per_method_positional(x, alpha: float, method: str):
    m = method.lower()
    if m == "ses":
        return _fc_ses_forecast_positional(x, alpha)
    elif m == "croston":
        return _fc_croston_or_sba_forecast_positional(x, alpha, variant="croston")
    elif m == "sba":
        return _fc_croston_or_sba_forecast_positional(x, alpha, variant="sba")
    else:
        raise ValueError(f"Méthode inconnue '{method}'. Use 'sba', 'ses', 'croston'.")

def _fc_interval_sum_next_days_positional(daily: pd.Series, start_idx: int, interval: int) -> float:
    if daily is None or daily.empty:
        return 0.0
    s = start_idx + 1
    e = s + int(max(0, interval))
    return float(pd.Series(daily).iloc[s:e].sum())

def _fc_rolling_with_rops_single_run_positional(
    xls_bytes: bytes,
    xls: pd.ExcelFile,
    product_code: str,
    method: str,
    alpha: float,
    window_ratio: float,
    interval: int,
    lead_time: int,
    lead_time_supplier: int,
    service_level: float,
    nb_sim: int,
    rng_seed: int,
):
    # Find the matching sheet (keeps your existing finder in other part of code)
    sheet = _fc_find_product_sheet(xls, product_code)

    # read time series strictly by position A,B,C
    stock_daily, cons_daily = _fc_daily_B_and_C_positional(xls_bytes, sheet)
    if cons_daily is None or cons_daily.empty:
        return pd.DataFrame()

    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        st.info(f"[{product_code}] Pas assez d'observations après window_ratio={window_ratio}.")
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows = []
    rop_carry_running = 0.0
    stock_running_cum = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            test_date = cons_daily.index[i]

            train = vals[:i]
            fc = _fc_forecast_per_method_positional(train, alpha=alpha, method=method)
            f = float(fc["forecast_per_period"])
            sigma_period = float(pd.Series(train).std(ddof=1))
            if not np.isfinite(sigma_period):
                sigma_period = 0.0

            real_demand = _fc_interval_sum_next_days_positional(cons_daily,  i, interval)
            stock_on_hand_interval = _fc_interval_sum_next_days_positional(stock_daily, i, interval)
            stock_running_cum += float(stock_on_hand_interval)

            can_cover = bool(stock_running_cum >= real_demand)
            order_qty_policy = 0.5 * real_demand if can_cover else max(real_demand - stock_running_cum, 0.0)
            stock_status = "holding" if can_cover else "rupture"

            forecast_for_interval = f * interval

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5

            if _SCIPY_OK:
                ROP_u = _nbinom_quantile_scipy(X_Lt,    var_u, service_level, nb_sim, rng)
                ROP_f = _nbinom_quantile_scipy(X_Lt_Lw, var_f, service_level, nb_sim, rng)
            else:
                ROP_u = _nb_quantile_fallback(X_Lt,    var_u, service_level, nb_sim, rng)
                ROP_f = _nb_quantile_fallback(X_Lt_Lw, var_f, service_level, nb_sim, rng)

            rop_carry_running += float(ROP_u - real_demand)

            rows.append({
                "method": method,
                "date": test_date.date(),
                "code": product_code,
                "interval": int(interval),
                "real_demand": float(real_demand),
                "stock_on_hand_interval": float(stock_on_hand_interval),
                "stock_on_hand_running": float(stock_running_cum),
                "can_cover_interval": bool(can_cover),
                "order_qty_policy": float(order_qty_policy),
                "forecast_per_period": f,
                "forecast_for_interval": float(forecast_for_interval),
                "forecast_error": float(real_demand - forecast_for_interval),
                "X_Lt": float(X_Lt),
                "reorder_point_usine": float(ROP_u),
                "lead_time_usine_days": int(lead_time),
                "lead_time_supplier_days": int(lead_time_supplier),
                "X_Lt_Lw": float(X_Lt_Lw),
                "reorder_point_fournisseur": float(ROP_f),
                "stock_status": stock_status,
                "rop_usine_minus_real_running": float(rop_carry_running),
                "z_t": float(fc.get("z_t", 0.0)),
                "p_t": float(fc.get("p_t", 1.0)),
            })

    return pd.DataFrame(rows)

def _fc_compute_metrics_positional(df_run: pd.DataFrame):
    if df_run.empty or "forecast_error" not in df_run:
        return np.nan, np.nan, np.nan, np.nan
    e = df_run["forecast_error"].astype(float)
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    return ME, absME, MSE, RMSE

def _fc_grid_search_and_final_for_method_positional(
    xls_bytes: bytes,
    xls: pd.ExcelFile,
    product_codes: list,
    method: str,
    pick_metric: str,
    alphas: list,
    window_ratios: list,
    intervals: list,
    lead_time: int,
    lead_time_supplier: int,
    service_level: float,
    nb_sim: int,
    rng_seed: int,
):
    all_results, best_rows_per_code = [], []
    for code in product_codes:
        metrics_rows = []
        for a in alphas:
            for w in window_ratios:
                for itv in intervals:
                    df_run = _fc_rolling_with_rops_single_run_positional(
                        xls_bytes=xls_bytes, xls=xls, product_code=code, method=method,
                        alpha=a, window_ratio=w, interval=itv,
                        lead_time=lead_time, lead_time_supplier=lead_time_supplier,
                        service_level=service_level, nb_sim=nb_sim, rng_seed=rng_seed,
                    )
                    ME, absME, MSE, RMSE = _fc_compute_metrics_positional(df_run)
                    metrics_rows.append({
                        "code": code, "method": method, "alpha": a, "window_ratio": w, "recalc_interval": itv,
                        "ME": ME, "absME": absME, "MSE": MSE, "RMSE": RMSE, "n_points": len(df_run)
                    })
                    all_results.append(metrics_rows[-1])

        df_metrics = pd.DataFrame(metrics_rows)
        best_ME_idx = df_metrics["absME"].idxmin() if df_metrics["absME"].notna().any() else None
        best_MSE_idx = df_metrics["MSE"].idxmin()   if df_metrics["MSE"].notna().any()  else None
        best_RMSE_idx= df_metrics["RMSE"].idxmin()  if df_metrics["RMSE"].notna().any() else None

        best_ME   = df_metrics.loc[best_ME_idx]   if best_ME_idx   is not None else None
        best_MSE  = df_metrics.loc[best_MSE_idx]  if best_MSE_idx  is not None else None
        best_RMSE = df_metrics.loc[best_RMSE_idx] if best_RMSE_idx is not None else None

        best_rows_per_code.append({
            "code": code, "method": method,
            "best_ME_alpha": None if best_ME is None else best_ME["alpha"],
            "best_ME_window": None if best_ME is None else best_ME["window_ratio"],
            "best_ME_interval": None if best_ME is None else best_ME["recalc_interval"],
            "best_ME": None if best_ME is None else best_ME["ME"],
            "best_absME": None if best_ME is None else best_ME["absME"],
            "best_MSE_alpha": None if best_MSE is None else best_MSE["alpha"],
            "best_MSE_window": None if best_MSE is None else best_MSE["window_ratio"],
            "best_MSE_interval": None if best_MSE is None else best_MSE["recalc_interval"],
            "best_MSE": None if best_MSE is None else best_MSE["MSE"],
            "best_RMSE_alpha": None if best_RMSE is None else best_RMSE["alpha"],
            "best_RMSE_window": None if best_RMSE is None else best_RMSE["window_ratio"],
            "best_RMSE_interval": None if best_RMSE is None else best_RMSE["recalc_interval"],
            "best_RMSE": None if best_RMSE is None else best_RMSE["RMSE"],
            "n_points_used": int(best_RMSE["n_points"]) if best_RMSE is not None else 0,
        })
    return pd.DataFrame(all_results), pd.DataFrame(best_rows_per_code)

def _fc_pick_params_positional(row: pd.Series, metric: str):
    if metric == "best_ME":  return row["best_ME_alpha"],  row["best_ME_window"],  int(row["best_ME_interval"])
    if metric == "best_MSE": return row["best_MSE_alpha"], row["best_MSE_window"], int(row["best_MSE_interval"])
    return row["best_RMSE_alpha"], row["best_RMSE_window"], int(row["best_RMSE_interval"])

def _fc_final_run_for_best_positional(
    xls_bytes: bytes, xls: pd.ExcelFile, method: str, df_best: pd.DataFrame, code: str,
    pick_metric: str, lead_time: int, lead_time_supplier: int, service_level: float, nb_sim: int, rng_seed: int
):
    row = df_best[df_best["code"].astype(str) == str(code)]
    if row.empty: return pd.DataFrame()
    a, w, itv = _fc_pick_params_positional(row.iloc[0], pick_metric)
    if pd.isna(a) or pd.isna(w) or pd.isna(itv): return pd.DataFrame()
    return _fc_rolling_with_rops_single_run_positional(
        xls_bytes=xls_bytes, xls=xls, product_code=code, method=method,
        alpha=float(a), window_ratio=float(w), interval=int(itv),
        lead_time=lead_time, lead_time_supplier=lead_time_supplier,
        service_level=service_level, nb_sim=nb_sim, rng_seed=rng_seed,
    )

def _display_table_with_order_positional(df_run: pd.DataFrame):
    if df_run.empty:
        return pd.DataFrame()
    tbl = df_run.copy()
    tbl["order"] = tbl["order_qty_policy"].round(2)
    if "order_policy" in tbl.columns:
        tbl = tbl.drop(columns=["order_policy"])
    cols = [
        "method","date","code","interval",
        "real_demand","stock_on_hand_interval","stock_on_hand_running","can_cover_interval",
        "order",
        "forecast_per_period","forecast_for_interval","forecast_error",
        "X_Lt","reorder_point_usine","lead_time_usine_days",
        "lead_time_supplier_days","X_Lt_Lw","reorder_point_fournisseur",
        "stock_status","z_t","p_t"
    ]
    cols = [c for c in cols if c in tbl.columns]
    return tbl[cols]

def summarize_hold_vs_rupture_positional(df_run: pd.DataFrame):
    if df_run.empty:
        return 0.0, 0.0, 0.0
    diff = (df_run["reorder_point_usine"].fillna(0) - df_run["real_demand"]).astype(float)
    holding = diff.clip(lower=0).sum()
    rupture = (-diff).clip(lower=0).sum()
    share = (diff >= 0).mean()
    return float(holding), float(rupture), float(share)


# ====================== Section: Prévisions & ROP (POSitional) ======================
st.markdown("---")
st.header("📈 Prévisions & ROP — SBA / SES / Croston (positional)")

_fc_src = uploaded_opt or uploaded
if _fc_src is None:
    st.info("Téléversez un classeur contenant des feuilles **time serie <CODE>**.")
else:
    xls_bytes = _get_excel_bytes(_fc_src)
    try:
        xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    except Exception as e:
        st.error(f"Impossible de lire le classeur de prévisions : {e}")
        xls = None

    if xls is not None:
        codes = _fc_list_time_serie_codes(xls)
        if not codes:
            st.warning("Aucune feuille 'time serie *' détectée.")
            codes = xls.sheet_names  # fallback manual selection

        if codes:
            c_left, c_right = st.columns([2,1])
            with c_left:
                selected_codes = st.multiselect("Codes produit / Feuilles", options=codes, default=codes)
                methods = st.multiselect("Méthodes", options=["sba", "ses", "croston"], default=["sba", "ses", "croston"])
                pick_metric = st.selectbox("Critère de sélection", ["best_RMSE", "best_MSE", "best_ME"], index=0)
                alphas = st.multiselect("Alphas", [0.1, 0.2, 0.3, 0.4], default=[0.1, 0.2, 0.3, 0.4])
                window_ratios = st.multiselect("Window ratios", [0.6, 0.7, 0.8], default=[0.6, 0.7, 0.8])
                recalc_intervals = st.multiselect("Intervalles (jours)", [5, 10, 20], default=[5, 10, 20])
            with c_right:
                lead_time = st.number_input("Lead time usine (jours)", min_value=0, value=1, step=1)
                lead_time_supplier = st.number_input("Lead time fournisseur + (jours)", min_value=0, value=3, step=1)
                base_service_level = st.slider("Niveau de service (table détaillée)", 0.50, 0.999, 0.95, 0.001)
                nb_sim = st.number_input("Taille simulation NB", min_value=100, step=100, value=1000)
                rng_seed = st.number_input("RNG seed", min_value=0, value=42, step=1)

            if not _SCIPY_OK:
                st.warning("⚠️ SciPy non dispo — ROP via **fallback Gamma–Poisson** (écarts possibles).")

            # ---------- Caching wrapper ----------
            @st.cache_data(show_spinner=False)
            def cache_grid_positional(xls_bytes: bytes, codes: list, method: str, pick_metric: str,
                                      alphas: list, window_ratios: list, intervals: list,
                                      lead_time: int, lead_time_supplier: int, service_level: float,
                                      nb_sim: int, rng_seed: int):
                xls_local = pd.ExcelFile(io.BytesIO(xls_bytes))
                return _fc_grid_search_and_final_for_method_positional(
                    xls_bytes=xls_bytes, xls=xls_local, product_codes=codes,
                    method=method, pick_metric=pick_metric, alphas=alphas,
                    window_ratios=window_ratios, intervals=intervals,
                    lead_time=int(lead_time), lead_time_supplier=int(lead_time_supplier),
                    service_level=float(service_level), nb_sim=int(nb_sim), rng_seed=int(rng_seed),
                )

            run = st.button("▶️ Lancer les prévisions (positional)")
            if run:
                if not selected_codes:
                    st.warning("Sélectionnez au moins un code/feuille.")
                elif not methods:
                    st.warning("Sélectionnez au moins une méthode.")
                else:
                    tabs = st.tabs([m.upper() for m in methods])
                    for m, tab in zip(methods, tabs):
                        with tab:
                            with st.spinner(f"Exécution {m.upper()} (positional)…"):
                                df_all, df_best = cache_grid_positional(
                                    xls_bytes, selected_codes, m, pick_metric,
                                    alphas, window_ratios, recalc_intervals,
                                    lead_time, lead_time_supplier, base_service_level,
                                    nb_sim, rng_seed
                                )
                            st.session_state[f"best_{m}"] = df_best.copy()

                            c1, c2 = st.columns([2,1], vertical_alignment="top")
                            with c1:
                                st.subheader("Meilleures combinaisons par article")
                                st.dataframe(df_best, use_container_width=True, hide_index=True)
                            with c2:
                                st.download_button(
                                    "Télécharger (best) CSV",
                                    data=df_best.to_csv(index=False).encode("utf-8"),
                                    file_name=f"best_combos_{m}_positional.csv",
                                    mime="text/csv",
                                )

                            st.subheader("Toutes les combinaisons testées")
                            st.dataframe(df_all, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Télécharger (grid) CSV",
                                data=df_all.to_csv(index=False).encode("utf-8"),
                                file_name=f"grid_search_{m}_positional.csv",
                                mime="text/csv",
                            )

                            # ---- Detailed table for one product ----
                            st.markdown("### Table détaillée (meilleurs paramètres) avec **order**")
                            if not df_best.empty:
                                code_for_table = st.selectbox(
                                    "Produit pour la table détaillée",
                                    options=sorted(df_best["code"].astype(str).unique()),
                                    key=f"{m}_table_code_pos"
                                )
                                if code_for_table:
                                    df_final_table = _fc_final_run_for_best_positional(
                                        xls_bytes=xls_bytes, xls=xls, method=m,
                                        df_best=df_best, code=code_for_table,
                                        pick_metric=pick_metric,
                                        lead_time=int(lead_time), lead_time_supplier=int(lead_time_supplier),
                                        service_level=float(base_service_level),
                                        nb_sim=int(nb_sim), rng_seed=int(rng_seed),
                                    )
                                    tbl = _display_table_with_order_positional(df_final_table)
                                    if tbl.empty or (tbl[["real_demand","stock_on_hand_interval","forecast_for_interval"]]
                                                     .sum().sum() == 0):
                                        st.warning("⚠️ Données nulles pour ce produit. Vérifiez la colonne C (consommation).")
                                    st.dataframe(tbl, use_container_width=True)
                                    st.download_button(
                                        "Télécharger la table (CSV)",
                                        data=tbl.to_csv(index=False).encode("utf-8"),
                                        file_name=f"details_{m}_{code_for_table}_positional.csv",
                                        mime="text/csv"
                                    )

# ====================== Comparison (Mean Holding & CT, positional) ======================
st.markdown("---")
st.header("📊 Comparaison: Mean Holding & CT (POSitional)")

# ---- CT Parameters ----
A_w, A_R = 50, 70
pi_c, tau_c = 1.0, 1.0
T_w = 1.0
C_w, C_R = 5.0, 8.0  # C_R' = 3

def compute_ct_positional(D, Qw, Qr):
    Iw_prime = Qw / 2.0
    Ir_prime = Qr / 2.0
    Cw_prime = C_w
    Cr_prime = C_R - C_w
    return (
        A_w * (D / Qw if Qw > 0 else 0)
        + pi_c * T_w * Cw_prime * Iw_prime
        + A_R * (D / Qr if Qr > 0 else 0)
        + tau_c * Cr_prime * Ir_prime
    )

SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

@st.cache_data(show_spinner=False)
def cache_final_run_for_sl_positional(
    xls_bytes: bytes, method: str, df_best: pd.DataFrame, code: str,
    pick_metric: str, lead_time: int, lead_time_supplier: int,
    service_level: float, nb_sim: int, rng_seed: int
):
    xls_local = pd.ExcelFile(io.BytesIO(xls_bytes))
    return _fc_final_run_for_best_positional(
        xls_bytes=xls_bytes, xls=xls_local, method=method, df_best=df_best, code=code,
        pick_metric=pick_metric, lead_time=lead_time, lead_time_supplier=lead_time_supplier,
        service_level=service_level, nb_sim=nb_sim, rng_seed=rng_seed
    )

if st.button("📈 Lancer la comparaison (Holding & CT, positional)"):
    df_best_sba = st.session_state.get("best_sba")
    df_best_croston = st.session_state.get("best_croston")
    df_best_ses = st.session_state.get("best_ses")

    if all([(df_best_sba is None or df_best_sba.empty),
            (df_best_croston is None or df_best_croston.empty),
            (df_best_ses is None or df_best_ses.empty)]):
        st.warning("Aucune 'best table' dispo. Lancez d’abord les prévisions.")
    else:
        records_holding, records_ct = [], []

        for method_name, df_best in [
            ("sba", df_best_sba),
            ("croston", df_best_croston),
            ("ses", df_best_ses),
        ]:
            if df_best is None or df_best.empty:
                continue

            for _, r in df_best.iterrows():
                code = r["code"]
                for sl in SERVICE_LEVELS:
                    df_run = cache_final_run_for_sl_positional(
                        xls_bytes, method_name, df_best, code, pick_metric,
                        int(lead_time), int(lead_time_supplier), float(sl),
                        int(nb_sim), int(rng_seed)
                    )
                    if df_run is None or df_run.empty:
                        mean_holding_val, CT_val = np.nan, np.nan
                    else:
                        mean_holding_val = df_run.loc[
                            df_run["stock_status"] == "holding",
                            "rop_usine_minus_real_running"
                        ].mean()

                        D = df_run["real_demand"].sum()
                        Qw = df_run["reorder_point_fournisseur"].mean()
                        Qr = df_run["reorder_point_usine"].mean()
                        CT_val = compute_ct_positional(D, Qw, Qr)

                    records_holding.append({
                        "product": code,
                        "method": method_name.upper(),
                        "service_level": f"{int(sl*100)}%",
                        "Mean_Holding": mean_holding_val,
                    })
                    records_ct.append({
                        "product": code,
                        "method": method_name.upper(),
                        "service_level": f"{int(sl*100)}%",
                        "CT": CT_val,
                    })

        df_holding = pd.DataFrame(records_holding)
        df_ct = pd.DataFrame(records_ct)

        if not df_holding.empty:
            table_holding = df_holding.pivot_table(
                index="product", columns=["method", "service_level"], values="Mean_Holding"
            )
            st.subheader("📊 Table de comparaison — Mean Holding (positional)")
            st.dataframe(table_holding, use_container_width=True)
            st.download_button(
                "Télécharger (Holding) CSV",
                data=table_holding.to_csv().encode("utf-8"),
                file_name="comparison_mean_holding_positional.csv",
                mime="text/csv"
            )

        if not df_ct.empty:
            table_ct = df_ct.pivot_table(
                index="product", columns=["method", "service_level"], values="CT"
            )
            st.subheader("💰 Table de comparaison — CT (positional)")
            st.dataframe(table_ct, use_container_width=True)
            st.download_button(
                "Télécharger (CT) CSV",
                data=table_ct.to_csv().encode("utf-8"),
                file_name="comparison_ct_positional.csv",
                mime="text/csv"
            )
