# ==================================================
# APPLICATION STREAMLIT - PFE HANIN
# Base Stock + Prévisions (SES / Croston / SBA)
# Sélection meilleure méthode + Simulation commandes
# + Analyse de sensibilité
# ==================================================

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import streamlit as st
import matplotlib.pyplot as plt

# ---------- PARAMÈTRES ----------
st.set_page_config(page_title="PFE HANIN - Base Stock", layout="wide")

st.title("📦 Application PFE HANIN")
st.markdown("Méthode **Base Stock** + Prévisions (SES / Croston / SBA)")

# Sidebar : paramètres
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("Chargez le fichier Excel", type=["xlsx"])

default_products = ["EM0400","EM1499","EM1091","EM1523","EM0392","EM1526"]
PRODUCT_CODE = st.sidebar.selectbox("Choisir le produit", default_products, index=0)

# Fixed parameters
NB_SIM = 1000
RNG_SEED = 42
LEAD_TIME = 10
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]
SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

# ==================================================
# PARTIE 1 : Qr*, Qw* et n* (Base Stock)
# ==================================================
def _find_product_sheet(excel_path, code: str) -> str:
    xls = pd.ExcelFile(excel_path)
    sheets = [s.strip().lower() for s in xls.sheet_names]
    targets = [f"time serie {code}".lower(), f"time series {code}".lower(), code.lower()]
    for t in targets:
        if t in sheets:
            return xls.sheet_names[sheets.index(t)]
    for s in sheets:
        if code.lower() in s:
            return xls.sheet_names[sheets.index(s)]
    raise ValueError(f"[Sheet] Onglet pour '{code}' introuvable.")

def compute_qstars(file_path, product_code):
    df_conso = pd.read_excel(file_path, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()
    qr_map, qw_map, n_map = {}, {}, {}
    for code in [product_code]:
        sheet = _find_product_sheet(file_path, code)
        df = pd.read_excel(file_path, sheet_name=sheet)
        C_r = df['Cr : cout stockage/article '].iloc[0]
        C_w = df['Cw : cout stockage\nchez F'].iloc[0]
        A_w = df['Aw : cout de\nlancement chez U'].iloc[0]
        A_r = df['Ar : cout de \nlancement chez F'].iloc[0]

        n = (A_w * C_r) / (A_r * C_w)
        n = 1 if n < 1 else round(n)
        n1, n2 = int(n), int(n) + 1
        F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
        F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
        n_star = n1 if F_n1 <= F_n2 else n2

        D = df_conso.get(code, 0)
        tau = 1
        Qr_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Qw_star = n_star * Qr_star

        qr_map[code] = round(Qr_star, 2)
        qw_map[code] = round(Qw_star, 2)
        n_map[code] = n_star

    return qr_map, qw_map, n_map

# ==================================================
# PARTIE 2 : Prévisions (SES / Croston / SBA)
# ==================================================
def croston_or_sba_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if (x == 0).all():
        return 0.0
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    z, p = x[nz_idx[0]], len(x)/len(nz_idx)
    psd = 0
    for t in range(nz_idx[0]+1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1-alpha) * z
            p = alpha * I_t + (1-alpha) * p
            psd = 0
    f = z / p
    if variant == "sba":
        f *= (1 - alpha/2.0)
    return f

def ses_forecast(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l

def load_matrix_timeseries(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    prod_col = df.columns[0]
    new_cols = [prod_col]
    for c in df.columns[1:]:
        try:
            new_cols.append(pd.to_datetime(c))
        except:
            new_cols.append(c)
    df.columns = new_cols
    return df, prod_col

def rolling_forecast_with_metrics(excel_path, product_code, sheet_name, alpha, window_ratio, interval, method):
    df, prod_col = load_matrix_timeseries(excel_path, sheet_name)
    row = df.loc[df[prod_col] == product_code]
    if row.empty:
        return pd.DataFrame()
    series = row.drop(columns=[prod_col]).T.squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(full_idx, fill_value=0.0).astype(float)
    values = daily.values
    split_index = int(len(values) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()
    out_rows = []
    for i in range(split_index, len(values)):
        if (i - split_index) % interval == 0:
            train = values[:i]
            real_demand = float(values[i])
            if method == "ses":
                f = ses_forecast(train, alpha)
            elif method == "croston":
                f = croston_or_sba_forecast(train, alpha, "croston")
            elif method == "sba":
                f = croston_or_sba_forecast(train, alpha, "sba")
            else:
                f = 0.0
            out_rows.append({"real_demand": real_demand, "forecast": f, "error": real_demand - f})
    return pd.DataFrame(out_rows)

def compute_metrics(df_run):
    if df_run.empty:
        return np.nan, np.nan, np.nan
    e = df_run["error"].astype(float)
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    absME = e.abs().mean()
    return absME, MSE, RMSE

def grid_search_all_methods(file_path, product_code):
    candidates = []
    for code in [product_code]:
        for method in ["ses","croston","sba"]:
            metrics_rows = []
            for a in ALPHAS:
                for w in WINDOW_RATIOS:
                    for itv in RECALC_INTERVALS:
                        df_run = rolling_forecast_with_metrics(file_path, code, "classification", a, w, itv, method)
                        absME, MSE, RMSE = compute_metrics(df_run)
                        metrics_rows.append({
                            "code": code, "method": method, "alpha": a,
                            "window_ratio": w, "recalc_interval": itv,
                            "absME": absME, "MSE": MSE, "RMSE": RMSE
                        })
            df_metrics = pd.DataFrame(metrics_rows)
            candidates.append(df_metrics)
    return pd.concat(candidates, ignore_index=True)

# ==================================================
# PARTIE 3 : Simulation finale avec ROP (improved)
# ==================================================
def simulate_orders(file_path, best_per_code, qr_map, service_level=SERVICE_LEVEL):
    results = []
    rng = np.random.default_rng(RNG_SEED)

    for _, row in best_per_code.iterrows():
        code = row["code"]; method = row["method"]
        alpha = row["alpha"]; window_ratio = row["window_ratio"]; interval = int(row["recalc_interval"])

        # Load product time series
        sheet = _find_product_sheet(file_path, code)
        df = pd.read_excel(file_path, sheet_name=sheet)
        dates = pd.to_datetime(df.iloc[:,0], errors="coerce")
        cons_col = pd.to_numeric(df.iloc[:,2], errors="coerce").fillna(0.0).astype(float)

        ts_cons = pd.DataFrame({"d":dates,"q":cons_col}).dropna().sort_values("d").set_index("d")["q"]
        full_idx = pd.date_range(ts_cons.index.min(), ts_cons.index.max(), freq="D")
        cons_daily = ts_cons.reindex(full_idx, fill_value=0.0)

        vals = cons_daily.values
        split_index = int(len(vals) * window_ratio)
        if split_index < 2: 
            continue

        # ---------- Inventory initialization ----------
        stock_on_hand = qr_map[code] * 2  # start with 2 lots as initial stock
        pipeline_orders = []              # list of {"arrival_time": idx, "qty": Qr*}

        # ---------- Simulation ----------
        for i in range(split_index, len(vals)):
            # --- Receive orders that arrive today ---
            arrivals = sum(o["qty"] for o in pipeline_orders if o["arrival_time"] == i)
            stock_on_hand += arrivals
            pipeline_orders = [o for o in pipeline_orders if o["arrival_time"] > i]

            # --- Forecast demand ---
            train = vals[:i]
            if method == "ses":
                f = ses_forecast(train, alpha)
            elif method == "croston":
                f = croston_or_sba_forecast(train, alpha, "croston")
            else:
                f = croston_or_sba_forecast(train, alpha, "sba")

            sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            sigma_period = sigma_period if np.isfinite(sigma_period) else 0.0

            # --- Reorder point (negative binomial) ---
            X_Lt = LEAD_TIME * f
            sigma_Lt = sigma_period * np.sqrt(max(LEAD_TIME, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt+1e-5
            p_nb = min(max(X_Lt/var_u, 1e-12),1-1e-12)
            r_nb = X_Lt**2/(var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(
                nbinom.rvs(r_nb, p_nb, size=NB_SIM, random_state=rng),
                100*service_level
            ))

            # --- Consume demand ---
            real_demand = float(vals[i])
            stock_on_hand -= real_demand

            # --- Check rupture ---
            stock_status = "rupture" if stock_on_hand < 0 else "holding"

            # --- Place new order if stock below ROP ---
            if stock_on_hand <= ROP_u:
                pipeline_orders.append({
                    "arrival_time": i + LEAD_TIME,
                    "qty": qr_map[code]
                })
                order_policy = f"order_Qr*_{qr_map[code]}"
            else:
                order_policy = "no_order"

            # --- Record results ---
            results.append({
                "date": cons_daily.index[i].date(),
                "code": code,
                "interval": interval,
                "real_demand": real_demand,
                "stock_on_hand": stock_on_hand,
                "order_policy": order_policy,
                "Qr_star": qr_map[code],
                "reorder_point_usine": ROP_u,
                "stock_status": stock_status,
                "service_level": service_level,
                "method": method
            })

    return pd.DataFrame(results)

# ==================================================
# PARTIE 4 : Analyse de sensibilité + Plot
# ==================================================
def run_sensitivity_with_methods(file_path, best_per_code, qr_map):
    all_results = []
    for sl in SERVICE_LEVELS:
        df_run = simulate_orders(file_path, best_per_code, qr_map, service_level=sl)
        if not df_run.empty:
            summary = df_run.groupby(["code","method"]).agg(
                holding_pct=("stock_status", lambda s: (s=="holding").mean()*100),
                rupture_pct=("stock_status", lambda s: (s=="rupture").mean()*100),
                Qr_star=("Qr_star","first")
            ).reset_index()
            summary["service_level"] = sl
            all_results.append(summary)
    return pd.concat(all_results, ignore_index=True)

def plot_tradeoff(df_summary):
    if df_summary.empty:
        st.warning("Pas de résultats pour tracer la sensibilité.")
        return
    
    plt.figure(figsize=(8,6))
    methods = df_summary["method"].unique()
    markers = {"ses":"o", "croston":"s", "sba":"^"}
    
    for method in methods:
        subset = df_summary[df_summary["method"] == method]
        plt.scatter(subset["holding_pct"], subset["rupture_pct"],
                    label=method, marker=markers.get(method,"o"))
        for _, row in subset.iterrows():
            plt.annotate(f"{row['code']} (SL={row['service_level']})",
                         (row["holding_pct"], row["rupture_pct"]),
                         fontsize=8, alpha=0.7)
    
    plt.xlabel("Holding %")
    plt.ylabel("Rupture %")
    plt.title("Trade-off Holding vs Rupture (%) – All Methods & SL")
    plt.legend()
    st.pyplot(plt)

# ==================================================
# MAIN STREAMLIT
# ==================================================
if uploaded_file is not None:
    with st.spinner("⏳ Calcul en cours..."):
        qr_map, qw_map, n_map = compute_qstars(uploaded_file, PRODUCT_CODE)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Base Stock", "🔮 Prévisions", "📦 Simulation", "📈 Sensibilité"])

        with tab1:
            st.subheader("Qr*, Qw* et n* (Base Stock)")
            df_base = pd.DataFrame({
                "Produit": list(qr_map.keys()),
                "Qr*": list(qr_map.values()),
                "Qw*": list(qw_map.values()),
                "n*": list(n_map.values())
            })
            st.dataframe(df_base)

        with tab2:
            st.subheader("Meilleure méthode de prévision")
            all_candidates = grid_search_all_methods(uploaded_file, PRODUCT_CODE)
            best_per_code = all_candidates.loc[all_candidates["RMSE"].idxmin()].to_frame().T.reset_index(drop=True)
            st.dataframe(best_per_code)

        with tab3:
            st.subheader(f"Simulation finale (SL={SERVICE_LEVEL:.2f})")
            final_results = simulate_orders(uploaded_file, best_per_code, qr_map, service_level=SERVICE_LEVEL)
            st.dataframe(final_results.head(50))

        with tab4:
            st.subheader("Analyse de sensibilité")
            sensitivity_summary = run_sensitivity_with_methods(uploaded_file, best_per_code, qr_map)
            st.dataframe(sensitivity_summary.head(50))
            plot_tradeoff(sensitivity_summary)
else:
    st.info("📥 Veuillez charger un fichier Excel pour commencer.")
