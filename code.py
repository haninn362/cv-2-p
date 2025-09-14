# app.py
import io
import re
from typing import Tuple, List

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

# ============================ Classification logic ============================
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

def excel_bytes(combined_df, stats_df, counts_df, methods_df) -> io.BytesIO:
    buf = io.BytesIO()
    for engine in ("openpyxl", "xlsxwriter", None):
        try:
            writer = pd.ExcelWriter(buf, engine=engine) if engine else pd.ExcelWriter(buf)
            with writer:
                sheet = "Résultats"
                stats_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
                r2 = len(stats_df) + 3
                counts_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
                r3 = r2 + len(counts_df) + 3
                combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r3, startcol=0)
                methods_df.reset_index().to_excel(writer, index=False, sheet_name="Méthodes")
            break
        except ModuleNotFoundError:
            buf = io.BytesIO()
            continue
    buf.seek(0)
    return buf

# ======================== Helpers / Excel I/O =======================
def _norm(s: str) -> str: return re.sub(r"\s+", " ", str(s).strip().lower())

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

# ======================== Optimisation (n*, Qr*, Qw*) =======================
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

# ============================== UI ==============================
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0
nonce = st.session_state["uploader_nonce"]

# ---------- fixed bar ----------
st.markdown('<div class="fixed-header"><div class="fixed-inner">', unsafe_allow_html=True)
st.markdown('<div class="controls-holder"><div class="controls-right">', unsafe_allow_html=True)

st.markdown('<div class="control">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Classeur **classification**",
    type=["xlsx", "xls"],
    key=f"clf_{nonce}",
    help="Feuille choisie = table large Produit × Périodes."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="control">', unsafe_allow_html=True)
uploaded_opt = st.file_uploader(
    "Classeur **optimisation** (optionnel)",
    type=["xlsx", "xls"],
    key=f"opt_{nonce}",
    help="Inclut 'consommation depots externe' + feuilles 'time serie *'."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="control">', unsafe_allow_html=True)
if st.button("🔄 Réinitialiser", key=f"reset_{nonce}", help="Efface les fichiers et la sélection."):
    st.session_state["uploader_nonce"] += 1
    for k in ["selected_product"]:
        st.session_state.pop(k, None)
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)  # end controls + holder
st.markdown('</div></div>', unsafe_allow_html=True)  # end header

st.title("Classification minimale — taille/fréquence → CV² & p → méthode")

# ===== classification sheet selector =====
sheet_name = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        noms = [s.lower() for s in xls.sheet_names]
        default_idx = noms.index("classification") if "classification" in noms else 0
        sheet_name = st.selectbox("Feuille (classeur de classification)", options=xls.sheet_names, index=default_idx)
    except Exception as e:
        st.error(f"Impossible de lire le classeur : {e}")

def compute_and_show(uploaded, sheet_name, uploaded_opt):
    if uploaded is None or sheet_name is None: return
    df_raw = pd.read_excel(uploaded, sheet_name=sheet_name)

    col_produit = df_raw.columns[0]
    produits = sorted(df_raw[col_produit].astype(str).dropna().unique().tolist())
    if not produits:
        st.warning("Aucun produit trouvé dans la première colonne.")
        return
    produit_sel = st.selectbox("Choisir un produit", options=produits, key="selected_product")

    combined_df, stats_df, counts_df, methods_df = compute_everything(df_raw)

    stats_one = stats_df.loc[[produit_sel]] if produit_sel in stats_df.index else stats_df.iloc[0:0]
    counts_one = counts_df.loc[[produit_sel]] if produit_sel in counts_df.index else counts_df.iloc[0:0]
    methods_one = methods_df.loc[[produit_sel]] if produit_sel in methods_df.index else methods_df.iloc[0:0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tableau 1 — moyenne / écart-type / CV² (sélection)**")
        st.dataframe(stats_one.reset_index(), use_container_width=True)
    with c2:
        st.markdown("**Tableau 2 — N périodes / N fréquences / p (sélection)**")
        st.dataframe(counts_one.reset_index(), use_container_width=True)

    st.markdown("**Combiné — taille / frequence (sélection)**")
    comb_sel = pd.DataFrame()
    if not combined_df.empty:
        mask_taille = (combined_df["Produit"] == produit_sel) & (combined_df["Type"] == "taille")
        if mask_taille.any():
            idx = combined_df.index[mask_taille][0]
            rows = [idx]
            if idx + 1 in combined_df.index: rows.append(idx + 1)
            comb_sel = combined_df.loc[rows]
    st.dataframe(comb_sel if not comb_sel.empty else pd.DataFrame(), use_container_width=True)

    st.markdown("**Graphe — p vs CV² avec seuils (sélection)**")
    if not methods_one.empty:
        fig = make_plot(methods_one); st.pyplot(fig, use_container_width=True)
    else:
        st.info("Pas de graphe pour ce produit.")
    st.markdown("**Méthode par produit (sélection)**")
    st.dataframe(methods_one.reset_index(), use_container_width=True)

    # Optimisation
    st.markdown("**Optimisation — n\\*, Qr\\*, Qw\\* (sélection)**")
    opt_source = uploaded_opt or uploaded
    st.caption("Classeur utilisé : " + ("optimisation séparé" if uploaded_opt is not None else "classification"))
    opt_df, info_msgs, warn_msgs = compute_qr_qw_from_workbook(opt_source)
    for msg in info_msgs: st.info(msg)
    for msg in warn_msgs: st.warning(msg)

    m = re.search(r"\b[A-Z]{2}\d{4}\b", str(produit_sel))
    opt_key = m.group(0) if m else str(produit_sel)
    opt_one = opt_df[opt_df["Code Produit"].astype(str) == opt_key]
    if opt_one.empty:
        st.info(f"Aucune ligne d’optimisation pour **{produit_sel}** (code recherché : '{opt_key}').")
    else:
        st.dataframe(opt_one, use_container_width=True)

    # Téléchargements
    xbuf = excel_bytes(combined_df, stats_df, counts_df, methods_df)
    st.download_button("Télécharger TOUS les résultats (Excel)", data=xbuf,
                       file_name="resultats_classification.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if not methods_one.empty:
        pbuf = io.BytesIO()
        fig.savefig(pbuf, format="png", bbox_inches="tight")
        pbuf.seek(0)
        st.download_button("Télécharger le graphe du produit (PNG)", data=pbuf,
                           file_name=f"classification_{opt_key or 'produit'}.png",
                           mime="image/png")

    if not opt_df.empty:
        st.download_button(
            "Télécharger TOUTES les optimisations (CSV)",
            data=opt_df.to_csv(index=False).encode("utf-8"),
            file_name="optimisation_qr_qw.csv",
            mime="text/csv"
        )

if uploaded is not None and sheet_name is not None:
    try:
        compute_and_show(uploaded, sheet_name, uploaded_opt)
    except Exception as e:
        st.error(f"Échec du traitement : {e}")
else:
    st.info("Téléversez d’abord le classeur de classification. (Vous pouvez aussi téléverser un classeur d’optimisation séparé.)")


# ============================================
# Unified Script: SBA + Croston + SES (unique names per method)
# Grid Search (ME, MSE, RMSE) + Final Recalc (display only selected columns)
# Optimized to reduce runtime & console noise
# ============================================

import numpy as np as _np2
import pandas as pd as _pd2
import re as _re2
from scipy.stats import nbinom as _nbinom2
from IPython.display import display

# ---------- GLOBAL PARAMETERS ----------
EXCEL_PATH_UNI = "PFE  HANIN (1).xlsx"
PRODUCT_CODES_UNI = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

# Optimized Grid Search (balanced speed/coverage)
ALPHAS_UNI = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
WINDOW_RATIOS_UNI = [0.6, 0.7, 0.8]
RECALC_INTERVALS_UNI = [1, 2, 5, 10, 15, 20]

# Supply / ROP
LEAD_TIME_UNI = 1
LEAD_TIME_SUPPLIER_UNI = 3
SERVICE_LEVEL_UNI = 0.95
NB_SIM_UNI = 800   # slight reduction for speed
RNG_SEED_UNI = 42

# Desired final columns to display
DISPLAY_COLUMNS_UNI = [
    "date", "code", "interval", "real_demand", "stock_on_hand_running",
    "stock_after_interval", "can_cover_interval", "order_policy",
    "reorder_point_usine", "lead_time_usine_days", "lead_time_supplier_days",
    "reorder_point_fournisseur", "stock_status", "rop_usine_minus_real_running"
]

# =====================================================================
# =========================  SBA SECTION  ==============================
# =====================================================================

def _find_product_sheet_sba(excel_path: str, code: str) -> str:
    xls = _pd2.ExcelFile(excel_path)
    sheets = xls.sheet_names
    target = f"time serie {code}"
    if target in sheets:
        return target
    patt = _re2.compile(r"time\s*ser(i|ie)s?\s*", _re2.IGNORECASE)
    cand = [s for s in sheets if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    for s in sheets:
        if s.strip().lower() == code.lower():
            return s
    raise ValueError(f"[SBA] Onglet pour '{code}' introuvable (attendu: 'time serie {code}').")

def _daily_consumption_and_stock_sba(excel_path: str, sheet_name: str):
    df = _pd2.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError(f"[SBA] Feuille '{sheet_name}': colonnes insuffisantes (A=date, B=stock, C=qté).")
    date_col, stock_col, cons_col = cols[0], cols[1], cols[2]

    dates = _pd2.to_datetime(df[date_col], errors="coerce")
    cons = _pd2.to_numeric(df[cons_col], errors="coerce").fillna(0.0).astype(float)
    stock = _pd2.to_numeric(df[stock_col], errors="coerce").astype(float)

    ts_cons  = (_pd2.DataFrame({"d": dates, "q": cons})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["q"])
    ts_stock = (_pd2.DataFrame({"d": dates, "s": stock})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["s"])

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = _pd2.date_range(min_date, max_date, freq="D")

    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    return cons_daily, stock_daily

def _interval_sum_next_days_sba(daily: _pd2.Series, start_idx: int, interval: int) -> float:
    s = start_idx + 1
    e = s + int(max(0, interval))
    return float(_pd2.Series(daily).iloc[s:e].sum())

def _croston_or_sba_forecast_array_sba(x, alpha: float, variant: str = "sba"):
    x = _pd2.Series(x).fillna(0.0).astype(float).values
    x = _np2.where(x < 0, 0.0, x)
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
    sheet = _find_product_sheet_sba(excel_path, product_code)
    cons_daily, stock_daily = _daily_consumption_and_stock_sba(excel_path, sheet)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return _pd2.DataFrame()

    rng = _np2.random.default_rng(rng_seed)
    rows = []
    rop_carry_running = 0.0
    stock_after_interval = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]

            fc = _croston_or_sba_forecast_array_sba(train, alpha=alpha, variant=variant)
            f = float(fc["forecast_per_period"])
            sigma_period = float(_pd2.Series(train).std(ddof=1)) if i > 1 else 0.0
            if not _np2.isfinite(sigma_period):
                sigma_period = 0.0

            real_demand = _interval_sum_next_days_sba(cons_daily, i, interval)              # Col C
            stock_on_hand_running = _interval_sum_next_days_sba(stock_daily, i, interval)   # Col B
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand

            next_real_demand = _interval_sum_next_days_sba(cons_daily, i + interval, interval) if (i + interval) < len(vals) else 0.0
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * _np2.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(_np2.percentile(_nbinom2.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * _np2.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(_np2.percentile(_nbinom2.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))

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

    return _pd2.DataFrame(rows)

def compute_metrics_sba(df_run: _pd2.DataFrame):
    if df_run.empty or "real_demand" not in df_run or "reorder_point_usine" not in df_run:
        return _np2.nan, _np2.nan, _np2.nan, _np2.nan
    est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, _np2.nan)
    e = df_run["real_demand"] - est
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = float(_np2.sqrt(MSE)) if _np2.isfinite(MSE) else _np2.nan
    return ME, absME, MSE, RMSE

def _grid_and_final_sba():
    best_rows = []
    for code in PRODUCT_CODES_UNI:
        best_row = None
        best_rmse = _np2.inf
        # fast grid (quiet)
        for a in ALPHAS_UNI:
            for w in WINDOW_RATIOS_UNI:
                for itv in RECALC_INTERVALS_UNI:
                    df_run = rolling_sba_with_rops_single_run(
                        excel_path=EXCEL_PATH_UNI, product_code=code,
                        alpha=a, window_ratio=w, interval=itv,
                        lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
                        service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
                        variant="sba",
                    )
                    _, _, _, RMSE = compute_metrics_sba(df_run)
                    if _pd2.notna(RMSE):
                        # prefer slightly larger interval/alpha/window when close (1% tolerance)
                        if (RMSE < best_rmse * 0.99) or (_np2.isclose(RMSE, best_rmse, rtol=0.01) and best_row is not None and (
                            (itv, a, w) > (best_row["recalc_interval"], best_row["alpha"], best_row["window_ratio"])
                        )):
                            best_rmse = RMSE
                            best_row = {"code": code, "alpha": a, "window_ratio": w, "recalc_interval": itv, "RMSE": RMSE}
        if best_row:
            best_rows.append(best_row)

    df_best_sba = _pd2.DataFrame(best_rows)
    print("\n✅ SBA — Best parameters (by RMSE):")
    display(df_best_sba)

    # Final recalculation and display desired columns only
    print("\n— SBA Final Tables (best params) —")
    for _, r in df_best_sba.iterrows():
        code = r["code"]; a = r["alpha"]; w = r["window_ratio"]; itv = r["recalc_interval"]
        print(f"Final SBA {code}: alpha={a}, window={w}, interval={itv}")
        df_final = rolling_sba_with_rops_single_run(
            excel_path=EXCEL_PATH_UNI, product_code=code,
            alpha=float(a), window_ratio=float(w), interval=int(itv),
            lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
            service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
            variant="sba"
        )
        display(df_final[DISPLAY_COLUMNS_UNI])
    return df_best_sba

# =====================================================================
# =======================  CROSTON SECTION  ============================
# =====================================================================

def _find_product_sheet_croston(excel_path: str, code: str) -> str:
    xls = _pd2.ExcelFile(excel_path)
    sheets = xls.sheet_names
    target = f"time serie {code}"
    if target in sheets:
        return target
    patt = _re2.compile(r"time\s*ser(i|ie)s?\s*", _re2.IGNORECASE)
    cand = [s for s in sheets if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    for s in sheets:
        if s.strip().lower() == code.lower():
            return s
    raise ValueError(f"[Croston] Onglet pour '{code}' introuvable (attendu: 'time serie {code}').")

def _daily_consumption_and_stock_croston(excel_path: str, sheet_name: str):
    df = _pd2.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError(f"[Croston] Feuille '{sheet_name}': colonnes insuffisantes (A=date, B=stock, C=qté).")
    date_col, stock_col, cons_col = cols[0], cols[1], cols[2]

    dates = _pd2.to_datetime(df[date_col], errors="coerce")
    cons = _pd2.to_numeric(df[cons_col], errors="coerce").fillna(0.0).astype(float)
    stock = _pd2.to_numeric(df[stock_col], errors="coerce").astype(float)

    ts_cons  = (_pd2.DataFrame({"d": dates, "q": cons})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["q"])
    ts_stock = (_pd2.DataFrame({"d": dates, "s": stock})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["s"])

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = _pd2.date_range(min_date, max_date, freq="D")

    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    return cons_daily, stock_daily

def _interval_sum_next_days_croston(daily: _pd2.Series, start_idx: int, interval: int) -> float:
    s = start_idx + 1
    e = s + int(max(0, interval))
    return float(_pd2.Series(daily).iloc[s:e].sum())

def _croston_forecast_array_croston(x, alpha: float):
    x = _pd2.Series(x).fillna(0.0).astype(float).values
    x = _np2.where(x < 0, 0.0, x)
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
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}

def rolling_croston_with_rops_single_run(
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
    sheet = _find_product_sheet_croston(excel_path, product_code)
    cons_daily, stock_daily = _daily_consumption_and_stock_croston(excel_path, sheet)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return _pd2.DataFrame()

    rng = _np2.random.default_rng(rng_seed)
    rows = []
    rop_carry_running = 0.0
    stock_after_interval = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]

            fc = _croston_forecast_array_croston(train, alpha=alpha)
            f = float(fc["forecast_per_period"])
            sigma_period = float(_pd2.Series(train).std(ddof=1)) if i > 1 else 0.0
            if not _np2.isfinite(sigma_period):
                sigma_period = 0.0

            real_demand = _interval_sum_next_days_croston(cons_daily, i, interval)              # Col C
            stock_on_hand_running = _interval_sum_next_days_croston(stock_daily, i, interval)   # Col B
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand

            next_real_demand = _interval_sum_next_days_croston(cons_daily, i + interval, interval) if (i + interval) < len(vals) else 0.0
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * _np2.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(_np2.percentile(_nbinom2.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * _np2.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(_np2.percentile(_nbinom2.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))

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

    return _pd2.DataFrame(rows)

def compute_metrics_croston(df_run: _pd2.DataFrame):
    if df_run.empty or "real_demand" not in df_run or "reorder_point_usine" not in df_run:
        return _np2.nan, _np2.nan, _np2.nan, _np2.nan
    est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, _np2.nan)
    e = df_run["real_demand"] - est
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = float(_np2.sqrt(MSE)) if _np2.isfinite(MSE) else _np2.nan
    return ME, absME, MSE, RMSE

def _grid_and_final_croston():
    best_rows = []
    for code in PRODUCT_CODES_UNI:
        best_row = None
        best_rmse = _np2.inf
        for a in ALPHAS_UNI:
            for w in WINDOW_RATIOS_UNI:
                for itv in RECALC_INTERVALS_UNI:
                    df_run = rolling_croston_with_rops_single_run(
                        excel_path=EXCEL_PATH_UNI, product_code=code,
                        alpha=a, window_ratio=w, interval=itv,
                        lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
                        service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
                    )
                    _, _, _, RMSE = compute_metrics_croston(df_run)
                    if _pd2.notna(RMSE):
                        if (RMSE < best_rmse * 0.99) or (_np2.isclose(RMSE, best_rmse, rtol=0.01) and best_row is not None and (
                            (itv, a, w) > (best_row["recalc_interval"], best_row["alpha"], best_row["window_ratio"])
                        )):
                            best_rmse = RMSE
                            best_row = {"code": code, "alpha": a, "window_ratio": w, "recalc_interval": itv, "RMSE": RMSE}
        if best_row:
            best_rows.append(best_row)

    df_best_croston = _pd2.DataFrame(best_rows)
    print("\n✅ Croston — Best parameters (by RMSE):")
    display(df_best_croston)

    print("\n— Croston Final Tables (best params) —")
    for _, r in df_best_croston.iterrows():
        code = r["code"]; a = r["alpha"]; w = r["window_ratio"]; itv = r["recalc_interval"]
        print(f"Final Croston {code}: alpha={a}, window={w}, interval={itv}")
        df_final = rolling_croston_with_rops_single_run(
            excel_path=EXCEL_PATH_UNI, product_code=code,
            alpha=float(a), window_ratio=float(w), interval=int(itv),
            lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
            service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
        )
        display(df_final[DISPLAY_COLUMNS_UNI])
    return df_best_croston

# =====================================================================
# ==========================  SES SECTION  =============================
# =====================================================================

def _find_product_sheet_ses(excel_path: str, code: str) -> str:
    xls = _pd2.ExcelFile(excel_path)
    sheets = xls.sheet_names
    target = f"time serie {code}"
    if target in sheets:
        return target
    patt = _re2.compile(r"time\s*ser(i|ie)s?\s*", _re2.IGNORECASE)
    cand = [s for s in sheets if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    for s in sheets:
        if s.strip().lower() == code.lower():
            return s
    raise ValueError(f"[SES] Onglet pour '{code}' introuvable (attendu: 'time serie {code}').")

def _daily_consumption_and_stock_ses(excel_path: str, sheet_name: str):
    df = _pd2.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError(f"[SES] Feuille '{sheet_name}': colonnes insuffisantes (A=date, B=stock, C=qté).")
    date_col, stock_col, cons_col = cols[0], cols[1], cols[2]

    dates = _pd2.to_datetime(df[date_col], errors="coerce")
    cons = _pd2.to_numeric(df[cons_col], errors="coerce").fillna(0.0).astype(float)
    stock = _pd2.to_numeric(df[stock_col], errors="coerce").astype(float)

    ts_cons  = (_pd2.DataFrame({"d": dates, "q": cons})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["q"])
    ts_stock = (_pd2.DataFrame({"d": dates, "s": stock})
                .dropna(subset=["d"]).sort_values("d").set_index("d")["s"])

    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = _pd2.date_range(min_date, max_date, freq="D")

    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    return cons_daily, stock_daily

def _interval_sum_next_days_ses(daily: _pd2.Series, start_idx: int, interval: int) -> float:
    s = start_idx + 1
    e = s + int(max(0, interval))
    return float(_pd2.Series(daily).iloc[s:e].sum())

def _ses_forecast_array_ses(x, alpha: float):
    x = _pd2.Series(x).fillna(0.0).astype(float).values
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
    sheet = _find_product_sheet_ses(excel_path, product_code)
    cons_daily, stock_daily = _daily_consumption_and_stock_ses(excel_path, sheet)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return _pd2.DataFrame()

    rng = _np2.random.default_rng(rng_seed)
    rows = []
    rop_carry_running = 0.0
    stock_after_interval = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]

            fc = _ses_forecast_array_ses(train, alpha=alpha)
            f = float(fc["forecast_per_period"])
            sigma_period = float(_pd2.Series(train).std(ddof=1)) if i > 1 else 0.0
            if not _np2.isfinite(sigma_period):
                sigma_period = 0.0

            real_demand = _interval_sum_next_days_ses(cons_daily, i, interval)              # Col C
            stock_on_hand_running = _interval_sum_next_days_ses(stock_daily, i, interval)   # Col B
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand

            next_real_demand = _interval_sum_next_days_ses(cons_daily, i + interval, interval) if (i + interval) < len(vals) else 0.0
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            X_Lt = lead_time * f
            sigma_Lt = sigma_period * _np2.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(_np2.percentile(_nbinom2.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))

            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * _np2.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(_np2.percentile(_nbinom2.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))

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

    return _pd2.DataFrame(rows)

def compute_metrics_ses(df_run: _pd2.DataFrame):
    if df_run.empty or "real_demand" not in df_run or "reorder_point_usine" not in df_run:
        return _np2.nan, _np2.nan, _np2.nan, _np2.nan
    est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, _np2.nan)
    e = df_run["real_demand"] - est
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = float(_np2.sqrt(MSE)) if _np2.isfinite(MSE) else _np2.nan
    return ME, absME, MSE, RMSE

def _grid_and_final_ses():
    best_rows = []
    for code in PRODUCT_CODES_UNI:
        best_row = None
        best_rmse = _np2.inf
        for a in ALPHAS_UNI:
            for w in WINDOW_RATIOS_UNI:
                for itv in RECALC_INTERVALS_UNI:
                    df_run = rolling_ses_with_rops_single_run(
                        excel_path=EXCEL_PATH_UNI, product_code=code,
                        alpha=a, window_ratio=w, interval=itv,
                        lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
                        service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
                    )
                    _, _, _, RMSE = compute_metrics_ses(df_run)
                    if _pd2.notna(RMSE):
                        if (RMSE < best_rmse * 0.99) or (_np2.isclose(RMSE, best_rmse, rtol=0.01) and best_row is not None and (
                            (itv, a, w) > (best_row["recalc_interval"], best_row["alpha"], best_row["window_ratio"])
                        )):
                            best_rmse = RMSE
                            best_row = {"code": code, "alpha": a, "window_ratio": w, "recalc_interval": itv, "RMSE": RMSE}
        if best_row:
            best_rows.append(best_row)

    df_best_ses = _pd2.DataFrame(best_rows)
    print("\n✅ SES — Best parameters (by RMSE):")
    display(df_best_ses)

    print("\n— SES Final Tables (best params) —")
    for _, r in df_best_ses.iterrows():
        code = r["code"]; a = r["alpha"]; w = r["window_ratio"]; itv = r["recalc_interval"]
        print(f"Final SES {code}: alpha={a}, window={w}, interval={itv}")
        df_final = rolling_ses_with_rops_single_run(
            excel_path=EXCEL_PATH_UNI, product_code=code,
            alpha=float(a), window_ratio=float(w), interval=int(itv),
            lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
            service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
        )
        display(df_final[DISPLAY_COLUMNS_UNI])
    return df_best_ses

# =====================================================================
# =======================  RUN ALL SECTIONS  ==========================
# =====================================================================

# minimal guard so this block doesn't fire automatically in Streamlit
RUN_FORECAST_STANDALONE = False

if __name__ == "__main__" and RUN_FORECAST_STANDALONE:
    # SBA
    df_best_sba = _grid_and_final_sba()
    # Croston
    df_best_croston = _grid_and_final_croston()
    # SES
    df_best_ses = _grid_and_final_ses()

    # (The comparison table can be built later using df_best_sba, df_best_croston, df_best_ses)



# ====================== BRIDGE TO STREAMLIT UI ======================
# Use the SAME uploaded Excel for the forecasting module

if uploaded is not None or uploaded_opt is not None:
    st.markdown("---")
    st.header("📈 Prévisions & ROP — SBA / Croston / SES (module concaténé)")

    # Choose the source (optimisation workbook if provided, else classification one)
    _fc_src = uploaded_opt or uploaded
    try:
        # Reassign EXCEL_PATH_UNI to a file-like BytesIO so the forecasting code uses it
        EXCEL_PATH_UNI = io.BytesIO(_get_excel_bytes(_fc_src))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("SBA — Best parameters / Final tables")
            df_best_sba = _grid_and_final_sba()
            if isinstance(df_best_sba, pd.DataFrame) and not df_best_sba.empty:
                st.dataframe(df_best_sba, use_container_width=True)

        with col2:
            st.subheader("Croston — Best parameters / Final tables")
            df_best_croston = _grid_and_final_croston()
            if isinstance(df_best_croston, pd.DataFrame) and not df_best_croston.empty:
                st.dataframe(df_best_croston, use_container_width=True)

        with col3:
            st.subheader("SES — Best parameters / Final tables")
            df_best_ses = _grid_and_final_ses()
            if isinstance(df_best_ses, pd.DataFrame) and not df_best_ses.empty:
                st.dataframe(df_best_ses, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors des prévisions/ROP: {e}")
else:
    st.info("Ajoutez un classeur pour activer la section Prévisions & ROP.")
