# app.py
import io
import re
from typing import Tuple
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
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

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

st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

st.title("Classification minimale — taille/fréquence → CV² & p → méthode")

# ===== classification sheet selector =====
sheet_name = None
if uploaded is not None:
    try:
        xls_classif = pd.ExcelFile(uploaded)
        noms = [s.lower() for s in xls_classif.sheet_names]
        default_idx = noms.index("classification") if "classification" in noms else 0
        sheet_name = st.selectbox(
            "Feuille (classeur de classification)",
            options=xls_classif.sheet_names,
            index=default_idx
        )
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

# ====================== Prévisions avancées — SBA / Croston / SES ======================
st.markdown("---")
st.header("📈 Prévisions avancées : SBA, Croston, SES")

_fc_src = uploaded_opt or uploaded
if _fc_src is None:
    st.info("Téléversez un classeur contenant des feuilles **time serie <CODE>** (A=Date, B=Stock, C=Consommation).")
else:
    xls_bytes = _get_excel_bytes(_fc_src)
    try:
        xls = pd.ExcelFile(io.BytesIO(xls_bytes))
    except Exception as e:
        st.error(f"Impossible de lire le classeur de prévisions : {e}")
        xls = None

    if xls is not None:

        # --------- robust sheet finder (matches 'time serie/série <CODE>', 'ts <CODE>', or just '<CODE>') ----------
        def _uni_find_product_sheet(xls: pd.ExcelFile, code: str) -> str:
            def norm(s: str) -> str:
                s = str(s).replace("\u00A0", " ")
                s = re.sub(r"\s+", " ", s).strip().lower()
                return s

            lc = norm(code)
            wanted_exact1 = f"time serie {lc}"
            wanted_exact2 = f"time série {lc}"
            wanted_ts     = f"ts {lc}"

            for s in xls.sheet_names:
                ns = norm(s)
                if ns in (wanted_exact1, wanted_exact2, wanted_ts, lc):
                    return s

            patt = re.compile(rf"\btime\s*s[ée]r(?:i|ie|ies)?\s*{re.escape(lc)}\b", re.IGNORECASE)
            for s in xls.sheet_names:
                if patt.search(s):
                    return s

            # last chance: any sheet that contains the code token
            for s in xls.sheet_names:
                if lc in re.sub(r"\s+", " ", str(s)).lower():
                    return s

            raise ValueError(
                f"Onglet pour '{code}' introuvable. Feuilles disponibles : {xls.sheet_names}"
            )

        # --------- read daily series (A=date, B=stock_on_hand/receipts, C=consommation) like your notebook ----------
        def _uni_daily_B_and_C(xls_bytes: bytes, sheet_name: str):
            df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=sheet_name)
            if df.shape[1] < 3:
                raise ValueError(f"Feuille '{sheet_name}' doit avoir au moins 3 colonnes (A,B,C).")
            date_col, stock_col, cons_col = df.columns[:3]

            dates = pd.to_datetime(df[date_col], errors="coerce")
            cons  = _parse_num_locale(df[cons_col])
            stock = _parse_num_locale(df[stock_col])

            ts_c = (pd.DataFrame({"d": dates, "q": cons})
                    .dropna(subset=["d"]).sort_values("d").set_index("d")["q"])
            ts_s = (pd.DataFrame({"d": dates, "s": stock})
                    .dropna(subset=["d"]).sort_values("d").set_index("d")["s"])

            min_d = min(ts_c.index.min(), ts_s.index.min())
            max_d = max(ts_c.index.max(), ts_s.index.max())
            full_idx = pd.date_range(min_d, max_d, freq="D")

            cons_daily  = ts_c.reindex(full_idx, fill_value=0.0)
            stock_daily = ts_s.reindex(full_idx).ffill().fillna(0.0)
            return cons_daily, stock_daily

        def _uni_interval_sum_next_days(daily: pd.Series, start_idx: int, interval: int) -> float:
            s = start_idx + 1
            e = s + int(max(0, interval))
            return float(pd.Series(daily).iloc[s:e].sum())

        # --------- forecasts (SBA/Croston/SES) exactly like your notebook ----------
        def _fc_sba_or_croston(x, alpha: float, variant: str):
            x = pd.Series(x).fillna(0.0).astype(float).values
            x = np.where(x < 0, 0.0, x)
            if (x == 0).all():
                return 0.0
            nz = [i for i, v in enumerate(x) if v > 0]
            first = nz[0]
            z = x[first]
            if len(nz) >= 2:
                p = sum([j - i for i, j in zip(nz[:-1], nz[1:])]) / len(nz)
            else:
                p = len(x) / len(nz)
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
            return float(f)

        def _fc_ses(x, alpha: float):
            x = pd.Series(x).fillna(0.0).astype(float).values
            if len(x) == 0:
                return 0.0
            l = x[0]
            for t in range(1, len(x)):
                l = alpha * x[t] + (1 - alpha) * l
            return float(l)

        def _rolling_method(xls_bytes: bytes, xls: pd.ExcelFile, code: str, alpha: float,
                            window_ratio: float, interval: int, forecast_func,
                            lead_time: int, lead_time_supplier: int, service_level: float,
                            nb_sim: int, rng_seed: int):
            sheet = _uni_find_product_sheet(xls, code)
            cons_daily, stock_daily = _uni_daily_B_and_C(xls_bytes, sheet)

            vals = cons_daily.values
            split_index = int(len(vals) * float(window_ratio))
            if split_index < 2:
                return pd.DataFrame()

            rng = np.random.default_rng(int(rng_seed))
            rows = []
            rop_carry_running = 0.0
            stock_after_interval = 0.0

            for i in range(split_index, len(vals)):
                if (i - split_index) % int(interval) != 0:
                    continue

                train = vals[:i]
                test_date = cons_daily.index[i]

                f = float(forecast_func(train, alpha))
                sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
                if not np.isfinite(sigma_period):
                    sigma_period = 0.0

                real_demand = _uni_interval_sum_next_days(cons_daily,  i, int(interval))
                stock_run   = _uni_interval_sum_next_days(stock_daily, i, int(interval))
                stock_after_interval = stock_after_interval + stock_run - real_demand

                next_real = _uni_interval_sum_next_days(cons_daily, i + int(interval), int(interval)) \
                            if (i + int(interval)) < len(vals) else 0.0
                can_cover_interval = "yes" if stock_after_interval >= next_real else "no"
                order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

                # ROP (NB quantile) – same shape as your notebook
                X_Lt = lead_time * f
                sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
                var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5

                totalL = lead_time + lead_time_supplier
                X_Lt_Lw = totalL * f
                sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
                var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5

                if _SCIPY_OK:
                    ROP_u = _nbinom_quantile_scipy(X_Lt, var_u, service_level, nb_sim, rng)
                    ROP_f = _nbinom_quantile_scipy(X_Lt_Lw, var_f, service_level, nb_sim, rng)
                else:
                    ROP_u = _nb_quantile_fallback(X_Lt, var_u, service_level, nb_sim, rng)
                    ROP_f = _nb_quantile_fallback(X_Lt_Lw, var_f, service_level, nb_sim, rng)

                rop_carry_running += float(ROP_u - real_demand)
                stock_status = "holding" if stock_after_interval > 0 else "rupture"

                rows.append({
                    "date": test_date.date(),
                    "code": code,
                    "interval": int(interval),
                    "real_demand": float(real_demand),
                    "stock_on_hand_running": float(stock_run),
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

        def _metrics_from_rop(df_run: pd.DataFrame):
            if df_run.empty:
                return np.nan, np.nan, np.nan, np.nan
            est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, np.nan)
            e = df_run["real_demand"] - est
            ME = e.mean()
            absME = e.abs().mean()
            MSE = (e**2).mean()
            RMSE = float(np.sqrt(MSE)) if np.isfinite(MSE) else np.nan
            return ME, absME, MSE, RMSE

        # --------- UI controls ----------
        discovered_codes = _fc_list_time_serie_codes(xls)
        if not discovered_codes:
            st.warning("Aucune feuille 'time serie *' détectée. Sélection manuelle permise.")
            discovered_codes = xls.sheet_names

        c_left, c_right = st.columns([2, 1])
        with c_left:
            selected_codes = st.multiselect("Codes produit / Feuilles",
                                            options=discovered_codes,
                                            default=discovered_codes)
            methods = st.multiselect("Méthodes", options=["sba", "croston", "ses"],
                                     default=["sba", "croston", "ses"])
            alphas = st.multiselect("Alphas", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                                    default=[0.1, 0.2, 0.3, 0.4])
            window_ratios = st.multiselect("Window ratios", [0.6, 0.7, 0.8],
                                           default=[0.6, 0.7, 0.8])
            recalc_intervals = st.multiselect("Intervalles (jours)", [1, 2, 5, 10, 15, 20],
                                              default=[1, 2, 5, 10, 15, 20])
        with c_right:
            lead_time = st.number_input("Lead time usine (jours)", min_value=0, value=1, step=1)
            lead_time_supplier = st.number_input("Lead time fournisseur + (jours)", min_value=0, value=3, step=1)
            service_level = st.slider("Niveau de service", 0.50, 0.999, 0.95, 0.001)
            nb_sim = st.number_input("Taille simulation NB", min_value=200, step=100, value=800)
            rng_seed = st.number_input("RNG seed", min_value=0, value=42, step=1)

        if not _SCIPY_OK:
            st.warning("SciPy indisponible — ROP via **fallback Gamma–Poisson**.")

        # --------- grid search + display ----------
        DISPLAY_COLUMNS_UNI = [
            "date", "code", "interval", "real_demand", "stock_on_hand_running",
            "stock_after_interval", "can_cover_interval", "order_policy",
            "reorder_point_usine", "lead_time_usine_days", "lead_time_supplier_days",
            "reorder_point_fournisseur", "stock_status", "rop_usine_minus_real_running",
        ]

        def _do_grid(method_name: str, codes: list[str]):
            all_rows, best_rows = [], []
            # map method to its forecaster
            if method_name == "sba":
                ffun = lambda x, a: _fc_sba_or_croston(x, a, "sba")
            elif method_name == "croston":
                ffun = lambda x, a: _fc_sba_or_croston(x, a, "croston")
            else:
                ffun = _fc_ses

            for code in codes:
                best_rmse, best = np.inf, None
                for a in alphas:
                    for w in window_ratios:
                        for itv in recalc_intervals:
                            df_run = _rolling_method(
                                xls_bytes, xls, code, a, w, itv, ffun,
                                int(lead_time), int(lead_time_supplier),
                                float(service_level), int(nb_sim), int(rng_seed)
                            )
                            ME, absME, MSE, RMSE = _metrics_from_rop(df_run)
                            all_rows.append({
                                "code": code, "method": method_name, "alpha": a,
                                "window_ratio": w, "recalc_interval": itv,
                                "ME": ME, "absME": absME, "MSE": MSE, "RMSE": RMSE,
                                "n_points": len(df_run)
                            })
                            if pd.notna(RMSE) and (
                                (RMSE < best_rmse * 0.99) or
                                (np.isclose(RMSE, best_rmse, rtol=0.01) and best is not None and
                                 (itv, a, w) > (best["recalc_interval"], best["alpha"], best["window_ratio"]))
                            ):
                                best_rmse = RMSE
                                best = {"code": code, "alpha": a, "window_ratio": w,
                                        "recalc_interval": itv, "RMSE": RMSE}
                if best:
                    best_rows.append(best)

            return pd.DataFrame(all_rows), pd.DataFrame(best_rows)

        run = st.button("▶️ Lancer les prévisions (grid search + résumé)")
        if run:
            if not selected_codes:
                st.warning("Sélectionnez au moins un code/feuille.")
            elif not methods:
                st.warning("Sélectionnez au moins une méthode.")
            else:
                tabs = st.tabs([m.upper() for m in methods])
                for m, tab in zip(methods, tabs):
                    with tab:
                        with st.spinner(f"Exécution {m.upper()}…"):
                            df_all, df_best = _do_grid(m, selected_codes)

                        st.subheader(f"{m.upper()} — meilleurs paramètres (par RMSE)")
                        st.dataframe(df_best, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Télécharger (best) CSV",
                            data=df_best.to_csv(index=False).encode("utf-8"),
                            file_name=f"best_{m}.csv", mime="text/csv"
                        )

                        st.subheader("Toutes les combinaisons testées")
                        st.dataframe(df_all, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Télécharger (grid) CSV",
                            data=df_all.to_csv(index=False).encode("utf-8"),
                            file_name=f"grid_{m}.csv", mime="text/csv"
                        )

                        st.markdown("### Table détaillée (meilleurs paramètres) — colonnes principales")
                        if not df_best.empty:
                            code_for_table = st.selectbox(
                                "Produit pour la table détaillée",
                                options=sorted(df_best["code"].astype(str).unique()),
                                key=f"{m}_table_code"
                            )
                            if code_for_table:
                                row = df_best[df_best["code"].astype(str) == str(code_for_table)].iloc[0]
                                df_final = _rolling_method(
                                    xls_bytes, xls, code_for_table,
                                    float(row["alpha"]), float(row["window_ratio"]), int(row["recalc_interval"]),
                                    (lambda x, a: _fc_sba_or_croston(x, a, "sba")) if m == "sba"
                                    else ((lambda x, a: _fc_sba_or_croston(x, a, "croston")) if m == "croston" else _fc_ses),
                                    int(lead_time), int(lead_time_supplier),
                                    float(service_level), int(nb_sim), int(rng_seed)
                                )
                                view = df_final[DISPLAY_COLUMNS_UNI] if not df_final.empty else pd.DataFrame()
                                if view.empty:
                                    st.warning("Aucune donnée dans la fenêtre d’évaluation pour ce produit.")
                                st.dataframe(view, use_container_width=True)
                                st.download_button(
                                    "Télécharger la table (CSV)",
                                    data=view.to_csv(index=False).encode("utf-8"),
                                    file_name=f"details_{m}_{code_for_table}.csv", mime="text/csv"
                                )
