import io
import math
import re
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- Syntetos & Boylan cut-offs ---
ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF       # ≈ 0.757576
CV2_CUTOFF = 0.49

st.set_page_config(page_title="Demand classification — p & CV²", layout="wide")

# ---------------- core logic ----------------
def choose_method(p: float, cv2: float) -> Tuple[str, str]:
    """Return (Category, Suggested)."""
    if pd.isna(p) or pd.isna(cv2):
        return "Insufficient data", ""
    if p <= 0:
        return "No demand", ""
    if p >= P_CUTOFF and cv2 <= CV2_CUTOFF:
        return "Smooth", "SES"
    if p >= P_CUTOFF and cv2 > CV2_CUTOFF:
        return "Erratic", "SES"
    if p < P_CUTOFF and cv2 <= CV2_CUTOFF:
        return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"

def compute_everything(df: pd.DataFrame):
    """
    df: first column = product, remaining columns = dates with numeric quantities.
    Returns: combined_df, stats_df, counts_df, methods_df
    """
    # dates/periods
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    # collect non-zero values & inter-arrivals
    combined_rows = []
    per_product_vals = {}
    max_len = 0

    for _, row in df.iterrows():
        product = str(row.iloc[0])
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
        combined_rows.append((product, vals, inter_arrivals))
        per_product_vals[product] = vals

    # Combined (taille/frequence)
    final_rows = []
    for product, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([product, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Product", "Type"] + list(range(max_len)))

    # Table 1 — moyenne, ecart-type, CV^2 (non-zero only)
    stats_rows = []
    for product, vals in per_product_vals.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            mean = s.mean()
            std = s.std(ddof=1)                # sample std (Excel STDEV.S)
            cv2 = (std / mean) ** 2 if mean != 0 else np.nan
        else:
            mean = std = cv2 = np.nan
        stats_rows.append([product, mean, std, cv2])

    stats_df = (
        pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "ecart-type", "CV^2"])
        .set_index("Produit")
        .sort_index()
    )

    # Table 2 — N périodes, N fréquence, p
    counts_rows = []
    for product, vals in per_product_vals.items():
        n_freq = len(vals)
        p = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([product, n_periods, n_freq, p])

    counts_df = (
        pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquence", "p"])
        .set_index("Produit")
        .sort_index()
    )

    # Methods (classification)
    methods_df = stats_df.join(counts_df, how="outer")
    cats = methods_df.apply(lambda r: choose_method(r["p"], r["CV^2"]),
                            axis=1, result_type="expand")
    methods_df["Category"] = cats[0]
    methods_df["Suggested"] = cats[1]
    methods_df = methods_df[["CV^2", "p", "Category", "Suggested"]]

    return combined_df, stats_df, counts_df, methods_df

def make_plot(methods_df: pd.DataFrame):
    """Matplotlib figure: p (x) vs CV^2 (y), with cutoffs and coordinates in labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = methods_df["p"].clip(lower=0, upper=1)
    y = methods_df["CV^2"]

    ax.scatter(x, y)
    for label, xi, yi in zip(methods_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(f"{label} (p={xi:.3f}, CV²={yi:.3f})",
                        (xi, yi), textcoords="offset points", xytext=(5, 5))

    ax.axvline(P_CUTOFF, linestyle="--")
    ax.axhline(CV2_CUTOFF, linestyle="--")
    ax.set_xlabel("p (share of non-zero periods)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("CV^2")
    ax.set_title("Demand classification (p vs CV^2) — Syntetos & Boylan")
    fig.tight_layout()
    return fig

def excel_bytes(combined_df, stats_df, counts_df, methods_df) -> io.BytesIO:
    """Create Excel with: Table 1, Table 2, Combined, Methods."""
    buf = io.BytesIO()
    for engine in ("openpyxl", "xlsxwriter", None):
        try:
            writer = pd.ExcelWriter(buf, engine=engine) if engine else pd.ExcelWriter(buf)
            with writer:
                sheet = "Results"
                stats_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
                r2 = len(stats_df) + 3
                counts_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
                r3 = r2 + len(counts_df) + 3
                combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r3, startcol=0)
                methods_df.reset_index().to_excel(writer, index=False, sheet_name="Methods")
            break
        except ModuleNotFoundError:
            buf = io.BytesIO()
            continue
    buf.seek(0)
    return buf

# =========================
# NEW FEATURE (ADD-ONLY): n*, Qr*, Qw* optimization based on your extra code
# =========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _find_first_col(df: pd.DataFrame, starts_with: str = None, contains: str = None):
    for c in df.columns:
        cn = _norm(c)
        if starts_with and cn.startswith(starts_with):
            return c
        if contains and contains in cn:
            return c
    return None

def _get_excel_bytes(file_like) -> bytes:
    """Safely obtain the raw bytes from an UploadedFile or file-like so we can read multiple times."""
    if hasattr(file_like, "getvalue"):
        return file_like.getvalue()
    try:
        return file_like.read()
    finally:
        try:
            file_like.seek(0)
        except Exception:
            pass

def compute_qr_qw_from_workbook(file_like, conso_sheet_hint: str = "consommation depots externe",
                                time_series_prefix: str = "time seri"):
    """
    Reads the uploaded Excel and computes n*, Qr*, Qw* for each 'time serie*' sheet.
    Returns (result_df, info_msgs, warn_msgs)
    """
    info_msgs, warn_msgs = [], []
    if file_like is None:
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    data_bytes = _get_excel_bytes(file_like)

    # --- pick the consumption sheet (default: 'consommation depots externe')
    xls = pd.ExcelFile(io.BytesIO(data_bytes))
    sheet_names_norm = {_norm(s): s for s in xls.sheet_names}
    conso_sheet = sheet_names_norm.get(_norm(conso_sheet_hint))
    if not conso_sheet:
        candidates = [s for s in xls.sheet_names if _norm(conso_sheet_hint) in _norm(s)]
        if candidates:
            conso_sheet = candidates[0]
    if not conso_sheet:
        warn_msgs.append("Sheet 'consommation depots externe' not found; select the correct sheet name or rename it.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    df_conso = pd.read_excel(io.BytesIO(data_bytes), sheet_name=conso_sheet)

    # find code & quantity columns (tolerant to accents/newlines/extra text)
    code_col = None
    for key in ["code produit", "code", "produit"]:
        code_col = next((c for c in df_conso.columns if key in _norm(c)), None) or code_col
    qty_col = None
    for key in ["quantite stial", "quantité stial", "quantite", "quantité", "qte"]:
        qty_col = next((c for c in df_conso.columns if key in _norm(c)), None) or qty_col

    if code_col is None or qty_col is None:
        warn_msgs.append("Could not locate 'Code Produit' and/or 'Quantite STIAL' columns in the consumption sheet.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    conso_series = df_conso.groupby(code_col, dropna=False)[qty_col].sum(numeric_only=True)
    info_msgs.append(f"Consumption sheet: '{conso_sheet}' (rows: {len(df_conso)})")

    # --- detect 'time serie*' or 'time series*' sheets
    ts_sheets = [s for s in xls.sheet_names if _norm(s).startswith(_norm(time_series_prefix))]
    if not ts_sheets:
        warn_msgs.append("No 'time serie*' sheets found (e.g., 'time serie EM0400').")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    results = []
    for sheet in ts_sheets:
        try:
            df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet)
            product_code = sheet.split()[-1]  # e.g., last token of 'time serie EM0400'

            # locate parameter columns by their leading tokens (robust to newlines/trailing spaces)
            cr_col = _find_first_col(df, starts_with="cr")
            cw_col = _find_first_col(df, starts_with="cw")
            aw_col = _find_first_col(df, starts_with="aw")
            ar_col = _find_first_col(df, starts_with="ar")

            if not all([cr_col, cw_col, aw_col, ar_col]):
                warn_msgs.append(f"[{sheet}] Missing one of CR/CW/AW/AR columns; skipped.")
                continue

            C_r = pd.to_numeric(df[cr_col].iloc[0], errors="coerce")
            C_w = pd.to_numeric(df[cw_col].iloc[0], errors="coerce")
            A_w = pd.to_numeric(df[aw_col].iloc[0], errors="coerce")
            A_r = pd.to_numeric(df[ar_col].iloc[0], errors="coerce")

            if any(pd.isna(v) for v in [C_r, C_w, A_w, A_r]) or any(v == 0 for v in [C_w, A_r]):
                warn_msgs.append(f"[{sheet}] Invalid parameter values (NaN or zero where not allowed); skipped.")
                continue

            # compute n (rounded to nearest integer ≥1)
            n = (A_w * C_r) / (A_r * C_w)
            n = 1 if n < 1 else round(n)
            n1, n2 = int(n), int(n) + 1

            F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
            F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
            n_star = n1 if F_n1 <= F_n2 else n2

            # demand D from consumption totals
            D = conso_series.get(product_code, 0)
            tau = 1  # as per your script

            denom = (n_star * C_w + C_r * tau)
            if denom <= 0:
                warn_msgs.append(f"[{sheet}] Non-positive denominator for Q*; skipped.")
                continue

            Q_r_star = ((2 * (A_r + A_w / n_star) * D) / denom) ** 0.5 if D > 0 else 0.0
            Q_w_star = n_star * Q_r_star

            results.append({
                "Code Produit": str(product_code),
                "n*": int(n_star),
                "Qr*": round(float(Q_r_star), 2),
                "Qw*": round(float(Q_w_star), 2),
            })
        except Exception as e:
            warn_msgs.append(f"[{sheet}] Failed: {e}")

    result_df = pd.DataFrame(results).sort_values("Code Produit") if results else pd.DataFrame(
        columns=["Code Produit", "n*", "Qr*", "Qw*"]
    )
    return result_df, info_msgs, warn_msgs

# ---------------- UI ----------------
st.title("Minimal demand classification — taille/frequence → CV² & p → method")

uploaded = st.file_uploader("Upload Excel (.xlsx/.xls). Col 0 = Product, cols 1..N = date headers with quantities.",
                            type=["xlsx", "xls"])

sheet_name = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        # prefer a sheet named 'classification' if present
        names_lower = [s.lower() for s in xls.sheet_names]
        default_idx = names_lower.index("classification") if "classification" in names_lower else 0
        sheet_name = st.selectbox("Sheet", options=xls.sheet_names, index=default_idx)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")

if uploaded is not None and sheet_name is not None:
    try:
        df_raw = pd.read_excel(uploaded, sheet_name=sheet_name)
        combined_df, stats_df, counts_df, methods_df = compute_everything(df_raw)

        # Table 1 & 2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Table 1 — moyenne / ecart-type / CV²**")
            st.dataframe(stats_df.reset_index(), use_container_width=True)
        with c2:
            st.markdown("**Table 2 — N périodes / N fréquence / p**")
            st.dataframe(counts_df.reset_index(), use_container_width=True)

        # Combined
        st.markdown("**Combined — taille / frequence**")
        st.dataframe(combined_df, use_container_width=True)

        # Plot
        st.markdown("**Graph — p vs CV² with cut-offs**")
        fig = make_plot(methods_df)
        st.pyplot(fig, use_container_width=True)

        # Methods
        st.markdown("**Method per product**")
        st.dataframe(methods_df.reset_index(), use_container_width=True)

        # --- New section: Optimization results (n*, Qr*, Qw*) ---
        st.markdown("**Optimisation — n\\*, Qr\\*, Qw\\* par produit**")
        opt_df, info_msgs, warn_msgs = compute_qr_qw_from_workbook(uploaded)

        for msg in info_msgs:
            st.info(msg)
        for msg in warn_msgs:
            st.warning(msg)

        st.dataframe(opt_df, use_container_width=True)

        # Downloads
        xbuf = excel_bytes(combined_df, stats_df, counts_df, methods_df)
        st.download_button("Download results (Excel)", data=xbuf,
                           file_name="results_minimal.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        pbuf = io.BytesIO()
        fig.savefig(pbuf, format="png", bbox_inches="tight")
        pbuf.seek(0)
        st.download_button("Download graph (PNG)", data=pbuf, file_name="classification_grid_p.png", mime="image/png")

        # Download CSV for the new results
        st.download_button(
            "Download optimisation results (CSV)",
            data=opt_df.to_csv(index=False).encode("utf-8"),
            file_name="optimisation_qr_qw.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info("Upload a file to start.")
