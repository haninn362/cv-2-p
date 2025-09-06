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

# ---------------- helpers & core logic ----------------
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

# -------- NEW: optimisation feature (n*, Qr*, Qw*) --------
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
    if file_like is None:
        return b""
    if hasattr(file_like, "getvalue"):
        try:
            return file_like.getvalue()
        except Exception:
            pass
    try:
        data = file_like.read()
        return data
    finally:
        try:
            file_like.seek(0)
        except Exception:
            pass

def compute_qr_qw_from_workbook(file_like, conso_sheet_hint: str = "consommation depots externe",
                                time_series_prefix: str = "time seri"):
    """Reads the uploaded Excel and computes n*, Qr*, Qw* for each 'time serie*' sheet."""
    info_msgs, warn_msgs = [], []
    if file_like is None:
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    data_bytes = _get_excel_bytes(file_like)
    if not data_bytes:
        warn_msgs.append("Optimisation workbook is empty or unreadable.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    xls = pd.ExcelFile(io.BytesIO(data_bytes))

    # pick the consumption sheet
    sheet_names_norm = {_norm(s): s for s in xls.sheet_names}
    conso_sheet = sheet_names_norm.get(_norm(conso_sheet_hint))
    if not conso_sheet:
        candidates = [s for s in xls.sheet_names if _norm(conso_sheet_hint) in _norm(s)]
        if candidates:
            conso_sheet = candidates[0]
    if not conso_sheet:
        warn_msgs.append("Sheet 'consommation depots externe' not found.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    df_conso = pd.read_excel(io.BytesIO(data_bytes), sheet_name=conso_sheet)

    # prefer 'Quantite STIAL' explicitly
    code_col = next((c for c in df_conso.columns if "code produit" in _norm(c)), None) or "Code Produit"
    qty_col = None
    # exact match
    for c in df_conso.columns:
        nc = _norm(c)
        if nc == "quantite stial" or nc == "quantité stial":
            qty_col = c
            break
    # substring match
    if qty_col is None:
        for c in df_conso.columns:
            nc = _norm(c)
            if "quantite stial" in nc or "quantité stial" in nc:
                qty_col = c
                break
    # generic fallback
    if qty_col is None:
        for key in ["quantite", "quantité", "qte"]:
            cand = next((c for c in df_conso.columns if key in _norm(c)), None)
            if cand:
                qty_col = cand
                break

    if code_col is None or qty_col is None:
        warn_msgs.append("Could not locate 'Code Produit' and/or 'Quantite STIAL' columns.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    conso_series = df_conso.groupby(code_col, dropna=False)[qty_col].sum(numeric_only=True)
    info_msgs.append(f"Consumption sheet: '{conso_sheet}' (rows: {len(df_conso)})")
    info_msgs.append(f"Quantity column used: '{qty_col}'")

    # detect 'time serie*' or 'time series*' sheets
    ts_sheets = [s for s in xls.sheet_names if _norm(s).startswith(_norm(time_series_prefix))]
    if not ts_sheets:
        warn_msgs.append("No 'time serie*' sheets found (e.g., 'time serie EM0400').")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    results = []
    for sheet in ts_sheets:
        try:
            df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet)
            product_code = sheet.split()[-1]  # last token of sheet name

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
                warn_msgs.append(f"[{sheet}] Invalid parameter values; skipped.")
                continue

            n = (A_w * C_r) / (A_r * C_w)
            n = 1 if n < 1 else round(n)
            n1, n2 = int(n), int(n) + 1
            F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
            F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
            n_star = n1 if F_n1 <= F_n2 else n2

            D = conso_series.get(product_code, 0)
            tau = 1
            denom = (n_star * C_w + C_r * tau)
            if denom <= 0:
                warn_msgs.append(f"[{sheet}] Non-positive denominator for Q*; skipped.")
                continue

            if D is None or D <= 0:
                warn_msgs.append(f"[{sheet}] Non-positive demand D={D} → set Q*=0.")
                Q_r_star = 0.0
            else:
                Q_r_star = ((2 * (A_r + A_w / n_star) * D) / denom) ** 0.5

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

# --- Sidebar controls: Reset & state ---
if "uploader_nonce" not in st.session_state:
    st.session_state["uploader_nonce"] = 0

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset (clear data)"):
        # Clear widget state by bumping the nonce; also drop our own selections
        st.session_state["uploader_nonce"] += 1
        for k in ["selected_product"]:
            st.session_state.pop(k, None)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

nonce = st.session_state.get("uploader_nonce", 0)

# Two uploaders: 1) classification workbook, 2) optimisation workbook (optional / different file)
uploaded = st.file_uploader(
    "Upload classification workbook (.xlsx/.xls). Col 0 = Product, cols 1..N = date headers with quantities.",
    type=["xlsx", "xls"],
    key=f"clf_{nonce}"
)
uploaded_opt = st.file_uploader(
    "Upload optimisation workbook (.xlsx/.xls) — optional if different from the first file",
    type=["xlsx", "xls"],
    key=f"opt_{nonce}"
)

sheet_name = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        # prefer a sheet named 'classification' if present
        names_lower = [s.lower() for s in xls.sheet_names]
        default_idx = names_lower.index("classification") if "classification" in names_lower else 0
        sheet_name = st.selectbox("Sheet (classification workbook)", options=xls.sheet_names, index=default_idx)
    except Exception as e:
        st.error(f"Could not read workbook: {e}")

if uploaded is not None and sheet_name is not None:
    try:
        df_raw = pd.read_excel(uploaded, sheet_name=sheet_name)

        # ---- Product selector (show only the selected product) ----
        prod_col = df_raw.columns[0]
        product_options = sorted(df_raw[prod_col].astype(str).dropna().unique().tolist())
        if not product_options:
            st.warning("No products found in the first column of the selected sheet.")
        selected_product = st.selectbox(
            "Select product",
            options=product_options,
            key="selected_product"
        )

        # compute and then filter all tables to the selected product
        combined_df, stats_df, counts_df, methods_df = compute_everything(df_raw)

        # Filtered Stats & Counts
        stats_one = stats_df.loc[[selected_product]] if selected_product in stats_df.index else stats_df.iloc[0:0]
        counts_one = counts_df.loc[[selected_product]] if selected_product in counts_df.index else counts_df.iloc[0:0]
        methods_one = methods_df.loc[[selected_product]] if selected_product in methods_df.index else methods_df.iloc[0:0]

        # Table 1 & 2 (filtered)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Table 1 — moyenne / ecart-type / CV² (selected)**")
            st.dataframe(stats_one.reset_index(), use_container_width=True)
        with c2:
            st.markdown("**Table 2 — N périodes / N fréquence / p (selected)**")
            st.dataframe(counts_one.reset_index(), use_container_width=True)

        # Combined (two rows: taille & frequence for the selected product)
        st.markdown("**Combined — taille / frequence (selected)**")
        comb_sel = pd.DataFrame()
        if not combined_df.empty:
            mask_taille = (combined_df["Product"] == selected_product) & (combined_df["Type"] == "taille")
            if mask_taille.any():
                idx = combined_df.index[mask_taille][0]
                rows = [idx]
                if idx + 1 in combined_df.index:
                    rows.append(idx + 1)  # frequence row right after
                comb_sel = combined_df.loc[rows]
        if comb_sel.empty:
            st.info("No combined rows found for the selected product.")
        else:
            st.dataframe(comb_sel, use_container_width=True)

        # Plot (selected point only)
        st.markdown("**Graph — p vs CV² with cut-offs (selected)**")
        if not methods_one.empty:
            fig = make_plot(methods_one)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No plot for the selected product.")

        # Methods (selected)
        st.markdown("**Method per product (selected)**")
        st.dataframe(methods_one.reset_index(), use_container_width=True)

        # --- Optimisation (n*, Qr*, Qw*) ---
        st.markdown("**Optimisation — n\\*, Qr\\*, Qw\\* (selected)**")
        # Use second file if provided, else fall back to the classification workbook
        opt_source = uploaded_opt or uploaded
        if uploaded_opt is not None:
            st.caption("Using the separate optimisation workbook you uploaded.")
        else:
            st.caption("No separate optimisation workbook uploaded — using the classification workbook.")

        opt_df, info_msgs, warn_msgs = compute_qr_qw_from_workbook(opt_source)
        for msg in info_msgs:
            st.info(msg)
        for msg in warn_msgs:
            st.warning(msg)

        # Try to map the selected label to a product code in optimisation results
        # If your product names contain codes like EM0400, we extract and match by that
        code_match = re.search(r"\b[A-Z]{2}\d{4}\b", str(selected_product))
        opt_key = code_match.group(0) if code_match else str(selected_product)
        opt_one = opt_df[opt_df["Code Produit"].astype(str) == opt_key]

        if opt_one.empty:
            st.info(f"No optimisation row found for **{selected_product}** (looked for code '{opt_key}').")
        else:
            st.dataframe(opt_one, use_container_width=True)

        # Downloads (full results still available if you want them)
        xbuf = excel_bytes(combined_df, stats_df, counts_df, methods_df)
        st.download_button("Download ALL results (Excel)", data=xbuf,
                           file_name="results_minimal.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if not methods_one.empty:
            pbuf = io.BytesIO()
            fig.savefig(pbuf, format="png", bbox_inches="tight")
            pbuf.seek(0)
            st.download_button("Download graph for selected (PNG)", data=pbuf,
                               file_name=f"classification_{opt_key or 'selected'}.png",
                               mime="image/png")

        if not opt_df.empty:
            st.download_button(
                "Download ALL optimisation results (CSV)",
                data=opt_df.to_csv(index=False).encode("utf-8"),
                file_name="optimisation_qr_qw.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info("Upload the classification workbook to start. (You can also upload a separate optimisation workbook.)")
