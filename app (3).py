import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# ==========================================
# 0) CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="Alliance Finance - Yield Dashboard", layout="wide")

START_YEAR  = 2026
START_MONTH = "July"
NUM_MONTHS  = 12
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

start_idx = MONTHS.index(START_MONTH)
WINDOW = [(START_YEAR + (start_idx + i) // 12, MONTHS[(start_idx + i) % 12])
          for i in range(NUM_MONTHS)]
PERIOD_LABELS = [f"{y} {m}" for y, m in WINDOW]

def find_header_row(raw_bytes, marker):
    text = raw_bytes.decode("utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines()):
        if line.replace('"', "").startswith(marker):
            return i
    return 0

# ==========================================
# 1) CACHED DATA PROCESSING (Upload Once)
# ==========================================
@st.cache_data
def process_data(stock_bytes, arr_bytes, arr_name):
    # --- Process Stock Depletion ---
    hdr = find_header_row(stock_bytes, "Year,Foracid")
    stock = pd.read_csv(io.BytesIO(stock_bytes), skiprows=hdr, low_memory=False)
    stock = stock[pd.to_numeric(stock["Year"], errors="coerce").notna()].copy()
    stock["Year"] = stock["Year"].astype(int)
    stock["Foracid"] = stock["Foracid"].astype("Int64")

    # --- Process Arrears ---
    NEEDED_ARR_COLS = ["facility_no", "sol_id", "branch", "int_rate", "product"]
    is_excel = str(arr_name).lower().endswith((".xlsx", ".xls"))

    if is_excel:
        preview = pd.read_excel(io.BytesIO(arr_bytes), header=None, nrows=15)
        header_row = next((i for i in range(len(preview)) 
                           if "facility_no" in preview.iloc[i].astype(str).tolist()), 0)
        arr = pd.read_excel(io.BytesIO(arr_bytes), header=header_row, usecols=NEEDED_ARR_COLS)
    else:
        hdr2 = find_header_row(arr_bytes, "facility_date")
        arr = pd.read_csv(io.BytesIO(arr_bytes), skiprows=hdr2, usecols=NEEDED_ARR_COLS, low_memory=False)

    arr = arr.drop_duplicates(subset="facility_no", keep="first")
    arr = arr.rename(columns={"facility_no": "Foracid"})
    arr["Foracid"] = arr["Foracid"].astype("Int64")

    # --- Calculate Yield ---
    all_years = sorted(stock["Year"].unique())
    full_periods = [(y, m) for y in all_years for m in MONTHS 
                    if not (y == START_YEAR and MONTHS.index(m) < start_idx)]
    period_seq = {p: i for i, p in enumerate(full_periods)}

    full_rows = []
    for y, m in full_periods:
        sub = stock[stock["Year"] == y][["Foracid", f"{m} Cap", f"{m} Int"]].copy()
        sub = sub.rename(columns={f"{m} Cap": "Cap", f"{m} Int": "Int"})
        sub["Cap"] = pd.to_numeric(sub["Cap"], errors="coerce").fillna(0)
        sub["Int"] = pd.to_numeric(sub["Int"], errors="coerce").fillna(0)
        sub["seq"] = period_seq[(y, m)]
        sub["Period"] = f"{y} {m}"
        full_rows.append(sub)

    full_long = pd.concat(full_rows, ignore_index=True).sort_values(["Foracid", "seq"])
    full_long["Opening_Bal"] = full_long.groupby("Foracid")["Cap"].transform(lambda s: s[::-1].cumsum()[::-1])

    long_df = full_long[full_long["Period"].isin(PERIOD_LABELS)].copy()
    long_df = long_df.merge(arr[["Foracid", "branch", "product", "int_rate"]], on="Foracid", how="left")

    yield_tbl = (long_df.groupby(["Period", "branch", "product"], dropna=False)
                 .agg(Total_Int=("Int", "sum"),
                      Total_Opening_Bal=("Opening_Bal", "sum"),
                      Avg_Contract_Rate=("int_rate", "mean"))
                 .reset_index())
    
    yield_tbl["Yield_pct"] = np.where(yield_tbl["Total_Opening_Bal"] > 0,
                                      (yield_tbl["Total_Int"] / yield_tbl["Total_Opening_Bal"] * 12 * 100).round(2),
                                      np.nan)
    yield_tbl["Avg_Contract_Rate"] = yield_tbl["Avg_Contract_Rate"].round(2)
    yield_tbl["Period"] = pd.Categorical(yield_tbl["Period"], categories=PERIOD_LABELS, ordered=True)
    return yield_tbl.sort_values(["Period", "branch", "product"]).reset_index(drop=True)


# ==========================================
# 2) UI: FILE UPLOAD
# ==========================================
st.title("🏦 Branch & Product Yield Dashboard")

with st.sidebar:
    st.header("1. Upload Reports")
    stock_file = st.file_uploader("Stock Depletion Report (CSV)", type=["csv"])
    arr_file = st.file_uploader("Arrears Report (CSV/Excel)", type=["csv", "xlsx", "xls"])

if stock_file and arr_file:
    # Process data only once thanks to @st.cache_data
    with st.spinner("Processing portfolios..."):
        yield_tbl = process_data(stock_file.getvalue(), arr_file.getvalue(), arr_file.name)
    
    st.success("Data loaded successfully!")

    # ==========================================
    # 3) UI: DASHBOARD & ANALYSIS
    # ==========================================
    st.header("2. Yield Trend Analysis")
    
    col1, col2 = st.columns(2)
    branches = ["All"] + sorted(yield_tbl["branch"].dropna().unique().tolist())
    products = ["All"] + sorted(yield_tbl["product"].dropna().unique().tolist())
    
    with col1:
        sel_branch = st.selectbox("Select Branch", branches)
    with col2:
        sel_product = st.selectbox("Select Product", products)

    # Filter logic
    df = yield_tbl.copy()
    if sel_branch != "All":
        df = df[df["branch"] == sel_branch]
    if sel_product != "All":
        df = df[df["product"] == sel_product]

    # Aggregate filtered data
    trend = df.groupby("Period", observed=True).apply(
        lambda g: pd.Series({
            "Total_Opening_Bal": g["Total_Opening_Bal"].sum(),
            "Total_Int": g["Total_Int"].sum(),
            "Yield_pct": round(g["Total_Int"].sum() / g["Total_Opening_Bal"].sum() * 12 * 100, 2) if g["Total_Opening_Bal"].sum() > 0 else np.nan,
            "Avg_Contract_Rate": round(g["Avg_Contract_Rate"].mean(), 2)
        })
    ).reset_index()

    st.dataframe(trend, use_container_width=True)

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend["Period"], trend["Yield_pct"], marker="o", label="Real Yield %")
    ax.plot(trend["Period"], trend["Avg_Contract_Rate"], marker="s", linestyle="--", label="Avg Contract Rate %")
    ax.set_title(f"Yield Trend - Branch: {sel_branch} | Product: {sel_product}")
    ax.set_ylabel("%")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # ==========================================
    # 4) UI: WHAT-IF CALCULATOR
    # ==========================================
    st.divider()
    st.header("3. Next-Month Investment Target")
    st.markdown("Calculate the required capital injection to shift the blended yield.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        C0 = st.number_input("Current Outstanding (Rs)", value=1000000.0, step=10000.0)
    with c2:
        Y0 = st.number_input("Current Yield %", value=15.0)
    with c3:
        Yt = st.number_input("Target Yield %", value=18.0)
    with c4:
        Rn = st.number_input("New Business Rate %", value=24.0)

    if st.button("Calculate Requirement", type="primary"):
        if Yt == Rn:
            st.error("Target Yield cannot equal the New Business Rate.")
        else:
            Cn = C0 * (Y0 - Yt) / (Yt - Rn)
            if Cn < 0:
                st.warning("Target unreachable by adding capital at this rate alone.")
            else:
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Required New Disbursement", f"Rs {Cn:,.2f}")
                col_res2.metric("New Total Capital Base", f"Rs {(C0 + Cn):,.2f}")

else:
    st.info("👈 Please upload both reports in the sidebar to begin analysis.")