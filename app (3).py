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
    with st.spinner("Processing portfolios..."):
        yield_tbl = process_data(stock_file.getvalue(), arr_file.getvalue(), arr_file.name)
    
    st.success("Data loaded successfully!")

    # ==========================================
    # 3) UI: DASHBOARD & ANALYSIS
    # ==========================================
    st.header("2. Yield & Portfolio Analysis")
    
    # 3-Way Filters
    col_p, col_b, col_pr = st.columns(3)
    periods = ["All"] + PERIOD_LABELS
    branches = ["All"] + sorted(yield_tbl["branch"].dropna().unique().tolist())
    products = ["All"] + sorted(yield_tbl["product"].dropna().unique().tolist())
    
    with col_p:
        sel_period = st.selectbox("Select Period", periods)
    with col_b:
        sel_branch = st.selectbox("Select Branch", branches)
    with col_pr:
        sel_product = st.selectbox("Select Product", products)

    st.write("---")
    
    # Toggle for what to show on the X-Axis
    view_by = st.radio("Analyze / Group by:", 
                       ["Period (Time Trend)", "Branch (Comparison)", "Product (Comparison)"], 
                       horizontal=True)

    # Apply Filters to Dataframe
    df = yield_tbl.copy()
    if sel_period != "All":
        df = df[df["Period"] == sel_period]
    if sel_branch != "All":
        df = df[df["branch"] == sel_branch]
    if sel_product != "All":
        df = df[df["product"] == sel_product]

    # Map the radio button choice to the actual dataframe column name
    if "Period" in view_by:
        group_col = "Period"
    elif "Branch" in view_by:
        group_col = "branch"
    else:
        group_col = "product"

    # Aggregate the filtered data based on the chosen grouping
    grouped = df.groupby(group_col, observed=True).apply(
        lambda g: pd.Series({
            "Total_Opening_Bal": g["Total_Opening_Bal"].sum(),
            "Total_Int": g["Total_Int"].sum(),
            "Yield_pct": round(g["Total_Int"].sum() / g["Total_Opening_Bal"].sum() * 12 * 100, 2) if g["Total_Opening_Bal"].sum() > 0 else np.nan,
            "Avg_Contract_Rate": round(g["Avg_Contract_Rate"].mean(), 2)
        })
    ).reset_index()

    # Remove empty groups (e.g., branches with no balance in that specific month)
    grouped = grouped[grouped["Total_Opening_Bal"] > 0]

    # Show Summary Metrics for the exact slice of data they filtered
    tot_bal = grouped["Total_Opening_Bal"].sum()
    tot_int = grouped["Total_Int"].sum()
    overall_yield = round(tot_int / tot_bal * 12 * 100, 2) if tot_bal > 0 else 0

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Filtered Portfolio Balance", f"Rs {tot_bal:,.2f}")
    mc2.metric("Filtered Total Interest", f"Rs {tot_int:,.2f}")
    mc3.metric("Blended Yield for Selection", f"{overall_yield}%")

    st.dataframe(grouped, use_container_width=True)

    # Dynamic Chart Rendering (Line for Time, Bar for Categories)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if group_col == "Period":
        # Line chart for time series
        ax.plot(grouped[group_col].astype(str), grouped["Yield_pct"], marker="o", label="Real Yield %", color="#1f77b4")
        ax.plot(grouped[group_col].astype(str), grouped["Avg_Contract_Rate"], marker="s", linestyle="--", label="Avg Contract Rate %", color="#ff7f0e")
    else:
        # Bar chart for category comparison
        x = np.arange(len(grouped))
        width = 0.35
        ax.bar(x - width/2, grouped["Yield_pct"], width, label="Real Yield %", color="#1f77b4")
        ax.bar(x + width/2, grouped["Avg_Contract_Rate"], width, label="Avg Contract Rate %", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped[group_col].astype(str), rotation=45, ha="right")

    ax.set_title(f"Yield Analysis by {group_col.capitalize()}")
    ax.set_ylabel("%")
    ax.legend()
    st.pyplot(fig)

    # ==========================================
    # 4) UI: WHAT-IF CALCULATOR (Tabs Added)
    # ==========================================
    st.divider()
    st.header("3. Next-Month Investment Target")
    st.markdown("Calculate either the required capital injection **OR** the required new business rate to hit your target yield.")
    
    tab1, tab2 = st.tabs(["📊 Calculate Required Disbursement", "📈 Calculate Required Rate (IRR)"])
    
    # --- TAB 1: Solve for Disbursement (Cn) ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            C0_1 = st.number_input("Current Outstanding (Rs)", value=1000000.0, step=10000.0, key="c0_1")
        with c2:
            Y0_1 = st.number_input("Current Yield %", value=15.0, key="y0_1")
        with c3:
            Yt_1 = st.number_input("Target Yield %", value=18.0, key="yt_1")
        with c4:
            Rn_1 = st.number_input("New Business Rate %", value=24.0, key="rn_1")

        if st.button("Calculate Disbursement Requirement", type="primary", key="btn1"):
            if Yt_1 == Rn_1:
                st.error("Target Yield cannot equal the New Business Rate.")
            else:
                Cn = C0_1 * (Y0_1 - Yt_1) / (Yt_1 - Rn_1)
                if Cn < 0:
                    st.warning("Target unreachable by adding capital at this rate alone.")
                else:
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("Required New Disbursement", f"Rs {Cn:,.2f}")
                    col_res2.metric("New Total Capital Base", f"Rs {(C0_1 + Cn):,.2f}")

    # --- TAB 2: Solve for Rate (Rn) ---
    with tab2:
        t2_c1, t2_c2, t2_c3, t2_c4 = st.columns(4)
        with t2_c1:
            C0_2 = st.number_input("Current Outstanding (Rs)", value=1000000.0, step=10000.0, key="c0_2")
        with t2_c2:
            Y0_2 = st.number_input("Current Yield %", value=15.0, key="y0_2")
        with t2_c3:
            Yt_2 = st.number_input("Target Yield %", value=18.0, key="yt_2")
        with t2_c4:
            Cn_2 = st.number_input("New Disbursement (Rs)", value=500000.0, step=10000.0, key="cn_2")

        if st.button("Calculate Required Rate", type="primary", key="btn2"):
            if Cn_2 <= 0:
                st.error("New Disbursement amount must be greater than zero.")
            else:
                # Formula: Rn = (Yt * (C0 + Cn) - C0 * Y0) / Cn
                Rn = (Yt_2 * (C0_2 + Cn_2) - (C0_2 * Y0_2)) / Cn_2
                
                col_res3, col_res4 = st.columns(2)
                col_res3.metric("Required New Business Rate (IRR)", f"{Rn:,.2f}%")
                col_res4.metric("New Total Capital Base", f"Rs {(C0_2 + Cn_2):,.2f}")

else:
    st.info("👈 Please upload both reports in the sidebar to begin analysis.")
