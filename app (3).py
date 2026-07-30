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
    hdr = find_header_row(stock_bytes, "Year,Foracid")
    stock = pd.read_csv(io.BytesIO(stock_bytes), skiprows=hdr, low_memory=False)
    stock = stock[pd.to_numeric(stock["Year"], errors="coerce").notna()].copy()
    stock["Year"] = stock["Year"].astype(int)
    stock["Foracid"] = stock["Foracid"].astype("Int64")

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

    periods = ["All"] + PERIOD_LABELS
    branches = ["All"] + sorted(yield_tbl["branch"].dropna().unique().tolist())
    products = ["All"] + sorted(yield_tbl["product"].dropna().unique().tolist())

    # ==========================================
    # 3) UI: DASHBOARD & ANALYSIS
    # ==========================================
    st.header("2. Yield & Portfolio Analysis")
    
    col_p, col_b, col_pr = st.columns(3)
    with col_p:
        sel_period = st.selectbox("Select Period (Chart)", periods, key="chart_period")
    with col_b:
        sel_branch = st.selectbox("Select Branch (Chart)", branches, key="chart_branch")
    with col_pr:
        sel_product = st.selectbox("Select Product (Chart)", products, key="chart_product")

    st.write("---")
    view_by = st.radio("Analyze / Group by:", 
                       ["Period (Time Trend)", "Branch (Comparison)", "Product (Comparison)"], 
                       horizontal=True, key="chart_groupby")

    df = yield_tbl.copy()
    if sel_period != "All":
        df = df[df["Period"] == sel_period]
    if sel_branch != "All":
        df = df[df["branch"] == sel_branch]
    if sel_product != "All":
        df = df[df["product"] == sel_product]

    if "Period" in view_by:
        group_col = "Period"
    elif "Branch" in view_by:
        group_col = "branch"
    else:
        group_col = "product"

    grouped = df.groupby(group_col, observed=True).apply(
        lambda g: pd.Series({
            "Total_Opening_Bal": g["Total_Opening_Bal"].sum(),
            "Total_Int": g["Total_Int"].sum(),
            "Yield_pct": round(g["Total_Int"].sum() / g["Total_Opening_Bal"].sum() * 12 * 100, 2) if g["Total_Opening_Bal"].sum() > 0 else np.nan,
            "Avg_Contract_Rate": round(g["Avg_Contract_Rate"].mean(), 2)
        })
    ).reset_index()

    grouped = grouped[grouped["Total_Opening_Bal"] > 0]

    tot_bal = grouped["Total_Opening_Bal"].sum()
    tot_int = grouped["Total_Int"].sum()
    overall_yield = round(tot_int / tot_bal * 12 * 100, 2) if tot_bal > 0 else 0.0

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Filtered Portfolio Balance", f"Rs {tot_bal:,.2f}")
    mc2.metric("Filtered Total Interest", f"Rs {tot_int:,.2f}")
    mc3.metric("Blended Yield for Selection", f"{overall_yield}%")

    st.dataframe(grouped, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    if group_col == "Period":
        ax.plot(grouped[group_col].astype(str), grouped["Yield_pct"], marker="o", label="Real Yield %", color="#1f77b4")
        ax.plot(grouped[group_col].astype(str), grouped["Avg_Contract_Rate"], marker="s", linestyle="--", label="Avg Contract Rate %", color="#ff7f0e")
    else:
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
    # 4) UI: WHAT-IF CALCULATOR & DEPLETION
    # ==========================================
    st.divider()
    st.header("3. Next-Month Growth & Pricing Target")
    st.markdown("Project natural depletion, early settlements, and calculate the required pricing on your target investment to grow the portfolio yield.")
    
    calc_c1, calc_c2, calc_c3 = st.columns(3)
    with calc_c1:
        calc_period = st.selectbox("Target Period", periods, key="calc_period_dropdown")
    with calc_c2:
        calc_branch = st.selectbox("Target Branch", branches, key="calc_branch_dropdown")
    with calc_c3:
        calc_product = st.selectbox("Target Product", products, key="calc_product_dropdown")

    # Get Current Balance & Yield
    calc_df = yield_tbl.copy()
    if calc_period != "All":
        calc_df = calc_df[calc_df["Period"] == calc_period]
    if calc_branch != "All":
        calc_df = calc_df[calc_df["branch"] == calc_branch]
    if calc_product != "All":
        calc_df = calc_df[calc_df["product"] == calc_product]

    calc_tot_bal = calc_df["Total_Opening_Bal"].sum()
    calc_tot_int = calc_df["Total_Int"].sum()
    calc_overall_yield = round(calc_tot_int / calc_tot_bal * 12 * 100, 2) if calc_tot_bal > 0 else 0.0

    # Calculate Natural Depletion
    natural_depletion = 0.0
    if calc_period == "All":
        st.info("💡 To calculate Natural Depletion, please select a specific Target Period (e.g., '2026 July') instead of 'All'.")
    elif calc_tot_bal > 0:
        try:
            curr_idx = PERIOD_LABELS.index(calc_period)
            if curr_idx < len(PERIOD_LABELS) - 1:
                next_period = PERIOD_LABELS[curr_idx + 1]
                
                next_df = yield_tbl.copy()
                next_df = next_df[next_df["Period"] == next_period]
                if calc_branch != "All":
                    next_df = next_df[next_df["branch"] == calc_branch]
                if calc_product != "All":
                    next_df = next_df[next_df["product"] == calc_product]
                    
                next_bal = next_df["Total_Opening_Bal"].sum()
                natural_depletion = calc_tot_bal - next_bal
            else:
                natural_depletion = calc_tot_bal 
        except ValueError:
            pass

    # --- STEP 1: DEPLETION PROJECTION ---
    st.write("### Step 1: Portfolio Depletion Projection")
    
    # Grid Row 1
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    with r1_col1:
        st.metric("Opening Cap Balance", f"Rs {calc_tot_bal:,.2f}")
    with r1_col2:
        st.metric("Natural Depletion", f"Rs {natural_depletion:,.2f}")
    with r1_col3:
        es_pct = st.number_input("Early Settlement %", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    # Grid Row 2
    r2_col1, r2_col2, r2_col3 = st.columns(3)
    es_cap = calc_tot_bal * (es_pct / 100.0)
    with r2_col1:
        st.metric("Early Settlement Cap", f"Rs {es_cap:,.2f}")
    with r2_col2:
        total_depletion = natural_depletion + es_cap
        st.metric("Total Depletion", f"Rs {total_depletion:,.2f}")
    with r2_col3:
        st.write("") # Blank space for alignment

    st.write("---")

    # --- STEP 2: GROWTH & PRICING TARGET ---
    st.write("### Step 2: Growth Strategy & Required IRR")
    
    # Grid Row 3
    r3_col1, r3_col2, r3_col3 = st.columns(3)
    with r3_col1:
        target_investment = st.number_input("Target Investment (New Disbursement Rs)", min_value=0.0, value=500000.0, step=50000.0)
        # Formatted caption hack to show thousand separators under the input box
        st.caption(f"**Value:** Rs {target_investment:,.2f}")
    with r3_col2:
        portfolio_growth = target_investment - total_depletion
        if portfolio_growth >= 0:
            st.metric("Portfolio Growth", f"Rs {portfolio_growth:,.2f}", delta="Positive Growth")
        else:
            st.metric("Portfolio Growth", f"Rs {portfolio_growth:,.2f}", delta="Negative Growth", delta_color="inverse")
    with r3_col3:
        st.metric("Current Yield %", f"{calc_overall_yield}%")

    # Grid Row 4
    r4_col1, r4_col2, r4_col3 = st.columns(3)
    with r4_col1:
        target_yield = st.number_input("Target Yield %", value=calc_overall_yield + 1.0, step=0.1)
    with r4_col2:
        st.write("") # Spacer
        st.write("") # Spacer
        calculate_btn = st.button("Calculate Required Rate", type="primary", use_container_width=True)
    with r4_col3:
        st.write("") # Spacer
        
    if calculate_btn:
        if target_investment <= 0:
            st.error("Target Investment must be greater than zero.")
        elif calc_tot_bal == 0:
            st.warning("Opening Balance is 0. Please select a valid branch/product.")
        else:
            Rn = (target_yield * (calc_tot_bal + target_investment) - (calc_tot_bal * calc_overall_yield)) / target_investment
            
            st.write("#### Result:")
            result_c1, result_c2 = st.columns(2)
            with result_c1:
                st.metric("Required New Business Rate (IRR)", f"{Rn:,.2f}%")
            with result_c2:
                st.metric("New Total Capital Base", f"Rs {(calc_tot_bal + target_investment):,.2f}")
            
            if Rn > 100:
                st.warning(f"⚠️ Note: The required rate is extremely high ({Rn:,.2f}%) because your Target Investment (Rs {target_investment:,.2f}) is too small compared to the massive size of the Opening Balance (Rs {calc_tot_bal:,.2f}). To raise the blended yield by {target_yield - calc_overall_yield:.2f}%, you need a significantly larger investment volume.")

else:
    st.info("👈 Please upload both reports in the sidebar to begin analysis.")
