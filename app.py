"""
SME Loan Book Simulator
Streamlit app to model Carbon's SME loan book growth over time.

- Multi-tenor starting book (6, 9, 12 month cohorts from actual BQ data)
- Monthly disbursements split by tenor product
- Simple interest (flat monthly rate on original principal)
- Straight-line principal amortization
- Constant monthly default rate (CDR)
- Three tabs: Portfolio Projection, Sales Planning, Risk Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Loan Book Simulator", layout="wide")
st.title("SME Loan Book Simulator")

TENORS = [6, 9, 12]

# ---------------------------------------------------------------------------
# Core calculation functions
# ---------------------------------------------------------------------------

def calculate_cohort_schedule(
    principal: float,
    monthly_rate: float,
    tenor_months: int,
    annual_default_rate: float,
) -> pd.DataFrame:
    """Month-by-month schedule for a single loan cohort.
    Simple interest on original principal, straight-line amortization, CDR defaults.
    """
    if principal <= 0 or tenor_months <= 0:
        return pd.DataFrame(columns=["month_offset", "principal_payment",
                                      "interest_income", "defaults",
                                      "outstanding_balance", "cumulative_defaults"])

    mdr = 1 - (1 - annual_default_rate) ** (1 / 12)

    rows = []
    performing_balance = principal
    cum_defaults = 0.0
    for month in range(1, tenor_months + 1):
        defaults = performing_balance * mdr
        performing_balance -= defaults
        cum_defaults += defaults

        remaining_months = tenor_months - month + 1
        principal_payment = performing_balance / remaining_months

        performing_share = performing_balance / principal if principal > 0 else 0
        interest_income = principal * monthly_rate * performing_share

        performing_balance -= principal_payment

        rows.append({
            "month_offset": month,
            "principal_payment": principal_payment,
            "interest_income": interest_income,
            "defaults": defaults,
            "outstanding_balance": max(performing_balance, 0),
            "cumulative_defaults": cum_defaults,
        })

    return pd.DataFrame(rows)


def simulate_portfolio(
    starting_cohorts: list[dict],
    disbursement_schedule: pd.DataFrame,
    horizon_months: int,
    stress_start: int = 0,
    stress_end: int = 0,
    stress_default_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full portfolio simulation. Returns summary_df and tenor_df."""

    cohorts: list[tuple[int, str, pd.DataFrame]] = []

    # Starting book cohorts
    for sc in starting_cohorts:
        if sc["balance"] <= 0:
            continue
        sch = calculate_cohort_schedule(
            sc["balance"], sc["monthly_rate"],
            sc["remaining_life"], sc["annual_default"],
        )
        cohorts.append((0, f"Starting {sc['tenor']}mo", sch))

    # New monthly disbursements
    for idx, row in disbursement_schedule.iterrows():
        month_idx = idx + 1
        for tenor in TENORS:
            amt = row.get(f"disb_{tenor}mo", 0)
            if amt <= 0:
                continue
            rate = row.get(f"rate_{tenor}mo", 4.0) / 100
            base_default = row.get(f"default_{tenor}mo", 5.0) / 100
            # Apply stress multiplier if this month falls in stress window
            if stress_start <= month_idx <= stress_end:
                effective_default = min(base_default * stress_default_multiplier, 1.0)
            else:
                effective_default = base_default
            sch = calculate_cohort_schedule(amt, rate, tenor, effective_default)
            cohorts.append((month_idx, f"M{month_idx} {tenor}mo", sch))

    # Consolidate
    months = list(range(0, horizon_months + 1))
    summary = {m: {"outstanding": 0.0, "interest": 0.0,
                    "principal_collected": 0.0, "defaults": 0.0}
               for m in months}

    tenor_groups = {}
    for tenor in TENORS:
        tenor_groups[f"{tenor}mo"] = {m: 0.0 for m in months}
    tenor_groups["Starting Book"] = {m: 0.0 for m in months}

    # Track disbursements per tenor per month for sales tab
    disb_by_tenor = {t: {m: 0.0 for m in months} for t in TENORS}

    for orig_month, label, sch in cohorts:
        is_starting = orig_month == 0
        tenor_label = label.split(" ")[1]  # e.g. "6mo"

        if len(sch) == 0:
            continue

        initial_bal = (sch["outstanding_balance"].iloc[0]
                       + sch["principal_payment"].iloc[0]
                       + sch["defaults"].iloc[0])

        if orig_month <= horizon_months:
            if is_starting:
                tenor_groups["Starting Book"][0] += initial_bal
            else:
                tenor_groups[tenor_label][orig_month] += initial_bal
                # Track disbursement
                tenor_num = int(tenor_label.replace("mo", ""))
                disb_by_tenor[tenor_num][orig_month] += initial_bal

        for _, srow in sch.iterrows():
            sim_month = orig_month + int(srow["month_offset"])
            if sim_month > horizon_months:
                break

            # Apply stress to existing cohorts too
            actual_defaults = srow["defaults"]
            if stress_start <= sim_month <= stress_end and stress_default_multiplier > 1.0:
                extra = srow["defaults"] * (stress_default_multiplier - 1.0)
                actual_defaults += extra

            summary[sim_month]["interest"] += srow["interest_income"]
            summary[sim_month]["principal_collected"] += srow["principal_payment"]
            summary[sim_month]["defaults"] += actual_defaults

            if is_starting:
                tenor_groups["Starting Book"][sim_month] += srow["outstanding_balance"]
            else:
                tenor_groups[tenor_label][sim_month] += srow["outstanding_balance"]

    # Build summary df
    summary_rows = []
    for m in months:
        total_outstanding = sum(tenor_groups[tg][m] for tg in tenor_groups)
        summary_rows.append({
            "month": m,
            "outstanding_balance": total_outstanding,
            "interest_income": summary[m]["interest"],
            "principal_collected": summary[m]["principal_collected"],
            "defaults_written_off": summary[m]["defaults"],
        })
    summary_df = pd.DataFrame(summary_rows)

    # Build tenor df
    tenor_df = pd.DataFrame(index=months)
    for group in ["Starting Book"] + [f"{t}mo" for t in TENORS]:
        tenor_df[group] = [tenor_groups[group][m] for m in months]

    # Attach disbursement tracking
    summary_df["_disb_by_tenor"] = [disb_by_tenor for _ in months]

    return summary_df, tenor_df


# ---------------------------------------------------------------------------
# Sidebar — Global Controls
# ---------------------------------------------------------------------------

st.sidebar.header("Simulation Settings")

start_month = st.sidebar.selectbox(
    "Start Month",
    options=pd.date_range("2025-01", periods=24, freq="MS"),
    index=14,
    format_func=lambda d: d.strftime("%B %Y"),
)

horizon = st.sidebar.slider("Projection Horizon (months)", 6, 60, 24)

annual_default = st.sidebar.number_input(
    "Annual Default Rate — all tenors (%)", 0.0, 50.0, 5.0, 1.0
)

st.sidebar.divider()
st.sidebar.header("New Loan Disbursements")

base_disb_str = st.sidebar.text_input("Base Monthly Disbursement (₦)", value="500,000,000")
try:
    base_disbursement = int(base_disb_str.replace(",", ""))
except ValueError:
    base_disbursement = 0
    st.sidebar.error("Enter a valid number")

st.sidebar.subheader("Tenor Mix (%)")
mix_6 = st.sidebar.slider("6mo mix (%)", 0, 100, 60, 5)
max_9 = 100 - mix_6
mix_9 = st.sidebar.slider("9mo mix (%)", 0, max_9, min(20, max_9), 5)
mix_12 = 100 - mix_6 - mix_9
st.sidebar.markdown(f"12mo mix: **{mix_12}%**")
tenor_mix = {6: mix_6 / 100, 9: mix_9 / 100, 12: mix_12 / 100}

st.sidebar.subheader("Monthly Growth Rate (%)")
growth_6 = st.sidebar.number_input("6mo growth (%)", -20.0, 50.0, 5.0, 1.0)
growth_9 = st.sidebar.number_input("9mo growth (%)", -20.0, 50.0, 5.0, 1.0)
growth_12 = st.sidebar.number_input("12mo growth (%)", -20.0, 50.0, 5.0, 1.0)
tenor_growth = {6: growth_6 / 100, 9: growth_9 / 100, 12: growth_12 / 100}

st.sidebar.subheader("Interest Rates (%)")
rate_6 = st.sidebar.number_input("6mo rate (%)", 0.0, 20.0, 4.35, 0.25)
rate_9 = st.sidebar.number_input("9mo rate (%)", 0.0, 20.0, 5.0, 0.25)
rate_12 = st.sidebar.number_input("12mo rate (%)", 0.0, 20.0, 5.0, 0.25)
tenor_rates = {6: rate_6, 9: rate_9, 12: rate_12}

st.sidebar.subheader("Average Loan Size (₦)")
avg_loan_6_str = st.sidebar.text_input("6mo avg ticket (₦)", value="3,247,943")
avg_loan_9_str = st.sidebar.text_input("9mo avg ticket (₦)", value="5,891,139")
avg_loan_12_str = st.sidebar.text_input("12mo avg ticket (₦)", value="6,135,069")
try:
    avg_ticket = {
        6: int(avg_loan_6_str.replace(",", "")),
        9: int(avg_loan_9_str.replace(",", "")),
        12: int(avg_loan_12_str.replace(",", "")),
    }
except ValueError:
    avg_ticket = {6: 3_247_943, 9: 5_891_139, 12: 6_135_069}
    st.sidebar.error("Enter valid numbers")

# ---------------------------------------------------------------------------
# Starting Book
# ---------------------------------------------------------------------------
st.header("Starting Book (as at simulation start)")
st.caption("Pre-filled from actual Fineract data. Edit to adjust.")

starting_book_data = pd.DataFrame({
    "Tenor": ["6 months", "9 months", "12 months"],
    "Principal Outstanding (₦)": [1_590_875_462, 473_039_228, 666_369_256],
    "Remaining Life (months)": [3, 6, 10],
    "Monthly Rate (%)": [4.35, 5.0, 4.98],
})

edited_starting = st.data_editor(
    starting_book_data,
    column_config={
        "Tenor": st.column_config.TextColumn("Tenor", disabled=True),
        "Principal Outstanding (₦)": st.column_config.NumberColumn(
            "Principal Outstanding (₦)", format="₦%,.0f", min_value=0),
        "Remaining Life (months)": st.column_config.NumberColumn(
            "Remaining Life (mo)", min_value=0, max_value=24),
        "Monthly Rate (%)": st.column_config.NumberColumn(
            "Monthly Rate (%)", format="%.2f%%", min_value=0, max_value=20),
    },
    use_container_width=True, hide_index=True,
)

total_starting = edited_starting["Principal Outstanding (₦)"].sum()
st.markdown(f"**Total starting book: ₦{total_starting:,.0f}**")

starting_cohorts = []
for idx, row in edited_starting.iterrows():
    starting_cohorts.append({
        "tenor": TENORS[idx],
        "balance": row["Principal Outstanding (₦)"],
        "remaining_life": int(row["Remaining Life (months)"]),
        "monthly_rate": row["Monthly Rate (%)"] / 100,
        "annual_default": annual_default / 100,
    })

# ---------------------------------------------------------------------------
# Disbursement Schedule
# ---------------------------------------------------------------------------
st.header("Monthly Disbursement Schedule")
st.caption("Auto-computed from sidebar controls. Edit cells to override.")

months_list = [(start_month + relativedelta(months=i)).strftime("%b %Y")
               for i in range(1, horizon + 1)]

row_labels = [f"{t}mo Disbursement (₦)" for t in TENORS] + ["Total Disbursement (₦)"]

computed_data = {}
for i, m in enumerate(months_list):
    col_vals = []
    total = 0.0
    for tenor in TENORS:
        amt = round(base_disbursement * tenor_mix[tenor] * (1 + tenor_growth[tenor]) ** i)
        col_vals.append(amt)
        total += amt
    col_vals.append(round(total))
    computed_data[m] = col_vals

transposed_data = pd.DataFrame(computed_data, index=row_labels)

col_cfg = {m: st.column_config.NumberColumn(m, format="%,.0f") for m in months_list}
edited_transposed = st.data_editor(
    transposed_data, column_config=col_cfg,
    use_container_width=True, num_rows="fixed",
)

schedule_rows = []
for i, m in enumerate(months_list):
    row = {"month": m}
    for tenor in TENORS:
        row[f"disb_{tenor}mo"] = edited_transposed.loc[f"{tenor}mo Disbursement (₦)", m]
        row[f"rate_{tenor}mo"] = tenor_rates[tenor]
        row[f"default_{tenor}mo"] = annual_default
    schedule_rows.append(row)
disbursement_schedule = pd.DataFrame(schedule_rows)

# ---------------------------------------------------------------------------
# Run Simulation (base case)
# ---------------------------------------------------------------------------
summary_df, tenor_df = simulate_portfolio(
    starting_cohorts=starting_cohorts,
    disbursement_schedule=disbursement_schedule,
    horizon_months=horizon,
)

date_labels = [(start_month + relativedelta(months=i)).strftime("%b %Y")
               for i in range(horizon + 1)]
summary_df["date"] = date_labels

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Projection", "💰 Sales Planning", "⚠️ Risk Dashboard"])

# ======================== TAB 1: Portfolio Projection ========================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Starting Book", f"₦{total_starting / 1e9:.1f}B")
    with col2:
        peak = summary_df["outstanding_balance"].max()
        st.metric("Peak Book Size", f"₦{peak / 1e9:.1f}B")
    with col3:
        ending = summary_df["outstanding_balance"].iloc[-1]
        st.metric("Ending Book Size", f"₦{ending / 1e9:.1f}B")
    with col4:
        total_defaults = summary_df["defaults_written_off"].sum()
        st.metric("Total Defaults", f"₦{total_defaults / 1e9:.2f}B")

    # Book balance chart
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(
        x=date_labels, y=summary_df["outstanding_balance"],
        mode="lines+markers", name="Outstanding Balance",
        line=dict(width=3, color="#1a5276"),
        fill="tozeroy", fillcolor="rgba(26, 82, 118, 0.15)",
    ))
    fig_balance.update_layout(
        title="Outstanding Loan Book Balance",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig_balance, use_container_width=True)

    # Balance table
    balance_row = summary_df.set_index("date")[["outstanding_balance"]].T
    balance_row.index = ["Outstanding Balance (₦)"]
    balance_row = balance_row.map(lambda x: f"{x:,.0f}")
    st.dataframe(balance_row, use_container_width=True)

    # Tenor stacked area
    fig_tenor = go.Figure()
    colors_map = {"Starting Book": "#7f8c8d", "6mo": "#2980b9",
                  "9mo": "#27ae60", "12mo": "#e74c3c"}
    for col_name in tenor_df.columns:
        fig_tenor.add_trace(go.Scatter(
            x=date_labels, y=tenor_df[col_name],
            mode="lines", name=col_name, stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors_map.get(col_name, "#95a5a6"),
        ))
    fig_tenor.update_layout(
        title="Book Balance by Tenor Product",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig_tenor, use_container_width=True)

    # Tenor table
    tenor_table = tenor_df.copy()
    tenor_table.index = date_labels
    tenor_table = tenor_table.T.map(lambda x: f"{x:,.0f}")
    st.dataframe(tenor_table, use_container_width=True)

    # Detailed breakdown
    with st.expander("Detailed Monthly Breakdown"):
        summary_display = summary_df.copy()
        summary_display["cumulative_interest"] = summary_display["interest_income"].cumsum()
        summary_display["cumulative_defaults"] = summary_display["defaults_written_off"].cumsum()
        summary_display["cumulative_collections"] = summary_display["principal_collected"].cumsum()

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Total Interest Earned",
                       f"₦{summary_display['cumulative_interest'].iloc[-1] / 1e9:.2f}B")
        with mcol2:
            st.metric("Total Principal Collected",
                       f"₦{summary_display['cumulative_collections'].iloc[-1] / 1e9:.2f}B")
        with mcol3:
            st.metric("Cumulative Defaults",
                       f"₦{summary_display['cumulative_defaults'].iloc[-1] / 1e9:.2f}B")

        display_cols = ["date", "outstanding_balance", "interest_income", "principal_collected",
                        "defaults_written_off", "cumulative_interest", "cumulative_defaults"]
        fmt_df = summary_display[display_cols].copy()
        for c in display_cols[1:]:
            fmt_df[c] = fmt_df[c].apply(lambda x: f"₦{x:,.0f}")
        fmt_df.columns = ["Month", "Outstanding Balance", "Interest Income", "Principal Collected",
                           "Defaults", "Cumulative Interest", "Cumulative Defaults"]
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)


# ======================== TAB 2: Sales Planning ========================
with tab2:
    st.subheader("Disbursement Targets & Loan Counts")

    # Build loan count and revenue table
    sales_rows = []
    for i, m in enumerate(months_list):
        row = {"Month": m}
        total_disb = 0
        total_loans = 0
        for tenor in TENORS:
            disb = disbursement_schedule.iloc[i][f"disb_{tenor}mo"]
            loans = int(disb / avg_ticket[tenor]) if avg_ticket[tenor] > 0 else 0
            row[f"{tenor}mo ₦"] = disb
            row[f"{tenor}mo #"] = loans
            total_disb += disb
            total_loans += loans
        row["Total ₦"] = total_disb
        row["Total #"] = total_loans
        sales_rows.append(row)

    sales_df = pd.DataFrame(sales_rows)

    # Key sales metrics
    total_disb_all = sales_df["Total ₦"].sum()
    total_loans_all = sales_df["Total #"].sum()
    avg_monthly_disb = sales_df["Total ₦"].mean()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Total Disbursements", f"₦{total_disb_all / 1e9:.1f}B")
    with sc2:
        st.metric("Total Loans", f"{total_loans_all:,}")
    with sc3:
        st.metric("Avg Monthly Disbursement", f"₦{avg_monthly_disb / 1e9:.2f}B")

    # Disbursement chart by tenor
    fig_disb = go.Figure()
    for tenor in TENORS:
        fig_disb.add_trace(go.Bar(
            x=months_list,
            y=sales_df[f"{tenor}mo ₦"],
            name=f"{tenor}mo",
            marker_color=colors_map.get(f"{tenor}mo", "#95a5a6"),
        ))
    fig_disb.update_layout(
        title="Monthly Disbursements by Tenor",
        barmode="stack",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_disb, use_container_width=True)

    # Loan count chart
    fig_loans = go.Figure()
    for tenor in TENORS:
        fig_loans.add_trace(go.Bar(
            x=months_list,
            y=sales_df[f"{tenor}mo #"],
            name=f"{tenor}mo",
            marker_color=colors_map.get(f"{tenor}mo", "#95a5a6"),
        ))
    fig_loans.update_layout(
        title="Monthly Loan Count by Tenor",
        barmode="stack",
        yaxis_title="Loans",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_loans, use_container_width=True)

    # Revenue projection
    st.subheader("Revenue Projection")

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=date_labels[1:],
        y=summary_df["interest_income"].iloc[1:],
        name="Interest Income",
        marker_color="#27ae60",
    ))
    fig_rev.update_layout(
        title="Monthly Interest Income",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Annualized yield
    ann_yield = []
    for i in range(len(summary_df)):
        bal = summary_df["outstanding_balance"].iloc[i]
        inc = summary_df["interest_income"].iloc[i]
        if bal > 0 and i > 0:
            ann_yield.append(inc / bal * 12 * 100)
        else:
            ann_yield.append(0)

    fig_yield = go.Figure()
    fig_yield.add_trace(go.Scatter(
        x=date_labels[1:], y=ann_yield[1:],
        mode="lines+markers", name="Annualized Yield",
        line=dict(width=2, color="#f39c12"),
    ))
    fig_yield.update_layout(
        title="Annualized Portfolio Yield (%)",
        yaxis_title="%", yaxis_tickformat=".1f",
        height=350, hovermode="x unified",
    )
    st.plotly_chart(fig_yield, use_container_width=True)

    # Sales summary table (transposed)
    sales_summary = pd.DataFrame(index=months_list)
    for tenor in TENORS:
        sales_summary[f"{tenor}mo Disbursement"] = sales_df[f"{tenor}mo ₦"].apply(lambda x: f"₦{x:,.0f}")
        sales_summary[f"{tenor}mo Loans"] = sales_df[f"{tenor}mo #"].values
    sales_summary["Total Disbursement"] = sales_df["Total ₦"].apply(lambda x: f"₦{x:,.0f}")
    sales_summary["Total Loans"] = sales_df["Total #"].values
    st.dataframe(sales_summary.T, use_container_width=True)


# ======================== TAB 3: Risk Dashboard ========================
with tab3:

    # --- NPL Ratio ---
    st.subheader("Portfolio Quality")

    # Compute cumulative defaults as proxy for NPL stock
    cum_defaults = summary_df["defaults_written_off"].cumsum()
    npl_ratio = []
    for i in range(len(summary_df)):
        bal = summary_df["outstanding_balance"].iloc[i]
        defaults_month = summary_df["defaults_written_off"].iloc[i]
        # NPL ratio = monthly defaults / outstanding balance (annualized)
        if bal > 0:
            npl_ratio.append(defaults_month / bal * 100)
        else:
            npl_ratio.append(0)

    fig_npl = go.Figure()
    fig_npl.add_trace(go.Scatter(
        x=date_labels, y=summary_df["outstanding_balance"],
        mode="lines", name="Outstanding Balance",
        line=dict(width=2, color="#1a5276"),
        yaxis="y",
    ))
    fig_npl.add_trace(go.Scatter(
        x=date_labels[1:], y=npl_ratio[1:],
        mode="lines+markers", name="Default Rate (% of book)",
        line=dict(width=2, color="#e74c3c", dash="dot"),
        yaxis="y2",
    ))
    fig_npl.update_layout(
        title="Outstanding Balance & Monthly Default Rate",
        yaxis=dict(title="₦", tickformat=",.0f"),
        yaxis2=dict(title="Default Rate (%)", overlaying="y", side="right", tickformat=".2f"),
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig_npl, use_container_width=True)

    # --- Concentration Risk ---
    st.subheader("Concentration Risk — Tenor Mix Over Time")

    # Compute tenor mix % over time
    tenor_pct = tenor_df.copy()
    row_totals = tenor_pct.sum(axis=1)
    for col in tenor_pct.columns:
        tenor_pct[col] = tenor_pct[col] / row_totals.replace(0, 1) * 100

    fig_conc = go.Figure()
    for col_name in tenor_pct.columns:
        fig_conc.add_trace(go.Scatter(
            x=date_labels, y=tenor_pct[col_name],
            mode="lines", name=col_name, stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors_map.get(col_name, "#95a5a6"),
        ))
    fig_conc.update_layout(
        title="Book Composition by Tenor (%)",
        yaxis_title="%", yaxis_range=[0, 100],
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_conc, use_container_width=True)

    # --- Stress Testing ---
    st.subheader("Stress Testing")
    st.caption("Shock default rates for a period and compare against base case.")

    stress_col1, stress_col2, stress_col3 = st.columns(3)
    with stress_col1:
        stress_multiplier = st.selectbox(
            "Default rate multiplier",
            options=[1.5, 2.0, 3.0, 5.0],
            index=1,
            format_func=lambda x: f"{x:.1f}x ({annual_default * x:.0f}% annual)",
        )
    with stress_col2:
        stress_start_month = st.slider("Stress starts (month)", 1, horizon, 3)
    with stress_col3:
        stress_duration = st.slider("Duration (months)", 1, 12, 3)

    stress_end_month = min(stress_start_month + stress_duration - 1, horizon)

    # Run stressed simulation
    stress_summary, stress_tenor = simulate_portfolio(
        starting_cohorts=starting_cohorts,
        disbursement_schedule=disbursement_schedule,
        horizon_months=horizon,
        stress_start=stress_start_month,
        stress_end=stress_end_month,
        stress_default_multiplier=stress_multiplier,
    )
    stress_summary["date"] = date_labels

    # Comparison chart
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(
        x=date_labels, y=summary_df["outstanding_balance"],
        mode="lines", name="Base Case",
        line=dict(width=2, color="#1a5276"),
    ))
    fig_stress.add_trace(go.Scatter(
        x=date_labels, y=stress_summary["outstanding_balance"],
        mode="lines", name=f"Stressed ({stress_multiplier:.0f}x defaults)",
        line=dict(width=2, color="#e74c3c", dash="dash"),
    ))
    # Shade stress period
    stress_start_label = date_labels[stress_start_month] if stress_start_month < len(date_labels) else date_labels[-1]
    stress_end_label = date_labels[stress_end_month] if stress_end_month < len(date_labels) else date_labels[-1]
    fig_stress.add_vrect(
        x0=stress_start_label, x1=stress_end_label,
        fillcolor="red", opacity=0.1,
        annotation_text="Stress Period", annotation_position="top left",
    )
    fig_stress.update_layout(
        title="Base Case vs Stress Scenario",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=450, hovermode="x unified",
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    # Stress impact metrics
    base_end = summary_df["outstanding_balance"].iloc[-1]
    stress_end_bal = stress_summary["outstanding_balance"].iloc[-1]
    base_defaults_total = summary_df["defaults_written_off"].sum()
    stress_defaults_total = stress_summary["defaults_written_off"].sum()

    im1, im2, im3 = st.columns(3)
    with im1:
        delta = stress_end_bal - base_end
        st.metric("Ending Book (Stressed)",
                   f"₦{stress_end_bal / 1e9:.2f}B",
                   delta=f"₦{delta / 1e9:.2f}B vs base")
    with im2:
        st.metric("Total Defaults (Stressed)",
                   f"₦{stress_defaults_total / 1e9:.2f}B",
                   delta=f"+₦{(stress_defaults_total - base_defaults_total) / 1e9:.2f}B")
    with im3:
        base_default_pct = base_defaults_total / total_starting * 100 if total_starting > 0 else 0
        stress_default_pct = stress_defaults_total / total_starting * 100 if total_starting > 0 else 0
        st.metric("Default as % of Starting Book",
                   f"{stress_default_pct:.1f}%",
                   delta=f"+{stress_default_pct - base_default_pct:.1f}pp")

    # --- Defaults breakdown ---
    st.subheader("Monthly Defaults — Base vs Stress")

    fig_def_compare = go.Figure()
    fig_def_compare.add_trace(go.Bar(
        x=date_labels[1:], y=summary_df["defaults_written_off"].iloc[1:],
        name="Base Case", marker_color="#3498db",
    ))
    fig_def_compare.add_trace(go.Bar(
        x=date_labels[1:], y=stress_summary["defaults_written_off"].iloc[1:],
        name="Stressed", marker_color="#e74c3c",
    ))
    fig_def_compare.update_layout(
        barmode="group",
        yaxis_title="₦", yaxis_tickformat=",.0f",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig_def_compare, use_container_width=True)
