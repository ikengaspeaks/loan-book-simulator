"""
SME Loan Book Simulator
Streamlit app to model Carbon's SME loan book growth over time.

- Multi-tenor starting book (6, 8, 10, 12 month cohorts from actual BQ data)
- Monthly disbursements split by tenor product
- Simple interest (flat monthly rate on original principal)
- Straight-line principal amortization
- Constant monthly default rate (CDR)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Loan Book Simulator", layout="wide")
st.title("SME Loan Book Simulator")

# Tenor products used throughout
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
    """Return month-by-month schedule for a single loan cohort.

    Uses simple interest on original principal and straight-line amortization.
    Defaults applied via constant monthly default rate on performing balance.
    """
    if principal <= 0 or tenor_months <= 0:
        return pd.DataFrame(columns=["month_offset", "principal_payment",
                                      "interest_income", "defaults", "outstanding_balance"])

    mdr = 1 - (1 - annual_default_rate) ** (1 / 12)

    rows = []
    performing_balance = principal
    for month in range(1, tenor_months + 1):
        defaults = performing_balance * mdr
        performing_balance -= defaults

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
        })

    return pd.DataFrame(rows)


def simulate_portfolio(
    starting_cohorts: list[dict],
    disbursement_schedule: pd.DataFrame,
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full portfolio simulation with multi-tenor starting book and disbursements.

    starting_cohorts: list of {tenor, balance, remaining_life, monthly_rate, annual_default}
    disbursement_schedule: DataFrame with columns for each tenor's disbursement + rates
    """
    # (origination_month, label, schedule_df)
    cohorts: list[tuple[int, str, pd.DataFrame]] = []

    # Starting book cohorts (month 0)
    for sc in starting_cohorts:
        if sc["balance"] <= 0:
            continue
        sch = calculate_cohort_schedule(
            sc["balance"], sc["monthly_rate"],
            sc["remaining_life"], sc["annual_default"],
        )
        cohorts.append((0, f"Starting {sc['tenor']}mo", sch))

    # New monthly disbursements — one cohort per tenor per month
    for idx, row in disbursement_schedule.iterrows():
        month_idx = idx + 1
        for tenor in TENORS:
            amt = row.get(f"disb_{tenor}mo", 0)
            if amt <= 0:
                continue
            rate = row.get(f"rate_{tenor}mo", 4.0) / 100
            default_rate = row.get(f"default_{tenor}mo", 5.0) / 100
            sch = calculate_cohort_schedule(amt, rate, tenor, default_rate)
            cohorts.append((month_idx, f"M{month_idx} {tenor}mo", sch))

    # Consolidate
    months = list(range(0, horizon_months + 1))
    summary = {m: {"outstanding": 0.0, "interest": 0.0,
                    "principal_collected": 0.0, "defaults": 0.0}
               for m in months}

    # Track by tenor group for the stacked chart
    tenor_groups = {}  # {tenor_label: {month: balance}}
    for tenor in TENORS:
        tenor_groups[f"{tenor}mo"] = {m: 0.0 for m in months}
    tenor_groups["Starting Book"] = {m: 0.0 for m in months}

    for orig_month, label, sch in cohorts:
        is_starting = orig_month == 0
        # Determine tenor group
        if is_starting:
            # Parse tenor from label like "Starting 6mo"
            tenor_label = label.split(" ")[1]  # "6mo"
        else:
            tenor_label = label.split(" ")[1]  # "6mo"

        # Set initial balance at origination
        if len(sch) > 0:
            initial_bal = (sch["outstanding_balance"].iloc[0]
                           + sch["principal_payment"].iloc[0]
                           + sch["defaults"].iloc[0])
        else:
            continue

        if orig_month <= horizon_months:
            if is_starting:
                tenor_groups["Starting Book"][0] += initial_bal
            # For non-starting, the balance appears when disbursed
            if not is_starting and orig_month <= horizon_months:
                tenor_groups[tenor_label][orig_month] += initial_bal

        for _, srow in sch.iterrows():
            sim_month = orig_month + int(srow["month_offset"])
            if sim_month > horizon_months:
                break
            summary[sim_month]["interest"] += srow["interest_income"]
            summary[sim_month]["principal_collected"] += srow["principal_payment"]
            summary[sim_month]["defaults"] += srow["defaults"]

            # Update tenor group balance: subtract what was paid/defaulted this month
            if is_starting:
                tenor_groups["Starting Book"][sim_month] += srow["outstanding_balance"]
            else:
                tenor_groups[tenor_label][sim_month] += srow["outstanding_balance"]

    # Build summary
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

    # Build tenor breakdown df
    tenor_df = pd.DataFrame(index=months)
    # Starting book first, then tenors in order
    for group in ["Starting Book"] + [f"{t}mo" for t in TENORS]:
        tenor_df[group] = [tenor_groups[group][m] for m in months]

    return summary_df, tenor_df


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

# Sidebar — Global Settings
st.sidebar.header("Simulation Settings")

start_month = st.sidebar.selectbox(
    "Start Month",
    options=pd.date_range("2025-01", periods=24, freq="MS"),
    index=14,  # March 2026
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

# ---------------------------------------------------------------------------
# Starting Book — from actual BQ data
# ---------------------------------------------------------------------------
st.header("Starting Book (as at simulation start)")
st.caption("Pre-filled from actual Fineract data. Edit to adjust.")

starting_book_data = pd.DataFrame(
    {
        "Tenor": ["6 months", "9 months", "12 months"],
        "Principal Outstanding (₦)": [1_590_875_462, 473_039_228, 666_369_256],
        "Remaining Life (months)": [3, 6, 10],
        "Monthly Rate (%)": [4.35, 5.0, 4.98],
    }
)

edited_starting = st.data_editor(
    starting_book_data,
    column_config={
        "Tenor": st.column_config.TextColumn("Tenor", disabled=True),
        "Principal Outstanding (₦)": st.column_config.NumberColumn(
            "Principal Outstanding (₦)", format="₦%,.0f", min_value=0
        ),
        "Remaining Life (months)": st.column_config.NumberColumn(
            "Remaining Life (mo)", min_value=0, max_value=24
        ),
        "Monthly Rate (%)": st.column_config.NumberColumn(
            "Monthly Rate (%)", format="%.2f%%", min_value=0, max_value=20
        ),
    },
    use_container_width=True,
    hide_index=True,
)

total_starting = edited_starting["Principal Outstanding (₦)"].sum()
st.markdown(f"**Total starting book: ₦{total_starting:,.0f}**")

# Build starting cohorts list
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
# Monthly Disbursement Schedule — auto-computed from sidebar levers
# ---------------------------------------------------------------------------
st.header("Monthly Disbursement Schedule")
st.caption("Auto-computed from sidebar controls. Edit cells to override specific months.")

months_list = [(start_month + relativedelta(months=i)).strftime("%b %Y")
               for i in range(1, horizon + 1)]

# Compute schedule from levers: disb_t_i = base × mix_t × (1 + growth_t)^i
row_labels = []
for tenor in TENORS:
    row_labels.append(f"{tenor}mo Disbursement (₦)")
row_labels.append("Total Disbursement (₦)")

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

col_cfg = {
    m: st.column_config.NumberColumn(m, format="%,.0f")
    for m in months_list
}

edited_transposed = st.data_editor(
    transposed_data,
    column_config=col_cfg,
    use_container_width=True,
    num_rows="fixed",
)

# Convert to schedule format
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
# Run Simulation
# ---------------------------------------------------------------------------
st.divider()
st.header("Results")

summary_df, tenor_df = simulate_portfolio(
    starting_cohorts=starting_cohorts,
    disbursement_schedule=disbursement_schedule,
    horizon_months=horizon,
)

date_labels = [(start_month + relativedelta(months=i)).strftime("%b %Y")
               for i in range(horizon + 1)]
summary_df["date"] = date_labels

# Key metrics
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

# Primary chart — Outstanding book balance
fig_balance = go.Figure()
fig_balance.add_trace(go.Scatter(
    x=date_labels,
    y=summary_df["outstanding_balance"],
    mode="lines+markers",
    name="Outstanding Balance",
    line=dict(width=3, color="#1a5276"),
    fill="tozeroy",
    fillcolor="rgba(26, 82, 118, 0.15)",
))
fig_balance.update_layout(
    title="Outstanding Loan Book Balance",
    yaxis_title="₦",
    yaxis_tickformat=",.0f",
    xaxis_title="",
    height=450,
    hovermode="x unified",
)
st.plotly_chart(fig_balance, use_container_width=True)

# Transposed balance table
balance_row = summary_df.set_index("date")[["outstanding_balance"]].T
balance_row.index = ["Outstanding Balance (₦)"]
balance_row = balance_row.map(lambda x: f"{x:,.0f}")
st.dataframe(balance_row, use_container_width=True)

# Stacked area by tenor group
fig_tenor = go.Figure()
colors_map = {
    "Starting Book": "#7f8c8d",
    "6mo": "#2980b9",
    "8mo": "#27ae60",
    "10mo": "#f39c12",
    "12mo": "#e74c3c",
}

for col_name in tenor_df.columns:
    fig_tenor.add_trace(go.Scatter(
        x=date_labels,
        y=tenor_df[col_name],
        mode="lines",
        name=col_name,
        stackgroup="one",
        line=dict(width=0.5),
        fillcolor=colors_map.get(col_name, "#95a5a6"),
    ))
fig_tenor.update_layout(
    title="Book Balance by Tenor Product",
    yaxis_title="₦",
    yaxis_tickformat=",.0f",
    xaxis_title="",
    height=450,
    hovermode="x unified",
)
st.plotly_chart(fig_tenor, use_container_width=True)

# Tenor breakdown table
tenor_table = tenor_df.copy()
tenor_table.index = date_labels
tenor_table = tenor_table.T
tenor_table = tenor_table.map(lambda x: f"{x:,.0f}")
st.dataframe(tenor_table, use_container_width=True)

# Secondary metrics
with st.expander("Detailed Monthly Breakdown"):
    col_left, col_right = st.columns(2)

    with col_left:
        fig_interest = go.Figure()
        fig_interest.add_trace(go.Bar(
            x=date_labels[1:],
            y=summary_df["interest_income"].iloc[1:],
            name="Interest Income",
            marker_color="#27ae60",
        ))
        fig_interest.update_layout(
            title="Monthly Interest Income",
            yaxis_title="₦",
            yaxis_tickformat=",.0f",
            height=350,
        )
        st.plotly_chart(fig_interest, use_container_width=True)

    with col_right:
        fig_defaults = go.Figure()
        fig_defaults.add_trace(go.Bar(
            x=date_labels[1:],
            y=summary_df["defaults_written_off"].iloc[1:],
            name="Defaults",
            marker_color="#e74c3c",
        ))
        fig_defaults.update_layout(
            title="Monthly Defaults / Write-offs",
            yaxis_title="₦",
            yaxis_tickformat=",.0f",
            height=350,
        )
        st.plotly_chart(fig_defaults, use_container_width=True)

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
