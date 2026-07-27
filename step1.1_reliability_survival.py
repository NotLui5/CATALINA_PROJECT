# Based on: https://github.com/gal-a/AppliedStatisticsForDataScience/blob/main/AppliedStatisticsForDataScience_08.ipynb
# Read more: https://towardsdatascience.com/survival-analysis-for-data-drift-and-ml-reliability/?utm_campaign=tds+daily&utm_medium=email&_hsenc=p2ANqtz-8S3rob-XzjGHXjXijV3AUMWSiRCzZW2SH8eWbMH_3fZnEcTq18hd566et1sSourjp9tzFeQHAFmbMtSFDD_SG_oHn-MQ&_hsmi=427545621&utm_source=newsletter
# ============================================
# Visualizations for Recurrent of Thyroid Cancer Dataset
# ============================================
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse 
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description = "It returns reliability based on survival analysis graphs."
    )

    parser.add_argument(
        "--time-followup",
        type=int,
        default=50,
        help="Time in years about following up max allowed. Default is 50 years (max reported 573.2 months)."
    )
    
    parser.add_argument(
        "--datalocated-fold",
        type=str,
        default = "database",
        help = "Path to the folder containing the datasets. Default is 'database'."
    )
    
    return parser.parse_args()
    

def export_plot(fig, plot_id, dpi=300, width=8, height=5, outdir="results"):
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.87, bottom=0.15)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{plot_id}.png", dpi=dpi, bbox_inches=None, pad_inches=0.45)
    
if __name__ == "__main__":
    args = parse_args()
    
    datalocated_fold = args.datalocated_fold
    time_followup = args.time_followup
    
    path_categorical_vars = './variable_selection/all_categorical_vars.txt'
    path_outdir = "results/reliabilty_survival"
    name_survival_distrib = "survival_dataset_plots"
    name_plot_joined = "sim_hazard_based_insights"    
    
    if time_followup >= 50:
        path_df = f"./{datalocated_fold}/database_complete.csv"
        path_to_cox_train_data = "./database/data_train_mice10.csv"
    else:
        path_df = f"./{datalocated_fold}/database_{time_followup}ly.csv"
        path_to_cox_train_data = f"./database/data{time_followup}ly_train_mice10.csv"
    
    thyroid_df = pd.read_csv(path_df)
    thyroid_df_to_cox = pd.read_csv(path_to_cox_train_data)
    thyroid_df_to_cox.info()
    with open(path_categorical_vars, "r", encoding="utf-8") as file:
        categorical_vars = [line.strip() for line in file if line.strip()]
        categorical_vars.remove("recurrence")
        categorical_vars.remove("thyrodectomy_approach")

    # Standardize event column name
    thyroid_df = thyroid_df.rename(columns={"recurrence": "event", "follow_months": "time"})
    thyroid_df_to_cox = thyroid_df_to_cox.rename(columns={"recurrence": "event", "follow_months": "time"})
    
    # Convert categorical variables to numeric (Cox PH cannot handle strings)
    thyroid_df = pd.get_dummies(thyroid_df, columns= categorical_vars, drop_first=True)
    thyroid_df_to_cox = pd.get_dummies(thyroid_df_to_cox, columns= categorical_vars.append("thyrodectomy_approach"), drop_first=True)
    
    # Dark pastel colors
    dark_pastel_blue = "#3b5f82"
    dark_pastel_red  = "#c96a6a"

    # Common grid style
    grid_style = dict(axis="x", linestyle="--", alpha=0.4)

    # Shared x-axis range
    time_min = thyroid_df["time"].min()
    time_max = thyroid_df["time"].max()

    # Common x-axis ticks
    x_ticks = range(0, int(time_max) + 1, 60)

    # Create combined figure
    fig = plt.figure(figsize=(8, 7))

    # --------------------------------------------
    # Visualization 1: Distribution of Survival Times
    # --------------------------------------------
    ax1 = fig.add_subplot(3, 1, 1)

    ax1.hist(
        [
            thyroid_df[thyroid_df["event"] == 1]["time"],   # Recurrences
            thyroid_df[thyroid_df["event"] == 0]["time"]    # Censored
        ],
        bins=30,
        stacked=True,
        color=[dark_pastel_red, dark_pastel_blue],
        edgecolor="black",
        alpha=0.85,
        label=["Recurrence", "Censored"]
    )

    ax1.set_title("Distribution of Survival Times Red=Recurrence Blue=Censored")
    ax1.set_xlabel("Survival Time (months)")
    ax1.set_ylabel("Count")
    ax1.set_xlim(time_min, time_max)
    ax1.set_xticks(x_ticks)
    ax1.grid(**grid_style)
    ax1.legend(loc="upper right", markerscale=1.0)

    # --------------------------------------------
    # Visualization 2: Events vs Censoring
    # --------------------------------------------
    ax2 = fig.add_subplot(3, 1, 2)

    ax2.scatter(
        thyroid_df[thyroid_df["event"] == 1]["time"],
        thyroid_df[thyroid_df["event"] == 1]["event"],
        color=dark_pastel_red,
        alpha=0.6,
        s=14,
        linewidth=0,
        label="Recurrence"
    )

    ax2.scatter(
        thyroid_df[thyroid_df["event"] == 0]["time"],
        thyroid_df[thyroid_df["event"] == 0]["event"],
        color=dark_pastel_blue,
        alpha=0.6,
        s=14,
        linewidth=0,
        label="Censored"
    )

    ax2.set_title("Events and Censoring \n Thyroid Cancer Recurrence Dataset")
    ax2.set_xlabel("Survival Time (months)")
    ax2.set_ylabel("Event Indicator")
    ax2.set_yticks([0, 1])
    ax2.set_xlim(time_min, time_max)
    ax2.set_xticks(x_ticks)
    ax2.grid(**grid_style)
    ax2.legend(loc="center right", markerscale=1.0)

    # --------------------------------------------
    # Visualization 3: Survival Time by trt = thyrodectomy_approach (Total vs Total + lymphadenoctomy)
    # --------------------------------------------
    ax3 = fig.add_subplot(3, 1, 3)

    treatments = sorted(thyroid_df["thyrodectomy_approach"].unique())
    colors = [dark_pastel_blue, dark_pastel_red]
    
    for i, treatment in enumerate(treatments):
        subset = thyroid_df[thyroid_df["thyrodectomy_approach"] == treatment]
        
        treatment_label = ("Total" if treatment == 0 else "Total + Lymphadenectomy")
            
        ax3.hist(
            subset["time"],
            bins=30,
            alpha=0.55,
            color=colors[i % len(colors)],
            label=f"Thyroidectomy Approach {treatment_label}",
            edgecolor="black"
        )

    ax3.set_title("Survival Time by Thyroidectomy Approach Group")
    ax3.set_xlabel("Survival Time (months)")
    ax3.set_ylabel("Count")
    ax3.set_xlim(time_min, time_max)
    ax3.set_xticks(x_ticks)
    ax3.grid(**grid_style)
    ax3.legend(loc="center right", markerscale=1.0)

    # Increase vertical spacing between rows
    plt.subplots_adjust(hspace=0.55)

    plt.tight_layout()

    export_plot(fig, name_survival_distrib, outdir=path_outdir, height=7)
    
    # ============================================================================================
    # Hazard-Based Insights for ML Reliability
    # ============================================================
    # 3x2 Survival Analysis Panel (8in × 7in)
    # ============================================================

    # Darker pastel palette
    dark_pastel_blue    = "#2f4d68"
    dark_pastel_red     = "#a84f4f"
    dark_pastel_purple  = "#5d4670"
    dark_pastel_green   = "#5f8f6b"
    dark_pastel_orange  = "#b36b3c"

    # --------------------------------------------
    # Fit models
    # --------------------------------------------
    km = KaplanMeierFitter()
    na = NelsonAalenFitter()
    subset_km_na = thyroid_df.dropna(subset=["time"])
    km.fit(subset_km_na["time"], subset_km_na["event"])
    na.fit(subset_km_na["time"], subset_km_na["event"])

    # --------------------------------------------
    # Exact Hazard at Event Times (NA Jump Sizes)
    # --------------------------------------------
    t_na = na.cumulative_hazard_.index.values
    H_na = na.cumulative_hazard_["NA_estimate"].values

    dH_exact = np.diff(H_na)
    t_exact = t_na[1:]

    # --------------------------------------------
    # Additional quantities
    # --------------------------------------------
    event_table = km.event_table

    t_risk = event_table.index.values
    n_at_risk = event_table["at_risk"].values

    cum_fail = event_table["observed"].cumsum().values
    t_fail = event_table.index.values

    # --------------------------------------------
    # Create 3x2 panel
    # --------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(8, 7))

    # Apply NASA-style formatting
    for ax in axes.flatten():
        ax.set_facecolor("white")
        ax.grid(True, which="both", axis="both",
                color="#b0b0b0", linestyle="-", linewidth=0.9, alpha=0.85)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("gray")

    # --------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------

    # 1A. Failure Counts
    bins = np.linspace(0, subset_km_na["time"].max(), 60)
    axes[0, 0].hist(
        subset_km_na.loc[subset_km_na["event"] == 1, "time"],
        bins=bins,
        color=dark_pastel_red,
        alpha=0.35,
        edgecolor="none"
    )
    axes[0, 0].set_title("Failure Counts Over Time")
    axes[0, 0].set_ylabel("Count")

    # 2A. Cumulative Failure Counts
    axes[1, 0].step(t_fail, cum_fail, where="post", color=dark_pastel_red, alpha=0.85)
    axes[1, 0].set_title("Cumulative Failure Counts")
    axes[1, 0].set_ylabel("Cumulative")

    # 3A. Number at Risk
    axes[2, 0].step(t_risk, n_at_risk, where="post", color=dark_pastel_purple)
    axes[2, 0].set_title("Number at Risk Over Time")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("At Risk")

    # --------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------

    # 1B. Survival Function
    axes[0, 1].plot(
        km.survival_function_.index,
        km.survival_function_["KM_estimate"],
        color=dark_pastel_blue
    )
    axes[0, 1].set_title("Survival Function Kaplan–Meier")
    axes[0, 1].set_ylabel("Survival Probability")

    # 2B. Cumulative Hazard
    axes[1, 1].plot(
        na.cumulative_hazard_.index,
        na.cumulative_hazard_["NA_estimate"],
        color=dark_pastel_green
    )
    axes[1, 1].set_title("Cumulative Hazard Nelson–Aalen")
    axes[1, 1].set_ylabel("Cumulative Hazard")

    # 3B. Exact Hazard (NA Jump Sizes)
    axes[2, 1].step(
        t_exact,
        dH_exact,
        where="post",
        color=dark_pastel_orange
    )
    axes[2, 1].set_title("Exact Hazard Function NA Jumps")
    axes[2, 1].set_xlabel("Time")
    axes[2, 1].set_ylabel("Hazard Jump")

    # Increase vertical spacing between rows
    plt.subplots_adjust(hspace=0.70)

    plt.tight_layout()

    export_plot(fig, name_plot_joined, outdir=path_outdir, height=7)
        

    # ============================================================
    # Spot‑Check: Last 3 Failures, Risk Set Size, and Exact Hazard
    # ============================================================

    # Step 1 — Extract the last three failures
    last_three_failures = (
        subset_km_na[subset_km_na["event"] == 1]
        .sort_values("time")
        .tail(3)
    )

    t1, t2, t3 = last_three_failures["time"].values

    # Step 2 — Number at risk at those times
    event_table = km.event_table
    n1 = event_table.loc[t1, "at_risk"]
    n2 = event_table.loc[t2, "at_risk"]
    n3 = event_table.loc[t3, "at_risk"]

    # Step 3 — Exact hazard at those times
    idx1 = np.where(t_exact == t1)[0][0]
    idx2 = np.where(t_exact == t2)[0][0]
    idx3 = np.where(t_exact == t3)[0][0]

    h1 = dH_exact[idx1]
    h2 = dH_exact[idx2]
    h3 = dH_exact[idx3]

    # Step 4 — Summary table
    print("=== Spot‑Check: Last Three Failures (Exact Hazard Values) ===\n")

    summary = pd.DataFrame({
        "failure_time": [t1, t2, t3],
        "n_at_risk": [n1, n2, n3],
        "exact_hazard_jump": [h1, h2, h3]
    })

    print(summary.to_markdown(index=False))

    # ====================================================================================
    # Cox Proportional Hazards Model
    # ====================================================================================
    
    # Fit model
    cph = CoxPHFitter()
    cph.fit(thyroid_df_to_cox, duration_col="time", event_col="event", weights_col="w")

    # Helper for rounding values safely
    def fmt(x):
        try:
            return f"{float(x):.3f}"
        except:
            return x

    # ---------------------------------------------------------
    # 1. UNIFIED SUMMARY TABLE (Model Summary + Global Tests)
    # ---------------------------------------------------------
    n_obs = cph._n_examples
    n_events = cph.event_observed.sum()
    partial_ll = cph._model.log_likelihood_

    # LR test
    lrt = cph.log_likelihood_ratio_test()
    lr_stat = fmt(lrt.test_statistic)
    lr_df = lrt.degrees_freedom
    lr_p = fmt(lrt.p_value)
    lr_log2p = fmt(-np.log2(lrt.p_value))

    summary_rows = [
        ["Model Summary", "model", "lifelines.CoxPHFitter"],
        ["Model Summary", "duration col", repr(cph.duration_col)],
        ["Model Summary", "event col", repr(cph.event_col)],
        ["Model Summary", "baseline estimation", "breslow"],
        ["Model Summary", "number of observations", n_obs],
        ["Model Summary", "number of events observed", n_events],
        ["Model Summary", "partial log-likelihood", fmt(partial_ll)],

        ["Global Tests", "Concordance", fmt(cph.concordance_index_)],
        ["Global Tests", "Partial AIC", fmt(cph.AIC_partial_)],
        ["Global Tests", "log-likelihood ratio test", f"{lr_stat} on {lr_df} df"],
        ["Global Tests", "-log2(p) of LR test", lr_log2p],
    ]

    summary_df = pd.DataFrame(summary_rows, columns=["Section", "Metric", "Value"])
    summary_md = summary_df.to_markdown(index=False)

    # ---------------------------------------------------------
    # 2. COEFFICIENTS TABLE (3 decimals, CLEAN MAPPING, TRANSPOSED, SPLIT)
    # ---------------------------------------------------------
    coef_df = cph.summary.copy()

    # Round numeric columns
    for col in coef_df.columns:
        if pd.api.types.is_numeric_dtype(coef_df[col]):
            coef_df[col] = coef_df[col].round(3)

    # Clean feature names
    clean_names = []
    for cov in coef_df.index:
        name = cov.lower()
        clean_names.append(name)

    coef_df.index = clean_names

    # Transpose
    coef_df = coef_df.T
    coef_df.index.name = "Metric"
    coef_df = coef_df.reset_index()

    # Split into two tables: first 5 features, next 6
    first_cols  = ["Metric"] + clean_names[:5]
    second_cols = ["Metric"] + clean_names[5:]

    coef_df_1 = coef_df[first_cols]
    coef_df_2 = coef_df[second_cols]

    coef_md_1 = coef_df_1.to_markdown(index=False)
    coef_md_2 = coef_df_2.to_markdown(index=False)

    # ---------------------------------------------------------
    # PRINT ALL TABLES
    # ---------------------------------------------------------
    print("# Cox Model Summary & Global Tests (Unified)\n")
    print(summary_md)

    print("\n\n# Cox Model Coefficients (Features 1–5)\n")
    print(coef_md_1)

    print("\n\n# Cox Model Coefficients (Features 6–11)\n")
    print(coef_md_2)