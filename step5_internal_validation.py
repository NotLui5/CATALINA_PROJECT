# =============================================================================================================
# INTERNAL VALIDATION USING SAVED BEST MODEL
# =============================================================================================================
# Use this section later when you have your internal validation datasets:
#
# ./database/imputation_mice/data_internal_valid_imp_1.csv
# ./database/imputation_mice/data_internal_valid_imp_2.csv
# ...
# ./database/imputation_mice/data_internal_valid_imp_10.csv
#
# Important:
# Do not refit MinMaxScaler.
# Do not refit the model.
# Use the saved fitted pipelines directly.
# =============================================================================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from step4_class import MODELS_SELECTION_DEVELOP 

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classification model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="SVC",
        help="Model name. Default: SVC"
    )

    parser.add_argument(
        "--featured",
        type=str,
        default="elasticnet",
        choices=["elasticnet", "lasso", "base"],
        help="Feature-selection method. Default: elasticnet"
    )

    parser.add_argument(
        "--model-selected-path",
        type=str,
        default=None,
        help=(
            "Path to the trained model .pkl"
        )
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "tuned"],
        help="Model type. Default: base"
    )
    
    parser.add_argument(
        "--path-data-iv",
        type=str,
        default="./database/imputation_mice/",
        help="Path to internal validation data files. Default: ./database/imputation_mice/"
    )
    
    parser.add_argument(
        "--time-followup",
        type= str,
        default = '50',
        help = "Define time in years of time_followup. Default all data."
    )
    
    return parser.parse_args()


Path("results").mkdir(exist_ok=True)
Path("results/metrics").mkdir(exist_ok=True)
Path("results/predictions").mkdir(exist_ok=True)
Path("results/plots").mkdir(exist_ok=True)

def run_internal_validation_from_saved_model(
    model_path,
    validation_file_pattern="./database/imputation_mice/data_internal_valid_imp_{i}.csv",
    threshold=0.50,
    calculate_ci=True,
    featured = "lasso" # or elasticnet
):
    saved_model = joblib.load(model_path)

    model_name = saved_model["model_name"]
    feature_cols_saved = saved_model["feature_cols"]
    fitted_pipelines = saved_model["fitted_pipelines"]
    
    valid_dfs = [
        pd.read_csv(validation_file_pattern.format(i=i))
        for i in range(1, len(fitted_pipelines) + 1)
    ]

    internal_val_pred_tables = []
    for i, df in enumerate(valid_dfs):
        if "id" not in df.columns:
            df["id"] = np.arange(len(df))
        df = pd.get_dummies(df, columns=['thy_disease_preop'], drop_first=False,dtype=float)
        df.rename({"thy_disease_preop_0": "euthyroidism", 
                    "thy_disease_preop_1": "hypothyroidism", 
                    "thy_disease_preop_2": "hyperthyroidism"},
                axis=1, inplace=True)
        valid_dfs[i] = df.copy()
        
    mdlsel = MODELS_SELECTION_DEVELOP(dfs = valid_dfs, selected_features = featured, 
                                      list_features =feature_cols_saved)
    
    for imp_idx, df_valid in enumerate(valid_dfs, start=1):
        print(f"Internal validation | {model_name} | Imputation {imp_idx}")

        fitted_pipe = fitted_pipelines[imp_idx - 1]

        X_valid_internal = df_valid[feature_cols_saved]
        y_valid_internal = df_valid["recurrence"].astype(int)

        p_valid_internal = fitted_pipe.predict_proba(X_valid_internal)[:, 1]

        internal_val_pred_tables.append(
            pd.DataFrame({
                "id": df_valid["id"].values,
                "y_true": y_valid_internal.values,
                "p_pred": p_valid_internal,
                "imputation": imp_idx
            })
        )

    internal_pooled_preds = (
        pd.concat(internal_val_pred_tables, ignore_index=True)
        .groupby("id")
        .agg(
            y_true=("y_true", "first"),
            p_pred=("p_pred", "mean")
        )
        .reset_index()
    )

    if calculate_ci:
        internal_validation_metrics = mdlsel.evaluate_predictions_with_ci(
            y_true=internal_pooled_preds["y_true"].values,
            p_pred=internal_pooled_preds["p_pred"].values,
            threshold=threshold,
            n_bins=N_BINS,
            n_bootstraps=N_BOOTSTRAPS,
            ci_level=CI_LEVEL,
            random_state=RANDOM_STATE
        )
    else:
        internal_validation_metrics = mdlsel.evaluate_predictions(
            y_true=internal_pooled_preds["y_true"].values,
            p_pred=internal_pooled_preds["p_pred"].values,
            threshold=threshold,
            n_bins=N_BINS
        )

    internal_validation_metrics["model"] = model_name
    internal_validation_metrics["validation_type"] = "internal_validation"
    internal_validation_metrics_dic = internal_validation_metrics
    internal_validation_metrics = pd.DataFrame([internal_validation_metrics])
    

    internal_validation_metrics.to_excel(
        f"results/metrics/{model_name}_internal_validation_metrics_with_ci_{featured}.xlsx",
        index=False
    )

    internal_pooled_preds.to_excel(
        f"results/predictions/{model_name}_internal_validation_pooled_predictions_{featured}.xlsx",
        index=False
    )

    internal_predictions_dict = {
        model_name: internal_pooled_preds
    }

    mdlsel.plot_roc_for_models(
        internal_predictions_dict,
        model_names=[model_name],
        save_path=f"results/plots/{model_name}_internal_validation_roc_curve_{featured}.png"
    )

    mdlsel.plot_calibration_for_models(
        internal_predictions_dict,
        model_names=[model_name],
        n_bins=10,
        save_path=f"results/plots/{model_name}_internal_validation_calibration_plot_{featured}.png"
    )

    mdlsel.plot_confusion_matrix_for_model(
        model_name,
        internal_predictions_dict,
        threshold=threshold,
        save_path=f"results/plots/{model_name}_internal_validation_confusion_matrix_{featured}.png"
    )

    return internal_validation_metrics_dic, internal_pooled_preds

# ======================================================
# 9.1. SAVING METRICS TABLES
# ======================================================
def format_metric_with_ci(results, metric, decimals=3, include_ci=True):
    """
    Format a point estimate and its bootstrap confidence interval.

    Example:
    0.840 (0.750, 0.920)
    """
    value = results.get(metric, np.nan)
    lower = results.get(f"{metric}_ci_lower", np.nan)
    upper = results.get(f"{metric}_ci_upper", np.nan)
    
    if pd.isna(value):
        return np.nan

    if not include_ci:
        return f"{float(value):.{decimals}f}"

    if pd.isna(lower) or pd.isna(upper):
        return f"{float(value):.{decimals}f}"

    return (
        f"{float(value):.{decimals}f} "
        f"({float(lower):.{decimals}f}, "
        f"{float(upper):.{decimals}f})"
    )
    
def create_evaluation_tables(results, decimals=3):
    """
    Evaluate predictions and return:

    1. 2 x 2 confusion matrix table
    2. Discrimination performance table
    3. Calibration performance table
    """
    # ----------------------------------------
    # 1. 2 x 2 table
    # ----------------------------------------
    confusion_table = pd.DataFrame(
        {
            "TP": [results.get("tp", np.nan)],
            "FP": [results.get("fp", np.nan)],
            "TN": [results.get("tn", np.nan)],
            "FN": [results.get("fn", np.nan)]
        },
        index=[results.get("model", "Unknown")]
    )

    # ----------------------------------------
    # 2. Discrimination performance
    # ----------------------------------------
    discrimination_table = pd.DataFrame(
        {
            "AUC (95% CI)": [
                format_metric_with_ci(
                    results,
                    "roc_auc_score",
                    decimals
                )
            ],
            "Sensitivity, recall": [
                format_metric_with_ci(
                    results,
                    "sensitivity",
                    decimals
                )
            ],
            "Specificity": [
                format_metric_with_ci(
                    results,
                    "specificity",
                    decimals
                )
            ],
            "Accuracy": [
                format_metric_with_ci(
                    results,
                    "accuracy",
                    decimals
                )
            ],
            "PPV, precision": [
                format_metric_with_ci(
                    results,
                    "ppv",
                    decimals
                )
            ],
            "NPV": [
                format_metric_with_ci(
                    results,
                    "npv",
                    decimals
                )
            ],
            "F1 score": [
                format_metric_with_ci(
                    results,
                    "f1",
                    decimals
                )
            ]
        },
        index=[results.get("model", "Unknown")]
    )

    # ----------------------------------------
    # 3. Calibration performance
    # ----------------------------------------
    observed = results.get("observed", np.nan)
    expected = results.get("expected", np.nan)

    calibration_table = pd.DataFrame(
        {
            "Observed": [
                int(observed) if pd.notna(observed) else np.nan
            ],
            "Expected": [
                (
                    f"{expected:.{0}f}"
                    if pd.notna(expected)
                    else np.nan
                )
            ],
            "OE ratio": [
                format_metric_with_ci(
                    results,
                    "oe_ratio",
                    decimals
                )
            ],
            "Brier score": [
                format_metric_with_ci(
                    results,
                    "brier_score",
                    decimals
                )
            ],
            "Log loss": [
                format_metric_with_ci(
                    results,
                    "log_loss",
                    decimals
                )
            ],
            "ECE": [
                format_metric_with_ci(
                    results,
                    "ece",
                    decimals
                )
            ],
            "Calibration intercept": [
                format_metric_with_ci(
                    results,
                    "calibration_intercept",
                    decimals
                )
            ],
            "Calibration slope": [
                format_metric_with_ci(
                    results,
                    "calibration_slope",
                    decimals
                )
            ]
        },
        index=[results.get("model", "Unknown")]
    )

    return {
        "confusion_table": confusion_table,
        "discrimination_table": discrimination_table,
        "calibration_table": calibration_table
    }
    
def append_table_to_excel(table, filepath, model, featured,model_type):
    """
    Append a model result to an Excel file.

    If the same combination of Model, Featured, and Model type already
    exists, the previous row is replaced.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    model_type = str(model_type).lower()

    if model_type not in {"base", "tuned"}:
        raise ValueError("model_type must be 'base' or 'tuned'.")

    new_table = table.copy().reset_index(drop=True)

    new_table.insert(0, "Model type", model_type)
    new_table.insert(0, "Featured", featured)
    new_table.insert(0, "Model", model)

    if filepath.exists():
        existing_table = pd.read_excel(filepath)

        required_columns = {"Model", "Featured", "Model type"}

        if required_columns.issubset(existing_table.columns):
            duplicate = (
                existing_table["Model"]
                .astype(str)
                .eq(str(model))
                &
                existing_table["Featured"]
                .astype(str)
                .eq(str(featured))
                &
                existing_table["Model type"]
                .astype(str)
                .str.lower()
                .eq(model_type)
            )

            existing_table = existing_table.loc[~duplicate]

        final_table = pd.concat(
            [existing_table, new_table],
            ignore_index=True
        )

    else:
        final_table = new_table

    final_table.to_excel(filepath, index=False)

    return final_table

if __name__ == "__main__":
    args = parse_args()

    MODEL = args.model
    FEATURED = args.featured
    RANDOM_STATE = 0
    N_BINS = 5
    N_BOOTSTRAPS = 1000
    CI_LEVEL = 0.95
    model_selected_path = args.model_selected_path
    MODEL_TYPE = args.model_type
    path_data_iv = args.path_data_iv
    time_followup = int(args.time_followup)
    
    if time_followup >= 50:
        path_data_internal_validation = path_data_iv + "data_internal_valid_imp_{i}.csv"
        path_confusion_table = "results/metrics/confusion_table.xlsx"
        path_discrimination_table = "results/metrics/discrimination_table.xlsx"
        path_calibration_table = "results/metrics/calibration_table.xlsx"
    else:
        path_data_internal_validation = path_data_iv + f"data{time_followup}ly" + "_internal_valid_imp_{i}.csv"
        path_confusion_table = f"results/metrics/confusion{time_followup}ly_table.xlsx"
        path_discrimination_table = f"results/metrics/discrimination{time_followup}ly_table.xlsx"
        path_calibration_table = f"results/metrics/calibration{time_followup}ly_table.xlsx"
        
        
    print("Configuration:")
    print(f"MODEL: {MODEL}")
    print(f"FEATURED: {FEATURED}")
    print(f"model_selected_path: {model_selected_path}")
    print(f"RANDOM_STATE: {RANDOM_STATE}")
    print(f"N_BINS: {N_BINS}")
    print(f"N_BOOTSTRAPS: {N_BOOTSTRAPS}")
    print(f"CI_LEVEL: {CI_LEVEL}")
    
    # To run internal validation later, uncomment this:
    internal_metrics_dict, internal_preds = run_internal_validation_from_saved_model(
        model_path=model_selected_path,
        validation_file_pattern=path_data_internal_validation,
        threshold=0.50,
        calculate_ci=True,
        featured=FEATURED
    )
    
    
    tables = create_evaluation_tables(internal_metrics_dict, decimals=3)
    confusion_results = append_table_to_excel(table=tables["confusion_table"], 
                                              filepath=path_confusion_table,
                                              model=MODEL,
                                              featured=FEATURED,
                                              model_type=MODEL_TYPE)

    discrimination_results = append_table_to_excel(table=tables["discrimination_table"],
                                                   filepath=path_discrimination_table,
                                                   model=MODEL, 
                                                   featured=FEATURED,
                                                   model_type=MODEL_TYPE)

    calibration_results = append_table_to_excel(table=tables["calibration_table"],
                                                filepath= path_calibration_table,
                                                model=MODEL,
                                                featured=FEATURED,
                                                model_type=MODEL_TYPE)