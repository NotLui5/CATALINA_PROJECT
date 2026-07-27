## =============================================================================================================
# ### 1: PACKAGES:
## =============================================================================================================

import json
import joblib
import warnings
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import zip_longest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    log_loss,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Output folders
Path("results").mkdir(exist_ok=True)
Path("results/models").mkdir(exist_ok=True)
Path("results/plots").mkdir(exist_ok=True)
Path("results/features").mkdir(exist_ok=True)
Path("results/predictions").mkdir(exist_ok=True)
Path("results/metrics").mkdir(exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Define time follow up in years, and variables selected."
    )
    parser.add_argument(
        "--time_followup", 
        type=int, 
        default=50, 
        help="Time follow up in years. Default 50 years (max reported 573.2 months).")
    
    parser.add_argument(
        "--features",
        type=str,
        default="elasticnet",
        help="type of features selected. lasso or elasticnet. Default is elasticnet"        
    )
    
    parser.add_argument(
        "--datalocated_fold", 
        type=str, 
        default="database/imputation_mice", 
        help="Folder containing the data")

    return parser.parse_args()


class MODELS_SELECTION_DEVELOP():
    
    # ==========================================================================
    # 2. SETTINGS
    # ==========================================================================
    def __init__(self, dfs, selected_features = "LASSO", list_features=list):        
        """Instance with dfs with inner dataset separated + str of model 
        selection features + lst of features used"""
        
        self.dfs = dfs
        self.SELECTED_FEATURES = selected_features.lower() #OR LASSO OR ELASTICNET
        self.feature_cols = list_features
        self.RANDOM_STATE = 0
        self.THRESHOLD = 0.50
        self.N_BINS = 5

        # Bootstrap settings for 95% confidence intervals
        self.N_BOOTSTRAPS = 1000
        self.CI_LEVEL = 0.95
    
    def model_enabled(self):
        self.models = {
            "LogisticRegression": LogisticRegression(
                max_iter=5000, penalty='l2', C=1.0, 
                solver='liblinear', class_weight="balanced", random_state=0,
                ), 
            "RandomForest": RandomForestClassifier(
                n_estimators=500, class_weight="balanced", max_depth=None,
                min_samples_split=2, min_samples_leaf=1, random_state=0, n_jobs=-1
                ),
            "KNN": KNeighborsClassifier(
                n_neighbors=15, weights='uniform'
                ),
            "SVC": SVC(
                kernel="rbf", probability=True, C=1.0,
                class_weight="balanced", random_state=0
                ),
            "GaussianNB": GaussianNB(
                var_smoothing = 1e-9
                ),
            "XGBoost": XGBClassifier(
                n_estimators=500, learning_rate=0.03, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, eval_metric="auc", #logloss for classification
                random_state=0, n_jobs=-1
                ),
            "LightGBM": LGBMClassifier(
                n_estimators=500, learning_rate=0.03, num_leaves=31,
                class_weight="balanced", random_state=0, n_jobs=-1,
                verbose=-1
                ),
            "CatBoost": CatBoostClassifier(
                iterations=500, learning_rate=0.03, depth=4,
                loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
                random_seed=0, verbose=0
                )
        }
        
        # ===================================================
        # HYPERPARAMETER PARAMETERS
        # ===================================================

        self.PARAM_DISTS = {
            "LogisticRegression": {
                "model__C": np.logspace(-3, 3, 10),
                "model__solver": ["liblinear", "saga"],
                "model__max_iter": [1000, 3000, 5000]
            },

            "RandomForest": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 3, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", "log2", None]
            },

            "KNN": {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            },

            "SVC": {
                "model__C": np.logspace(-3, 3, 10),
                "model__gamma": ["scale", "auto"]
            },

            "GaussianNB": {
                "model__var_smoothing": np.logspace(-9, -7, 10)
            },

            "XGBoost": {
                "model__n_estimators": [200, 500, 800],
                "model__learning_rate": np.logspace(-3, -0.5, 10),
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0]
            },


            "LightGBM": {
                "model__n_estimators": [200, 500, 800],
                "model__learning_rate": np.logspace(-3, -0.5, 10),
                "model__num_leaves": [7, 15, 23, 31],
                "model__max_depth": [-1, 3, 5, 10],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0]
            },

            "CatBoost": {
                "model__iterations": [200, 500, 800],
                "model__learning_rate": np.logspace(-3, -0.5, 10),
                "model__depth": [2, 3, 4, 5, 6],
                "model__l2_leaf_reg": [1, 3, 5, 10]
            }
        }
            
        f"Models enabled: {list(self.models.keys())}"
        return self.models, self.PARAM_DISTS

    # ==============================================================
    # 4. TRAIN / DEVELOP SPLIT
    # ==============================================================
    def sep_train_and_develop_data(self):
        id_target = self.dfs[0][["id", "recurrence"]].copy()

        train_ids, valid_ids = train_test_split(
            id_target["id"],
            test_size=0.176,
            random_state=self.RANDOM_STATE,
            stratify=id_target["recurrence"]
        )

        self.train_ids = set(train_ids)
        self.valid_ids = set(valid_ids)

        print("N train:", len(self.train_ids))
        print("N validation:", len(self.valid_ids))

    # =================================================
    # 5. LASSO or ELASTICNET SELECTED FEATURES
    # =================================================
    def save_features(self, path_save_features = "results/features/{selector}_selected_features.json"):
        
        selected_features_info = {
            "selection_method": self.SELECTED_FEATURES,
            "n_features": len(self.feature_cols),
            "selected_features": self.feature_cols,
        }

        with open(path_save_features.format(selector=self.SELECTED_FEATURES), "w") as f:
            json.dump(selected_features_info, f, indent=4)

        joblib.dump(
            self.feature_cols,
            f"results/features/{self.SELECTED_FEATURES}_selected_features.pkl"
        )

        print(f"{self.SELECTED_FEATURES}-selected features saved.")

    # =============================================
    # 7. PIPELINE FUNCTION
    # =============================================
    def make_pipeline(self, estimator):
        """
        All models use MinMaxScaler.

        Important:
        The scaler is inside the pipeline, so during cross-validation it is fitted
        only on the training fold. This avoids data leakage.
        """
        return Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("model", clone(estimator))
            ]
        )

    # ================================================
    # 8. METRICS FUNCTIONS
    # ================================================
    def calculate_ece(self, y_true, p_pred, n_bins=5):
        """
        Expected Calibration Error.
        """
        y_true = np.asarray(y_true).astype(int)
        p_pred = np.asarray(p_pred).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
            mask = (p_pred >= bin_lower) & (p_pred < bin_upper)

            if np.sum(mask) > 0:
                bin_confidence = np.mean(p_pred[mask])
                bin_accuracy = np.mean(y_true[mask])
                bin_weight = np.sum(mask) / len(y_true)

                ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

        return ece


    def calculate_calibration_metrics(self,y_true, p_pred, n_bins=5):
        """
        Calibration metrics:
        - Observed events
        - Expected events
        - Observed:Expected ratio
        - Brier score
        - Log loss
        - Expected Calibration Error
        - Calibration intercept
        - Calibration slope
        """
        y_true = np.asarray(y_true).astype(int)
        p_pred = np.asarray(p_pred).astype(float)

        eps = 1e-6
        p_pred = np.clip(p_pred, eps, 1 - eps)

        observed = np.sum(y_true)
        expected = np.sum(p_pred)
        oe_ratio = observed / expected if expected > 0 else np.nan

        brier = brier_score_loss(y_true, p_pred)
        lloss = log_loss(y_true, p_pred)
        ece = self.calculate_ece(y_true, p_pred, n_bins=n_bins)

        if len(np.unique(y_true)) == 2:
            logit_p = np.log(p_pred / (1 - p_pred)).reshape(-1, 1)

            try:
                cal_model = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=5000
                )
                cal_model.fit(logit_p, y_true)

                calibration_intercept = cal_model.intercept_[0]
                calibration_slope = cal_model.coef_[0][0]

            except Exception:
                calibration_intercept = np.nan
                calibration_slope = np.nan

        else:
            calibration_intercept = np.nan
            calibration_slope = np.nan

        return {
            "observed": observed,
            "expected": expected,
            "oe_ratio": oe_ratio,
            "brier_score": brier,
            "log_loss": lloss,
            "ece": ece,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope
        }


    def evaluate_predictions(self, y_true, p_pred, threshold=0.50, n_bins=5):
        """
        Calculate classification, discrimination, and calibration metrics.

        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        PPV         = TP / (TP + FP)
        NPV         = TN / (TN + FN)
        """
        y_true = np.asarray(y_true).astype(int)
        p_pred = np.asarray(p_pred).astype(float)

        y_pred = (p_pred >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "threshold": threshold
        }

        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, p_pred)

            results.update({
                "roc_auc_score": roc_auc_score(y_true, p_pred),
                "auc": auc(fpr, tpr)
            })

        else:
            results.update({
                "roc_auc_score": np.nan,
                "auc": np.nan
            })

        calibration_results = self.calculate_calibration_metrics(
            y_true,
            p_pred,
            n_bins=n_bins
        )

        results.update(calibration_results)

        return results
    # ===================================================
    # 9. BOOTSTRAP 95% CONFIDENCE INTERVALS
    # ===================================================
    def bootstrap_metric_ci(
        self,
        y_true,
        p_pred,
        threshold=0.50,
        n_bins=5,
        n_bootstraps=1000,
        ci_level=0.95,
        random_state=0
    ):
        """
        Bootstrap percentile confidence intervals for performance metrics.

        Applied to:
        - accuracy
        - precision / PPV
        - recall / sensitivity
        - specificity
        - NPV
        - f1
        - roc_auc_score
        - auc
        - brier_score
        - log_loss
        - ece
        - oe_ratio
        - calibration_intercept
        - calibration_slope

        Not applied to:
        - tn, fp, fn, tp
        - observed, expected

        These are counts rather than normalized performance measures.
        """
        rng = np.random.default_rng(random_state)

        y_true = np.asarray(y_true).astype(int)
        p_pred = np.asarray(p_pred).astype(float)

        if len(y_true) != len(p_pred):
            raise ValueError("y_true and p_pred must have the same length.")

        if len(y_true) == 0:
            raise ValueError("y_true and p_pred cannot be empty.")

        n = len(y_true)

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "roc_auc_score",
            "auc",
            "brier_score",
            "log_loss",
            "ece",
            "oe_ratio",
            "calibration_intercept",
            "calibration_slope"
        ]

        boot_values = {metric: [] for metric in metric_names}

        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)

            y_boot = y_true[idx]
            p_boot = p_pred[idx]

            try:
                metrics_boot = self.evaluate_predictions(
                    y_boot,
                    p_boot,
                    threshold=threshold,
                    n_bins=n_bins
                )

                for metric in metric_names:
                    value = metrics_boot.get(metric, np.nan)
                    boot_values[metric].append(value)

            except Exception:
                for metric in metric_names:
                    boot_values[metric].append(np.nan)

        alpha = 1 - ci_level
        lower_q = 100 * alpha / 2
        upper_q = 100 * (1 - alpha / 2)

        ci_results = {}

        for metric in metric_names:
            values = np.asarray(boot_values[metric], dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                ci_results[f"{metric}_ci_lower"] = np.nan
                ci_results[f"{metric}_ci_upper"] = np.nan
            else:
                ci_results[f"{metric}_ci_lower"] = np.percentile(
                    values,
                    lower_q
                )
                ci_results[f"{metric}_ci_upper"] = np.percentile(
                    values,
                    upper_q
                )

        return ci_results

    #It is neccesary??????????????????? yes, it seems
    def evaluate_predictions_with_ci(self, y_true, p_pred, threshold=0.50, n_bins=5, 
                                     n_bootstraps=1000, ci_level=0.95, random_state=0):
        """
        Point estimates + bootstrap 95% confidence intervals.
        """
        point_metrics = self.evaluate_predictions(
            y_true=y_true,
            p_pred=p_pred,
            threshold=threshold,
            n_bins=n_bins
        )

        ci_metrics = self.bootstrap_metric_ci(
            y_true=y_true,
            p_pred=p_pred,
            threshold=threshold,
            n_bins=n_bins,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            random_state=random_state
        )

        point_metrics.update(ci_metrics)

        return point_metrics
        
        
    # ======================================================
    # 10. SAFE CROSS-VALIDATION
    # ======================================================

    def make_safe_stratified_cv(self, y, desired_splits=5, random_state=0):
        """
        Prevents CV errors when the minority class is too small.
        """
        y = pd.Series(y)
        min_class_count = y.value_counts().min()

        n_splits = min(desired_splits, min_class_count)

        if n_splits < 2:
            raise ValueError(
                f"Cannot perform stratified CV. "
                f"The smallest class has only {min_class_count} observation(s)."
            )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )


    # =========================================================
    # 11. BASE MODEL BENCHMARKING ACROSS IMPUTATIONS
    # =========================================================
    def unique_outcome(self, series):
        values = series.dropna().unique()

        if len(values) != 1:
            raise ValueError(
                f"Inconsistent y_true for the same ID: {values.tolist()}"
            )

        return values[0]

    def benchmark_model_across_imputations(self, model_name, estimator, dfs, threshold=0.50, 
                                           calculate_ci=True):
        per_imp_rows = []
        val_pred_tables = []
        fitted_models = []

        for imp_idx, df in enumerate(dfs, start=1):
            print(f"{model_name} | Base model | Imputation {imp_idx}")

            train_df = df[df["id"].isin(self.train_ids)].copy()
            valid_df = df[df["id"].isin(self.valid_ids)].copy()

            X_train = train_df[self.feature_cols]
            y_train = train_df["recurrence"].astype(int)

            X_valid = valid_df[self.feature_cols]
            y_valid = valid_df["recurrence"].astype(int)

            pipe = self.make_pipeline(estimator)

            pipe.fit(X_train, y_train)

            p_valid = pipe.predict_proba(X_valid)[:, 1]

            ## It will be exported
            metrics_imp = self.evaluate_predictions(
                y_true=y_valid,
                p_pred=p_valid,
                threshold=threshold,
                n_bins=self.N_BINS
            )

            metrics_imp["model"] = model_name
            metrics_imp["imputation"] = imp_idx

            per_imp_rows.append(metrics_imp)

            val_pred_tables.append(
                pd.DataFrame({
                    "id": valid_df["id"].values,
                    "y_true": y_valid.values,
                    "p_pred": p_valid,
                    "imputation": imp_idx
                })
            )

            fitted_models.append(pipe)

        pooled_preds = (
            pd.concat(val_pred_tables, ignore_index=True)
            .groupby("id")
            .agg( 
                y_true=("y_true", lambda x: self.unique_outcome(x)),###################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! serious weak, "first" here all must be equal predictions, but "mode" is taked when it differs
                p_pred=("p_pred", "mean")
            )
            .reset_index()
        )

        if calculate_ci:
            pooled_metrics = self.evaluate_predictions_with_ci(
                y_true=pooled_preds["y_true"].values,
                p_pred=pooled_preds["p_pred"].values,
                threshold=threshold,
                n_bins=self.N_BINS,
                n_bootstraps=self.N_BOOTSTRAPS,
                ci_level=self.CI_LEVEL,
                random_state=self.RANDOM_STATE
            )
        else:
            pooled_metrics = self.evaluate_predictions(
                y_true=pooled_preds["y_true"].values,
                p_pred=pooled_preds["p_pred"].values,
                threshold=threshold,
                n_bins=self.N_BINS
            )

        pooled_metrics["model"] = model_name
        pooled_metrics["model_type"] = "base"
        pooled_metrics["n_imputations"] = len(dfs)

        return pooled_metrics, pd.DataFrame(per_imp_rows), pooled_preds, fitted_models



    # =============================================================================================================
    # 15. PLOT FUNCTIONS
    # =============================================================================================================
    def plot_roc_for_models(self, pooled_predictions, model_names=None, save_path=None):
        if model_names is None:
            model_names = list(pooled_predictions.keys())

        plt.figure(figsize=(7, 6))

        for name in model_names:
            df_pred = pooled_predictions[name]

            y_true = df_pred["y_true"].values
            p_pred = df_pred["p_pred"].values

            fpr, tpr, _ = roc_curve(y_true, p_pred)
            model_auc = roc_auc_score(y_true, p_pred)

            plt.plot(
                fpr,
                tpr,
                label=f"{name} AUC={model_auc:.3f}"
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Chance"
        )

        plt.xlabel("1 - Specificity / False Positive Rate")
        plt.ylabel("Sensitivity / True Positive Rate")
        plt.title("ROC curves - pooled predictions across imputations")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


    def plot_calibration_for_models(self, pooled_predictions, model_names=None, n_bins=10, save_path=None):
        if model_names is None:
            model_names = list(pooled_predictions.keys())

        fig, ax = plt.subplots(figsize=(9, 8))

        metric_lines = []

        for name in model_names:
            df_pred = pooled_predictions[name]

            y_true = df_pred["y_true"].values
            p_pred = df_pred["p_pred"].values

            frac_pos, mean_pred = calibration_curve(
                y_true,
                p_pred,
                n_bins=n_bins,
                strategy="quantile"
            )

            ax.plot(
                mean_pred,
                frac_pos,
                marker="o",
                label=name
            )

            # Use your existing calibration metrics function
            metrics = self.calculate_calibration_metrics(
                y_true=y_true,
                p_pred=p_pred,
                n_bins=n_bins
            )

            metric_lines.append(
                f"{name}: "
                f"Brier={metrics['brier_score']:.3f} | "
                f"LogLoss={metrics['log_loss']:.3f} | "
                f"ECE={metrics['ece']:.3f}"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration"
        )

        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event proportion")
        ax.set_title("Calibration plot - pooled predictions across imputations")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # Build bottom text block in 2 columns
        # ------------------------------------------------------------------
        half = math.ceil(len(metric_lines) / 2)
        left_col = metric_lines[:half]
        right_col = metric_lines[half:]

        footer_lines = []
        for left, right in zip_longest(left_col, right_col, fillvalue=""):
            footer_lines.append(f"{left:<65} {right}")

        footer_text = "\n".join(footer_lines)

        # Leave space at bottom and right
        plt.tight_layout(rect=[0, 0.18, 0.78, 1])

        # Add the footer text
        fig.text(
            0.02,                      # x position
            0.02,                      # y position
            footer_text,
            ha="left",
            va="bottom",
            fontsize=8,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray"
            )
        )

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        

    def plot_confusion_matrix_for_model(self, model_name, pooled_predictions, threshold=0.50, save_path=None):
        df_pred = pooled_predictions[model_name]

        y_true = df_pred["y_true"].values
        p_pred = df_pred["p_pred"].values

        y_pred = (p_pred >= threshold).astype(int)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No recurrence", "Recurrence"]
        )

        disp.plot(values_format="d")
        plt.title(f"Confusion matrix - {model_name} - threshold={threshold}")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


    # =============================================================================================================
    # 18. TUNING + VALIDATION ACROSS IMPUTATIONS
    # =============================================================================================================

    def tune_and_evaluate_across_imputations(self, model_name, estimator, dfs, param_dist, 
                                             n_iter=20, threshold=0.50, calculate_ci=True):
        per_imp_rows = []
        val_pred_tables = []
        best_params_rows = []
        fitted_tuned_models = []

        for imp_idx, df in enumerate(dfs, start=1):
            print(f"{model_name} | Tuning | Imputation {imp_idx}")

            train_df = df[df["id"].isin(self.train_ids)].copy()
            valid_df = df[df["id"].isin(self.valid_ids)].copy()

            X_train = train_df[self.feature_cols]
            y_train = train_df["recurrence"].astype(int)

            X_valid = valid_df[self.feature_cols]
            y_valid = valid_df["recurrence"].astype(int)

            pipe = self.make_pipeline(estimator)

            cv = self.make_safe_stratified_cv(
                y_train,
                desired_splits=5,
                random_state=self.RANDOM_STATE
            )

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                random_state=self.RANDOM_STATE,
                n_jobs=-1,
                refit=True,
                error_score=np.nan
            )

            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_

            p_valid = best_pipe.predict_proba(X_valid)[:, 1]

            metrics_imp = self.evaluate_predictions(
                y_true=y_valid,
                p_pred=p_valid,
                threshold=threshold,
                n_bins=self.N_BINS
            )

            metrics_imp["model"] = model_name + "_tuned"
            metrics_imp["imputation"] = imp_idx
            metrics_imp["best_cv_auc"] = search.best_score_

            per_imp_rows.append(metrics_imp)

            best_params_rows.append({
                "model": model_name,
                "imputation": imp_idx,
                "best_cv_auc": search.best_score_,
                "best_params": search.best_params_
            })

            val_pred_tables.append(
                pd.DataFrame({
                    "id": valid_df["id"].values,
                    "y_true": y_valid.values,
                    "p_pred": p_valid,
                    "imputation": imp_idx
                })
            )

            fitted_tuned_models.append(best_pipe)

        pooled_preds = (
            pd.concat(val_pred_tables, ignore_index=True)
            .groupby("id")
            .agg(
                y_true=("y_true", "first"),
                p_pred=("p_pred", "mean")
            )
            .reset_index()
        )

        if calculate_ci:
            pooled_metrics = self.evaluate_predictions_with_ci(
                y_true=pooled_preds["y_true"].values,
                p_pred=pooled_preds["p_pred"].values,
                threshold=threshold,
                n_bins=self.N_BINS,
                n_bootstraps=self.N_BOOTSTRAPS,
                ci_level=self.CI_LEVEL,
                random_state=self.RANDOM_STATE
            )
        else:
            pooled_metrics = self.evaluate_predictions(
                y_true=pooled_preds["y_true"].values,
                p_pred=pooled_preds["p_pred"].values,
                threshold=threshold,
                n_bins=self.N_BINS
            )

        pooled_metrics["model"] = model_name + "_tuned"
        pooled_metrics["model_type"] = "tuned"
        pooled_metrics["n_imputations"] = len(dfs)

        return (
            pooled_metrics,
            pd.DataFrame(per_imp_rows),
            pooled_preds,
            pd.DataFrame(best_params_rows),
            fitted_tuned_models
        )

    # =============================================================================================================
    # 25. FEATURE IMPORTANCE FOR TOP 4 FINAL MODELS
    # =============================================================================================================

    def get_feature_importance_from_pipeline(self, pipe, feature_cols):
        """
        Extract feature importance from one fitted sklearn Pipeline.

        Works for:
        - LogisticRegression: absolute coefficient values
        - RandomForest, XGBoost, LightGBM, CatBoost: feature_importances_
        - Other models: returns None
        """
        model = pipe.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = model.coef_

            # Binary LogisticRegression usually has shape (1, n_features)
            if coef.ndim == 2:
                importance = np.abs(coef[0])
            else:
                importance = np.abs(coef)

        else:
            return None

        return pd.DataFrame({
            "feature": feature_cols,
            "importance": importance
        })


    def get_pooled_feature_importance(self, model_name, fitted_models, feature_cols):
        """
        Pool feature importance across the 10 imputations.

        Returns:
        - mean importance
        - standard deviation
        - min and max importance
        """
        importance_tables = []

        for imp_idx, pipe in enumerate(fitted_models, start=1):

            imp_df = self.get_feature_importance_from_pipeline(
                pipe=pipe,
                feature_cols=feature_cols
            )

            if imp_df is None:
                print(f"{model_name}: feature importance not available.")
                return None

            imp_df["imputation"] = imp_idx
            importance_tables.append(imp_df)

        importance_all = pd.concat(importance_tables, ignore_index=True)

        pooled_importance = (
            importance_all
            .groupby("feature")
            .agg(
                mean_importance=("importance", "mean"),
                sd_importance=("importance", "std"),
                min_importance=("importance", "min"),
                max_importance=("importance", "max")
            )
            .reset_index()
            .sort_values("mean_importance", ascending=False)
        )

        pooled_importance["model"] = model_name

        return pooled_importance


    def plot_feature_importance(self, importance_df, model_name, top_n=15, save_path=None):
        """
        Plot top N pooled feature importances for one model.
        """
        plot_df = (
            importance_df
            .sort_values("mean_importance", ascending=False)
            .head(top_n)
            .sort_values("mean_importance", ascending=True)
        )

        plt.figure(figsize=(8, 6))

        plt.barh(
            plot_df["feature"],
            plot_df["mean_importance"],
            xerr=plot_df["sd_importance"]
        )

        plt.xlabel("Mean feature importance across imputations")
        plt.ylabel("Feature")
        plt.title(f"Feature importance - {model_name}")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


    def get_and_plot_feature_importance_top_models(self, final_ranking, fitted_model_store, 
                                                   tuned_model_store, feature_cols, top_n_models=4,
                                                   top_n_features=15, save_folder="results/plots",
                                                   save_tables_folder="results/features"):
        """
        Get and plot feature importance for the top final models.

        Important:
        This function automatically detects whether each model is base or tuned
        by checking fitted_model_store and tuned_model_store.
        """
        Path(save_folder).mkdir(exist_ok=True)
        Path(save_tables_folder).mkdir(exist_ok=True)

        top_models = final_ranking.head(top_n_models)["model"].tolist()

        all_importance_tables = {}

        for model_name in top_models:

            print(f"Processing feature importance for: {model_name}")

            if model_name in tuned_model_store:
                fitted_models = tuned_model_store[model_name]

            elif model_name in fitted_model_store:
                fitted_models = fitted_model_store[model_name]

            else:
                print(f"{model_name} not found in fitted model stores. Skipped.")
                continue

            importance_df = self.get_pooled_feature_importance(
                model_name=model_name,
                fitted_models=fitted_models,
                feature_cols=feature_cols
            )

            if importance_df is None:
                print(f"{model_name}: skipped because importance is not available.")
                continue

            all_importance_tables[model_name] = importance_df

            importance_df.to_csv(
                f"{save_tables_folder}/{model_name}_{self.SELECTED_FEATURES}_feature_importance.csv",
                index=False
            )

            self.plot_feature_importance(
                importance_df=importance_df,
                model_name=model_name,
                top_n=top_n_features,
                save_path=f"{save_folder}/{model_name}_{self.SELECTED_FEATURES}_feature_importance.png"
            )

        return all_importance_tables


    # # ==============================================
    # # 25. DESICION CURVE ANALYSIS
    # # ==============================================
    # https://mskcc-epi-bio.github.io/decisioncurveanalysis/dca-tutorial-python.html#Survival_Outcomes

if __name__ == "__main__":
    args = parse_args()
    time = args.time_followup
        
    SELECTED_FEATURES = args.features # "ELASTICNET"
    N_IMPUTATIONS = 10
    THRESHOLD = 0.5
    
    model_path_prefix = "results/models/"
    path_p_pred_prefix = "results/predictions/"
    
    if time >= 50:
        data_train_file = "/data_train_imp_{i}.csv"
        path = Path(f"./variable_selection/selected_features_{SELECTED_FEATURES}.txt")
        
        benchmark_summary_path = f"results/metrics/{SELECTED_FEATURES}_benchmark_summary_base_models_with_ci.xlsx"
        per_imputation_results_path = f"results/metrics/{SELECTED_FEATURES}_per_imputation_metrics_base_models.xlsx"
        
        model_path_sufix = f"_{SELECTED_FEATURES}_base_10_imputations.pkl"
        
        path_p_pred_sufix = f"_{SELECTED_FEATURES}_base_pooled_predictions.csv"
        
        path_roc_curve = f"results/plots/{SELECTED_FEATURES}_base_models_roc_curve.png"
        path_calibration_plot = f"results/plots/{SELECTED_FEATURES}_base_models_calibration_plot.png"
        path_confusion_matrix = f"results/plots/{SELECTED_FEATURES}_best_base_model_confusion_matrix.png"
        
        path_table_tuned_bechmarks = f"results/metrics/{SELECTED_FEATURES}_benchmark_summary_tuned_models_with_ci.xlsx"
        path_table_tuned_per_imp = f"results/metrics/{SELECTED_FEATURES}_per_imputation_metrics_tuned_models.xlsx"
        path_table_tuned_best_params = f"results/metrics/{SELECTED_FEATURES}_tuned_best_params.xlsx"
        
        model_tuned_path_sufix = f"_{SELECTED_FEATURES}_tuned_10_imputations.pkl"
        
        path_tuned_p_pred_sufix =f"_{SELECTED_FEATURES}_tuned_pooled_predictions.xlsx"
        
        path_final_ranking = f"results/metrics/{SELECTED_FEATURES}_final_model_ranking_with_ci.xlsx"
        
        path_tuned_roc_curve = f"results/plots/tuned_roc_curve_{SELECTED_FEATURES}.png"
        path_calibration_tuned = f"results/plots/tuned_calibration_plot_{SELECTED_FEATURES}.png"
        path_confusion_matrix_tuned = f"results/plots/tuned_confusion_matrix_{SELECTED_FEATURES}.png"
        
        path_best_final_model_training = f"results/models/best_final_model_10_imputations_{SELECTED_FEATURES}.pkl"

    else: 
        data_train_file = f"/data{time}ly_train_imp_"+"{i}.csv"
        path = Path(f"./variable_selection/selected_features_{SELECTED_FEATURES}{time}ly.txt")
        
        benchmark_summary_path = f"results/metrics/{SELECTED_FEATURES}{time}ly_benchmark_summary_base_models_with_ci.xlsx"
        per_imputation_results_path = f"results/metrics/{SELECTED_FEATURES}{time}ly_per_imputation_metrics_base_models.xlsx"
        
        model_path_sufix = f"_{SELECTED_FEATURES}{time}ly_base_10_imputations.pkl"
        
        path_p_pred_sufix = f"_{SELECTED_FEATURES}{time}ly_base_pooled_predictions.csv"
        
        path_roc_curve = f"results/plots/{SELECTED_FEATURES}{time}ly_base_models_roc_curve.png"
        path_calibration_plot = f"results/plots/{SELECTED_FEATURES}{time}ly_base_models_calibration_plot.png"
        path_confusion_matrix = f"results/plots/{SELECTED_FEATURES}{time}ly_best_base_model_confusion_matrix.png"
        
        path_table_tuned_bechmarks = f"results/metrics/{SELECTED_FEATURES}{time}ly_benchmark_summary_tuned_models_with_ci.xlsx"
        path_table_tuned_per_imp = f"results/metrics/{SELECTED_FEATURES}{time}ly_per_imputation_metrics_tuned_models.xlsx"
        path_table_tuned_best_params = f"results/metrics/{SELECTED_FEATURES}{time}ly_tuned_best_params.xlsx"
        
        model_tuned_path_sufix = f"_{SELECTED_FEATURES}{time}ly_tuned_10_imputations.pkl"
        
        path_tuned_p_pred_sufix =f"_{SELECTED_FEATURES}{time}ly_tuned_pooled_predictions.xlsx"
        
        path_final_ranking = f"results/metrics/{SELECTED_FEATURES}{time}ly_final_model_ranking_with_ci.xlsx"
        
        path_tuned_roc_curve = f"results/plots/tuned_roc_curve_{SELECTED_FEATURES}{time}ly.png"
        path_calibration_tuned = f"results/plots/tuned_calibration_plot_{SELECTED_FEATURES}{time}ly.png"
        path_confusion_matrix_tuned = f"results/plots/tuned_confusion_matrix_{SELECTED_FEATURES}{time}ly.png"
        
        path_best_final_model_training = f"results/models/best_final_model_10_imputations_{SELECTED_FEATURES}{time}ly.pkl"
        
    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    feature_cols = lines[2:-1]
    
    path_train_data = args.datalocated_fold + data_train_file
    
    dfs = [
            pd.read_csv(path_train_data.format(i=i))
            for i in range(1, N_IMPUTATIONS + 1)
        ]

    for i, df in enumerate(dfs):
        if "id" not in df.columns:
            df["id"] = np.arange(len(df))
        df = pd.get_dummies(df, columns=['thy_disease_preop'], drop_first=False,dtype=float)
        df.rename({"thy_disease_preop_0": "euthyroidism", 
                    "thy_disease_preop_1": "hypothyroidism", 
                    "thy_disease_preop_2": "hyperthyroidism"},
                axis=1, inplace=True)
        dfs[i] = df.copy()
        
    mdlsel = MODELS_SELECTION_DEVELOP(dfs=dfs, selected_features=SELECTED_FEATURES, 
                                      list_features=feature_cols)
      
    models, parameters = mdlsel.model_enabled()
    mdlsel.sep_train_and_develop_data()
    mdlsel.save_features()
    # make_pipeline() #no, return pieline
    
    # ======================================================
    # 12. RUN BASE MODEL BENCHMARKING
    # ======================================================
    summary_rows = [] ## Initial variable to all metrics, then are saved from benchmarks_summary
    per_imputation_results = [] ## Dataframes with metric for each imputation for each model
    pooled_predictions = {} ## y_pred, p_pred for each model, pooled across imputations
    fitted_model_store = {} ## 

    for model_name, estimator in models.items():
        print(f"\nTraining base model: {model_name}")

        pooled_metrics, per_imp_df, pooled_preds, fitted_models = mdlsel.benchmark_model_across_imputations(
            model_name=model_name,
            estimator=estimator,
            dfs=dfs,
            threshold=THRESHOLD,
            calculate_ci=True
        )

        summary_rows.append(pooled_metrics)
        per_imputation_results.append(per_imp_df)
        pooled_predictions[model_name] = pooled_preds
        fitted_model_store[model_name] = fitted_models

    benchmark_summary = pd.DataFrame(summary_rows)

    benchmark_summary = benchmark_summary.sort_values(
        by=["roc_auc_score", "brier_score"],
        ascending=[False, True]
    ).reset_index(drop=True)

    per_imputation_results = pd.concat(
        per_imputation_results,
        ignore_index=True
    )

    benchmark_summary.to_excel(
        benchmark_summary_path,
        index=False
    )

    per_imputation_results.to_excel(
        per_imputation_results_path,
        index=False
    )

    print("Base model benchmarking completed.")
    benchmark_summary

    # =============================================================================================================
    # 13. SAVE BASE MODELS
    # =============================================================================================================

    for model_name, fitted_models in fitted_model_store.items():
        model_path = model_path_prefix + model_name + model_path_sufix

        joblib.dump(
            {
                "model_name": model_name,
                "model_type": "base",
                "n_imputations": len(fitted_models),
                "feature_selection_method": SELECTED_FEATURES,
                "feature_cols": feature_cols,
                "fitted_pipelines": fitted_models,
                "params": fitted_models[0].get_params() if len(fitted_models) > 0 else None
            },
            model_path
        )

        print(f"Saved base model: {model_path}")


    # =============================================================================================================
    # 14. SAVE BASE MODEL P-PREDICTIONS
    # =============================================================================================================

    for model_name, preds in pooled_predictions.items():
        preds.to_csv(
            path_p_pred_prefix + model_name + path_p_pred_sufix,
            index=False
        )


    # =============================================================================================================
    # 16. BASE MODEL PLOTS
    # =============================================================================================================

    top_models = benchmark_summary.head(8)["model"].tolist()
    
    mdlsel.plot_roc_for_models(
        pooled_predictions,
        model_names=top_models,
        save_path=path_roc_curve
    )

    mdlsel.plot_calibration_for_models(
        pooled_predictions,
        model_names=top_models,
        n_bins=10,
        save_path=path_calibration_plot
    )

    best_base_model_name = benchmark_summary.iloc[0]["model"]

    mdlsel.plot_confusion_matrix_for_model(
        best_base_model_name,
        pooled_predictions,
        threshold=THRESHOLD,
        save_path=path_confusion_matrix
    )


        
    # =============================================================================================================
    # 19. RUN TUNING FOR TOP BASE MODELS
    # =============================================================================================================

    top_models_for_tuning = benchmark_summary.head(8)["model"].tolist()

    tuned_summary_rows = []
    tuned_per_imp_results = []
    tuned_predictions = {}
    tuned_best_params = []
    tuned_model_store = {}

    for model_name in top_models_for_tuning:
        if model_name not in parameters:
            print(f"No tuning grid for {model_name}. Skipped.")
            continue

        print(f"\nTuning model: {model_name}")

        pooled_metrics, per_imp_df, pooled_preds, best_params_df, fitted_tuned_models = mdlsel.tune_and_evaluate_across_imputations(
            model_name=model_name,
            estimator=models[model_name],
            dfs=dfs,
            param_dist=parameters[model_name],
            n_iter=20,
            threshold=THRESHOLD,
            calculate_ci=True
        )

        tuned_name = model_name + "_tuned"

        tuned_summary_rows.append(pooled_metrics)
        tuned_per_imp_results.append(per_imp_df)
        tuned_predictions[tuned_name] = pooled_preds
        tuned_best_params.append(best_params_df)
        tuned_model_store[tuned_name] = fitted_tuned_models


    tuned_summary = pd.DataFrame(tuned_summary_rows)

    if len(tuned_summary) > 0:
        tuned_summary = tuned_summary.sort_values(
            by=["roc_auc_score", "brier_score"],
            ascending=[False, True]
        ).reset_index(drop=True)

        tuned_per_imp_results = pd.concat(
            tuned_per_imp_results,
            ignore_index=True
        )

        tuned_best_params = pd.concat(
            tuned_best_params,
            ignore_index=True
        )

        tuned_summary.to_excel(
            path_table_tuned_bechmarks,
            index=False
        )

        tuned_per_imp_results.to_excel(
            path_table_tuned_per_imp,
            index=False
        )

        tuned_best_params.to_excel(
            path_table_tuned_best_params,
            index=False
        )

    print("Tuning completed.")
    tuned_summary


    # =============================================================================================================
    # 20. SAVE TUNED MODELS
    # =============================================================================================================
    
    for model_name, fitted_models in tuned_model_store.items():
        model_path = model_path_prefix + model_name + model_tuned_path_sufix

        joblib.dump(
            {
                "model_name": model_name,
                "model_type": "tuned",
                "n_imputations": len(fitted_models),
                "feature_selection_method": SELECTED_FEATURES,
                "feature_cols": feature_cols,
                "fitted_pipelines": fitted_models,
                "params": fitted_models[0].get_params() if len(fitted_models) > 0 else None

            },
            model_path
        )

        print(f"Saved tuned model: {model_path}")


    # =============================================================================================================
    # 21. SAVE TUNED MODEL PREDICTIONS
    # =============================================================================================================
    for model_name, preds in tuned_predictions.items():
        preds.to_excel(
            path_p_pred_prefix + model_name + path_tuned_p_pred_sufix,
            index=False
        )


    # =============================================================================================================
    # 22. FINAL RANKING: BASE VS TUNED
    # =============================================================================================================

    if len(tuned_summary) > 0:
        final_ranking = pd.concat(
            [benchmark_summary, tuned_summary],
            ignore_index=True
        )
    else:
        final_ranking = benchmark_summary.copy()

    final_ranking = final_ranking.sort_values(
        by=["roc_auc_score", "brier_score"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    final_ranking.to_excel(
        path_final_ranking,
        index=False
    )

    final_ranking


    # =============================================================================================================
    # 23. TUNED PREDICTIONS
    # =============================================================================================================

    all_predictions = {}
    all_predictions.update(pooled_predictions)
    all_predictions.update(tuned_predictions)

    top_final_models = tuned_summary.head(8)["model"].tolist()

    mdlsel.plot_roc_for_models(
        all_predictions,
        model_names=top_final_models,
        save_path=path_tuned_roc_curve
    )

    mdlsel.plot_calibration_for_models(
        all_predictions,
        model_names=top_final_models,
        n_bins=10,
        save_path=path_calibration_tuned
    )

    best_final_model_name = final_ranking.iloc[0]["model"]

    mdlsel.plot_confusion_matrix_for_model(
        best_final_model_name,
        all_predictions,
        threshold=THRESHOLD,
        save_path=path_confusion_matrix_tuned
    )
    
    feature_importance_results = mdlsel.get_and_plot_feature_importance_top_models(
    final_ranking=final_ranking,
    fitted_model_store=fitted_model_store,
    tuned_model_store=tuned_model_store,
    feature_cols=feature_cols,
    top_n_models=4,
    top_n_features=15,
    save_folder="results/plots",
    save_tables_folder="results/features"
    )   
    
    # =============================================================================================================
    # 24. SAVE BEST FINAL MODEL
    # =============================================================================================================

    if best_final_model_name in tuned_model_store:
        best_fitted_models = tuned_model_store[best_final_model_name]
        best_model_type = "tuned"

    elif best_final_model_name in fitted_model_store:
        best_fitted_models = fitted_model_store[best_final_model_name]
        best_model_type = "base"

    else:
        raise ValueError(f"Best model {best_final_model_name} not found in fitted model stores.")
    
    
    joblib.dump(
        {
            "model_name": best_final_model_name,
            "model_type": best_model_type,
            "n_imputations": len(best_fitted_models),
            "feature_selection_method": SELECTED_FEATURES,
            "feature_cols": feature_cols,
            "fitted_pipelines": best_fitted_models,
            "final_ranking": final_ranking,
            "params": best_fitted_models[0].get_params() if len(best_fitted_models) > 0 else None
        },
        path_best_final_model_training
    )

    print(f"Best final model saved: {best_final_model_name}")
