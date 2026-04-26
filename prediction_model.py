import argparse
import logging
import joblib
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class ModelResult:
    """Store model results with complete scope management."""
    def __init__(self, model_name, balanced_state, variables_used):
        self.model_name = model_name
        self.balanced_state = balanced_state
        self.variables_used = variables_used
        self.test_accuracy = None
        self.test_precision = None
        self.test_recall = None
        self.test_f1 = None
        self.test_roc_auc = None
        self.y_scores_test = None
        self.y_test = None
        self.y_pred_test = None
        self.train_accuracy = None
        self.train_roc_auc = None
        self.fine_tune = None
    
    def to_dict(self):
        """Convert to dictionary for CSV export."""
        return {
            'Model': self.model_name,
            'Balanced_State': self.balanced_state,
            'Variables': self.variables_used,
            'Train_Accuracy': self.train_accuracy,
            'Train_ROC_AUC': self.train_roc_auc,
            'Test_Accuracy': self.test_accuracy,
            'Test_Precision': self.test_precision,
            'Test_Recall': self.test_recall,
            'Test_F1_Score': self.test_f1,
            'Test_ROC_AUC': self.test_roc_auc,
            'fine_tune': self.fine_tune
        }


class PredictionModelPipeline:
    """
    End-to-end prediction model pipeline for thyroid cancer recurrence prediction.
    Encapsulates data loading, preprocessing, feature selection, model training, and evaluation.
    """
    
    def __init__(self, args=None):
        """
        Initialize the pipeline with command-line arguments.
        
        Args:
            args: parsed argparse arguments or None
        """
        self.args = args or self._parse_arguments()
        self.logger = None
        self.scaler = StandardScaler()
        
        # Data containers
        self.df = None
        self.df_imputed = None
        self.df_log = None
        self.X = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_smote = None
        self.y_smote = None
        self.variables_picked = None
        
        # Results containers
        self.results = {}
        self.best_model_xgb_imbalanced = None
        self.best_model_xgb_balanced = None
        self.tune_strategy = self.args.tune_strategy if hasattr(self.args, 'tune_strategy') else 'cv'
        self.bootstrap_results = None
        
        # Variable mappings
        self.map_variables = self._define_variable_mappings()
        
    # =====================================================================
    # SETUP & CONFIGURATION METHODS
    # =====================================================================
    
    def setup_environment(self):
        """Create necessary directories and configure logging."""
        dirs = [
            "./variable_selection", "./models", "./results", 
            "./models_comparison", "./distribution", "./logs",
            "./results/confusion_matrix", "./results/roc_curves",
            "./results/feature_importances"
        ]
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/model_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        return self.logger
    
    @staticmethod
    def _parse_arguments():
        """Parse command-line arguments for model training and prediction."""
        parser = argparse.ArgumentParser(
            description='Train thyroid cancer recurrence prediction models'
        )
        parser.add_argument(
            '--variables',
            type=str,
            choices=['all', 'selected'],
            default='selected',
            help='Use all or selected features (default: selected)'
        )
        parser.add_argument(
            '--use-random-search',
            action='store_true',
            default=False,
            help='Use RandomizedSearchCV for faster and initial tuning'
        )
        parser.add_argument(
            '--skip-plots',
            action='store_true',
            default=False,
            help='Skip distribution plots to save memory'
        )
        parser.add_argument(
            '--data-path',
            type=str,
            default='./database/hee_brazil_ambato_peru_base.csv',
            help='Path to input data (default: ./database/hee_brazil_ambato_peru_base.csv)'
        )
        parser.add_argument(
            '--tune-strategy',
            type=str,
            choices=['cv', 'bootstrap'],
            default='cv',
            help='Tuning strategy: cv (cross-validation) or bootstrap (default: cv)'
        )
        return parser.parse_args()
    
    @staticmethod
    def _define_variable_mappings():
        """Define variable mappings for categorical variables."""
        return {
            "SEX": {0: "Female", 1: "Male"},
            "RADIOTHERAPY EXPOSURE": {1: "Yes", 0: "No"},
            "FAMILY HISTORY OF THYROID CANCER": {1: "Yes", 0: "No"},
            "EUTHYROIDISM": {0: "No", 1: "Yes"},
            "HYPOTHYROIDISM": {0: "No", 1: "Yes"},
            "HYPERTHYROIDISM": {0: "No", 1: "Yes"},
            "THYROIDECTOMY APPROACH": {0: "Total", 1: "Total + Lymphadenectomy"},
            "TYPEOFRESECTION": {0: "R0", 1: "R1", 2: "R2"},
            "HISTOLOGY": {0: "Papilar", 1: "Folicular", 2: "Hurtle Cells"},
            "SUBTYPE_FOLLI_PAPIL": {0: "Minimally invasive", 1: "Encapsulated invasive", 2: "Widely invasive",
                3: "Classic", 4: "Follicular variant", 5: "Encapsulated", 6: "Diffuse sclerosant",
                7: "High cells", 8: "Colunar cells", 9: "Cribiform-morular", 10: "Hobnail", 
                11: "Warthin-like", 12: "Oncocytic", 13: "Trabecular/Solid", 14: "Classic and Follicular",
                15: "Follicular and oncocytic"},
            "EXTRATHYROIDALEXTENSION": {0: "Absent", 1: "Microscopic", 2: "Macroscopic"},
            "MULTICENTRIC": {1: "Yes", 0: "No"},
            "MULTICENTER_BILATERAL": {1: "Yes", 0: "No"},
            "VASCULARINVASION": {1: "Yes", 0: "No"},
            "PERINEURALINVASION": {1: "Yes", 0: "No"},
            "POSITIVELYMPHNODEN1": {0: "No excision", 2: "Yes", 1: "No"},
            "EXTRANODALEXTENSION": {1: "Yes", 0: "No"},
            "TNMT": {0: "Tx", 1: "T1", 2: "T2", 3: "T3", 4: "T4"},
            "HASHIMOTO THYROIDITIS": {1: "Yes", 0: "No"},
            "TNMN": {1: "N0", 2: "N1a", 3: "N1b", 0: "Nx"},
            "TNMM": {0: "M0", 1: "M1"},
            "STAGE": {0: "I", 1: "II", 2: "III", 3: "IV"},
            "ATA_2015_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio", 3: "Alto"},
            "ATA_2025_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio o Bajo", 3: "Intermedio o Alto", 4: "Alto"},
            "RAI": {1: "Yes", 0: "No"},
            "ANTI TG PRE RAI (POSITIVE or NEGATIVE)": {1: "Positivo", 0: "Negativo"},
            "ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)": {1: "Positivo", 0: "Negativo"},
            "RECURRENCE": {0: "No Recurrence", 1: "Recurrence"}
        }
    
    # =====================================================================
    # DATA LOADING & PREPROCESSING METHODS
    # =====================================================================
    
    def load_data(self, data_path):
        """Load data from CSV file."""
        self.logger.info(f"Loading data from: {data_path}")
        try:
            self.df = pd.read_csv(data_path)
            self.logger.info(f"Data loaded: {self.df.shape}")
        except Exception as e:
            self.logger.error(f"Error while loading data: {e}")
            return None
        return self.df
    
    def define_variables(self):
        """Classify variables into categorical and continuous."""
        categorical_variable = []
        continuous_variable = []
        
        for col in self.df.columns:
            if col in ['record_id']:
                continue
            if col == "SUBTYPE_FOLLI_PAPIL" or self.df[col].nunique() < 6:
                categorical_variable.append(col)
            else:
                continuous_variable.append(col)
        
        self.logger.info(f"Categorical variables: {len(categorical_variable)}")
        self.logger.info(f"Continuous variables: {len(continuous_variable)}")
        
        return categorical_variable, continuous_variable
    
    def report_missing_data(self):
        """Generate and log missing data report."""
        total = self.df.isnull().sum().sort_values(ascending=False)
        percent_total = (self.df.isnull().sum() / len(self.df)).sort_values(ascending=False) * 100
        missing = pd.concat([total, round(percent_total, 2)], axis=1, 
                           keys=['Total', 'Percent'])
        missing = missing[missing['Total'] > 0]
        
        if len(missing) > 0:
            self.logger.info("\nMissing Data Report:")
            self.logger.info(f"\n{missing}")
            missing.to_csv("./logs/missing_data_report.csv")
        else:
            self.logger.info("No missing data found.")
        
        return missing
    
    def handle_missing_data(self, categorical_variables):
        """Impute missing values with validation."""
        self.df_imputed = self.df.copy()
        imputation_log = {}
        
        for col in self.df_imputed.columns:
            if self.df_imputed[col].isnull().sum() == 0:
                continue
            
            try:
                if col in categorical_variables:
                    mode_val = self.df_imputed[col].mode()
                    if len(mode_val) > 0:
                        self.df_imputed[col].fillna(mode_val[0], inplace=True)
                        self.df_imputed[col] = self.df_imputed[col].astype('uint8')
                        imputation_log[col] = f"mode: {mode_val[0]}"
                        self.logger.info(f"Imputed categorical '{col}' with mode: {mode_val[0]}")
                    else:
                        self.logger.warning(f"No mode found for {col}. Filling with 0.")
                        self.df_imputed[col].fillna(0, inplace=True)
                        imputation_log[col] = "mode: 0 (default)"
                else:
                    median_val = self.df_imputed[col].median()
                    self.df_imputed[col].fillna(median_val, inplace=True)
                    imputation_log[col] = f"median: {median_val}"
                    self.logger.info(f"Imputed continuous '{col}' with median: {median_val}")
            except Exception as e:
                self.logger.error(f"Error imputing {col}: {e}")
        
        if imputation_log:
            with open("./logs/imputation_log.txt", "w") as f:
                for col, method in imputation_log.items():
                    f.write(f"{col}: {method}\n")
        
        return self.df_imputed
    
    def safe_log_transform(self, df, columns, epsilon=2.220446049250313e-16):
        """Apply log transformation with safety checks for zeros/negatives."""
        df_transformed = df.copy()
        
        for col in columns:
            if col not in df_transformed.columns:
                continue
            
            min_val = df_transformed[col].min()
            
            if min_val <= 0:
                n_min = ((df_transformed[col] == min_val).sum())/len(df_transformed)
                df_transformed[col] = np.log(df_transformed[col] + epsilon)
                self.logger.warning(
                    f"Column '{col}' had min value {min_val} in {n_min:.2%}. Shifted by {epsilon:.2e}"
                )
            else:
                df_transformed[col] = np.log(df_transformed[col])
        
        return df_transformed
    
    def generate_distribution_plots(self, df, categorical_vars, continuous_vars, 
                                   skip_plots=False, max_plots=50, log_state=False):
        """Generate distribution plots with memory management."""
        if skip_plots:
            self.logger.info("Skipping distribution plots to save memory")
            return
        
        plot_count = 0
        plot_dir = Path("./distribution")
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for col in df.columns:
            if plot_count >= max_plots:
                self.logger.info(f"Reached max plots limit ({max_plots})")
                break
            
            if col == "RECURRENCE" or col not in categorical_vars + continuous_vars:
                continue
            
            try:
                plt.figure(figsize=(10, 6))
                
                if col in categorical_vars:
                    ax = sns.countplot(data=df, x=col, hue="RECURRENCE")
                    plt.title(f'Frequency of Recurrence by {col}')
                    if col in self.map_variables:
                        ax.set_xticklabels(self.map_variables[col].values())
                else:
                    sns.histplot(data=df, x=col, hue="RECURRENCE", kde=False, 
                               element="step", common_norm=False, alpha=0.6)
                    plt.title(f'Distribution of Recurrence by {col}')
                
                plt.legend(title="Recurrence", labels=["No", "Yes"], prop={'size': 10})
                plt.tight_layout()
                
                path = plot_dir / f"freq_{col}{'_norm' if log_state else ''}.png"
                plt.savefig(path, dpi=150)
                plt.close()
                plot_count += 1
                self.logger.info(f"Saved plot: {path}")
                
            except Exception as e:
                self.logger.error(f"Error plotting {col}: {e}")
                plt.close()
    
    # =====================================================================
    # FEATURE SELECTION METHODS
    # =====================================================================
    
    def spearman_correlation_analysis(self, X_df, imputation_name, skip=False):
        """Calculate and save Spearman correlations."""
        if skip:
            return
        
        corr_matrix = X_df.corr(method='spearman')
        output_path = f"./variable_selection/spearman_corr_{imputation_name}.xlsx"
        corr_matrix.to_excel(output_path)
        
        plt.figure(figsize=(12, 12))
        correlation = X_df.corr()
        sns.heatmap((correlation), annot=False, cmap=sns.color_palette("mako", as_cmap=True))
        plt.savefig("./variable_selection/heatmap_correlation.png")
        plt.close()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append(f"{col1} - {col2}: {corr_value:.3f}")
        
        txt_path = f"./variable_selection/spearman_high_results_{imputation_name}.txt"
        with open(txt_path, "w") as f:
            f.write(f"Correlaciones de Spearman > |0.8| ({len(high_corr_pairs)} pares):\n")
            f.write("="*50 + "\n")
            for pair in high_corr_pairs:
                f.write(pair + "\n")
        
        self.logger.info(f"Spearman correlations for {imputation_name} saved. "
                        f"{len(high_corr_pairs)} high correlations found.")
        return high_corr_pairs
    
    def select_features_lasso(self, X_scaled, y, X_columns, cv=10):
        """Select features using LassoCV with cross-validation."""
        self.logger.info(f"Starting LASSO feature selection with cv={cv}")
        
        lasso_cv = LassoCV(cv=cv, random_state=0, max_iter=10000)
        lasso_cv.fit(X_scaled, y)
        
        self.logger.info(f"LassoCV best alpha: {lasso_cv.alpha_}")
        
        sfm = SelectFromModel(lasso_cv, threshold=None, prefit=True)
        selected_idx = sfm.get_support(indices=True)
        selected_features = list(X_columns[selected_idx])
        
        path = Path("./variable_selection/selected_features_lasso.txt")
        with open(path, "a") as f:
            f.write(f"LassoCV selected {len(selected_features)} features with alpha={lasso_cv.alpha_}:\n")
            for feat in selected_features:
                f.write(f"{feat}\n")
        
        self.logger.info(f"Selected {len(selected_features)} features via LASSO")
        return selected_features
    
    # =====================================================================
    # MODEL TUNING METHODS
    # =====================================================================
    
    def tune_model(self, model, param_grid, X_train, y_train, use_random=False, cv=5):
        """Tune model hyperparameters with GridSearchCV or RandomizedSearchCV."""
        if use_random:
            self.logger.info(f"RandomizedSearchCV for {model.__class__.__name__}")
            search = RandomizedSearchCV(
                model, param_grid, cv=cv, n_iter=10,
                n_jobs=-1, random_state=0, scoring='roc_auc', verbose=1
            )
        else:
            self.logger.info(f"GridSearchCV for {model.__class__.__name__}")
            search = GridSearchCV(
                model, param_grid, cv=cv,
                n_jobs=-1, scoring='roc_auc', verbose=1
            )
        
        search.fit(X_train, y_train)
        self.logger.info(f"Best params: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search
    
    def tune_model_bootstrap(self, model, X_train, y_train, X_test, y_test, 
                            n_bootstrap=50, random_seed=0):
        """
        Tune model hyperparameters using bootstrap sampling.
        
        Args:
            model: model instance
            X_train: training features
            y_train: training labels
            X_test: test features
            y_test: test labels
            n_bootstrap: number of bootstrap samples
            random_seed: random state
        
        Returns:
            Tuple of (best_model, bootstrap_stats_df)
        """
        self.logger.info(f"Starting Bootstrap tuning for {model.__class__.__name__} "
                        f"with {n_bootstrap} iterations")
        
        bootstrap_stats = []
        best_auc = 0
        best_model = None
        
        for i in range(n_bootstrap):
            # Bootstrap sample from training data (use RNG for reproducibility)
            rng = np.random.RandomState(random_seed + i)
            indices = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]

            # Out-of-bag (OOB) samples for validation
            oob_mask = np.ones(len(X_train), dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.nonzero(oob_mask)[0]
            if len(oob_indices) > 0:
                X_oob = X_train[oob_indices]
                y_oob = y_train.iloc[oob_indices] if hasattr(y_train, 'iloc') else y_train[oob_indices]
            else:
                X_oob = X_test
                y_oob = y_test

            # Train a fresh estimator on the bootstrap sample
            try:
                est = clone(model)
                est.fit(X_boot, y_boot)

                # Evaluate on OOB data
                if hasattr(est, 'predict_proba'):
                    y_scores_oob = est.predict_proba(X_oob)[:, 1]
                else:
                    y_scores_oob = est.predict(X_oob)

                oob_auc = roc_auc_score(y_oob, y_scores_oob)

                # Evaluate on test data
                if hasattr(est, 'predict_proba'):
                    y_scores_test = est.predict_proba(X_test)[:, 1]
                else:
                    y_scores_test = est.predict(X_test)

                test_auc = roc_auc_score(y_test, y_scores_test)

                bootstrap_stats.append({
                    'iteration': i,
                    'oob_auc': oob_auc,
                    'test_auc': test_auc
                })

                if test_auc > best_auc:
                    best_auc = test_auc
                    best_model = est

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Bootstrap iteration {i+1}/{n_bootstrap}: "
                                   f"OOB AUC={oob_auc:.4f}, Test AUC={test_auc:.4f}")
            except Exception as e:
                self.logger.error(f"Error in bootstrap iteration {i}: {e}")
                continue
        
        bootstrap_df = pd.DataFrame(bootstrap_stats)
        
        if len(bootstrap_stats) > 0:
            self.logger.info(f"Bootstrap tuning completed.")
            self.logger.info(f"Mean OOB AUC: {bootstrap_df['oob_auc'].mean():.4f} "
                           f"(±{bootstrap_df['oob_auc'].std():.4f})")
            self.logger.info(f"Mean Test AUC: {bootstrap_df['test_auc'].mean():.4f} "
                           f"(±{bootstrap_df['test_auc'].std():.4f})")
        
        self.bootstrap_results = bootstrap_df
        return best_model, bootstrap_df
    
    # =====================================================================
    # MODEL EVALUATION METHODS
    # =====================================================================
    
    def compute_model_metrics(self, y_train, y_test, train_preds, test_preds, model_name,
                             fine_tune, balanced_state, variables_used, y_scores_test=None,
                             y_train_scores=None):
        """Compute and log model metrics."""
        result = ModelResult(model_name, balanced_state, variables_used)
        result.fine_tune = fine_tune
        
        result.train_accuracy = accuracy_score(y_train, train_preds)
        result.train_roc_auc = roc_auc_score(y_train, y_train_scores) if y_train_scores is not None else None
        
        result.test_accuracy = accuracy_score(y_test, test_preds)
        result.test_precision = precision_score(y_test, test_preds, zero_division=0)
        result.test_recall = recall_score(y_test, test_preds, zero_division=0)
        result.test_f1 = f1_score(y_test, test_preds, zero_division=0)
        
        if y_scores_test is not None:
            result.test_roc_auc = roc_auc_score(y_test, y_scores_test)
        else:
            self.logger.warning("y_scores_test is None, setting test_roc_auc to None")
            result.test_roc_auc = None
        
        result.y_scores_test = y_scores_test
        result.y_test = y_test
        result.y_pred_test = test_preds
        
        self.logger.info(f"\n{model_name} - {'Tuned' if fine_tune else 'Base'}")
        self.logger.info(f"Train Accuracy: {result.train_accuracy:.4f}")
        if result.train_roc_auc:
            self.logger.info(f"Train ROC AUC: {result.train_roc_auc:.4f}")
        self.logger.info(f"Test Accuracy: {result.test_accuracy:.4f}")
        self.logger.info(f"Test Precision: {result.test_precision:.4f}")
        self.logger.info(f"Test Recall: {result.test_recall:.4f}")
        self.logger.info(f"Test F1: {result.test_f1:.4f}")
        self.logger.info(f"Test ROC AUC: {result.test_roc_auc:.4f}")
        
        self._save_confusion_matrix(y_train, y_test, train_preds, test_preds,
                                    model_name, fine_tune, balanced_state, variables_used)
        
        return result
    
    def _save_confusion_matrix(self, y_train, y_test, train_preds, test_preds,
                              model_name, fine_tune, balanced_state, variables_used):
        """Plot and save confusion matrix."""
        train_confusion_matrix = confusion_matrix(y_train, train_preds)
        test_confusion_matrix = confusion_matrix(y_test, test_preds)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        labels = ['0', '1']
        sns.heatmap(train_confusion_matrix, annot=True, cmap='Blues', ax=axes[0], 
                   fmt="d", xticklabels=labels, yticklabels=labels)
        axes[0].set_xlabel('Predicted labels')
        axes[0].set_ylabel('True labels')
        axes[0].set_title('Train Confusion Matrix')
        sns.heatmap(test_confusion_matrix, annot=True, cmap='Blues', ax=axes[1], 
                   fmt="d", xticklabels=labels, yticklabels=labels)
        axes[1].set_xlabel('Predicted labels')
        axes[1].set_ylabel('True labels')
        axes[1].set_title('Test Confusion Matrix')
        plt.title(f"{model_name} - {'Fine-tuned' if fine_tune else 'Base'}")
        
        cm_dir = Path(f"./results/confusion_matrix/{balanced_state}/{variables_used}_variables")
        cm_dir.mkdir(parents=True, exist_ok=True)
        path = cm_dir / f"cm_{model_name}_{'tuned' if fine_tune else 'base'}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Confusion matrix saved: {path}")
    
    def plot_roc_curves(self, results_list, balanced_state, n_vars):
        """Plot ROC curves for all models in comparison."""
        plt.figure(figsize=(12, 9))
        
        for result in results_list:
            if result.y_scores_test is not None:
                fpr, tpr, _ = roc_curve(result.y_test, result.y_scores_test)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{result.model_name} (AUC = {roc_auc:.3f})", 
                        alpha=0.8, linewidth=2)
        
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves Comparison - {balanced_state} ({n_vars} variables)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        roc_dir = Path("./results/roc_curves")
        roc_dir.mkdir(parents=True, exist_ok=True)
        path = roc_dir / f"roc_{balanced_state}_{n_vars}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ROC plot saved: {path}")
    
    def plot_feature_importance(self, best_model, feature_names, balanced_state, 
                               n_vars, model_name="XGBoost"):
        """Plot feature importance for best model only."""
        if not hasattr(best_model, 'feature_importances_'):
            self.logger.warning(f"{best_model.__class__.__name__} has no feature_importances_")
            return
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name} (Best Model)', 
                 fontsize=16, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
        plt.yticks(range(len(indices)), np.array(feature_names)[indices])
        plt.xlabel('Relative Importance', fontsize=12, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        fi_dir = Path("./results/feature_importances")
        fi_dir.mkdir(parents=True, exist_ok=True)
        path = fi_dir / f"feature_importance_{balanced_state}_{n_vars}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Feature importance plot saved: {path}")
    
    # =====================================================================
    # MODEL PERSISTENCE METHODS
    # =====================================================================
    
    def save_best_hyperparameters(self, model_name, best_params, variables_used, 
                                 balanced_state, searched_params):
        """Save best hyperparameters to log file."""
        path = f"./results/best_params.txt"
        with open(path, "a") as file:
            file.write(f"{model_name}, {searched_params}, {variables_used}, "
                      f"{balanced_state}: {best_params}\n")
    
    def save_best_model(self, model, model_name, balanced_state, n_vars, 
                       features, metrics):
        """Save best model with metadata using joblib."""
        model_dir = Path("./models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        filename = model_dir / f"best_model_{model_name}_{balanced_state}_{n_vars}.pkl"
        
        metadata = {
            'model': model,
            'model_name': model_name,
            'balanced_state': balanced_state,
            'variables': list(features),
            'n_variables': len(features),
            'metrics': metrics,
            'timestamp': pd.Timestamp.now()
        }
        
        joblib.dump(metadata, filename)
        self.logger.info(f"Model saved with joblib: {filename}")
    
    def load_model(self, model_path):
        """Load model from joblib or pickle file."""
        try:
            if str(model_path).endswith('.pkl'):
                try:
                    metadata = joblib.load(model_path)
                    self.logger.info(f"Model loaded with joblib: {model_path}")
                except:
                    with open(model_path, 'rb') as f:
                        metadata = pickle.load(f)
                    self.logger.info(f"Model loaded with pickle: {model_path}")
            else:
                raise ValueError("Model file must be .pkl")
            
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    # =====================================================================
    # MODEL TRAINING METHODS
    # =====================================================================
    
    def get_models_config(self):
        """Return model configurations with hyperparameters."""
        return {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=0),
                'params': {
                    'penalty': ['l1', 'l2'],
                    'class_weight': [None, 'balanced'],
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=0),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=0, verbose=-1),
                'params': {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [20, 30, 40]
                }
            },
            'CatBoost': {
                'model': cb.CatBoostClassifier(random_state=0, verbose=0),
                'params': {
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 150]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            },
            'SVC': {
                'model': SVC(probability=True, random_state=0),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            },
            'GaussianNB': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            }
        }
    
    def _build_rnn_model(self, input_dim):
        """Build and compile RNN model for classification."""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy', 'AUC'])
        return model
    
    def _build_lstm_model(self, input_dim):
        """Build and compile LSTM model for classification."""
        # Reshape for LSTM: (samples, timesteps, features)
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(1, input_dim), return_sequences=True),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy', 'AUC'])
        return model
    
    def train_neural_model(self, model_type, X_train, y_train, X_test, y_test, epochs=50):
        """Train RNN or LSTM model and return predictions."""
        input_dim = X_train.shape[1]
        
        if model_type == 'RNN':
            model = self._build_rnn_model(input_dim)
            X_train_nn = X_train
            X_test_nn = X_test
        elif model_type == 'LSTM':
            model = self._build_lstm_model(input_dim)
            # Reshape for LSTM: (samples, 1 timestep, features)
            X_train_nn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_nn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train_nn, y_train, 
                 validation_data=(X_test_nn, y_test),
                 epochs=epochs, batch_size=32, 
                 callbacks=[early_stop], verbose=0)
        
        y_pred = (model.predict(X_test_nn, verbose=0) > 0.5).astype(int).flatten()
        y_scores = model.predict(X_test_nn, verbose=0).flatten()
        
        # Get training predictions
        y_pred_train = (model.predict(X_train_nn, verbose=0) > 0.5).astype(int).flatten()
        y_train_scores = model.predict(X_train_nn, verbose=0).flatten()
        
        return y_pred_train, y_pred, y_train_scores, y_scores
    
    def train_all_models(self, X_train_scaled, X_test_scaled, y_train, y_test, 
                        X_smote_scaled, y_smote, variables_picked, n_vars, use_random=False):
        """Train all models on both imbalanced and balanced data."""
        
        results = {}
        imbalanced_results = []
        balanced_results = []
        models_config = self.get_models_config()
        
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING MODELS ON IMBALANCED DATA")
        self.logger.info("="*70)
        
        # Train on imbalanced data
        for model_name, config in models_config.items():
            self.logger.info(f"\n--- Training {model_name} (Imbalanced) ---")
            
            # Base model
            model = config['model']
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_train_scores = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
            y_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
            
            result = self.compute_model_metrics(y_train, y_test, y_pred_train, y_pred_test,
                                              model_name, fine_tune=False, 
                                              balanced_state="Imbalanced",
                                              variables_used=X_train_scaled.shape[1],
                                              y_scores_test=y_scores,
                                              y_train_scores=y_train_scores)
            imbalanced_results.append(result)
            
            # Tuned model with CV
            best_model, best_params, _ = self.tune_model(
                config['model'], config['params'],
                X_train_scaled, y_train, use_random=use_random
            )
            y_pred_train_tuned = best_model.predict(X_train_scaled)
            y_pred_test_tuned = best_model.predict(X_test_scaled)
            y_train_scores_tuned = best_model.predict_proba(X_train_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_train_tuned
            y_scores_tuned = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_test_tuned
            
            result_tuned = self.compute_model_metrics(y_train, y_test, y_pred_train_tuned, 
                                                     y_pred_test_tuned, model_name, fine_tune=True,
                                                     balanced_state="Imbalanced",
                                                     variables_used=X_train_scaled.shape[1],
                                                     y_scores_test=y_scores_tuned,
                                                     y_train_scores=y_train_scores_tuned)
            imbalanced_results.append(result_tuned)
            
            self.save_best_hyperparameters(model_name, best_params, X_train_scaled.shape[1], 
                                          "imbalanced data.", "Random Search" if use_random else "Grid Search")
            
            # Optional: Bootstrap refinement
            if self.tune_strategy == 'bootstrap':
                self.logger.info(f"Applying bootstrap tuning for {model_name}...")
                best_model_boot, boot_stats = self.tune_model_bootstrap(
                    config['model'], X_train_scaled, y_train, X_test_scaled, y_test, n_bootstrap=50
                )
            
            if model_name == 'XGBoost':
                self.best_model_xgb_imbalanced = best_model
                metrics_dict = result_tuned.to_dict()
                self.save_best_model(best_model, model_name, "Imbalanced", n_vars,
                                   variables_picked, metrics_dict)
        
        # Train RNN and LSTM models
        for nn_model_name in ['RNN', 'LSTM']:
            self.logger.info(f"\n--- Training {nn_model_name} (Imbalanced) ---")
            try:
                y_pred_train, y_pred_test, y_train_scores, y_scores = self.train_neural_model(
                    nn_model_name, X_train_scaled, y_train, X_test_scaled, y_test, epochs=50
                )
                
                result = self.compute_model_metrics(y_train, y_test, y_pred_train, y_pred_test,
                                                  nn_model_name, fine_tune=True, 
                                                  balanced_state="Imbalanced",
                                                  variables_used=X_train_scaled.shape[1],
                                                  y_scores_test=y_scores,
                                                  y_train_scores=y_train_scores)
                imbalanced_results.append(result)
            except Exception as e:
                self.logger.error(f"Error training {nn_model_name}: {e}")
        
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING MODELS ON BALANCED DATA (SMOTE)")
        self.logger.info("="*70)
        
        # Train on SMOTE balanced data
        for model_name, config in models_config.items():
            self.logger.info(f"\n--- Training {model_name} (Balanced SMOTE) ---")
            
            # Base model
            model = config['model']
            model.fit(X_smote_scaled, y_smote)
            y_pred_train = model.predict(X_smote_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_train_scores = model.predict_proba(X_smote_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
            y_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
            
            result = self.compute_model_metrics(y_smote, y_test, y_pred_train, y_pred_test,
                                              model_name, fine_tune=False,
                                              balanced_state="Balanced_SMOTE",
                                              variables_used=X_smote_scaled.shape[1],
                                              y_scores_test=y_scores,
                                              y_train_scores=y_train_scores)
            balanced_results.append(result)
            
            # Tuned model with CV
            best_model, best_params, _ = self.tune_model(
                config['model'], config['params'],
                X_smote_scaled, y_smote, use_random=use_random
            )
            y_pred_train_tuned = best_model.predict(X_smote_scaled)
            y_pred_test_tuned = best_model.predict(X_test_scaled)
            y_train_scores_tuned = best_model.predict_proba(X_smote_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_train_tuned
            y_scores_tuned = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_test_tuned
            
            result_tuned = self.compute_model_metrics(y_smote, y_test, y_pred_train_tuned,
                                                     y_pred_test_tuned, model_name, fine_tune=True,
                                                     balanced_state="Balanced_SMOTE",
                                                     variables_used=X_smote_scaled.shape[1],
                                                     y_scores_test=y_scores_tuned,
                                                     y_train_scores=y_train_scores_tuned)
            balanced_results.append(result_tuned)
            
            self.save_best_hyperparameters(model_name, best_params, X_smote_scaled.shape[1], 
                                          "balanced data.", "Random Search" if use_random else "Grid Search")

            if model_name == 'XGBoost':
                self.best_model_xgb_balanced = best_model
                metrics_dict = result_tuned.to_dict()
                self.save_best_model(best_model, model_name, "Balanced_SMOTE", n_vars,
                                   variables_picked, metrics_dict)
        
        # Train RNN and LSTM models on balanced data
        for nn_model_name in ['RNN', 'LSTM']:
            self.logger.info(f"\n--- Training {nn_model_name} (Balanced SMOTE) ---")
            try:
                y_pred_train, y_pred_test, y_train_scores, y_scores = self.train_neural_model(
                    nn_model_name, X_smote_scaled, y_smote, X_test_scaled, y_test, epochs=50
                )
                
                result = self.compute_model_metrics(y_smote, y_test, y_pred_train, y_pred_test,
                                                  nn_model_name, fine_tune=True, 
                                                  balanced_state="Balanced_SMOTE",
                                                  variables_used=X_smote_scaled.shape[1],
                                                  y_scores_test=y_scores,
                                                  y_train_scores=y_train_scores)
                balanced_results.append(result)
            except Exception as e:
                self.logger.error(f"Error training {nn_model_name}: {e}")
        
        results['imbalanced'] = imbalanced_results
        results['balanced'] = balanced_results

        return results

    def save_tuning_summary(self):
        """Save a comprehensive summary of tuning strategy and results."""
        summary_path = Path("./results/tuning_summary.txt")
        
        with open(summary_path, "w") as f:
            f.write("="*70 + "\n")
            f.write("MODEL TUNING STRATEGY SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Tuning Strategy: {self.tune_strategy.upper()}\n\n")
            
            if self.tune_strategy == 'cv':
                f.write("Cross-Validation (CV) Tuning:\n")
                f.write("-" * 70 + "\n")
                f.write("Strategy: Uses GridSearchCV or RandomizedSearchCV\n")
                f.write("Objective: Find optimal hyperparameters using k-fold cross-validation\n")
                f.write("Advantages:\n")
                f.write("  - Fast and efficient\n")
                f.write("  - Reduces overfitting through multiple folds\n")
                f.write("  - Standard approach in scikit-learn\n")
                f.write("  - Provides cross-validation scores\n\n")
            
            elif self.tune_strategy == 'bootstrap':
                f.write("Bootstrap Tuning with CV:\n")
                f.write("-" * 70 + "\n")
                f.write("Strategy: Two-stage approach\n")
                f.write("  Stage 1: GridSearchCV/RandomizedSearchCV for initial tuning\n")
                f.write("  Stage 2: Bootstrap sampling for stability assessment\n")
                f.write("Objective: Find robust hyperparameters with uncertainty estimates\n")
                f.write("Advantages:\n")
                f.write("  - Provides both optimal and robust estimates\n")
                f.write("  - Bootstrap statistics give confidence intervals\n")
                f.write("  - Assesses parameter stability\n")
                f.write("  - Better for small/medium datasets\n\n")
                
                if self.bootstrap_results is not None:
                    f.write("Bootstrap Results Summary:\n")
                    f.write(f"  Mean OOB AUC: {self.bootstrap_results['oob_auc'].mean():.4f} "
                           f"(±{self.bootstrap_results['oob_auc'].std():.4f})\n")
                    f.write(f"  Mean Test AUC: {self.bootstrap_results['test_auc'].mean():.4f} "
                           f"(±{self.bootstrap_results['test_auc'].std():.4f})\n\n")
            
            f.write("Models Trained:\n")
            f.write("-" * 70 + "\n")
            f.write("Sklearn Models:\n")
            f.write("  - LogisticRegression\n")
            f.write("  - RandomForest\n")
            f.write("  - XGBoost\n")
            f.write("  - LightGBM (NEW)\n")
            f.write("  - CatBoost (NEW)\n")
            f.write("  - KNN\n")
            f.write("  - SVC\n")
            f.write("  - GaussianNB\n\n")
            
            f.write("Deep Learning Models (NEW):\n")
            f.write("  - RNN (Recurrent Neural Network)\n")
            f.write("  - LSTM (Long Short-Term Memory)\n\n")
            
            f.write("Data Balance States:\n")
            f.write("-" * 70 + "\n")
            f.write("  - Imbalanced (original class distribution)\n")
            f.write("  - Balanced with SMOTE (synthetic minority oversampling)\n\n")
            
            f.write("Model Evaluation:\n")
            f.write("-" * 70 + "\n")
            f.write("  Base models: Trained without hyperparameter tuning\n")
            f.write("  Tuned models: Fine-tuned with CV/Bootstrap\n")
            f.write("  Metrics: Accuracy, Precision, Recall, F1, ROC-AUC\n")
            f.write("  Visualizations: Confusion matrices, ROC curves, Feature importance\n\n")
        
        self.logger.info(f"Tuning summary saved: {summary_path}")
    
    # =====================================================================
    # RESULTS & EXPORT METHODS
    # =====================================================================
    
    def save_results_to_csv(self, results, n_vars, tune_by="gridsearch"):
        """Save model results to CSV files for analysis."""
        
        imbalanced_df = pd.DataFrame([r.to_dict() for r in results['imbalanced']])
        balanced_df = pd.DataFrame([r.to_dict() for r in results['balanced']])
        
        imbalanced_path = Path(f"./results/model_results_imbalanced_{n_vars}_{tune_by}.csv")
        balanced_path = Path(f"./results/model_results_balanced_{n_vars}_{tune_by}.csv")
        
        imbalanced_df.to_csv(imbalanced_path, index=False)
        balanced_df.to_csv(balanced_path, index=False)
        
        self.logger.info(f"Results saved: {imbalanced_path}, {balanced_path}")
    
    # =====================================================================
    # MAIN PIPELINE EXECUTION
    # =====================================================================
    
    def run(self):
        """Execute the complete prediction model pipeline."""
        self.setup_environment()
        
        # Load and explore data
        self.load_data(self.args.data_path)
        categorical_vars, continuous_vars = self.define_variables()
        
        # Generate initial distribution plots
        self.generate_distribution_plots(self.df, categorical_vars, continuous_vars, 
                                        skip_plots=self.args.skip_plots, max_plots=50, log_state=False)
        
        # Handle missing data
        self.report_missing_data()
        self.handle_missing_data(categorical_vars)
        
        # Log transformation
        self.df_log = self.safe_log_transform(self.df_imputed, continuous_vars)
        
        # Generate distribution plots after transformation
        self.generate_distribution_plots(self.df_log, categorical_vars, continuous_vars, 
                                        skip_plots=self.args.skip_plots, max_plots=50, log_state=True)
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            self.df_log.drop("RECURRENCE", axis=1), 
            self.df_log["RECURRENCE"], 
            test_size=0.2, 
            stratify=self.df_log["RECURRENCE"], 
            random_state=0
        )
        
        self.x_train, self.x_test = x_train, x_test
        self.y_train, self.y_test = y_train, y_test
        
        # Feature extraction
        no_include = ['RECURRENCE', 'ATA_2015_RISCO_INICIAL', 'ATA_2025_RISCO_INICIAL']
        self.X = self.x_train.drop(no_include, axis=1, errors='ignore')
        self.y = self.y_train
        
        self.logger.info(f"Recurrence counts: {self.y.value_counts()}")
        self.logger.info(f"Recurrence proportion: {self.y.mean():.2%}")
        
        # Scale and analyze correlations
        X_scaled = self.scaler.fit_transform(self.X)
        self.spearman_correlation_analysis(self.X, "median", skip=False)
        
        # Feature selection
        if self.args.variables == 'selected':
            self.variables_picked = self.select_features_lasso(X_scaled, self.y, self.X.columns)
            self.logger.info(f"Training with {len(self.variables_picked)} selected features via LASSO.")
        else:
            self.variables_picked = list(self.X.columns)
            self.logger.info(f"Using all {len(self.variables_picked)} features.")
        
        # Prepare train/test data with selected features
        x_train_sel, x_test_sel, y_train_sel, y_test_sel = train_test_split(
            self.X[self.variables_picked], self.y, 
            test_size=0.1, stratify=self.y, random_state=0
        )
        
        # Scale selected features
        X_train_scaled = self.scaler.fit_transform(x_train_sel)
        X_test_scaled = self.scaler.transform(x_test_sel)
        
        # SMOTE balancing
        smote = SMOTE(random_state=0)
        X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train_sel)
        X_smote_scaled = self.scaler.fit_transform(X_smote)
        
        # Train models
        self.logger.info("Training all models...")
        self.results = self.train_all_models(
            X_train_scaled, X_test_scaled, y_train_sel, y_test_sel,
            X_smote, y_smote,
            self.variables_picked, self.args.variables,
            use_random=self.args.use_random_search
        )
        
        # Plot comparisons
        self.plot_roc_curves(self.results['imbalanced'], 'Imbalanced', self.args.variables)
        self.plot_roc_curves(self.results['balanced'], 'Balanced_SMOTE', self.args.variables)
        
        # Plot feature importance
        if self.best_model_xgb_imbalanced is not None:
            self.plot_feature_importance(self.best_model_xgb_imbalanced,
                                        self.variables_picked, 'Imbalanced',
                                        self.args.variables, 'XGBoost')
        if self.best_model_xgb_balanced is not None:
            self.plot_feature_importance(self.best_model_xgb_balanced,
                                        self.variables_picked, 'Balanced_SMOTE',
                                        self.args.variables, 'XGBoost')
        
        # Save results
        tune_method = "random" if self.args.use_random_search else "grid"
        self.save_results_to_csv(self.results, self.args.variables, tune_by=tune_method)
        
        # Save tuning summary
        self.save_tuning_summary()
        
        self.logger.info("Training pipeline completed!")


def main():
    """Main entry point for the prediction model pipeline."""
    pipeline = PredictionModelPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
