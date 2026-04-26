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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP LOGGING & DIRECTORIES
# ============================================================================
class Predictionmodel():
    def __init__(self, args):
        self.args = args
def setup_environment():
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
    return logging.getLogger(__name__)

logger = setup_environment()

# ============================================================================
# 1. ARGUMENT PARSER - Replace input()
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments for model training and prediction.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
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
    
    return parser.parse_args()

# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================
def load_data(data_path):
    logger.info(f"Loading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        return
    
    return df

def define_variables(df):
    """
    Classify variables into categorical and continuous.
    
    Args:
        df: DataFrame
    
    Returns:
        Tuple of (categorical_variables, continuous_variables)
    """
    categorical_variable = []
    continuous_variable = []
    
    for col in df.columns:
        if col in ['record_id']:
            continue
        if col == "SUBTYPE_FOLLI_PAPIL" or df[col].nunique() < 6:
            categorical_variable.append(col)
        else:
            continuous_variable.append(col)
    
    logger.info(f"Categorical variables: {len(categorical_variable)}")
    logger.info(f"Continuous variables: {len(continuous_variable)}")
    
    return categorical_variable, continuous_variable

def handle_missing_data(df, categorical_variables):
    """
    Impute missing values with validation.
    
    Args:
        df: DataFrame
        categorical_variables: list of categorical columns
        continuous_variables: list of continuous columns
    
    Returns:
        Imputed DataFrame
    """
    df_imputed = df.copy()
    imputation_log = {}
    
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() == 0:
            continue
        
        try:
            if col in categorical_variables:
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col].fillna(mode_val[0], inplace=True)
                    df_imputed[col] = df_imputed[col].astype('uint8')
                    imputation_log[col] = f"mode: {mode_val[0]}"
                    logger.info(f"Imputed categorical '{col}' with mode: {mode_val[0]}")
                else:
                    logger.warning(f"No mode found for {col}. Filling with 0.")
                    df_imputed[col].fillna(0, inplace=True)
                    imputation_log[col] = "mode: 0 (default)"
            else:
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
                imputation_log[col] = f"median: {median_val}"
                logger.info(f"Imputed continuous '{col}' with median: {median_val}")
        except Exception as e:
            logger.error(f"Error imputing {col}: {e}")
    
    # Save imputation log
    if imputation_log:
        with open("./logs/imputation_log.txt", "w") as f:
            for col, method in imputation_log.items():
                f.write(f"{col}: {method}\n")
    
    return df_imputed

# ============================================================================
# 3. SAFE LOG TRANSFORMATION
# ============================================================================

def safe_log_transform(df, columns, epsilon=2.220446049250313e-16):
    """
    Apply log transformation with safety checks for zeros/negatives.
    
    Args:
        df: DataFrame
        columns: list of column names to transform
        epsilon: small value to avoid log(0)
    
    Returns:
        Transformed DataFrame
    """
    df_transformed = df.copy()
    
    for col in columns:
        if col not in df_transformed.columns:
            continue
        
        min_val = df_transformed[col].min()
        
        if min_val <= 0:
            n_min = ((df_transformed[col] == min_val).sum())/len(df_transformed)
            df_transformed[col] = np.log(df_transformed[col] + epsilon)
            logger.warning(
                f"Column '{col}' had min value {min_val} in {n_min:.2%}. Shifted by {epsilon:.2e}"
            )
        else:
            df_transformed[col] = np.log(df_transformed[col])
        
    return df_transformed

# ============================================================================
# 4. MEMORY-EFFICIENT PLOT GENERATION
# ============================================================================

def generate_distribution_plots(df, categorical_vars, continuous_vars, 
                               map_variables, skip_plots=False, max_plots=50, log_state=False):
    """
    Generate distribution plots with memory management.
    
    Args:
        df: DataFrame
        categorical_vars: list of categorical columns
        continuous_vars: list of continuous columns
        map_variables: dictionary for recoding variable values
        skip_plots: if True, skip plot generation
        max_plots: maximum number of plots to generate
    """
    if skip_plots:
        logger.info("Skipping distribution plots to save memory")
        return
    
    plot_count = 0
    plot_dir = Path("./distribution")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for col in df.columns:
        if plot_count >= max_plots:
            logger.info(f"Reached max plots limit ({max_plots})")
            break
        
        if col == "RECURRENCE" or col not in categorical_vars + continuous_vars:
            continue
        
        try:
            plt.figure(figsize=(10, 6))
            
            if col in categorical_vars:
                ax = sns.countplot(data=df, x=col, hue="RECURRENCE")
                plt.title(f'Frequency of Recurrence by {col}')
                if col in map_variables:
                    ax.set_xticklabels(map_variables[col].values())
            else:
                sns.histplot(data=df, x=col, hue="RECURRENCE", kde=False, 
                           element="step", common_norm=False, alpha=0.6)
                plt.title(f'Distribution of Recurrence by {col}')
            
            plt.legend(title="Recurrence", labels=["No", "Yes"], prop={'size': 10})
            plt.tight_layout()
            
            path = plot_dir / f"freq_{col}{"_norm"if log_state else ""}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            plot_count += 1
            logger.info(f"Saved plot: {path}")
            
        except Exception as e:
            logger.error(f"Error plotting {col}: {e}")
            plt.close()

# ============================================================================
# 5. MISSING DATA REPORT
# ============================================================================

def report_missing_data(df):
    """
    Generate and log missing data report.
    
    Args:
        df: DataFrame
    
    Returns:
        DataFrame with missing data summary
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent_total = (df.isnull().sum() / len(df)).sort_values(ascending=False) * 100
    missing = pd.concat([total, round(percent_total, 2)], axis=1, 
                       keys=['Total', 'Percent'])
    missing = missing[missing['Total'] > 0]
    
    if len(missing) > 0:
        logger.info("\nMissing Data Report:")
        logger.info(f"\n{missing}")
        missing.to_csv("./logs/missing_data_report.csv")
    else:
        logger.info("No missing data found.")
    
    return missing

# ============================================================================
# 6. FEATURE SELECTION (LASSO)
# ============================================================================
def sperman_by_imputed(X_df, imputation_name, state = False):
    """Calcula y guarda correlaciones de Spearman"""
    if state:
        return
    corr_matrix = X_df.corr(method='spearman', min_periods=159)
    output_path = f"./variable_selection/spearman_corr_{imputation_name}.xlsx"
    corr_matrix.to_excel(output_path)
    
    ## Chart Correlation Heatmap
    plt.figure(figsize=(12,12))
    correlation = X_df.corr()
    sns.heatmap((correlation), annot=False, cmap=sns.color_palette("mako", as_cmap=True))
    plt.savefig("./variable_selection/heatmap_correlation.png")
    # plt.show(block=True)
    plt.close()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.6:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append(f"{col1} - {col2}: {corr_value:.3f}")
    
    txt_path = f"./variable_selection/spearman_high_results_{imputation_name}.txt"
    # with open(txt_path, "w") as f:
    #     f.write(f"Correlaciones de Spearman > |0.8| ({len(high_corr_pairs)} pares):\n")
    #     f.write("="*50 + "\n")
    #     for pair in high_corr_pairs:
    #         f.write(pair + "\n")
    
    print(f"Spearman correlations for {imputation_name} saved. {len(high_corr_pairs)} high correlations found.")
    return high_corr_pairs, corr_matrix

def select_features_lasso(X_scaled, y, X_columns, cv=10):
    """
    Select features using LassoCV with cross-validation.
    
    Args:
        X_scaled: scaled feature matrix
        y: target variable
        X_columns: column names
        cv: cross-validation folds
    
    Returns:
        List of selected features
    """
    logger.info(f"Starting LASSO feature selection with cv={cv}")
    
    lasso_cv = LassoCV(cv=cv, random_state=0, max_iter=10000)
    lasso_cv.fit(X_scaled, y)
    
    logger.info(f"LassoCV best alpha: {lasso_cv.alpha_}")
    
    sfm = SelectFromModel(lasso_cv, threshold=None, prefit=True)
    selected_idx = sfm.get_support(indices=True)
    selected_features = list(X_columns[selected_idx])
    
    path = Path("./variable_selection/selected_features_lasso.txt")
    with open(path, "a") as f:
        f.write(f"LassoCV selected {len(selected_features)} features with alpha={lasso_cv.alpha_}:\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    logger.info(f"Selected {len(selected_features)} features via LASSO")
    return selected_features

# ============================================================================
# 7. HYPERPARAMETER TUNING - GridSearch & RandomSearch & Bootstrap
# ============================================================================
def tune_model(model, param_grid, X_train, y_train, use_random=False, cv=5):
    """
    Tune model hyperparameters with GridSearchCV or RandomizedSearchCV.
    
    Args:
        model: sklearn model instance
        param_grid: parameter dictionary
        X_train: training features
        y_train: training labels
        use_random: use RandomizedSearchCV if True, else GridSearchCV
        cv: cross-validation folds
    
    Returns:
        Tuple of (best_model, best_params, search_object)
    """
    if use_random:
        logger.info(f"RandomizedSearchCV for {model.__class__.__name__}")
        search = RandomizedSearchCV(
            model, param_grid, cv=cv, n_iter=10,
            n_jobs=-1, random_state=0, scoring='roc_auc', verbose=1
        )
    else:
        logger.info(f"GridSearchCV for {model.__class__.__name__}")
        search = GridSearchCV(
            model, param_grid, cv=cv,
            n_jobs=-1, scoring='roc_auc', verbose=1
        )
    
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search

def tune_model_bootstrap(model, param_grid, x_data, y_data, n_bootstrap=300):
    """
    Tune model hyperparameters using bootstrap sampling.
    Default n_bootstrap = 300 
    https://stats.stackexchange.com/questions/14516/understanding-bootstrapping-for-validation-and-model-selection
    
    Args:
        model: model instance
        param_grid: parameter dictionary possible
        X_train: training features
        y_train: training labels
        n_bootstrap: number of bootstrap samples
    
    Returns:
        Tuple of (search.best_estimator_, search.best_params_, search)
        Dataframe with stats for each bootstrap iteration (optional)
    """
    roc_auc_score = None
    best_model = None
    best_params = None
    # Initializing DataFrame, to hold bootstrapped statistics
    bootstrapped_stats = pd.DataFrame()
    
    for i in range(n_bootstrap):
        x_train, y_train = resample(x_data, y_data, replace = True, n_samples=len(x_data), random_state=i) #random state int in [0, 2**32 - 1]
        x_test = x_data[~x_data.index.isin(x_train.index)] 
        y_test = y_data[~y_data.index.isin(y_train.index)] 
        
        model.fit(x_train, y_train)    

        bootstrapped_stats_i = pd.DataFrame(
            data=dict(
                best_model=best_model,
                best_params=best_params,
                roc_auc_score=roc_auc_score
            )
        )

        bootstrapped_stats = pd.concat(objs=[bootstrapped_stats, bootstrapped_stats_i])
        
    logger.info(f"Bootstrap tuning completed. Best ROC AUC score: {roc_auc_score:.4f}")
    logger.info(f"Best params from bootstrap: {best_params}")

    return best_model, best_params
    
# ============================================================================
# 8. MODEL METRICS WITH SCOPE MANAGEMENT !!!!!!!!!!
# ============================================================================

class ModelResult:
    """
    Store model results with complete scope management.
    Ensures all metrics and predictions are accessible throughout pipeline.
    """
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

def model_metrics(y_train, y_test, train_preds, test_preds, model_name,
                  fine_tune, balanced_state, variables_used, y_scores_test=None,
                  y_train_scores=None):
    result = ModelResult(model_name, balanced_state, variables_used)
    result.fine_tune = fine_tune   # agregar atributo en ModelResult
    
    result.train_accuracy = accuracy_score(y_train, train_preds)
    result.train_roc_auc = roc_auc_score(y_train, y_train_scores) if y_train_scores is not None else None
    
    result.test_accuracy = accuracy_score(y_test, test_preds)
    result.test_precision = precision_score(y_test, test_preds, zero_division=0)
    result.test_recall = recall_score(y_test, test_preds, zero_division=0)
    result.test_f1 = f1_score(y_test, test_preds, zero_division=0)
    
    # CORRECCIÓN: usar scores para ROC AUC
    if y_scores_test is not None:
        result.test_roc_auc = roc_auc_score(y_test, y_scores_test)
    else:
        logger.warning("y_scores_test is None, setting test_roc_auc to None")
        result.test_roc_auc = None
    
    result.y_scores_test = y_scores_test
    result.y_test = y_test
    result.y_pred_test = test_preds
    
    logger.info(f"\n{model_name} - {'Tuned' if fine_tune else 'Base'}")
    logger.info(f"Train Accuracy: {result.train_accuracy:.4f}")
    if result.train_roc_auc:
        logger.info(f"Train ROC AUC: {result.train_roc_auc:.4f}")
    logger.info(f"Test Accuracy: {result.test_accuracy:.4f}")
    logger.info(f"Test Precision: {result.test_precision:.4f}")
    logger.info(f"Test Recall: {result.test_recall:.4f}")
    logger.info(f"Test F1: {result.test_f1:.4f}")
    logger.info(f"Test ROC AUC: {result.test_roc_auc:.4f}")
    
    # Plot and Save Confusion Matrix
    train_confusion_matrix = confusion_matrix(y_train, train_preds)
    test_confusion_matrix = confusion_matrix(y_test, test_preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ['0', '1']
    sns.heatmap(train_confusion_matrix, annot=True, cmap='Blues', ax=axes[0], fmt="d", xticklabels=labels, yticklabels=labels)
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')
    axes[0].set_title('Train Confusion Matrix')
    sns.heatmap(test_confusion_matrix, annot=True, cmap='Blues', ax=axes[1], fmt="d", xticklabels=labels, yticklabels=labels)
    axes[1].set_xlabel('Predicted labels')
    axes[1].set_ylabel('True labels')
    axes[1].set_title('Test Confusion Matrix')
    plt.title(f"{model_name} - {'Fine-tuned' if fine_tune else 'Base'}")
    
    cm_dir = Path(f"./results/confusion_matrix/{balanced_state}/{variables_used}_variables")
    cm_dir.mkdir(parents=True, exist_ok=True)
    path = cm_dir / f"cm_{model_name}_{'tuned' if fine_tune else 'base'}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved: {path}")
    
    return result

# ============================================================================
# 9. ROC PLOTTING - ALL MODELS COMPARISON
# ============================================================================

def plot_roc_curves(results_list, balanced_state, n_vars):
    """
    Plot ROC curves for all models in comparison.
    
    Args:
        results_list: list of ModelResult objects with y_scores_test populated
        balanced_state: data balance state ('Imbalanced' or 'Balanced_SMOTE')
        n_vars: 'all' or 'selected'
    """
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
    logger.info(f"ROC plot saved: {path}")

# ============================================================================
# 10. FEATURE IMPORTANCE - BEST MODEL ONLY
# ============================================================================

def plot_feature_importance(best_model, feature_names, balanced_state, n_vars, model_name="XGBoost"):
    """
    Plot feature importance for best model only.
    
    Args:
        best_model: trained model with feature_importances_ attribute
        feature_names: list of feature names
        balanced_state: data balance state
        n_vars: 'all' or 'selected'
        model_name: name of model
    """
    if not hasattr(best_model, 'feature_importances_'):
        logger.warning(f"{best_model.__class__.__name__} has no feature_importances_")
        return
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
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
    logger.info(f"Feature importance plot saved: {path}")

# ============================================================================
# 11. MODEL SAVING WITH JOBLIB !!!!!!!!!! hyperparam
# ============================================================================
def save_best_hyperparameters(model_name, best_params, variables_used, balanced_state, searched_params):
    path = f"./results/best_params.txt"
    with open(path, "a") as file:
        file.write(f"{model_name}, {searched_params}, {variables_used}, {balanced_state}: {best_params}\n")


def save_best_model(model, model_name, balanced_state, n_vars, features, metrics):
    """
    Save best model with metadata using joblib and pickle.
    
    Args:
        model: trained model instance
        model_name: name of model
        balanced_state: data balance state
        n_vars: 'all' or 'selected'
        features: list of features used
        metrics: model performance metrics (dict)
    """
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
    
    # Save with joblib
    joblib.dump(metadata, filename)
    logger.info(f"Model saved with joblib: {filename}")
    
    # Also save with pickle for compatibility
    pickle_name = model_dir / f"best_model_{model_name}_{balanced_state}_{n_vars}_pickle.pkl"
    with open(pickle_name, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Pickle backup saved: {pickle_name}")

def load_model(model_path):
    """
    Load model from joblib or pickle file.
    
    Args:
        model_path: path to model file
    
    Returns:
        Dictionary with model and metadata
    """
    try:
        if str(model_path).endswith('.pkl'):
            # Try joblib first
            try:
                metadata = joblib.load(model_path)
                logger.info(f"Model loaded with joblib: {model_path}")
            except:
                # Fall back to pickle
                with open(model_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Model loaded with pickle: {model_path}")
        else:
            raise ValueError("Model file must be .pkl")
        
        return metadata
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# ============================================================================
# 12. TRAIN ALL MODELS
# ============================================================================

def train_all_models(X_train, X_test, y_train, y_test, X_smote, y_smote, 
                    variables_picked, n_vars, use_random=False):
    """
    Train all models on both imbalanced and balanced data.
    
    Args:
        X_train, X_test: train/test features (imbalanced)
        y_train, y_test: train/test labels (imbalanced)
        X_smote, y_smote: SMOTE balanced data
        variables_picked: selected features
        n_vars: 'all' or 'selected'
        use_random: use RandomizedSearchCV
    
    Returns:
        Dictionary with all model results organized by balance state
    """
    
    results = {}
    scaler = StandardScaler()
    
    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_smote_scaled = scaler.fit_transform(X_smote)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODELS ON IMBALANCED DATA")
    logger.info("="*70)
    
    imbalanced_results = []
    balanced_results = []
    best_model_xgb_imbalanced = None
    best_model_xgb_balanced = None
    
    # Define models and hyperparameters
    models_config = {
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
    
    # Train models on imbalanced data
    for model_name, config in models_config.items():
        logger.info(f"\n--- Training {model_name} (Imbalanced) ---")
        
        # Base model
        model = config['model']
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_train_scores = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
        y_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
        
        result = model_metrics(y_train, y_test, y_pred_train, y_pred_test,
                             model_name, fine_tune=False, 
                             balanced_state="Imbalanced",
                             variables_used=X_train_scaled.shape[1],
                             y_scores_test=y_scores,
                            #  y_train_preds=y_pred_train,
                             y_train_scores=y_train_scores)
        imbalanced_results.append(result)
        
        # Tuned model
        best_model, best_params, _ = tune_model(
            config['model'], config['params'],
            X_train_scaled, y_train, use_random=use_random
        )
        y_pred_train_tuned = best_model.predict(X_train_scaled)
        y_pred_test_tuned = best_model.predict(X_test_scaled)
        y_train_scores_tuned = best_model.predict_proba(X_train_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_train_tuned
        y_scores_tuned = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_test_tuned
        
        result_tuned = model_metrics(y_train, y_test, y_pred_train_tuned, 
                                    y_pred_test_tuned, model_name, fine_tune=True,
                                    balanced_state="Imbalanced",
                                    variables_used=X_train_scaled.shape[1],
                                    y_scores_test=y_scores_tuned,
                                    # y_train_preds=y_pred_train_tuned,
                                    y_train_scores=y_train_scores_tuned)
        imbalanced_results.append(result_tuned)
        
        save_best_hyperparameters(model_name, best_params, X_train_scaled.shape[1], "imbalanced data.", "Random Search" if use_random else "Grid Search")
        
        # Keep XGBoost for feature importance !!!!!!!!!!!!!!!
        if model_name == 'XGBoost':
            best_model_xgb_imbalanced = best_model
            metrics_dict = result_tuned.to_dict()
            save_best_model(best_model, model_name, "Imbalanced", n_vars,
                          variables_picked, metrics_dict)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODELS ON BALANCED DATA (SMOTE)")
    logger.info("="*70)
    
    # Train models on SMOTE balanced data
    for model_name, config in models_config.items():
        logger.info(f"\n--- Training {model_name} (Balanced SMOTE) ---")
        
        # Base model
        model = config['model']
        model.fit(X_smote_scaled, y_smote)
        y_pred_train = model.predict(X_smote_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_train_scores = model.predict_proba(X_smote_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
        y_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred_test
        
        result = model_metrics(y_smote, y_test, y_pred_train, y_pred_test,
                             model_name, fine_tune=False,
                             balanced_state="Balanced_SMOTE",
                             variables_used=X_smote_scaled.shape[1],
                             y_scores_test=y_scores,
                            #  y_train_preds=y_pred_train,
                             y_train_scores=y_train_scores)
        balanced_results.append(result)
        
        # Tuned model
        best_model, best_params, _ = tune_model(
            config['model'], config['params'],
            X_smote_scaled, y_smote, use_random=use_random
        )
        y_pred_train_tuned = best_model.predict(X_smote_scaled)
        y_pred_test_tuned = best_model.predict(X_test_scaled)
        y_train_scores_tuned = best_model.predict_proba(X_smote_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_train_tuned
        y_scores_tuned = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred_test_tuned
        
        result_tuned = model_metrics(y_smote, y_test, y_pred_train_tuned,
                                    y_pred_test_tuned, model_name, fine_tune=True,
                                    balanced_state="Balanced_SMOTE",
                                    variables_used=X_smote_scaled.shape[1],
                                    y_scores_test=y_scores_tuned,
                                    # y_train_preds=y_pred_train_tuned,
                                    y_train_scores=y_train_scores_tuned)
        balanced_results.append(result_tuned)
        
        save_best_hyperparameters(model_name, best_params, X_train_scaled.shape[1], "imbalanced data.", "Random Search" if use_random else "Grid Search")

        # Keep XGBoost for feature importance !!!!!!!!!!!!
        if model_name == 'XGBoost':
            best_model_xgb_balanced = best_model
            metrics_dict = result_tuned.to_dict()
            save_best_model(best_model, model_name, "Balanced_SMOTE", n_vars,
                          variables_picked, metrics_dict)
    
    results['imbalanced'] = imbalanced_results
    results['balanced'] = balanced_results
    results['best_model_xgb_imbalanced'] = best_model_xgb_imbalanced
    results['best_model_xgb_balanced'] = best_model_xgb_balanced
    
    return results

# ============================================================================
# 14. SAVE RESULTS TO CSV
# ============================================================================

def save_results_to_csv(results, n_vars, tune_by = "grisearch"):
    """
    Save model results to CSV files for analysis.
    
    Args:
        results: dictionary with model results
        n_vars: 'all' or 'selected'
    
    Returns:
        Tuple of (imbalanced_df, balanced_df)
    """
    
    imbalanced_df = pd.DataFrame([r.to_dict() for r in results['imbalanced']])
    balanced_df = pd.DataFrame([r.to_dict() for r in results['balanced']])
    
    imbalanced_path = Path(f"./results/model_results_imbalanced_{n_vars}_{tune_by}.csv")
    balanced_path = Path(f"./results/model_results_balanced_{n_vars}_{tune_by}.csv")
    
    imbalanced_df.to_csv(imbalanced_path, index=False)
    balanced_df.to_csv(balanced_path, index=False)





# ============================================================================
# 13. MAIN EXECUTION BLOCK
# ============================================================================

def main():
    logger = setup_environment()

    args = parse_arguments()    

    df = load_data(args.data_path)
    
    map_variables = { #Last Changes 
        "SEX": {0: "Female", 1: "Male"}, #1-> 0, 2->1
        "RADIOTHERAPY EXPOSURE": {1: "Yes", 0: "No"}, #2->0
        "FAMILY HISTORY OF THYROID CANCER": {1: "Yes", 0: "No"}, #2->0,
        "EUTHYROIDISM": {0: "No", 1: "Yes"}, #2->0
        "HYPOTHYROIDISM": {0: "No", 1: "Yes"},
        "HYPERTHYROIDISM": {0: "No", 1: "Yes"},
        "THYROIDECTOMY APPROACH": {0: "Total", 1: "Total + Lymphadenectomy"}, #1->0, 2->1
        "TYPEOFRESECTION": {0: "R0", 1: "R1", 2: "R2"},#1->0, 2->1, 3->2   
        "HISTOLOGY": {0: "Papilar", 1: "Folicular", 2: "Hurtle Cells"}, #1->0, 2->1, 3->2
        "SUBTYPE_FOLLI_PAPIL": {0: "Minimally invasive", 1: "Encapsulated invasive", 2: "Widely invasive",
            3: "Classic", 4: "Follicular variant", 5: "Encapsulated", 6: "Diffuse sclerosant",
            7: "High cells", 8: "Colunar cells", 9: "Cribiform-morular", 10: "Hobnail", 
            11: "Warthin-like", 12: "Oncocytic", 13: "Trabecular/Solid", 14: "Classic and Follicular",
            15: "Follicular and oncocytic"}, #1->0, 2->1, 3->2, 15->3, 16->4, 17->5, 5->6, 6->7, 7->8, 8->9, 9->10, 10->11, 11->12, 12->13, 13->14, 14->15
        "EXTRATHYROIDALEXTENSION": {0: "Absent", 1: "Microscopic", 2: "Macroscopic"}, #1->0, 2->1, 3->2
        "MULTICENTRIC": {1: "Yes", 0: "No"}, #2->0
        "MULTICENTER_BILATERAL": {1: "Yes", 0: "No"}, #2->0
        "VASCULARINVASION": {1: "Yes", 0: "No"}, #2->0
        "PERINEURALINVASION": {1: "Yes", 0: "No"}, #2->0
        "POSITIVELYMPHNODEN1": {0: "No excision", 2: "Yes", 1: "No"}, #2->1, 1->2
        "EXTRANODALEXTENSION": {1: "Yes", 0: "No"}, #2->0
        "TNMT": {0: "Tx", 1: "T1", 2: "T2", 3: "T3", 4: "T4"},
        "HASHIMOTO THYROIDITIS": {1: "Yes", 0: "No"}, #2->0
        "TNMN": {1: "N0", 2: "N1a", 3: "N1b", 0: "Nx"}, #3->0, 0->1, 1->2, 2->3
        "TNMM": {0: "M0", 1: "M1"}, 
        "STAGE": {0: "I", 1: "II", 2: "III", 3: "IV"}, #1->0, 2->1, 3->2, 4->3
        "ATA_2015_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio", 3: "Alto"}, 
        "ATA_2025_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio o Bajo", 3: "Intermedio o Alto", 4: "Alto"},
        "RAI": {1: "Yes", 0: "No"}, #2->0
        "ANTI TG PRE RAI (POSITIVE or NEGATIVE)": {1: "Positivo", 0: "Negativo"}, #2->0
        "ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)": {1: "Positivo", 0: "Negativo"}, #2->0
        "RECURRENCE": {0: "No Recurrence", 1: "Recurrence"}
        }
    
    ### Define variables by n output unique values
    categorical_variable, continuous_variable = define_variables(df)
    
    ### distribution plots before imputation and log transformation
    generate_distribution_plots(df, categorical_variable, continuous_variable, 
                                map_variables, skip_plots = args.skip_plots, max_plots=50, log_state=False)
    
    ### About missing values
    report_missing_data(df)

    ### imputation data 
    df_imputed = handle_missing_data(df, categorical_variable)
    
    ### Attempt to normalize by log transformation
    df_log = safe_log_transform(df_imputed, continuous_variable)
    
    ### distribution plots after log transformation
    generate_distribution_plots(df_log, categorical_variable, continuous_variable, 
                               map_variables, skip_plots = args.skip_plots, max_plots=50, log_state=True)
    
    ### 80/20 train/test (Internal validation)
    x_train, x_test, y_train, y_test = train_test_split(df_log.drop("RECURRENCE", axis=1), df_log["RECURRENCE"], test_size=0.2, stratify=df_log["RECURRENCE"], random_state=0)
    df_log, df_ival, df_log["RECURRENCE"], df_ival["RECURRENCE"] = x_train, x_test, y_train, y_test

    no_include = ['RECURRENCE', 'ATA_2015_RISCO_INICIAL', 'ATA_2025_RISCO_INICIAL'] #the lsat vars are for comparative analysis
    X = df_log.drop(no_include, axis=1)
    y = df_log['RECURRENCE']  # outcome variable

    print(f"Recurrence counts: {y.value_counts()}")
    print(f"Recurrence proportion: {y.mean():.2%}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sperman_by_imputed(X, "median", state = False)  # True = computed

    # ---------------------------
    # variables: all vs selected
    # ---------------------------
    if args.variables == 'selected':
        # Escalar para LASSO
        variables_picked = select_features_lasso(X_scaled, y, X.columns)
        logger.info(f"Training {len(variables_picked)} features with LASSO.")
        
    else:
        variables_picked = list(X.columns)
        logger.info(f"Using all {len(variables_picked)} features.")
    
    x_train, x_test, y_train, y_test = train_test_split(X[variables_picked], y, test_size=0.1, stratify=y, random_state=0)

    # Balanceo con SMOTE
    smote = SMOTE(random_state=0)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)

    ### Training model
    logger.info("Training models...")
    results = train_all_models(
        x_train, x_test, y_train, y_test,
        x_smote, y_smote,
        variables_picked, args.variables,
        use_random=args.use_random_search
    )
    
    # Plot ROC comparatives
    plot_roc_curves(results['imbalanced'], 'Imbalanced', args.variables)
    plot_roc_curves(results['balanced'], 'Balanced_SMOTE', args.variables)
    
    # Features importance for best XGBoost models???????????
    if results['best_model_xgb_imbalanced'] is not None:
        plot_feature_importance(results['best_model_xgb_imbalanced'],
                                variables_picked, 'Imbalanced',
                                args.variables, 'XGBoost')
    if results['best_model_xgb_balanced'] is not None:
        plot_feature_importance(results['best_model_xgb_balanced'],
                                variables_picked, 'Balanced_SMOTE',
                                args.variables, 'XGBoost')
    
    # Guardar resultados en CSV
    tune_method = "random" if args.use_random_search else "grid"
    save_results_to_csv(results, args.variables, tune_by=tune_method)
    
    logger.info("Training done!")

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================
if __name__ == "__main__":
    main()