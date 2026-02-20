###########################################################################
### PRE-PROCESSING
###########################################################################

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

### Work paths
os.makedirs("./variable_selection", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./models_comparison", exist_ok=True)
os.makedirs("./distribution", exist_ok=True)

### Import data TRAIN
data_base1 = pd.read_excel("./database/Base_pos_limpeza_V9_with Record ID.xlsx")
data_base2 = pd.read_excel("./database/HEE_limpia 6.xlsx")
data_base3 = pd.read_excel("./database/AMBATO_Base_limpia_3.xlsx")

data_base2["ATA2025LAST_ULTIMA_CONSULTA"] = data_base2["ATA2015_ULTIMA_CONSULTA"]#We won't use ata2015, but HEEE didn't reported it so, we taked 2025 in all bases except HEEE (is for concatenate level)
data_base2.rename(columns = {'BMI ': 'BMI', "ID": "record_id"}, inplace=True)

### Join just one database:
data_base = pd.concat([data_base1, data_base2], ignore_index=True, names= list(data_base1.columns), verify_integrity=True, sort = False)
path_base = "./database/HEE_Brazil_Ambato.xlsx"
if not os.path.exists(path_base):
    data_base.to_excel("./database/HEE_and_Brazil.xlsx", index=False)

# THYROIDECTOMY APPROACH, QUE SIGNIFICA 3? SOLO HAY 1 Y 2 EN EL LIBRO DE CODIGOS
# TNMM, QUE SIGNIFICA 3? SOLO HAY 1 Y 2 EN EL LIBRO DE CODIGOS
# ANTI TG FOLLOW UP (POSITIVE or NEGATIVE), QUE SIGNIFICA 2.38? SOLO HAY 1 Y 2 EN EL LIBRO DE CODIGOS
# BRAZIL
# RAI QUE SIGNIFICA 0, PORQUE EL LIBRO DE COD HAY 1 Y 2 NO 0 Y 1 COMO AQUI
# ANTI TG FOLLOW UP (POSITIVE or NEGATIVE), QUE SIGNIFICA 3? SOLO HAY 1 Y 2 EN EL LIBRO DE CODIGOS

## Fix col values
# d1 = pd.get_dummies(data_base, drop_first=True,) # If all were like yes/not 
from sklearn.preprocessing import OrdinalEncoder
d1 = data_base.copy()

d1['SEX'] = d1['SEX'].replace({'Female': 1, 'Male': 2, 
                               '1': 1, '2': 2})
d1['TYPEOFRESECTION'] = d1['TYPEOFRESECTION'].replace({'R0': 1, 'R1': 2, 'R2': 3,
                                                       '1': 1, '2': 2, '3': 3})
print("SEX and TYPEOFRESECTION transformed to ordinal encoder")

d1['RECURRENCE'] = d1['ATA2025LAST_ULTIMA_CONSULTA'].replace([1,2,3],0).replace(4, 1)
idx= 0

for col in d1.columns:
    # sns.countplot(data=d1, x=col)
    if d1[col].dtype == 'object':
        # d1[col]
        d1[col] = pd.to_numeric(d1[col], errors='coerce')
           
    path_dens = f"./distribution/dens_{col}.png"
    path_count = f"./distribution/freq_{col}.png"
    if not os.path.exists(path_dens): 
        sns.histplot(data=d1, x=col, hue="RECURRENCE", multiple="dodge", shrink=.8)
        plt.title(f'Distribution of {col}')
        plt.legend(prop={'size': 10})
        plt.savefig(path_dens)
        plt.show(block=True)
        
        sns.histplot(data=d1, x=col, hue="RECURRENCE", stat="density", multiple="dodge", shrink=.8)
        plt.title(f'Distribution of {col}')
        plt.legend(prop={'size': 10})
        plt.savefig(path_count)
        plt.show(block=True)
        
    print(f"type of cols transformed to {d1[col].dtype}")

path_dis = "./distribution/d1_distributions.png"
if not os.path.exists(path_dis):
    sns.pairplot(d1, hue="RECURRENCE")
    plt.savefig(path_dis)
### Organize NA values
# combine follicular subtype with papillar subtype, so to this we change value follicular with the following number of pappillary en after combine them in one col
d1["FOLLICULARSUBTYPE"] = d1["FOLLICULARSUBTYPE"].replace({1: 15, 2: 16, 3: 17})
d1['SUBTYPE_FOLLI_PAPIL'] = d1['FOLLICULARSUBTYPE'].combine_first(d1['PAPILLARYSUBTYPE'])
print("New SUBTYPE_FOLLI_PAPIL col from FOLLICULARSUBTYPE _ and _ PAPILLARYSUBTYPE cols")
d1.isna().sum() /777 * 100
# fillna = 0 porque no era neesario que se reporte ese valor
col_fill_na = [
    "NUMBEROFLYMPHNODEEXCISION",
    "NUMBEROFPOSITIVELYMPHNODEEXCISION",
    "LN RATIO",
    "SIZEOFPOSITIVELYMPHNODE(cm)",
    "EXTRANODALEXTENSION",
    "TSH PRE RAI",
    "TG PRE RAI",
    "ANTI TG PRE RAI (POSITIVE or NEGATIVE)",
    "ANTI TG PRE RAI "
]
for col in col_fill_na:
    if col in d1.columns:
        d1[col] = d1[col].fillna(0)
    # d1[col_fill_na] = d1[col_fill_na].fillna(0)
    print(f"Na values in {col} filled with 0")

# Remove variables innecesary:
cols_remove = [
    "record_id", 
    "ATA2015_ULTIMA_CONSULTA", 
    "ATA2025LAST_ULTIMA_CONSULTA", 
    "ATA_2025_RISCO_INICIAL", 
    "ATA_2015_RISCO_INICIAL", 
    "PAPILLARYSUBTYPE", 
    "FOLLICULARSUBTYPE", 
    "MUTATION", 
    "TSH POST OP ", 
    "TG POST OP", 
    "ANTI TG POST OP (POSITIVE or NEGATIVE)", 
    "ANTI TG POST OP VALUE",
    "TNMN", "TNMM", "STAGE" #AUTHOR (PS) decision!!!
    ]
d1 = d1.drop(cols_remove, axis=1) # Remove col innecesary
print(f"columns: {cols_remove} were removed")
# d1.isna().sum() /777 * 100

# Sperman Correlation and LASSO to select best variables
from sklearn.model_selection import train_test_split
##### 80/10/10 train/dev/test

X = d1.drop('RECURRENCE', axis=1)
y = d1['RECURRENCE']  # outcome variable
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2,stratify=y, random_state=0)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=0.5,stratify=y_dev, random_state=0)
print(f"\nDistribución de clases:")
print(y.value_counts())
print(f"Proporción: {y.mean():.2%} de recurrencia")

X_mean = X_train.copy()
X_median = X_train.copy()
X_iter = X_train.copy()

# Fit LassoCV to find the best alpha but with NaN value we have to impute    
# Simple imputation
# Columnas a imputar
imputation_median_mean = ['TNMT', 'BMI', 'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY', 'TUMORSIZE (cm)', 'RAIDOSE', 'ANTI TG FOLLOW UP','TG FOLLOW UP', 'TSH FOLLOW UP']
imputation_moda = ['TYPEOFRESECTION', 'MULTICENTRIC', 'RAI', 'MULTICENTER_BILATERAL', 'RADIOTHERAPY EXPOSURE', 'FAMILY HISTORY OF THYROID CANCER', 'THYROID DISEASE PREOP', 'EXTRATHYROIDALEXTENSION', 'POSITIVELYMPHNODEN1', 'HASHIMOTO THYROIDITIS', 'ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)', 'SUBTYPE_FOLLI_PAPIL', 'VASCULARINVASION', 'PERINEURALINVASION']


# 2. Aplicar SimpleImputer (Media)
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
imputer_mode = SimpleImputer(strategy='most_frequent')

X_mean[imputation_median_mean] = imputer_mean.fit_transform(X_mean[imputation_median_mean])
X_mean[imputation_moda] = imputer_mode.fit_transform(X_mean[imputation_moda])

# 3. Aplicar SimpleImputer (Mediana) ## We choose mean
# imputer_median = SimpleImputer(strategy='median')
# X_median[imputation_median_mean] = imputer_median.fit_transform(X_median[imputation_median_mean])
# X_median[imputation_moda] = imputer_mode.fit_transform(X_median[imputation_moda])

# Iterative imputation
# from sklearn.experimental import enable_iterative_imputer  #Regression imputation​​
# from sklearn.impute import IterativeImputer  #Regression imputation​​
# imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior= False) #
# X_iter = imp.fit_transform(X_iter)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.feature_selection import SelectFromModel
scaler = StandardScaler()
# X_scaled_iter = scaler.fit_transform(X_iter)
X_scaled_mean = scaler.fit_transform(X_mean)
# X_scaled_median = scaler.fit_transform(X_median)

# Calculate Spearman correlation matrix with pandas, also we can w spermancor but there is an error about minimun data  https://www.yourdatateacher.com/2021/05/05/feature-selection-in-machine-learning-using-lasso-regression/
def sperman_by_imputed(X_df, imputation_name):
    """Calcula y guarda correlaciones de Spearman"""
    X_df = pd.DataFrame(X_df)
    corr_matrix = X_df.corr(method='spearman')
    output_path = f"./variable_selection/spearman_corr_{imputation_name}.xlsx"
    corr_matrix.to_excel(output_path)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append(f"{col1} - {col2}: {corr_value:.3f}")
    
    txt_path = f"./variable_selection/spearman_high_results_{imputation_name}.txt"
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"Correlaciones de Spearman > |0.8| ({len(high_corr_pairs)} pares):\n")
            f.write("="*50 + "\n")
            for pair in high_corr_pairs:
                f.write(pair + "\n")
    
    print(f"Spearman correlations for {imputation_name} saved. {len(high_corr_pairs)} high correlations found.")
    return high_corr_pairs

# sperman_by_imputed(X_iter, "iterative")
sperman_by_imputed(X_mean, "mean")
# sperman_by_imputed(X_median, "median")

#Selection
from sklearn.linear_model import LassoCV, ElasticNetCV
def select_laso_imput(X_scaled_imputed, y, imput, X_columns, cv_laso=5, random_state_laso=0):
    lasso_cv = LassoCV(cv=cv_laso, random_state=random_state_laso)
    lasso_cv.fit(X_scaled_imputed, y)
    print("Best alpha:", lasso_cv.alpha_)    
    sfm = SelectFromModel(lasso_cv, threshold = None, prefit=True)
    selected_feature_idx = sfm.get_support(indices=True)
    selected_features_lasso = list(X_columns[selected_feature_idx])
    if "LN RATIO" not in selected_features_lasso:
        selected_features_lasso.append("LN RATIO")
    path_variables = f"./variable_selection/variables_{imput}.txt"
    
    if not os.path.exists(path_variables):
        with open(path_variables, "w") as file:
            file.write(f"With {lasso_cv} selected {imput}: \n")
            for selected in selected_features_lasso:
                file.write(f"{selected}\n")
    print(f"Features selected imputation {imput}: {selected_features_lasso}")
    
    return selected_features_lasso

X_columns = X_train.columns
# var_iter = select_laso_imput(X_scaled_iter, y_train, "iterative", X_columns)
var_mean = select_laso_imput(X_scaled_mean, y_train, "mean", X_columns)
# var_median = select_laso_imput(X_scaled_median, y_train, "median", X_columns)


# def select_elastic_net_cv(X_imputed, y, cv_k = 10, X_cols = list, path = str):
#     l1_r = [.05, .1, .3, .6, .9, .93, .96, .98, .99, 1]
    
#     EN_model_mean = ElasticNetCV(l1_ratio=l1_r, random_state=0, cv = cv_k,  n_jobs = -1, fit_intercept = True)
#     EN_model_mean.fit(X_imputed, y)
    
#     print("Model Coefficients:", EN_model_mean.coef_)
#     selected_indices = np.where(EN_model_mean.coef_ != 0)[0]
#     print(f"\nNumber of selected variables: {len(selected_indices)}")
#     vars = list(X_cols[selected_indices])
#     with open(path, "a") as file:
#         file.write(f"ElasticNETCV with alpha: {EN_model_mean.alpha_} and l1_ratio: {EN_model_mean.l1_ratio_}" 
#                    f"by {cv_k} folds, we selected {len(selected_indices)} features: \n"
#                    f"{vars}\n")
#     print(f"Variables: {vars}")
    
#     return vars

# en_mean = select_elastic_net_cv(X_mean, y_train, X_cols = X_columns, path=f"./variable_selection/EN_variables_mean.txt")
# en_median = select_elastic_net_cv(X_median, y_train, X_cols = X_columns, path=f"./variable_selection/EN_variables_median.txt")
# en_iter = select_elastic_net_cv(X_iter, y_train, X_cols = X_columns, path=f"./variable_selection/EN_variables_iter.txt")

# Eventos en cada base antes de LASSO 80/20 Training/Testing 
# XGBoost / RandomForest / LightGBM / Bayesian Models with Priors with pymc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve, 
                           auc, classification_report)
from sklearn.pipeline import Pipeline

print("\n" + "="*60)
print("ENTRENAMIENTO DE MODELOS DE REGRESIÓN LOGÍSTICA")
print("="*60)
# strategies = ['mean', 'median', 'iterative'] # iF DISTRIBUTION NORMAL WILL BE MEAN

results = {}
results['mean'] = {'X_imputed': X_mean, 'X_scaled': X_scaled_mean, 'selected_features': var_mean}
# X_iter = pd.DataFrame(X_iter, columns=X_columns)
# results['iterative'] = {'X_imputed': X_iter, 'X_scaled': X_scaled_iter, 'selected_features': var_iter}
# results['median'] = {'X_imputed': X_median, 'X_scaled': X_scaled_median, 'selected_features': var_median}

# Usar imputación iterativa como estrategia inicial
# imp = IterativeImputer(max_iter=20, random_state=42)
# X = X_dev.copy()
# X_imputed_dev = imp.fit_transform(X)
# X_imputed_dev = pd.DataFrame(X_imputed_dev, columns=X.columns)

from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
imputer_mode = SimpleImputer(strategy='most_frequent')
X_mean_dev = X_dev.copy()
X_mean_dev[imputation_median_mean] = imputer_mean.fit_transform(X_mean_dev[imputation_median_mean])
X_mean_dev[imputation_moda] = imputer_mode.fit_transform(X_mean_dev[imputation_moda])

# # 3. Aplicar SimpleImputer (Mediana)
# imputer_median = SimpleImputer(strategy='median')
# X_median_dev = X_dev.copy()
# X_median_dev[imputation_median_mean] = imputer_median.fit_transform(X_median_dev[imputation_median_mean])
# X_median_dev[imputation_moda] = imputer_mode.fit_transform(X_median_dev[imputation_moda])

# Dividir datos
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_dev.shape}")
print(f"Recurrence in train: {y_train.mean():.2%}")
print(f"Recurrence in test: {y_dev.mean():.2%}")

### 2. DEFINIR Y PROBAR MÚLTIPLES MODELOS
print("\n" + "="*60)
print("ENTRENANDO Y COMPARANDO MÚLTIPLES MODELOS")
print("="*60)

all_metrics = []

from sklearn.model_selection import cross_validate
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, eval = False):
    """Evalúa un modelo y retorna métricas"""
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    #ROC_AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)  # o roc_auc_score(y_test, y_pred_proba)

    # Graficar
    path_graph = f"./auc_graphs/auc_dev_{model_name}"
    if eval == True:
        path_graph = f"./auc_graphs/auc_test_{model_name}"
    if not os.path.exists(path_graph):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve \n {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(path_graph)
        plt.show()
    
    # También puedes obtener el AUC directamente
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc_score:.3f}")
    # Métricas
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Validación cruzada para obtener métricas más robustas
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, 
        cv=cv, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    
    metrics.update({
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_precision_mean': cv_results['test_precision'].mean(),
        'cv_precision_std': cv_results['test_precision'].std(),
        'cv_recall_mean': cv_results['test_recall'].mean(),
        'cv_recall_std': cv_results['test_recall'].std(),
        'cv_f1_mean': cv_results['test_f1'].mean(),
        'cv_f1_std': cv_results['test_f1'].std(),
        'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
        'cv_roc_auc_std': cv_results['test_roc_auc'].std()
    })
    
    return metrics, model


trained_models = {}

# A. LOGISTIC REGRESSION
# https://www.youtube.com/watch?v=BHok3wJpmf0
# https://www.youtube.com/watch?v=3giTXZbyf1Q
print("\n1. Logistic Regression")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Crear pipeline para regresión logística
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        penalty = "l2",#"elasticnet",
        solver= 'sag',#'saga',
        C=0.018, # Regularización más fuerte
        # l1_ratio=0.65
    ))
])


# lr_metrics, lr_model = evaluate_model(
#     lr_pipeline, X_median[var_median], X_median_dev[var_median], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
#     "Logistic_Regression Median"
# )
# all_metrics.append(lr_metrics)
# trained_models['Logistic_Regression Median'] = lr_model
# var_mean=['AGEATDIGNOSIS', 'BMI', 'TUMORSIZE (cm)', 'NUMBEROFLYMPHNODEEXCISION', 'NUMBEROFPOSITIVELYMPHNODEEXCISION', 'LN RATIO', 'TNMT', 'RAIDOSE', 'TSH PRE RAI', 'TG PRE RAI', 'ANTI TG PRE RAI ', 'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY', 'TSH FOLLOW UP', 'TG FOLLOW UP', 'ANTI TG FOLLOW UP', 'SUBTYPE_FOLLI_PAPIL']

lr_metrics, lr_model = evaluate_model(
    lr_pipeline, X_mean[var_mean], X_mean_dev[var_mean], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
    "Logistic_Regression Mean"
)

all_metrics.append(lr_metrics)
trained_models['Logistic_Regression Mean'] = lr_model
print("intercept (b):", lr_model[1].intercept_)
print("pendiente (w):", lr_model[1].coef_)
# For binary classification, model.coef_ will be a 2D array of shape (1, n_features)
coefficients = lr_model[1].coef_[0]

# 4. Create a pandas Series to associate coefficients with feature names
coef_series = pd.Series(coefficients, index=X_train.columns)

# 5. Order the variables by the absolute magnitude of their coefficients
# Higher absolute value indicates greater importance
ordered_importance = coef_series.abs().sort_values(ascending=False)

# 6. Display the ordered list
print("Ordered Variable Importance (by absolute coefficient magnitude):")
print(ordered_importance)

# To see the actual coefficients and their signs:
ordered_coefficients = coef_series.reindex(ordered_importance.index)
print("\nOrdered Coefficients:")
print(ordered_coefficients)

# lr_metrics, lr_model = evaluate_model(
#     lr_pipeline, X_iter[var_iter], X_imputed_dev[var_iter], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
#     "Logistic_Regression Iter"
# )
# all_metrics.append(lr_metrics)
# trained_models['Logistic_Regression Iter'] = lr_model

# print(all_metrics)

# B. RANDOM FOREST
print("\n2. Random Forest")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

rf_metrics, rf_trained = evaluate_model(
    rf_model, X_train, X_test, y_train, y_test,
    "Random_Forest"
)
all_metrics.append(rf_metrics)
trained_models['Random_Forest'] = rf_trained

# C. XGBOOST
print("\n3. XGBoost")
try:
    from xgboost import XGBClassifier
    
    # Calcular scale_pos_weight para manejar desbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        # use_label_encoder=False,
        eval_metric='logloss',
        enable_categorical=False
    )
    
    xgb_metrics, xgb_trained = evaluate_model(
        xgb_model, X_train, X_test, y_train, y_test,
        "XGBoost"
    )
    all_metrics.append(xgb_metrics)
    trained_models['XGBoost'] = xgb_trained
    
except ImportError:
    print("XGBoost no está instalado. Instala con: pip install xgboost")

# D. LIGHTGBM
print("\n4. LightGBM")
try:
    from lightgbm import LGBMClassifier
    
    lgb_model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1  # Silenciar output
    )
    
    lgb_metrics, lgb_trained = evaluate_model(
        lgb_model, X_train, X_test, y_train, y_test,
        "LightGBM"
        )
    all_metrics.append(lgb_metrics)
    trained_models['LightGBM'] = lgb_trained
    
except ImportError:
    print("LightGBM no está instalado. Instala con: pip install lightgbm")

# E. BAYESIAN LOGISTIC REGRESSION (aproximación)
print("\n5. Bayesian Logistic Regression (aproximado)")
try:
    # Usamos regresión logística con regularización tipo "l2" como aproximación bayesiana
    from sklearn.linear_model import LogisticRegressionCV
    
    bayesian_lr = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        l1_ratios=None
    )
    
    bayesian_metrics, bayesian_trained = evaluate_model(
        bayesian_lr, X_train, X_test, y_train, y_test,
        "Bayesian_Logistic"
    )
    all_metrics.append(bayesian_metrics)
    trained_models['Bayesian_Logistic'] = bayesian_trained
    
except Exception as e:
    print(f"Error con Bayesian Logistic: {e}")

# F. GRADIENT BOOSTING (scikit-learn)
print("\n6. Gradient Boosting (scikit-learn)")
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gb_metrics, gb_trained = evaluate_model(
    gb_model, X_train, X_test, y_train, y_test,
    "Gradient_Boosting"
)
all_metrics.append(gb_metrics)
trained_models['Gradient_Boosting'] = gb_trained

# Configurar validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Probar diferentes escaladores
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'maxabs': MaxAbsScaler()
}

all_results1 = []

for strategy in strategies:
    print(f"\n{'='*50}")
    print(f"Resultados para imputación: {strategy}")
    print('='*50)
    
    X_original = results[strategy]['X_imputed']
    selected_features = results[strategy]['selected_features']
    
    if len(selected_features) == 0:
        print(f"No se seleccionaron características con {strategy}. Usando todas.")
        selected_features = X_original.columns
    
    # Filtrar características seleccionadas
    X_selected = X_original[selected_features]
    
    for scaler_name, scaler in scalers.items():
        print(f"\nEscalador: {scaler_name}")
        
        # Crear pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Para manejar clases desbalanceadas
            ))
        ])
        
        # Validación cruzada
        cv_scores = cross_val_score(
            pipeline, X_selected, y, 
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        print(f"AUC-ROC (CV): {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Entrenar en conjunto completo para obtener métricas adicionales
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, 
            random_state=42, stratify=y
        )
        
        # Ajustar pipeline
        pipeline.fit(X_train, y_train)
        
        # Predicciones
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {
            'imputation': strategy,
            'scaler': scaler_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'n_features': len(selected_features)
        }
        
        all_results1.append(metrics)
        
        # Mostrar resultados
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        print(f"AUC-ROC (test): {metrics['auc_roc']:.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de confusión:")
        print(cm)
        
        # Reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['No Recurrencia', 'Recurrencia']))

### Análisis de resultados y selección del mejor modelo
results_df = pd.DataFrame(all_results1)
print("\n" + "="*60)
print("COMPARACIÓN DE TODOS LOS MODELOS")
print("="*60)
print(results_df.sort_values('auc_roc', ascending=False).to_string())

# Guardar resultados
results_df.to_csv("./results/comparacion_modelos_lbfgf.csv", index=False)

# Encontrar el mejor modelo
best_idx = results_df['auc_roc'].idxmax()
best_model = results_df.loc[best_idx]
print(f"\n{'='*60}")
print("MEJOR MODELO ENCONTRADO:")
print('='*60)
for key, value in best_model.items():
    print(f"{key}: {value}")

# Entrenar el mejor modelo final con todos los datos
# print("\n" + "="*60)
# print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
# print("="*60)

# best_strategy = best_model['imputation']
# best_scaler_name = best_model['scaler']
# selected_features = results[best_strategy]['selected_features']

# if len(selected_features) == 0:
#     selected_features = X.columns

# X_final = results[best_strategy]['X_imputed'][selected_features]

# # Crear pipeline final
# final_pipeline = Pipeline([
#     ('scaler', scalers[best_scaler_name]),
#     ('classifier', LogisticRegression(
#         solver='lbfgs',
#         max_iter=1000,
#         random_state=42,
#         class_weight='balanced'
#     ))
# ])

# # Entrenar con todos los datos
# final_pipeline.fit(X_final, y)

# # Guardar el modelo final
# import joblib
# joblib.dump(final_pipeline, "./models/logistic_regression_final_model.pkl")
# joblib.dump(selected_features, "./models/selected_features.pkl")

# print(f"Modelo final guardado en: ./models/logistic_regression_final_model.pkl")
# print(f"Características utilizadas: {list(selected_features)}")

# # Obtener coeficientes del modelo final
# if hasattr(final_pipeline.named_steps['classifier'], 'coef_'):
#     coefficients = final_pipeline.named_steps['classifier'].coef_[0]
#     feature_importance = pd.DataFrame({
#         'feature': selected_features,
#         'coefficient': coefficients,
#         'abs_coefficient': abs(coefficients)
#     }).sort_values('abs_coefficient', ascending=False)
    
#     print("\nImportancia de características (coeficientes):")
#     print(feature_importance.to_string())
    
#     # Guardar importancia de características
#     feature_importance.to_csv("./results/feature_importance.csv", index=False)

# print("\n¡Análisis completado!")

# RANDOM FOREST
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression

# ...