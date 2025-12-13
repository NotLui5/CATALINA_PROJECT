## Logistic Regression
import pandas as pd
import numpy as np 
import os

### Import the data
data_base = pd.read_excel("database/Base_pos_limpeza_V9_with Record ID.xlsx")
# data_base.sample(5)
## Fix col values
# d1 = pd.get_dummies(data_base, drop_first=True,) # If all were like yes/not 
from sklearn.preprocessing import OrdinalEncoder
d1 = data_base

sex_code = ['Female', 'Male']
enc = OrdinalEncoder(categories = [sex_code])
d1['SEX'] = enc.fit_transform(d1[['SEX']])
print("SEX transformed to ordinal encoder")

typeresection_code = ['R0', 'R1', 'R2']
enc = OrdinalEncoder(categories = [typeresection_code])
d1['TYPEOFRESECTION'] = enc.fit_transform(d1[['TYPEOFRESECTION']])
print("TYPEOFRESECTION transformed to ordinal encoder")

d1['RECURRENCE'] = d1['ATA2025LAST_ULTIMA_CONSULTA'].replace([1,2,3],0).replace(4, 1)
for col in d1.columns:
    if d1[col].dtype == 'object':
        # d1[col]
        d1[col] = pd.to_numeric(d1[col], errors='coerce')
print("type of cols transformed to numeric type")

### Organize NA values
# combine follicular subtype with papillar subtype, so to this we change value follicular with the following number of pappillary en after combine them in one col
d1["FOLLICULARSUBTYPE"] = d1["FOLLICULARSUBTYPE"].replace(1, 15).replace(2, 16).replace(3, 17)
d1["SUBTYPE_FOLLI_PAPIL"]  = d1['FOLLICULARSUBTYPE'].fillna(d1['PAPILLARYSUBTYPE'])
print("New SUBTYPE_FOLLI_PAPIL col from FOLLICULARSUBTYPE _ and _ PAPILLARYSUBTYPE cols")

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
d1.loc[:, col_fill_na] = d1.loc[:, col_fill_na].fillna(0)

print("Na value filled with 0")

# Remove variables innecesary:
cols_remove = ["record_id", "ATA2015_ULTIMA_CONSULTA", "ATA2025LAST_ULTIMA_CONSULTA", "ATA_2025_RISCO_INICIAL", "ATA_2015_RISCO_INICIAL", "PAPILLARYSUBTYPE", "FOLLICULARSUBTYPE", "MUTATION", "TSH POST OP ", "TG POST OP", "ANTI TG POST OP (POSITIVE or NEGATIVE)", "ANTI TG POST OP VALUE", "ANTI TG POST OP (POSITIVE or NEGATIVE)", "ANTI TG POST OP VALUE", ]
d1 = d1.drop(cols_remove, axis=1) # Remove col innecesary
# d1.isna().sum() /359 * 100

# Sperman Correlation and LASSO to select best variables
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  #Regression imputation​​
from sklearn.impute import IterativeImputer  #Regression imputation​​

X = d1.drop('RECURRENCE', axis=1)
y = d1['RECURRENCE']  # outcome variable
            
# Calculate Spearman correlation matrix with pandas, also we can w spermancor but there is an error about minimun data  https://www.yourdatateacher.com/2021/05/05/feature-selection-in-machine-learning-using-lasso-regression/
corr_matrix_spearman = X.corr(method='spearman')
corr_matrix_spearman.to_excel("./variable_selection/spearman_corr.xlsx")

sperman_high = []
for col in corr_matrix_spearman.columns:
    for value in corr_matrix_spearman[col]:
        if value > 0.8:
            if not corr_matrix_spearman[corr_matrix_spearman[col] > 0.8].index[0] == col:
                # print(f"For {col}, has a {value} sperman correlation with {corr_matrix_spearman[corr_matrix_spearman[col] > 0.8].index[0]}")
                sperman_high.append(f"{col} - {value} - {corr_matrix_spearman[corr_matrix_spearman[col] > 0.8].index[0]}")
        elif value < -0.8:
            if not corr_matrix_spearman[corr_matrix_spearman[col] < -0.8].index[0] == col:
                # print(f"For {col}, has a {value} sperman correlation with {corr_matrix_spearman[corr_matrix_spearman[col] < -0.8].index[0]}")
                sperman_high.append(f"{col} - {value} - {corr_matrix_spearman[corr_matrix_spearman[col] < -0.8].index[0]}")

path_sperman = "./variable_selection/sperman_high_results.txt"
if not os.path.exists(path_sperman):
    with open(path_sperman, "w") as file:
        for _ in sperman_high:
            file.write(f"{_} \n")
print("sperman_high_results.txt created to consider in variable selection.")

# Load data and split into training and testing sets to LASSO #########
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit LassoCV to find the best alpha but with NaN value we have to impute    
# Simple imputation
imputation_moda = ['RADIOTHERAPY EXPOSURE', 'FAMILY HISTORY OF THYROID CANCER', 'THYROID DISEASE PREOP', 'EXTRATHYROIDALEXTENSION', 'POSITIVELYMPHNODEN1', 'HASHIMOTO THYROIDITIS', 'ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)', 'SUBTYPE_FOLLI_PAPIL']
imputation_median_mean = ['BMI', 'TUMORSIZE (cm)', 'RAIDOSE', 'ANTI TG FOLLOW UP','TG FOLLOW UP']

imputer = SimpleImputer(strategy='median')
array1 = imputer.fit_transform(X[imputation_median_mean])
imputer = SimpleImputer(strategy='most_frequent')
array2 = imputer.fit_transform(X[imputation_moda])
X_imputed_median = np.hstack((array1, array2))

imputer = SimpleImputer(strategy='mean')
array1 = imputer.fit_transform(X[imputation_median_mean])
X_imputed_mean = np.hstack((array1, array2))

# Iterative imputation
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior= False) #
X_imputed_iter = imp.fit_transform(X)

scaler = StandardScaler()
X_scaled_iter = scaler.fit_transform(X_imputed_iter)
X_scaled_mean = scaler.fit_transform(X_imputed_mean)
X_scaled_median = scaler.fit_transform(X_imputed_median)

#Selection
def select_laso_imput(X_scaled_imputed, y, imput, cv_laso=5, random_state_laso=0):
    lasso_cv = LassoCV(cv=cv_laso, random_state=random_state_laso)
    lasso_cv.fit(X_scaled_imputed, y)
    print("Best alpha:", lasso_cv.alpha_)    
    sfm = SelectFromModel(lasso_cv, threshold = None, prefit=True)
    selected_feature_idx = sfm.get_support(indices=True)
    selected_features_lasso = X.columns[selected_feature_idx]
    
    path_variables = "./variable_selection/variables.txt"
    with open(path_variables, "a") as file:
        file.write(f"With {lasso_cv} selected {imput}: {list(selected_features_lasso)}\n")
    print(f"Features selected imputation {imput}: {list(selected_features_lasso)}")
    
    return selected_features_lasso
    
var_iter = select_laso_imput(X_scaled_iter, y, "iterative")
var_mean = select_laso_imput(X_scaled_mean, y, "mean")
var_median = select_laso_imput(X_scaled_median, y, "median")



# Eventos en cada base antes de LASSO 80/20 Training/Testing 
# XGBoost / RandomForest / LightGBM / Bayesian Models with Priors with pymc

# Test train split LOGISTIC REGRESION / naive bayes
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X_imputed_mean, y, train_size=0.9, random_state=42) # pmsampsize’ and ‘pmvalsampsize
LogReg = LogisticRegression(solver = 'lbfgs')
LogReg.fit(X_train, y_train)

#Scoring the model 
LogReg.score(X_test, y_test)

#Understanding score
y_pred = LogReg.predict(X_test)
np.sum(y_pred == y_test) / len(y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
classification_report(y_test, y_pred)



# RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

...