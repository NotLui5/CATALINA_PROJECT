## Logistic Regression
import pandas as pd
import numpy as np 

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
d1["NUMBEROFLYMPHNODEEXCISION"] = d1["NUMBEROFLYMPHNODEEXCISION"].fillna(0)
d1["NUMBEROFPOSITIVELYMPHNODEEXCISION"] = d1["NUMBEROFPOSITIVELYMPHNODEEXCISION"].fillna(0)
d1["LN RATIO"] = d1["LN RATIO"].fillna(0)
d1["SIZEOFPOSITIVELYMPHNODE(cm)"] = d1["SIZEOFPOSITIVELYMPHNODE(cm)"].fillna(0)
d1["EXTRANODALEXTENSION"] = d1["EXTRANODALEXTENSION"].fillna(0)
d1["TSH PRE RAI"] = d1["TSH PRE RAI"].fillna(0) 
d1["TG PRE RAI"] = d1["TG PRE RAI"].fillna(0)                                 
d1["ANTI TG PRE RAI (POSITIVE or NEGATIVE)"] = d1["ANTI TG PRE RAI (POSITIVE or NEGATIVE)"].fillna(0)
d1["ANTI TG PRE RAI "] = d1["ANTI TG PRE RAI "].fillna(0)
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
corr_matrix_spearman.to_excel("spearman_corr.xlsx")

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


with open("./variable_selection/sperman_high_results.txt", "w") as file:
    for _ in sperman_high:
        file.write(f"{_} \n")
print("sperman_high_results.txt created to consider in variable selection.")

# Load data and split into training and testing sets to LASSO #########
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit LassoCV to find the best alpha but with NaN value we have to impute
#imputer = SimpleImputer(strategy='median')  # o 'mean', 'most_frequent' ALL WERE SAME RESULTS multiple imputation (replacing missing values with multiple plausible estimates),52 regression imputation (using fitted models to predict missing values), reference values (eg, mean or median age-sex values)
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)
#X_imputed = imputer.fit_transform(X)

#Iterative Imputer (regression imputation)
imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior= False) #
X_imputed = imp.fit_transform(X)

scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_imputed)
# X_test_scaled = scaler.transform(X_test_imputed)
X_scaled = scaler.fit_transform(X_imputed)

lasso_cv = LassoCV(cv=5, random_state=0)

# lasso_cv.fit(X_train_scaled, y_train)
lasso_cv.fit(X_scaled, y)
print("Best alpha:", lasso_cv.alpha_)

# Select features with LASSO
sfm = SelectFromModel(lasso_cv, threshold = None, prefit=True)
# sfm.estimator.coef_
# X_train_selected = sfm.transform(X_train_scaled)
# X_test_selected = sfm.transform(X_test_scaled)
# X_selected = sfm.transform(X_scaled)
# Obtener nombres de características seleccionadas
selected_feature_idx = sfm.get_support(indices=True)
selected_features_lasso = X.columns[selected_feature_idx]

with open("./variable_selection/variables.txt", "a") as file:
    file.write(f"With {lasso_cv} selected: {list(selected_features_lasso)}")
print("Features selected:", list(selected_features_lasso))


# Eventos en cada base antes de LASSO 80/20 Training/Testing 
# XGBoost / RandomForest / LightGBM / Bayesian Models with Priors with pymc

# Test train split LOGISTIC REGRESION / naive bayes
from sklearn.linear_model import LogisticRegression
x = list(selected_features_lasso)
x.append('RECURRENCE')
d1_lasso = d1[x]
X_train, X_test, y_train, y_test = train_test_split(d1_lasso.drop('RECURRENCE', axis = 1), d1_lasso['RECURRENCE'], train_size=0.9, random_state=42) # pmsampsize’ and ‘pmvalsampsize
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