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

id="multi-imputation-loop"
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "knn": KNeighborsClassifier(),
    "svc": SVC(probability=True),
    "nb": GaussianNB(),
    "lgbm": LGBMClassifier(),
    "cat": CatBoostClassifier(verbose=0)
}

id="loop-training"
results = {name: [] for name in models}

for i in range(10):  # tus imputaciones
    df_i = imputed_datasets[i]
    
    X = df_i.drop(columns=[target])
    y = df_i[target]
    
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]
        
        auc = roc_auc_score(y, y_pred)
        results[name].append(auc)
        
        
id="combine"
import numpy as np

for name in results:
    print(name, np.mean(results[name]), np.std(results[name]))