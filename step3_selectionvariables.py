import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Follow up time in years to generate selection variables with specific data."
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

def sperman_by_imputed(X_df, imputation_name, state = False):
    """Calcula y guarda correlaciones de Spearman"""
    if state:
        return
    corr_matrix = X_df.corr(method='spearman', min_periods=159)
    output_path = f"./variable_selection/spearman_corr_{imputation_name}.xlsx"
    corr_matrix.to_excel(output_path)
    
    ## Chart Correlation Heatmap
    plt.figure(figsize=(12,12))
    correlation = X_df.corr(method='spearman')
    sns.heatmap((correlation), annot=False, cmap=sns.color_palette("mako", as_cmap=True))
    plt.savefig("./variable_selection/heatmap_correlation.png")
    # plt.show(block=True)
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
    
    print(f"Spearman correlations for {imputation_name} saved. {len(high_corr_pairs)} high correlations found.")
    return high_corr_pairs, corr_matrix

if __name__ == "__main__":
    args = parse_args()
    time_followup = args.time_followup
    datalocated_fold = args.datalocated_fold
    
    if time_followup >= 50:
        df = pd.read_csv(f"./{datalocated_fold}/data_train_mice10.csv") #It arrives from R, all 10 imputations. data_m10  <- rbind(data_m10, complete(gm,1:10))}
    else:
        df = pd.read_csv(f"./{datalocated_fold}/data{time_followup}ly_train_mice10.csv") 

    target = "recurrence"
    # Variables to decide if we want to include them in the model that accept missing values 
    col_nan_default = ["number_posit_ln", "ln_ratio", "size_posit_ln", "raidose", "tg_pre_rai"]
    col_to_compare = ['ata_2015', 'ata_2025']
    col_time = ["follow_months"] ### there is not include in last step of mice

    df = pd.get_dummies(df, columns=['thy_disease_preop'], drop_first=False,dtype=float)
    df.rename({"thy_disease_preop_0": "euthyroidism", 
                "thy_disease_preop_1": "hypothyroidism", 
                "thy_disease_preop_2": "hyperthyroidism"},
            axis=1, inplace=True)
    # df1.drop(columns=col_to_compare).corr()[target].sort_values(ascending=False)
    sperman_by_imputed(df.drop(columns=col_to_compare), "mice", state = False)     

    df["w"] = 1/10

    X = df.drop(columns=[target, "w", *col_nan_default, *col_to_compare])
    y = df[target]
    w = df["w"]

    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X)

    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=10000,
            random_state=42
        ))
    ])

    param_grid = {
    "model__C": np.logspace(-4, 2, 20),
    "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X, y, model__sample_weight=w)

    best_model = search.best_estimator_.named_steps["model"]

    coef = best_model.coef_.ravel()
    selected_features = X.columns[np.abs(coef) > 1e-6].tolist()

    print("Best parameters:", search.best_params_)
    print("Best cross-validated AUC:", search.best_score_)
    print("Selected variables:", selected_features)
    # for x in penalties:
    #     lasso = LogisticRegression(
    #         l1_ratio=x, # Use l1_ratio=1 is lasso, 0 <= l1_ratio <= 1 'elasticnet'
    #         solver='saga',
    #         max_iter=5000
    #     )

    #     lasso.fit(X_scaled, y, sample_weight=w)

    #     coef = lasso.coef_[0]
    #     selected_features = X.columns[np.abs(coef) > 1e-6]

    #     print(selected_features)

    #     selected_features = list(selected_features)
        
    best_l1_ratio = search.best_params_["model__l1_ratio"]

    if np.isclose(best_l1_ratio, 1.0):
        method = "lasso"
    else:
        method = "elasticnet"
    
    if time_followup >= 50:
        path = Path(f"./variable_selection/selected_features_{method}.txt") ######################## chage according to size data followup
    else: 
        path = Path(f"./variable_selection/selected_features_{method}{time_followup}ly.txt") ######################## chage according to size data followup

    
    with open(path, "w") as f:
        f.write(f"==============================================================\n")
        f.write(f"{method.upper()} selected {len(selected_features)} features:\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
        f.write(f"==============================================================\n")
        


    ##### With all cases of follow up.
    ### Lasso
    # x = ['sex', 'thyrodectomy_approach', 'type_resection', 'extra_thy_exten',
    #    'multicentric', 'multicent_bilat', 'vascular_inv', 'positive_ln',
    #    'extranod_exten', 'tnm_t', 'tnm_m', 'stage', 'rai', 'tsh_follow',
    #    'tg_follow'],


    # ### ElasticNet
    # y = ['sex', 'age', 'radiotherapy', 'family_history', 'thy_disease_preop',
    #        'bmi', 'thyrodectomy_approach', 'type_resection', 'histology',
    #        'tumor_size', 'extra_thy_exten', 'multicentric', 'multicent_bilat',
    #        'vascular_inv', 'perineural_inv', 'positive_ln', 'number_ln_exc',
    #        'extranod_exten', 'hashimoto', 'tnm_t', 'tnm_n', 'tnm_m', 'stage',
    #        'rai', 'follow_months', 'tsh_follow', 'tg_follow', 'subtype']