# # https://github.com/kartik2433/Code_Unnati_Marathon/blob/main/server/model/Heart%20Pediction.ipynb
# https://github.com/Navjotkhatri/CARDIOVASCULAR-RISK-PREDICTION/blob/main/Cardiovascular_Risk_Prediction.ipynb
# https://github.com/Preetirai-tech/Cardiovascular-Risk-Prediction/blob/main/Cardiovascular_Risk_Prediction.ipynb
###############################################################################################
### DATASET LOADING
###############################################################################################

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

### Work paths
os.makedirs("./variable_selection", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./models_comparison", exist_ok=True)
os.makedirs("./distribution", exist_ok=True)

## Some functions 
def define_variables(df): 
    ## Classify variables in categorical and continuous:
    categorical_variable = []
    continuous_variable = []
    
    for i in df.columns:
        if i in ['record_id']:
            pass
        if i == "SUBTYPE_FOLLI_PAPIL" or df[i].nunique() <6: ### PAPILLARY SUBTYPE nunique=14
            categorical_variable.append(i)
        else:
            continuous_variable.append(i)
    return categorical_variable, continuous_variable
 
## Dictionary coder
map_variables = {
    "SEX": {1: "Female", 2: "Male"},
    "RADIOTHERAPY EXPOSURE": {1: "Yes", 2: "No"},
    "FAMILY HISTORY OF THYROID CANCER": {1: "Yes", 2: "No"},
    "THYROID DISEASE PREOP": {1: "Euthyroidism", 2: "Hypothyroidism", 3: "Hyperthyroidism"},
    "THYROIDECTOMY APPROACH": {1: "Total", 2: "Total + Lymphadenectomy"},
    "TYPEOFRESECTION": {1: "R0", 2: "R1", 3: "R2"},
    "HISTOLOGY": {1: "Papilar", 2: "Folicular", 3: "Hurtle Cells"},
    "SUBTYPE_FOLLI_PAPIL": {1: "Minimally invasive", 2: "Encapsulated invasive", 3: "Widely invasive",
        5: "Diffuse sclerosant", 6: "High cells", 7: "Colunar cells",
        8: "Cribiform-morular", 9: "Hobnail", 10: "Warthin-like",
        11: "Oncocytic", 12: "Trabecular/Solid", 13: "Classic and Follicular",
        14: "Follicular and oncocytic", 15: "Classic", 16: "Follicular variant", 
        17: "Encapsulated"},
    "EXTRATHYROIDALEXTENSION": {1: "Absent", 2: "Microscopic", 3: "Macroscopic"},
    "MULTICENTRIC": {1: "Yes", 2: "No"},
    "MULTICENTER_BILATERAL": {1: "Yes", 2: "No"}, 
    "VASCULARINVASION": {1: "Yes", 2: "No"},
    "PERINEURALINVASION": {1: "Yes", 2: "No"},
    "POSITIVELYMPHNODEN1": {0: "No excision", 1: "Yes", 2: "No"},
    "EXTRANODALEXTENSION": {1: "Yes", 2: "No"},
    "TNMT": {0: "Tx", 1: "T1", 2: "T2", 3: "T3", 4: "T4"},
    "HASHIMOTO THYROIDITIS": {1: "Yes", 2: "No"},
    "TNMN": {0: "N0", 1: "N1a", 2: "N1b", 3: "Nx"},
    "TNMM": {1: "M0", 2: "M1"},
    "STAGE": {1: "I", 2: "II", 3: "III", 4: "IV"},
    "ATA_2015_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio", 3: "Alto"},
    "ATA_2025_RISCO_INICIAL": {1: "Bajo", 2: "Intermedio o Bajo", 3: "Intermedio o Alto", 4: "Alto"},
    "RAI": {1: "Yes", 2: "No"},
    "ANTI TG PRE RAI (POSITIVE or NEGATIVE)": {1: "Positivo", 2: "Negativo"},
    "ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)": {1: "Positivo", 2: "Negativo"},
    "RECURRENCE": {0: "No Recurrence", 1: "Recurrence"}
}

### Import data 
path_base = "./database/hee_brazil_ambato_base.csv"
if not os.path.exists(path_base):
    data_base1 = pd.read_csv("./database/Base_pos_limpeza_V9_with Record ID.csv", sep = ";", encoding='utf-8-sig')
    data_base2 = pd.read_csv("./database/HEE_limpia 6.csv", sep = ";", encoding='utf-8-sig')
    data_base3 = pd.read_csv("./database/AMBATO_Base_limpia_3.csv", sep = ";")

    data_base1["RAI"].replace("0,00", 2, inplace=True) 
    
    ##!!! We won't use ata2015, but HEEE and AMBATO bases didn't reported ATA2025.
    data_base2["ATA2025LAST_ULTIMA_CONSULTA"] = data_base2["ATA2015_ULTIMA_CONSULTA"]
    data_base3["ATA2025LAST_ULTIMA_CONSULTA"] = data_base3["ATA2015_ULTIMA_CONSULTA"]

    ##! Diferent colnames
    # for x in data_base1.columns:
    #     if x not in data_base3.columns:
    #         print(x)
    data_base2.rename(columns={"ID": "record_id", "BMI ": "BMI"}, inplace=True)
    data_base3.rename(columns={"TNMT ": "TNMT"}, inplace=True)
    if not data_base1.columns.equals(data_base2.columns):
        print("Check colnames database1 and 2")
    if not data_base2.columns.equals(data_base3.columns):
        print("Check colnames database2 and 3")
        
    ### Join just one database:
    df = pd.concat([data_base1, data_base2, data_base3], ignore_index=True, 
                names= list(data_base1.columns), verify_integrity=True, 
                sort = False)
    
    ## Create RECURRENCE outcome variable by ATA.
    df['RECURRENCE'] = df['ATA2025LAST_ULTIMA_CONSULTA']

    ### Data wrangling
    # ANTI TG FOLLOW UP (POSITIVE or NEGATIVE), QUE SIGNIFICA 2.38? SOLO HAY 1 Y 2 EN EL LIBRO DE CODIGOS
    ## Fix col values
    sex_map = {'Female': 1,
            'Male': 2}
    typeresection_map = {'R0': 1, 
                        'R1': 2, 
                        'R2': 3}
    thy_map = {3: 1} # TNMM 3 is unknown so it will be 1, and THYROIDECTOMY APPROACH 3 is updated so it will be 1, explained by PS
    
    df['SEX'] = df['SEX'].replace(sex_map) 
    df['TYPEOFRESECTION'] = df['TYPEOFRESECTION'].replace(typeresection_map)
    df["TNMM"] = df["TNMM"].replace(thy_map)
    df["THYROIDECTOMY APPROACH"] = df["THYROIDECTOMY APPROACH"].replace(thy_map)
        
    # Recode 'Follicular' values to avoid corresponding 'Papillary' numeric code and merge both into a single column
    df["FOLLICULARSUBTYPE"] = df["FOLLICULARSUBTYPE"].replace({1: 15, 2: 16, 3: 17})
    df['SUBTYPE_FOLLI_PAPIL'] = df['FOLLICULARSUBTYPE'].combine_first(df['PAPILLARYSUBTYPE'])
    
    # Remove variables innecesary:
    cols_remove = [
        "record_id", 
        "ATA2015_ULTIMA_CONSULTA", 
        "ATA2025LAST_ULTIMA_CONSULTA", 
        # "ATA_2025_RISCO_INICIAL", # COMPARATIVO
        # "ATA_2015_RISCO_INICIAL", # COMPARATIVO
        "PAPILLARYSUBTYPE", #it's merged in SUBTYPE_FOLLI_PAPIL
        "FOLLICULARSUBTYPE", #it's merged in SUBTYPE_FOLLI_PAPIL
        "MUTATION", 
        "TSH POST OP ", 
        "TG POST OP", 
        "ANTI TG POST OP (POSITIVE or NEGATIVE)", 
        "ANTI TG POST OP VALUE",
        
        ]
    df = df.drop(cols_remove, axis=1) # Remove col innecesary
    
    categorical_variable, continuous_variable = define_variables(df)
    for i in df.columns:
        df[i] = df[i].astype(str).str.replace(',', '.', regex=False)
        if i in continuous_variable:
            df[i] = pd.to_numeric(df[i], errors='coerce')
        else:
            df[i] = pd.to_numeric(df[i], errors='coerce').astype("Int64")            
    
    df['RECURRENCE'] = df['RECURRENCE'].replace([1,2,3],0).replace(4, 1)
    df['RECURRENCE'] = df.pop('RECURRENCE')
    #Data final to start imputation and modeling
    df.to_csv(path_base, index=False)

else:
    #Data final to start imputation and modeling
    df = pd.read_csv(path_base)
    categorical_variable, continuous_variable = define_variables(df)

### Dataset first View 
df.head() #.tail

### Dataset Rows & Columns count
df.shape

### Dataset information
df.info()

### Dataset Describe 
df.describe(include = "all")
for i in df.columns:
    print("No. of unique values in", i , "is" , df[i].nunique())

### Duplicate Values 
len(df[df.duplicated()])

### NaN Values 
print('Missing Data')
cero_nodes = ["POSITIVELYMPHNODEN1", "NUMBEROFPOSITIVELYMPHNODEEXCISION", "LN RATIO", "SIZEOFPOSITIVELYMPHNODE(cm)", "NUMBEROFLYMPHNODEEXCISION"]
df.loc[df["NUMBEROFLYMPHNODEEXCISION"] == 0, cero_nodes] = 0
# If we impute to 0 these will be the lowest value between another high values. So it will be imputed then by mean or median.
# cero_rai = ["TSH PRE RAI", "TG PRE RAI", "ANTI TG PRE RAI"]
# df.loc[df["RAI"] == 2, "ANTI TG PRE RAI (POSITIVE or NEGATIVE)"] 





####################################################################################################
##### Missing data
####################################################################################################
total = df.isnull().sum().sort_values(ascending=False)
percent_total = (df.isnull().sum()/len(df)).sort_values(ascending=False)*100
missing = pd.concat([total, round(percent_total, 2)], axis=1, keys=['Total', 'Percent'])
missing = missing[missing['Total']>0]
missing

### Boxplots and distribution of variables with missing data to decide imputation strategy
# nan_columns_list = missing.index.tolist()
# n_rows = 8

# fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(15, 42))
# x = len(nan_columns_list) // n_rows

# for i in range(n_rows):
#     start = i * x
#     end = (i + 1) * x if i < n_rows - 1 else len(nan_columns_list)
    
#     subset = nan_columns_list[start:end]
    
#     df[subset].boxplot(ax=axes[i])
#     axes[i].set_title(f" ")

# plt.tight_layout()
# plt.save(figsize=(20, 50), fname="./distribution/boxplots_nan.png")
# plt.show()

# colors = sns.color_palette("rocket", len(nan_columns_list))
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(30, 20))
# axes = axes.flatten()

# for i, column in enumerate(nan_columns_list):
#     ax = axes[i]
#     sns.distplot(df[column], ax=ax, color=colors[i])
# for j in range(len(nan_columns_list), len(axes)):
#     axes[j].remove()

# plt.save(figsize=(20, 50), fname="./distribution/distribution_nan.png")
# plt.show()

#######################
### IMPUTATION
#######################
for i in df.columns: 
    if i in categorical_variable:
        df.fillna({i: df[i].mode()[0]}, inplace=True)
        df[i] = df[i].astype('uint8') ### encoding to improve performance
    else: 
        df.fillna({i: df[i].median()}, inplace=True)
        
#####################################################################################################      
##### CHARTS TO SEE ASSOCIATIONS AND CORRELATIONS, AND OUTLIERS AFTER IMPUTATION, BEFORE MODELING
#####################################################################################################
# Chart age by sex and recurrence 
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(x="SEX", y="AGEATDIGNOSIS", hue="RECURRENCE", data= df, ax=ax)
ax.set_title("Age Distribution of Patients by Sex and Recurrence Status")
ax.set_xticklabels(map_variables["SEX"].values())
ax.set_xlabel("SEX")
ax.set_ylabel("AGE")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["No Recurrence", "Recurrence"], loc="best")
plt.savefig("./distribution/boxplot_age_sex.png")
# plt.show()

#Frequencies
for col in df.columns:
    # sns.countplot(data=df, x=col)           
    path_count = f"./distribution/freq_{col}.png"
    if col == "RECURRENCE":
        continue
    
    elif col in categorical_variable:
        ax = sns.countplot(data=df, x=col, hue="RECURRENCE")
        plt.title(f'Frequency of Recurrence cases by {col}')
        ax.set_xticklabels(map_variables[col].values())
        if col == "SUBTYPE_FOLLI_PAPIL":
            plt.xticks(rotation=45, ha='right')

    else:
        if col in ["ANTI TG FOLLOW UP", "TG FOLLOW UP", "TSH FOLLOW UP", "OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY", "ANTI TG PRE RAI", "TG PRE RAI", "TSH PRE RAI", "RAIDOSE (mCi)"]:
            plt.figure(figsize=(10,8))
            sns.countplot(x=col, hue='RECURRENCE', data= df)
            plt.title(f'Frequency of Recurrence cases by {col}')               

        else:
            sns.histplot(data=df, x=col, hue="RECURRENCE", kde=False, element="step", common_norm=False, alpha=0.6)
            plt.title(f'Frequency of Recurrence cases by {col}')
            plt.xlabel(col, fontsize=12)

    plt.legend(title="Recurrence", labels=["No", "Yes"], prop={'size': 10})
    plt.tight_layout()
    
    plt.savefig(path_count, dpi=300) # Alta resolución para presentaciones
    # plt.show(block=True)
    plt.close()
    
## Pairplots betweem continuous variables and recurrence
path_dis = "./distribution/df_continuous_distributions.png"
if not os.path.exists(path_dis):
    sns.pairplot(df[continuous_variable + ["RECURRENCE"]], hue="RECURRENCE")
    plt.savefig(path_dis)
    # plt.show(block=True)
    plt.close()
    
######################################################################################################
###### Handling Outliers & Outlier treatments
######################################################################################################
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()
for ax, col in zip(axes, continuous_variable):
    sns.boxplot(df[col], ax=ax)
    ax.set_title(col.title(), weight='bold')
plt.tight_layout()
# plt.show()

df[continuous_variable] = np.log(df[continuous_variable] + 1)  # Log-transform to handle skewness and outliers
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()
for ax, col in zip(axes, continuous_variable):
    sns.boxplot(df[col], ax=ax)
    ax.set_title(col.title(), weight='bold')
plt.tight_layout()
# plt.show()
        
        

####################################################################################################
##### Sperman Correlation and LASSO to select best variables
####################################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
## 80/20 train/test (Internal validation)
x_train, x_test, y_train, y_test = train_test_split(df.drop("RECURRENCE", axis=1), df["RECURRENCE"], test_size=0.2, stratify=df["RECURRENCE"], random_state=0)
df, df_ival, df["RECURRENCE"], df_ival["RECURRENCE"] = x_train, x_test, y_train, y_test

no_include = ['RECURRENCE', 'ATA_2015_RISCO_INICIAL', 'ATA_2025_RISCO_INICIAL'] #the lsat vars are for comparative analysis
X = df.drop(no_include, axis=1)
y = df['RECURRENCE']  # outcome variable

print(f"Recurrence counts: {y.value_counts()}")
print(f"Recurrence proportion: {y.mean():.2%}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### Calculate Spearman correlation matrix with pandas, also we can w spermancor but there is an error about minimun data  https://www.yourdatateacher.com/2021/05/05/feature-selection-in-machine-learning-using-lasso-regression/
def sperman_by_imputed(X_df, imputation_name):
    """Calcula y guarda correlaciones de Spearman"""
    X_df = pd.DataFrame(X_df)
    corr_matrix = X_df.corr(method='spearman')
    output_path = f"./variable_selection/spearman_corr_{imputation_name}.xlsx"
    corr_matrix.to_excel(output_path)
    
    ## Chart Correlation Heatmap
    plt.figure(figsize=(12,12))
    correlation = df.corr()
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
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"Correlaciones de Spearman > |0.8| ({len(high_corr_pairs)} pares):\n")
            f.write("="*50 + "\n")
            for pair in high_corr_pairs:
                f.write(pair + "\n")
    
    print(f"Spearman correlations for {imputation_name} saved. {len(high_corr_pairs)} high correlations found.")
    return high_corr_pairs

sperman_by_imputed(X, "median")



### Selection
from sklearn.linear_model import LassoCV#, ElasticNetCV
def select_laso_imput(X_scaled_imputed, y, imput, X_columns, cvf = 5, random_state_laso=0):
    lasso_cv = LassoCV(cv=cvf, random_state=random_state_laso)
    lasso_cv.fit(X_scaled_imputed, y)
    print("Best alpha:", lasso_cv.alpha_)    
    sfm = SelectFromModel(lasso_cv, threshold = None, prefit=True)
    selected_feature_idx = sfm.get_support(indices=True)
    selected_features_lasso = list(X_columns[selected_feature_idx])
    # if "LN RATIO" not in selected_features_lasso:
    #     selected_features_lasso.append("LN RATIO") #author decision
    # if "AGEATDIGNOSIS" not in selected_features_lasso:
    #     selected_features_lasso.append("AGEATDIGNOSIS")
    path_variables = f"./variable_selection/variables_{imput}.txt"
    
    if not os.path.exists(path_variables):
        with open(path_variables, "w") as file:
            file.write(f"With {lasso_cv} selected {imput}: \n")
            for selected in selected_features_lasso:
                file.write(f"{selected}\n")
    print(f"Features selected imputation {imput}: {selected_features_lasso}")
    
    return selected_features_lasso

var_median = select_laso_imput(X_scaled, y, "median", X.columns, cvf = 10)


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


##############################################################################################################
##### Training/Development 90/10 from last database NOT internal validation 
##############################################################################################################
n_vars = input("Press Enter 'all' or 'selected' variables to start training models...\n")
if n_vars == "all":
    variables_picked = X.columns
else:
    variables_picked = var_median
    
x_train, x_test, y_train, y_test = train_test_split(X[variables_picked], y, test_size=0.1, stratify=y, random_state=0)

##########################################
##### SMOTE for Imbalanced Classification 
##########################################
import imblearn
from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from imblearn.under_sampling import RandomUnderSampler

print(y_train.value_counts())

# over = SMOTE(sampling_strategy=0.1)
# under = RandomUnderSampler(sampling_strategy=0.5)
# steps = [('o', over), ('u', under)]
# pipeline = Pipeline(steps=steps)
## transform the dataset
# x_smote_ou, y_smote_ou = pipeline.fit_resample(x_train, y_train)
smote = SMOTE(random_state=0)

x_smote_ou, y_smote_ou = smote.fit_resample(x_train, y_train)

print(y_smote_ou.value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, roc_curve, 
                           auc) #, classification_report)
# from sklearn.pipeline import Pipeline

### Start!!!
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

def model_metrics(y_train, y_test, train_preds, test_preds, 
                  model: str, fine_tune: bool, performance: dict,
                  balanced_state: str, variables_used: int):
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    train_precision = precision_score(y_train, train_preds)
    test_precision = precision_score(y_test, test_preds)
    train_recall = recall_score(y_train, train_preds)
    test_recall = recall_score(y_test, test_preds)
    train_f1 = f1_score(y_train, train_preds)
    test_f1 = f1_score(y_test, test_preds)
    train_roc_auc = roc_auc_score(y_train, train_preds)
    test_roc_auc = roc_auc_score(y_test, test_preds)
    
    print(f"{'Train Accuracy':<20}{train_accuracy:.4f}")
    print(f"{'Test Accuracy':<20}{test_accuracy:.4f}")
    print(f"{'Train Precision':<20}{train_precision:.4f}")
    print(f"{'Test Precision':<20}{test_precision:.4f}")
    print(f"{'Train Recall':<20}{train_recall:.4f}")
    print(f"{'Test Recall':<20}{test_recall:.4f}")
    print(f"{'Train F1 Score':<20}{train_f1:.4f}")
    print(f"{'Test F1 Score':<20}{test_f1:.4f}")
    print(f"{'Train ROC AUC':<20}{train_roc_auc:.4f}")
    print(f"{'Test ROC AUC':<20}{test_roc_auc:.4f}")
    print("\n"+"-"*50)  

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

    dir_cm = f"./results/confusion_matrix/{balanced_state}/{variables_used}_variables"
    if os.path.exists(dir_cm):
        pass
    else:
        os.makedirs(dir_cm, exist_ok=True)
    
    plt.title(f"{model} - {'Fine-tuned' if fine_tune else 'Base'}")
    plt.savefig(f"{dir_cm}/cm_{model}_{'fine_tune' if fine_tune else 'base'}.png", dpi=300)
    # plt.show()
    
    key = "1" if fine_tune else "0"
        
    data = {'Model' : model,
            'balanced_state': balanced_state,
            'Test_Accuracy'  : test_accuracy,
            'Test_Precision' : test_precision,
            'Test_Recall'    : test_recall,
            'Test_F1_Score'  : test_f1,
            'Test_ROC_AUC'   : test_roc_auc}
    
    performance[key].append(data)
    
    return performance

# XGBoost / RandomForest / LightGBM / Bayesian Models with Priors with pymc
performance_base = {"0": [], #Without-hyperparameter-tuning
                    "1": []} #With-hyperparameter-tuning

print("\n" + "="*60)

################################################
### LogisticRegression Model - 1 Implementation
################################################

##Imbalanced data
logistic_classifier= LogisticRegression()
# Fit the Algorithm
logistic_classifier.fit(x_train,y_train)
# Predict on the model
y_train_logistic_pred= logistic_classifier.predict(x_train)
y_test_logistic_pred= logistic_classifier.predict(x_test)

model_metrics(y_train, y_test, y_train_logistic_pred, y_test_logistic_pred, 
              model="Logistic Regression", fine_tune=False, performance=performance_base,
              balanced_state="imbalanced", variables_used=x_train.shape[1])

##Balanced SMOTE - Undersampled data
logistic_classifier= LogisticRegression()
# Fit the Algorithm
logistic_classifier.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_logistic_pred= logistic_classifier.predict(x_smote_ou)
y_test_logistic_pred= logistic_classifier.predict(x_test)

model_metrics(y_smote_ou, y_test, y_train_logistic_pred, y_test_logistic_pred, 
              model="Logistic Regression", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

# ML Model - 1 Implementation with hyperparameter optimization techniques (GridSearch CV)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
def save_best_hyperparameters(model_name, best_params, variables_used, balanced_state):
    path = f"./results/best_params.txt"
    with open(path, "a") as file:
        file.write(f"{model_name}, {variables_used}, {balanced_state}: {best_params}\n")

logistic_regression = LogisticRegression()
# set up the parameter grid for hyperparameter tuning
param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.1, 1.0, 10.0],
              'solver': ['liblinear', 'saga']}
# Fit the Algorithm
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("Logistic Regression", best_params, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters Logistic Regression:', best_params)
# use the best hyperparameters to fit the model and make predictions
logistic_regression_best = LogisticRegression(**best_params)
# perform cross-validation on the model with the best hyperparameters
cv_scores = cross_val_score(logistic_regression_best, x_train, y_train, cv=5)
# fit the final model using all the training data and the best hyperparameters
logistic_regression_best.fit(x_train, y_train)
y_train_logistic_pred_cv = logistic_regression_best.predict(x_train)
y_test_logistic_pred_cv  = logistic_regression_best.predict(x_test)
y_score_logistic_pred_cv = logistic_regression_best.predict_proba(x_test)[:, 1]

model_metrics(y_train, y_test, y_train_logistic_pred_cv, y_test_logistic_pred_cv, 
              model="Logistic Regression", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


logistic_regression = LogisticRegression()
# set up the parameter grid for hyperparameter tuning
# Fit the Algorithm equal to the last param_grid but with balanced data
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)
grid_search.fit(x_smote_ou, y_smote_ou)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("Logistic Regression", best_params, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters Logistic Regression:', best_params)
# use the best hyperparameters to fit the model and make predictions
logistic_regression_best = LogisticRegression(**best_params)
# perform cross-validation on the model with the best hyperparameters
cv_scores = cross_val_score(logistic_regression_best, x_smote_ou, y_smote_ou, cv=5)
# fit the final model using all the training data and the best hyperparameters
logistic_regression_best.fit(x_smote_ou, y_smote_ou)
y_train_logistic_pred_cv = logistic_regression_best.predict(x_smote_ou)
y_test_logistic_pred_cv  = logistic_regression_best.predict(x_test)
y_score_logistic_pred_cv_ = logistic_regression_best.predict_proba(x_test)[:, 1]

model_metrics(y_smote_ou, y_test, y_train_logistic_pred_cv, y_test_logistic_pred_cv, 
              model="Logistic Regression", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

################################################
### RandomForestClassifier Model - 2  Implementation
################################################

##Imbalanced data
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
# Fit the Algorithm
random_forest.fit(x_train, y_train)

# Predict on the model
y_train_rf_pred = random_forest.predict(x_train)
y_test_rf_pred = random_forest.predict(x_test)

model_metrics(y_train, y_test, y_train_rf_pred, y_test_rf_pred, 
              model="Random Forest", fine_tune=False, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])

##Balanced SMOTE - Undersampled data
random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
# Fit the Algorithm
random_forest.fit(x_smote_ou, y_smote_ou)

# Predict on the model
y_train_rf_pred = random_forest.predict(x_smote_ou)
y_test_rf_pred = random_forest.predict(x_test)

model_metrics(y_smote_ou, y_test, y_train_rf_pred, y_test_rf_pred, 
              model="Random Forest", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])


# ML Model - 2 Implementation with hyperparameter optimization techniques (GridSearch CV)
random_forest = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [5, 10, 15, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}
# Fit the Algorithm
grid_search = GridSearchCV(random_forest, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("Random Forest", best_params, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters Random Forest:', best_params)
# use the best hyperparameters to fit the model to the training data
random_forest_best = RandomForestClassifier(**best_params)
random_forest_best.fit(x_train, y_train)
# Predict on the model
y_train_rf_pred_gs = random_forest_best.predict(x_train)
y_test_rf_pred_gs  = random_forest_best.predict(x_test)
y_score_rf_pred_gs = random_forest_best.predict_proba(x_test)[:, 1]


model_metrics(y_train, y_test, y_train_rf_pred_gs, y_test_rf_pred_gs, 
              model="Random Forest", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


random_forest = RandomForestClassifier()
# Fit the Algorithm equal to the last param_grid but with balanced data
grid_search = GridSearchCV(random_forest, param_grid, cv=5)
grid_search.fit(x_smote_ou, y_smote_ou)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("Random Forest", best_params, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters Random Forest:', best_params)
# use the best hyperparameters to fit the model to the training data
random_forest_best = RandomForestClassifier(**best_params)
random_forest_best.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_rf_pred_gs = random_forest_best.predict(x_smote_ou)
y_test_rf_pred_gs  = random_forest_best.predict(x_test)
y_score_rf_pred_gs_ = random_forest_best.predict_proba(x_test)[:, 1]

model_metrics(y_smote_ou, y_test, y_train_rf_pred_gs, y_test_rf_pred_gs, 
              model="Random Forest", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

##################################################
### XGBClassifier Model - 3 Implementation
##################################################

xgb = XGBClassifier()
# Fit the Algorithm
xgb.fit(x_train, y_train)
# Predict on the model
y_train_xgb_pred = xgb.predict(x_train)
y_test_xgb_pred = xgb.predict(x_test)

model_metrics(y_train, y_test, y_train_xgb_pred, y_test_xgb_pred,
              model="XGBoost", fine_tune=False, performance=performance_base,
              balanced_state="imbalanced", variables_used=x_train.shape[1])


xgb = XGBClassifier()
# Fit the Algorithm
xgb.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_xgb_pred = xgb.predict(x_smote_ou)
y_test_xgb_pred = xgb.predict(x_test)

model_metrics(y_smote_ou, y_test, y_train_xgb_pred, y_test_xgb_pred,
              model="XGBoost", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])



# ML Model - 3 Implementation with hyperparameter optimization techniques (GridSearch CV)
# set up the parameter grid for hyperparameter tuning
xgb = XGBClassifier()
param_grid = {'max_depth': [3, 5, 7],
              'learning_rate': [0.01, 0.1, 0.3],
              'n_estimators': [50, 100, 200]}
# Fit the Algorithm
grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
# print the best hyperparameters
save_best_hyperparameters("XGBoost", grid_search.best_params_, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters XGBoost:', grid_search.best_params_)
# Predict on the model
best_estimator = grid_search.best_estimator_
y_train_xgb_pred_gs = best_estimator.predict(x_train)
y_test_xgb_pred_gs  = best_estimator.predict(x_test)
y_score_xgb_pred_gs = best_estimator.predict_proba(x_test)[:, 1]

model_metrics(y_train, y_test, y_train_xgb_pred_gs, y_test_xgb_pred_gs, 
              model="XGBoost", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


# set up the parameter grid for hyperparameter tuning
xgb = XGBClassifier()
# Fit the Algorithm equal to the last param_grid but with balanced data
grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_smote_ou, y_smote_ou)
# print the best hyperparameters
save_best_hyperparameters("XGBoost", grid_search.best_params_, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters XGBoost:', grid_search.best_params_)
# Predict on the model
best_estimator_b = grid_search.best_estimator_
y_train_xgb_pred_gs = best_estimator_b.predict(x_smote_ou)
y_test_xgb_pred_gs  = best_estimator_b.predict(x_test)
y_score_xgb_pred_gs_ = best_estimator_b.predict_proba(x_test)[:, 1]

model_metrics(y_smote_ou, y_test, y_train_xgb_pred_gs, y_test_xgb_pred_gs, 
              model="XGBoost", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

######################################################
### KNeighborsClassifier Model - 4 Implementation
######################################################

knn = KNeighborsClassifier(n_neighbors=5)
# Fit the Algorithm
knn.fit(x_train, y_train)
# Predict on the model
y_train_knn_pred = knn.predict(x_train)
y_test_knn_pred = knn.predict(x_test)

model_metrics(y_train, y_test, y_train_knn_pred, y_test_knn_pred, 
              model="KNN", fine_tune=False, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


knn = KNeighborsClassifier(n_neighbors=5)
# Fit the Algorithm
knn.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_knn_pred = knn.predict(x_smote_ou)
y_test_knn_pred = knn.predict(x_test)

model_metrics(y_smote_ou, y_test, y_train_knn_pred, y_test_knn_pred, 
              model="KNN", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])


# ML Model - 4  Implementation with hyperparameter optimization techniques (GridSearch CV)
# set up the parameter grid for hyperparameter tuning
knn = KNeighborsClassifier(n_neighbors=5)
param_grid = {'n_neighbors': [3, 5, 7],
              'weights': ['uniform', 'distance']}
# Fit the Algorithm
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("KNN", best_params, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters KNN:', best_params)
# train the classifier with the best hyperparameters on the full training set
knn_best = KNeighborsClassifier(**best_params)
knn_best.fit(x_train, y_train)
# Predict on the model
y_test_knn_pred_gs  = knn_best.predict(x_test)
y_train_knn_pred_gs = knn_best.predict(x_train)
y_score_knn_pred_gs = knn_best.predict_proba(x_test)[:, 1]

model_metrics(y_train, y_test, y_train_knn_pred_gs, y_test_knn_pred_gs, 
              model="KNN", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


knn = KNeighborsClassifier(n_neighbors=5)
# Fit the Algorithm
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(x_smote_ou, y_smote_ou)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("KNN", best_params, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters KNN:', best_params)
# train the classifier with the best hyperparameters on the full training set
knn_best = KNeighborsClassifier(**best_params)
knn_best.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_test_knn_pred_gs  = knn_best.predict(x_test)
y_train_knn_pred_gs = knn_best.predict(x_smote_ou)
y_score_knn_pred_gs_ = knn_best.predict_proba(x_test)[:, 1]

model_metrics(y_smote_ou, y_test, y_train_knn_pred_gs, y_test_knn_pred_gs, 
              model="KNN", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

######################################################
### SVC Model - 5 Implementation
######################################################

svc = SVC(kernel='rbf', C=1, gamma='scale')
# Fit the Algorithm
svc.fit(x_train, y_train)
# Predict on the model
y_train_svc_pred = svc.predict(x_train)
y_test_svc_pred = svc.predict(x_test)

model_metrics(y_train, y_test, y_train_svc_pred, y_test_svc_pred, 
              model="SVC", fine_tune=False, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])
     
svc = SVC(kernel='rbf', C=1, gamma='scale')
# Fit the Algorithm
svc.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_svc_pred = svc.predict(x_smote_ou)
y_test_svc_pred = svc.predict(x_test)

model_metrics(y_smote_ou, y_test, y_train_svc_pred, y_test_svc_pred, 
              model="SVC", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])
     
# ML Model - 5  Implementation with hyperparameter optimization techniques (GridSearch CV)
svc = SVC(probability=True)
# set up the parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto']}
# perform a grid search with 5-fold cross-validation to find the best hyperparameters
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("SVC", best_params, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters SVC:', best_params)
# train the classifier with the best hyperparameters on the full training set
svc_best = SVC(**best_params, probability=True)
svc_best.fit(x_train, y_train)
# Predict on the model
y_test_svc_pred_gs = svc_best.predict(x_test)
y_train_svc_pred_gs = svc_best.predict(x_train)
y_score_svc_pred_gs = svc_best.predict_proba(x_test)[:, 1]     


model_metrics(y_train, y_test, y_train_svc_pred_gs, y_test_svc_pred_gs,
              model="SVC", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


svc = SVC(probability=True)
# set up the parameter grid for hyperparameter tuning
# perform a grid search equal to the last param_grid but with balanced data
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(x_smote_ou, y_smote_ou)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("SVC", best_params, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters SVC:', best_params)
# train the classifier with the best hyperparameters on the full training set
svc_best = SVC(**best_params, probability=True)
svc_best.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_test_svc_pred_gs = svc_best.predict(x_test)
y_train_svc_pred_gs = svc_best.predict(x_smote_ou)
y_score_svc_pred_gs_ = svc_best.predict_proba(x_test)[:, 1]     


model_metrics(y_smote_ou, y_test, y_train_svc_pred_gs, y_test_svc_pred_gs,
              model="SVC", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

######################################################
### GaussianNB Model - 6 Implementation
######################################################

# create an instance of the Gaussian Naive Bayes classifier
nb = GaussianNB()
# Fit the Algorithm
nb.fit(x_train, y_train)
# Predict on the model
y_train_nb_pred = nb.predict(x_train)
y_test_nb_pred = nb.predict(x_test)
model_metrics(y_train, y_test, y_train_nb_pred, y_test_nb_pred, 
              model="NB Classifier", fine_tune=False, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])

nb = GaussianNB()
# Fit the Algorithm
nb.fit(x_smote_ou, y_smote_ou)
# Predict on the model
y_train_nb_pred = nb.predict(x_smote_ou)
y_test_nb_pred = nb.predict(x_test)
model_metrics(y_smote_ou, y_test, y_train_nb_pred, y_test_nb_pred, 
              model="NB Classifier", fine_tune=False, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])


# ML Model - 6  Implementation with hyperparameter optimization techniques (GridSearch CV)
nb = GaussianNB()
# set up the parameter grid for hyperparameter tuning
param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
# perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(nb, param_grid, cv=5)
grid_search.fit(x_train, y_train)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("GaussianNB", best_params, x_train.shape[1], "imbalanced data.")
print('Best hyperparameters GaussianNB:', best_params)
# create a new instance of the classifier using the best hyperparameters
nb_best = GaussianNB(**best_params)
# evaluate the classifier using cross-validation
scores = cross_val_score(nb_best, x_train, y_train, cv=5)
# print the cross-validation scores
print('Cross-validation scores:', scores)
# train the classifier on the entire training set using the best hyperparameters
nb_best.fit(x_train, y_train)
# make predictions on the training and test sets
y_train_nb_pred_gs = nb_best.predict(x_train)
y_test_nb_pred_gs = nb_best.predict(x_test)
y_score_nb_pred_gs = nb_best.predict_proba(x_test)[:, 1]

model_metrics(y_train, y_test, y_train_nb_pred_gs, y_test_nb_pred_gs, 
              model="NB Classifier", fine_tune=True, performance=performance_base,
              balanced_state="Imbalanced", variables_used=x_train.shape[1])


nb = GaussianNB()
# set up the parameter grid for hyperparameter tuning
# perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(nb, param_grid, cv=5)
grid_search.fit(x_smote_ou, y_smote_ou)
# get the best hyperparameters and print them
best_params = grid_search.best_params_
save_best_hyperparameters("GaussianNB", best_params, x_smote_ou.shape[1], "Balanced_SMOTE data.")
print('Best hyperparameters GaussianNB:', best_params)
# create a new instance of the classifier using the best hyperparameters
nb_best = GaussianNB(**best_params)
# evaluate the classifier using cross-validation
scores = cross_val_score(nb_best, x_smote_ou, y_smote_ou, cv=5)
# print the cross-validation scores
print('Cross-validation scores:', scores)
# train the classifier on the entire training set using the best hyperparameters
nb_best.fit(x_smote_ou, y_smote_ou)
# make predictions on the training and test sets
y_train_nb_pred_gs = nb_best.predict(x_smote_ou)
y_test_nb_pred_gs = nb_best.predict(x_test)
y_score_nb_pred_gs_ = nb_best.predict_proba(x_test)[:, 1]

model_metrics(y_smote_ou, y_test, y_train_nb_pred_gs, y_test_nb_pred_gs, 
              model="NB Classifier", fine_tune=True, performance=performance_base,
              balanced_state="Balanced_SMOTE", variables_used=x_smote_ou.shape[1])

print("="*60)

######################################################################
# define the classifiers
classifiers = [ ("Logistic Regression", LogisticRegression()),
                ("Random Forest Classifier", RandomForestClassifier()),
                ("XGB Classifier", XGBClassifier()),
                ("KNN", KNeighborsClassifier()),
                ("SVC", SVC(probability=True)),
                ("NB Classifier", GaussianNB())]

# iterate through classifiers and plot ROC curves
plt.figure(figsize=(10, 8))
for name, classifier in classifiers:
    classifier.fit(x_train, y_train)
    y_score = classifier.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC with Imbalanced Data Baseline')
plt.legend(loc="lower right")
plt.savefig(f"./results/ROC_Imbalanced_Baseline_{n_vars}.png", dpi=300)
# plt.show()


plt.figure(figsize=(10, 8))
for name, classifier in classifiers:
    classifier.fit(x_smote_ou, y_smote_ou)
    y_score = classifier.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC with Balanced SMOTE Undersampled Data Baseline')
plt.legend(loc="lower right")
plt.savefig(f"./results/ROC_Balanced_SMOTE_baseline_{n_vars}.png", dpi=300)
# plt.show()


# after cross validation and hyperparameter tuning
Model = ["Logistic Regression", "Random Forest Classifier", "XGBoost", "KNN", "SVC","NBClassifier"]
Y_SCORE = [y_score_logistic_pred_cv, y_score_rf_pred_gs, y_score_xgb_pred_gs, 
           y_score_knn_pred_gs, y_score_svc_pred_gs,y_score_nb_pred_gs]

# Create dataframe from the lists
data = {'MODEL': Model, 'Y_SCORE': Y_SCORE}
Metric_df = pd.DataFrame(data)

# plot the ROC curves for each model
plt.figure(figsize=(10, 8))
for i, row in Metric_df.iterrows():
    fpr, tpr, _ = roc_curve(y_test, row['Y_SCORE'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{row['MODEL']} (AUC = {roc_auc:.2f})", alpha=0.8)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC with Imbalanced Data after Hyperparameter Tuning')
plt.legend(loc="lower right")
plt.savefig(f"./results/ROC_Imbalanced_Hypertuned_{n_vars}.png", dpi=300)
# plt.show()



# after cross validation and hyperparameter tuning
Model = ["Logistic Regression", "Random Forest Classifier", "XGBoost", "KNN", "SVC","NBClassifier"]
Y_SCORE = [y_score_logistic_pred_cv_, y_score_rf_pred_gs_, y_score_xgb_pred_gs_, 
           y_score_knn_pred_gs_, y_score_svc_pred_gs_,y_score_nb_pred_gs_]

# Create dataframe from the lists
data = {'MODEL': Model, 'Y_SCORE': Y_SCORE}
Metric_df_ = pd.DataFrame(data)

# plot the ROC curves for each model
plt.figure(figsize=(10, 8))
for i, row in Metric_df_.iterrows():
    fpr, tpr, _ = roc_curve(y_test, row['Y_SCORE'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{row['MODEL']} (AUC = {roc_auc:.2f})", alpha=0.8)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC with Balanced SMOTE Undersampled Data after Hyperparameter Tuning')
plt.legend(loc="lower right")
plt.savefig(f"./results/ROC_Balanced_SMOTE_Hypertuned_{n_vars}.png", dpi=300)
# plt.show()


# Storing metrics in order to make dataframe of metrics 
# (after cross validation and hyperparameter tuning)
# Create dataframe from the lists
Metric_df = pd.DataFrame(performance_base["1"])

# Printing dataframe
Metric_df_imbalanced = Metric_df[Metric_df["balanced_state"] == "Imbalanced"]
Metric_df_imbalanced.to_csv(f"./results/metrics_imbalanced_{n_vars}.csv", index=False)
Metric_df_imbalanced

Metric_df_Balanced_SMOTE = Metric_df[Metric_df["balanced_state"] == "Balanced_SMOTE"]
Metric_df_Balanced_SMOTE.to_csv(f"./results/metrics_Balanced_SMOTE_{n_vars}.csv", index=False)
Metric_df_Balanced_SMOTE


# Plotting the barplot to determine which feature is contributing the most
features = np.array(variables_picked)  
importances = best_estimator.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,8))
plt.grid(zorder=0)
plt.title('Feature Importances', fontsize=20)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance Imbalanced Data')
plt.savefig(f"./results/feature_importances/fi_{n_vars}_imbalanced.png", dpi=300)



importances = best_estimator_b.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,8))
plt.grid(zorder=0)
plt.title('Feature Importances', fontsize=20)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance Balanced SMOTE Undersampled')
plt.savefig(f"./results/feature_importances/fi_{n_vars}_balanced_smote_undersampled.png", dpi=300)

############################################################################
##### Saving the best model
############################################################################

import pickle
# Save the File
filename=f'thyroid_recurrence_{n_vars}_prediction_model.pkl'
# serialize process (wb=write byte)
pickle.dump(best_estimator,open(filename,'wb'))
     

print("="*60 + "Finished" + "="*60)
# strategies = ['mean', 'median', 'iterative'] # iF DISTRIBUTION NORMAL WILL BE MEAN

# results = {}
# results['mean'] = {'X_imputed': X_mean, 'X_scaled': X_scaled_mean, 'selected_features': var_mean}
# # X_iter = pd.DataFrame(X_iter, columns=X_columns)
# # results['iterative'] = {'X_imputed': X_iter, 'X_scaled': X_scaled_iter, 'selected_features': var_iter}
# # results['median'] = {'X_imputed': X_median, 'X_scaled': X_scaled_median, 'selected_features': var_median}

# # Usar imputación iterativa como estrategia inicial
# # imp = IterativeImputer(max_iter=20, random_state=42)
# # X = X_dev.copy()
# # X_imputed_dev = imp.fit_transform(X)
# # X_imputed_dev = pd.DataFrame(X_imputed_dev, columns=X.columns)

# from sklearn.impute import SimpleImputer
# imputer_mean = SimpleImputer(strategy='mean')
# imputer_mode = SimpleImputer(strategy='most_frequent')
# X_mean_dev = X_dev.copy()
# X_mean_dev[imputation_median_mean] = imputer_mean.fit_transform(X_mean_dev[imputation_median_mean])
# X_mean_dev[imputation_moda] = imputer_mode.fit_transform(X_mean_dev[imputation_moda])

# # # 3. Aplicar SimpleImputer (Mediana)
# # imputer_median = SimpleImputer(strategy='median')
# # X_median_dev = X_dev.copy()
# # X_median_dev[imputation_median_mean] = imputer_median.fit_transform(X_median_dev[imputation_median_mean])
# # X_median_dev[imputation_moda] = imputer_mode.fit_transform(X_median_dev[imputation_moda])

# # Dividir datos
# print(f"Training set: {X_train.shape}")
# print(f"Test set: {X_dev.shape}")
# print(f"Recurrence in train: {y_train.mean():.2%}")
# print(f"Recurrence in test: {y_dev.mean():.2%}")

# ### 2. DEFINIR Y PROBAR MÚLTIPLES MODELOS
# print("\n" + "="*60)
# print("ENTRENANDO Y COMPARANDO MÚLTIPLES MODELOS")
# print("="*60)

# all_metrics = []

# from sklearn.model_selection import cross_validate
# def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, eval = False):
#     """Evalúa un modelo y retorna métricas"""
    
#     # Entrenar modelo
#     model.fit(X_train, y_train)
    
#     # Predicciones
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
    
#     #ROC_AUC
#     fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
#     roc_auc = auc(fpr, tpr)  # o roc_auc_score(y_test, y_pred_proba)

#     # Graficar
#     path_graph = f"./auc_graphs/auc_dev_{model_name}"
#     if eval == True:
#         path_graph = f"./auc_graphs/auc_test_{model_name}"
#     if not os.path.exists(path_graph):
#         plt.figure(figsize=(8, 6))
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title(f'Receiver Operating Characteristic (ROC) Curve \n {model_name}')
#         plt.legend(loc="lower right")
#         plt.grid(True, alpha=0.3)
#         plt.savefig(path_graph)
#         plt.show()
    
#     # También puedes obtener el AUC directamente
#     auc_score = roc_auc_score(y_test, y_pred_proba)
#     print(f"AUC-ROC Score: {auc_score:.3f}")
#     # Métricas
#     metrics = {
#         'model': model_name,
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred, zero_division=0),
#         'recall': recall_score(y_test, y_pred, zero_division=0),
#         'f1': f1_score(y_test, y_pred, zero_division=0),
#         'roc_auc': roc_auc_score(y_test, y_pred_proba),
#         'confusion_matrix': confusion_matrix(y_test, y_pred)
#     }
    
#     # Validación cruzada para obtener métricas más robustas
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     cv_results = cross_validate(
#         model, X_train, y_train, 
#         cv=cv, 
#         scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
#         n_jobs=-1
#     )
    
#     metrics.update({
#         'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
#         'cv_accuracy_std': cv_results['test_accuracy'].std(),
#         'cv_precision_mean': cv_results['test_precision'].mean(),
#         'cv_precision_std': cv_results['test_precision'].std(),
#         'cv_recall_mean': cv_results['test_recall'].mean(),
#         'cv_recall_std': cv_results['test_recall'].std(),
#         'cv_f1_mean': cv_results['test_f1'].mean(),
#         'cv_f1_std': cv_results['test_f1'].std(),
#         'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
#         'cv_roc_auc_std': cv_results['test_roc_auc'].std()
#     })
    
#     return metrics, model


# trained_models = {}

# # A. LOGISTIC REGRESSION
# # https://www.youtube.com/watch?v=BHok3wJpmf0
# # https://www.youtube.com/watch?v=3giTXZbyf1Q
# print("\n1. Logistic Regression")
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# # Crear pipeline para regresión logística
# lr_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', LogisticRegression(
#         max_iter=1000,
#         random_state=42,
#         class_weight='balanced',
#         penalty = "l2",#"elasticnet",
#         solver= 'sag',#'saga',
#         C=0.018, # Regularización más fuerte
#         # l1_ratio=0.65
#     ))
# ])


# # lr_metrics, lr_model = evaluate_model(
# #     lr_pipeline, X_median[var_median], X_median_dev[var_median], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
# #     "Logistic_Regression Median"
# # )
# # all_metrics.append(lr_metrics)
# # trained_models['Logistic_Regression Median'] = lr_model
# # var_mean=['AGEATDIGNOSIS', 'BMI', 'TUMORSIZE (cm)', 'NUMBEROFLYMPHNODEEXCISION', 'NUMBEROFPOSITIVELYMPHNODEEXCISION', 'LN RATIO', 'TNMT', 'RAIDOSE', 'TSH PRE RAI', 'TG PRE RAI', 'ANTI TG PRE RAI ', 'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY', 'TSH FOLLOW UP', 'TG FOLLOW UP', 'ANTI TG FOLLOW UP', 'SUBTYPE_FOLLI_PAPIL']

# lr_metrics, lr_model = evaluate_model(
#     lr_pipeline, X_mean[var_mean], X_mean_dev[var_mean], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
#     "Logistic_Regression Mean"
# )

# all_metrics.append(lr_metrics)
# trained_models['Logistic_Regression Mean'] = lr_model
# print("intercept (b):", lr_model[1].intercept_)
# print("pendiente (w):", lr_model[1].coef_)
# # For binary classification, model.coef_ will be a 2D array of shape (1, n_features)
# coefficients = lr_model[1].coef_[0]

# # 4. Create a pandas Series to associate coefficients with feature names
# coef_series = pd.Series(coefficients, index=X_train.columns)

# # 5. Order the variables by the absolute magnitude of their coefficients
# # Higher absolute value indicates greater importance
# ordered_importance = coef_series.abs().sort_values(ascending=False)

# # 6. Display the ordered list
# print("Ordered Variable Importance (by absolute coefficient magnitude):")
# print(ordered_importance)

# # To see the actual coefficients and their signs:
# ordered_coefficients = coef_series.reindex(ordered_importance.index)
# print("\nOrdered Coefficients:")
# print(ordered_coefficients)

# # lr_metrics, lr_model = evaluate_model(
# #     lr_pipeline, X_iter[var_iter], X_imputed_dev[var_iter], y_train, y_dev,  #X_ -> imputed, mean, median, what selected?
# #     "Logistic_Regression Iter"
# # )
# # all_metrics.append(lr_metrics)
# # trained_models['Logistic_Regression Iter'] = lr_model

# # print(all_metrics)

# # B. RANDOM FOREST
# print("\n2. Random Forest")
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=10,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_features='sqrt',
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1,
# )

# rf_metrics, rf_trained = evaluate_model(
#     rf_model, X_train, X_test, y_train, y_test,
#     "Random_Forest"
# )
# all_metrics.append(rf_metrics)
# trained_models['Random_Forest'] = rf_trained

# # C. XGBOOST
# print("\n3. XGBoost")
# try:
#     from xgboost import XGBClassifier
    
#     # Calcular scale_pos_weight para manejar desbalance
#     scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
#     xgb_model = XGBClassifier(
#         n_estimators=200,
#         max_depth=6,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=scale_pos_weight,
#         random_state=42,
#         n_jobs=-1,
#         # use_label_encoder=False,
#         eval_metric='logloss',
#         enable_categorical=False
#     )
    
#     xgb_metrics, xgb_trained = evaluate_model(
#         xgb_model, X_train, X_test, y_train, y_test,
#         "XGBoost"
#     )
#     all_metrics.append(xgb_metrics)
#     trained_models['XGBoost'] = xgb_trained
    
# except ImportError:
#     print("XGBoost no está instalado. Instala con: pip install xgboost")

# # D. LIGHTGBM
# print("\n4. LightGBM")
# try:
#     from lightgbm import LGBMClassifier
    
#     lgb_model = LGBMClassifier(
#         n_estimators=200,
#         max_depth=8,
#         learning_rate=0.05,
#         num_leaves=31,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1,
#         verbose=-1  # Silenciar output
#     )
    
#     lgb_metrics, lgb_trained = evaluate_model(
#         lgb_model, X_train, X_test, y_train, y_test,
#         "LightGBM"
#         )
#     all_metrics.append(lgb_metrics)
#     trained_models['LightGBM'] = lgb_trained
    
# except ImportError:
#     print("LightGBM no está instalado. Instala con: pip install lightgbm")

# # E. BAYESIAN LOGISTIC REGRESSION (aproximación)
# print("\n5. Bayesian Logistic Regression (aproximado)")
# try:
#     # Usamos regresión logística con regularización tipo "l2" como aproximación bayesiana
#     from sklearn.linear_model import LogisticRegressionCV
    
#     bayesian_lr = LogisticRegressionCV(
#         Cs=10,
#         cv=5,
#         penalty='l2',
#         solver='lbfgs',
#         max_iter=1000,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1,
#         l1_ratios=None
#     )
    
#     bayesian_metrics, bayesian_trained = evaluate_model(
#         bayesian_lr, X_train, X_test, y_train, y_test,
#         "Bayesian_Logistic"
#     )
#     all_metrics.append(bayesian_metrics)
#     trained_models['Bayesian_Logistic'] = bayesian_trained
    
# except Exception as e:
#     print(f"Error con Bayesian Logistic: {e}")

# # F. GRADIENT BOOSTING (scikit-learn)
# print("\n6. Gradient Boosting (scikit-learn)")
# from sklearn.ensemble import GradientBoostingClassifier

# gb_model = GradientBoostingClassifier(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=5,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     subsample=0.8,
#     random_state=42
# )

# gb_metrics, gb_trained = evaluate_model(
#     gb_model, X_train, X_test, y_train, y_test,
#     "Gradient_Boosting"
# )
# all_metrics.append(gb_metrics)
# trained_models['Gradient_Boosting'] = gb_trained

# # Configurar validación cruzada
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Probar diferentes escaladores
# scalers = {
#     'standard': StandardScaler(),
#     'minmax': MinMaxScaler(),
#     'maxabs': MaxAbsScaler()
# }

# all_results1 = []

# for strategy in strategies:
#     print(f"\n{'='*50}")
#     print(f"Resultados para imputación: {strategy}")
#     print('='*50)
    
#     X_original = results[strategy]['X_imputed']
#     selected_features = results[strategy]['selected_features']
    
#     if len(selected_features) == 0:
#         print(f"No se seleccionaron características con {strategy}. Usando todas.")
#         selected_features = X_original.columns
    
#     # Filtrar características seleccionadas
#     X_selected = X_original[selected_features]
    
#     for scaler_name, scaler in scalers.items():
#         print(f"\nEscalador: {scaler_name}")
        
#         # Crear pipeline
#         pipeline = Pipeline([
#             ('scaler', scaler),
#             ('classifier', LogisticRegression(
#                 solver='lbfgs',
#                 max_iter=1000,
#                 random_state=42,
#                 class_weight='balanced'  # Para manejar clases desbalanceadas
#             ))
#         ])
        
#         # Validación cruzada
#         cv_scores = cross_val_score(
#             pipeline, X_selected, y, 
#             cv=cv, scoring='roc_auc', n_jobs=-1
#         )
        
#         print(f"AUC-ROC (CV): {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
#         # Entrenar en conjunto completo para obtener métricas adicionales
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_selected, y, test_size=0.2, 
#             random_state=42, stratify=y
#         )
        
#         # Ajustar pipeline
#         pipeline.fit(X_train, y_train)
        
#         # Predicciones
#         y_pred = pipeline.predict(X_test)
#         y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
#         # Calcular métricas
#         metrics = {
#             'imputation': strategy,
#             'scaler': scaler_name,
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, zero_division=0),
#             'recall': recall_score(y_test, y_pred, zero_division=0),
#             'f1': f1_score(y_test, y_pred, zero_division=0),
#             'auc_roc': roc_auc_score(y_test, y_pred_proba),
#             'cv_auc_mean': cv_scores.mean(),
#             'cv_auc_std': cv_scores.std(),
#             'n_features': len(selected_features)
#         }
        
#         all_results1.append(metrics)
        
#         # Mostrar resultados
#         print(f"Accuracy: {metrics['accuracy']:.3f}")
#         print(f"Precision: {metrics['precision']:.3f}")
#         print(f"Recall: {metrics['recall']:.3f}")
#         print(f"F1-Score: {metrics['f1']:.3f}")
#         print(f"AUC-ROC (test): {metrics['auc_roc']:.3f}")
        
#         # Matriz de confusión
#         cm = confusion_matrix(y_test, y_pred)
#         print("Matriz de confusión:")
#         print(cm)
        
#         # Reporte de clasificación
#         print("\nReporte de clasificación:")
#         print(classification_report(y_test, y_pred, target_names=['No Recurrencia', 'Recurrencia']))

# ### Análisis de resultados y selección del mejor modelo
# results_df = pd.DataFrame(all_results1)
# print("\n" + "="*60)
# print("COMPARACIÓN DE TODOS LOS MODELOS")
# print("="*60)
# print(results_df.sort_values('auc_roc', ascending=False).to_string())

# # Guardar resultados
# results_df.to_csv("./results/comparacion_modelos_lbfgf.csv", index=False)

# # Encontrar el mejor modelo
# best_idx = results_df['auc_roc'].idxmax()
# best_model = results_df.loc[best_idx]
# print(f"\n{'='*60}")
# print("MEJOR MODELO ENCONTRADO:")
# print('='*60)
# for key, value in best_model.items():
#     print(f"{key}: {value}")

# # Entrenar el mejor modelo final con todos los datos
# # print("\n" + "="*60)
# # print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
# # print("="*60)

# # best_strategy = best_model['imputation']
# # best_scaler_name = best_model['scaler']
# # selected_features = results[best_strategy]['selected_features']

# # if len(selected_features) == 0:
# #     selected_features = X.columns

# # X_final = results[best_strategy]['X_imputed'][selected_features]

# # # Crear pipeline final
# # final_pipeline = Pipeline([
# #     ('scaler', scalers[best_scaler_name]),
# #     ('classifier', LogisticRegression(
# #         solver='lbfgs',
# #         max_iter=1000,
# #         random_state=42,
# #         class_weight='balanced'
# #     ))
# # ])

# # # Entrenar con todos los datos
# # final_pipeline.fit(X_final, y)

# # # Guardar el modelo final
# # import joblib
# # joblib.dump(final_pipeline, "./models/logistic_regression_final_model.pkl")
# # joblib.dump(selected_features, "./models/selected_features.pkl")

# # print(f"Modelo final guardado en: ./models/logistic_regression_final_model.pkl")
# # print(f"Características utilizadas: {list(selected_features)}")

# # # Obtener coeficientes del modelo final
# # if hasattr(final_pipeline.named_steps['classifier'], 'coef_'):
# #     coefficients = final_pipeline.named_steps['classifier'].coef_[0]
# #     feature_importance = pd.DataFrame({
# #         'feature': selected_features,
# #         'coefficient': coefficients,
# #         'abs_coefficient': abs(coefficients)
# #     }).sort_values('abs_coefficient', ascending=False)
    
# #     print("\nImportancia de características (coeficientes):")
# #     print(feature_importance.to_string())
    
# #     # Guardar importancia de características
# #     feature_importance.to_csv("./results/feature_importance.csv", index=False)

# # print("\n¡Análisis completado!")

# # RANDOM FOREST
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.datasets import make_regression

# # ...