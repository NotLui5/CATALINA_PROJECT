import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def define_variables(df): 
    """Return categorical (Dicotomics and ordinals), and continuous 
    variables, provide a dataframe cleaned, its based on nunique <6,
    except for 'SUBTYPE_FOLLI_PAPIL', 'NUMBEROFLYMPHNODEEXCISION', 
    'NUMBEROFPOSITIVELYMPHNODEEXCISION' """
    ## Classify variables in categorical and continuous:
    categorical_variable = []
    continuous_variable = []
    not_continuous = ["SUBTYPE_FOLLI_PAPIL", "NUMBEROFLYMPHNODEEXCISION", "NUMBEROFPOSITIVELYMPHNODEEXCISION"]
    for i in df.columns:
        
        if i in ['record_id']:
            pass
        if i in not_continuous or df[i].nunique() <6: ### TNMT have 5
            categorical_variable.append(i)
        else:
            continuous_variable.append(i)
    return categorical_variable, continuous_variable

data_base1 = pd.read_csv("./database/Base_pos_limpeza_V9_with Record ID.csv", sep = ";", encoding='utf-8-sig')
data_base1["RAI"].replace("0,00", 2, inplace=True) 

data_base2 = pd.read_csv("./database/HEE_limpia 6.csv", sep = ";", encoding='utf-8-sig')

data_base3 = pd.read_csv("./database/AMBATO_Base_limpia_3.csv", sep = ";")

data_base4 = pd.read_csv("./database/BASE DE DATOS CDT HNERM FINAL 2018 Paola.csv", sep = ";", encoding='utf-8-sig', nrows=195, usecols=range(49))
data_base4["EXTRANODALEXTENSION"].replace(0, 2, inplace=True)
data_base4["SEX"].replace(0, 2, inplace=True)


##!!! We won't use ata2015, but HEEE and AMBATO bases didn't reported ATA2025.
data_base2["ATA2025LAST_ULTIMA_CONSULTA"] = data_base2["ATA2015_ULTIMA_CONSULTA"]
data_base3["ATA2025LAST_ULTIMA_CONSULTA"] = data_base3["ATA2015_ULTIMA_CONSULTA"]

##! Diferent colnames
# for x in data_base1.columns:
#     if x not in data_base3.columns:
#         print(x)
data_base2.rename(columns={"ID": "record_id", "BMI ": "BMI"}, inplace=True)
data_base3.rename(columns={"TNMT ": "TNMT"}, inplace=True)
data_base4.rename(columns={"ID": "record_id", "BMI ": "BMI"}, inplace=True)
if not data_base1.columns.equals(data_base2.columns):
    print("Check colnames database1 and 2")
elif not data_base2.columns.equals(data_base3.columns):
    print("Check colnames database2 and 3")
elif not data_base3.columns.equals(data_base4.columns):
    print("Check colnames database3 and 4")
else:
    print("Colnames are the same in all databases, we can proceed to join them.")    
    
### Join just one database:
df = pd.concat([data_base1, data_base2, data_base3, data_base4], ignore_index=True, 
            names= list(data_base1.columns), verify_integrity=True, 
            sort = False)

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

## Create RECURRENCE outcome variable by ATA.
df['RECURRENCE'] = df['ATA2025LAST_ULTIMA_CONSULTA']

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
    ### Remove by amount of missing data and desicion of authors
    "ANTI TG PRE RAI ",
    "ANTI TG PRE RAI (POSITIVE or NEGATIVE)",
    "TSH PRE RAI",
    "ANTI TG FOLLOW UP",
    "ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)"
    ]

df = df.drop(cols_remove, axis=1) # Remove col innecesary

categorical_variable, continuous_variable = define_variables(df)
for i in df.columns:
    df[i] = df[i].astype(str).str.replace(',', '.', regex=False)
    if i in continuous_variable:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    else:
        df[i] = pd.to_numeric(df[i], errors='coerce').astype("Int64")            

categorical_variable, continuous_variable = define_variables(df)
# Label encoding for the outcome variable, 0 for absence
map_recurrence = {"SEX": {1:0, 2: 1}, #1-> 0, 2->1
    "RADIOTHERAPY EXPOSURE": {1:1, 2: 0}, #2->0
    "FAMILY HISTORY OF THYROID CANCER": {1: 1, 2: 0}, #2->0
    "THYROID DISEASE PREOP": {1: "EUTHYROIDISM", 2: "HYPOTHYROIDISM", 3: "HYPERTHYROIDISM"}, #ONE HOT ENCODE
    "THYROIDECTOMY APPROACH": {1: 0, 2: 1}, #1->0, 2->1
    "TYPEOFRESECTION": {1: 0, 2: 1, 3: 2},#1->0, 2->1, 3->2   
    "HISTOLOGY": {1: 0, 2: 1, 3: 2}, #1->0, 2->1, 3->2
    "SUBTYPE_FOLLI_PAPIL": {1: 0, 2: 1, 3: 2,
        15: 3, 16: 4, 17: 5, 5: 6, 6: 7, 7: 8,
        8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15}, #1->0, 2->1, 3->2, 15->3, 16->4, 17->5, 5->6, 6->7, 7->8, 8->9, 9->10, 10->11, 11->12, 12->13, 13->14, 14->15
    "EXTRATHYROIDALEXTENSION": {1: 0, 2: 1, 3: 2}, #1->0, 2->1, 3->2
    "MULTICENTRIC": {1: 1, 2: 0}, #2->0
    "MULTICENTER_BILATERAL": {1: 1, 2: 0}, #2->0
    "VASCULARINVASION": {1: 1, 2: 0}, #2->0
    "PERINEURALINVASION": {1: 1, 2: 0}, #2->0
    "POSITIVELYMPHNODEN1": {0: 0, 1: 2, 2: 1}, #2->1, 1->2
    "EXTRANODALEXTENSION": {1: 1, 2: 0}, #2->0
    "TNMT": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    "HASHIMOTO THYROIDITIS": {1: 1, 2: 0}, #2->0
    "TNMN": {0: 1, 1: 2, 2: 3, 3: 0}, #3->0, 0->1, 1->2, 2->3
    "TNMM": {1: 0, 2: 1}, 
    "STAGE": {1: 0, 2: 1, 3: 2, 4: 3}, #1->0, 2->1, 3->2, 4->3
    "ATA_2015_RISCO_INICIAL": {1: 1, 2: 2, 3: 3}, 
    "ATA_2025_RISCO_INICIAL": {1: 1, 2: 2, 3: 3, 4: 4},
    "RAI": {1: 1, 2: 0}, #2->0
    # "ANTI TG PRE RAI (POSITIVE or NEGATIVE)": {1: 1, 2: 0}, #2->0
    # "ANTI TG FOLLOW UP (POSITIVE or NEGATIVE)": {1: 1, 2:0},
}

for col, mapping in map_recurrence.items():
    df[col] = df[col].replace(mapping)

df['RECURRENCE'] = df['RECURRENCE'].replace([1,2,3],0).replace(4, 1)

map_variables = { #Meanings 
        "THYROID DISEASE PREOP": {"EUTHYROIDISM": "EUTHYROIDISM", "HYPOTHYROIDISM": "HYPOTHYROIDISM", "HYPERTHYROIDISM": "HYPERTHYROIDISM"},
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

# df.describe(include="all").T

def create_summary_table(df, categorical,mapvar):
    df = df.copy()
    rows = []
    other_cont = ["NUMBEROFLYMPHNODEEXCISION", "NUMBEROFPOSITIVELYMPHNODEEXCISION"]
    categorical = [var for var in categorical if var not in other_cont]
    measure = {'AGEATDIGNOSIS': "in years",
               'BMI': "kg/m^2", 
                'TUMORSIZE (cm)': "in centimeters", 
                'LN RATIO': "in proportion lymph/nodes positives", 
                'SIZEOFPOSITIVELYMPHNODE(cm)': "in centimeters", 
                'RAIDOSE': "in units zzzzz",
                'TG PRE RAI': "in units zzzzz", 
                "NUMBEROFLYMPHNODEEXCISION": "n",
                "NUMBEROFPOSITIVELYMPHNODEEXCISION": "n",
                'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY': "in months",
                'TSH FOLLOW UP': "in units zzzzz", 
                'TG FOLLOW UP': "in units zzzzz"}
    # Demographics
    for col, mapping in mapvar.items():
        if col in df.columns:
            df[col] = df[col].astype(object).replace(mapping)
            
    for i in df.columns:        
        if i in categorical:
            rows.append((f'{i} (n, {df[i].isna().sum()} missing)', ''))
            for val, count in df[i].value_counts().items():
                val_tag = mapping.get(val, val) if i in mapvar else val
                pct = count / df[i].notna().sum() * 100
                # Indentamos con 4 espacios
                rows.append((f'    {val}', f"{count} ({pct:.0f}%)"))       
        else:                    
            data_i = df[i].dropna()
            rows.append((f'{i} ({measure[i]}, {df[i].isna().sum()} missing)',
                         f"{data_i.median():.0f} [{data_i.quantile(0.25):.0f}–{data_i.quantile(0.75):.0f}]"))

    return pd.DataFrame(rows, columns=['Features', 'Stats'])

df.to_csv("./database/database_summaried.csv", index=False)

table_stats_intial = create_summary_table(df, categorical_variable, map_variables)
table_stats_intial.to_excel("./results/initial_summary_table.xlsx", index=False)

data_set = pd.read_csv("./database/database_summaried.csv")
data1 = data_set.copy()
data1["THYROID DISEASE PREOP"] = data1["THYROID DISEASE PREOP"].replace({"EUTHYROIDISM": 0, "HYPOTHYROIDISM": 1, "HYPERTHYROIDISM": 2})
categ, continous = define_variables(data1)
for x in data1.columns:
    if x in categ:
        data1[x] = data1[x].astype('Int64')
    
nodes_vars = ['NUMBEROFPOSITIVELYMPHNODEEXCISION', 'LN RATIO', 'SIZEOFPOSITIVELYMPHNODE(cm)']
rai_vars = ['RAIDOSE', 'TG PRE RAI']

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Linear model import
from sklearn.linear_model import BayesianRidge
# Ensemble model imports
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


# Condición previa a la imputación: si NUMBEROFLYMPHNODEEXCISION == 0 -> nodes_vars = NA
cond_nodes = data1['NUMBEROFLYMPHNODEEXCISION'] == 0
for col in nodes_vars:
    data1.loc[cond_nodes, col] = np.nan

# Si RAI == 0 -> rai_vars = NA (antes de imputar)
cond_rai = data1['RAI'] == 0
for col in rai_vars:
    data1.loc[cond_rai, col] = np.nan

# Imputadores
imputers = {
    'bayesian_ridge': IterativeImputer(estimator=BayesianRidge(), random_state=42),
    'extra_trees': IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42), random_state=42),
    'rf_regressor': IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), random_state=42)
}

imputed_datasets = {}

for name, imputer in imputers.items():
    # Imputar todo el data1
    imputed_array = imputer.fit_transform(data1)
    imputed_df = pd.DataFrame(imputed_array, columns=data1.columns, index=data1.index)
    
    # Reaplicar condiciones con los valores imputados de las variables condición
    cond_nodes_post = imputed_df['NUMBEROFLYMPHNODEEXCISION'] == 0
    for col in nodes_vars:
        imputed_df.loc[cond_nodes_post, col] = np.nan
    
    cond_rai_post = imputed_df['RAI'] == 0
    for col in rai_vars:
        imputed_df.loc[cond_rai_post, col] = np.nan
    
    imputed_datasets[name] = imputed_df

# print("\n3. Imputed Dataset Versions based on Different Estimators:")
for name, dataset in imputed_datasets.items():
    print(f"{name}: THYROID DISEASE PREOP = {dataset['THYROID DISEASE PREOP'].value_counts()}")
    
    
#Data final to start imputation and modeling
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',dtype=int )
encoded_array = encoder.fit_transform(df[['THYROID DISEASE PREOP']])

feature_names = encoder.get_feature_names_out(['THYROID DISEASE PREOP'])
print(f"Nombres de columnas: {feature_names}")  # Ver cuántos hay
print(f"Forma del array codificado: {encoded_array.shape}")  # Debe coincidir

# Crear DataFrame con las nuevas columnas
encoded_df = pd.DataFrame(
    encoded_array,
    columns=feature_names,   # Aquí usamos exactamente los nombres devueltos
    index=df.index
)
encoded_df = encoded_df.drop(columns=['THYROID DISEASE PREOP_nan'], errors='ignore')
encoded_df.columns = encoded_df.columns.str.replace('THYROID DISEASE PREOP_', '')
# Unir con el resto de columnas (excluyendo la original)
df = pd.concat([df.drop('THYROID DISEASE PREOP', axis=1), encoded_df], axis=1)

path_base = "./database/hee_brazil_ambato_peru_base.csv"
df.to_csv(path_base, index=False)
