import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(
        description = "It returns fixed dataframes with all the variables, to use as training and testing to each model."
    )

    parser.add_argument(
        "--time-followup",
        type=int,
        default=50,
        help="Time in years about following up max allowed. Default is 50 years (max reported 573.2 months)."
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default = 0.20,
        help = "Using an Hold-out method to split training and testing datasets. Default is 0.20 to test."
    )
    
    parser.add_argument(
        "--datalocated-fold",
        type=str,
        default = "database",
        help = "Path to the folder containing the datasets. Default is 'database'."
    )
    
    return parser.parse_args()
    
def define_variables(df): 
    """Return categorical (Dicotomics and ordinals), and continuous 
    variables, provide a dataframe cleaned, its based on nunique <6,
    except for 'SUBTYPE_FOLLI_PAPIL', 'NUMBEROFLYMPHNODEEXCISION', 
    'NUMBEROFPOSITIVELYMPHNODEEXCISION' """
    ## Classify variables in categorical and continuous:
    categorical_variable = []
    continuous_variable = []
    not_continuous = ["SUBTYPE_FOLLI_PAPIL", "NUMBEROFLYMPHNODEEXCISION", "NUMBEROFPOSITIVELYMPHNODEEXCISION", "number_ln_exc", "number_posit_ln", "subtype"]
    for i in df.columns:
        
        if i in ['record_id']:
            pass
        if i in not_continuous or df[i].nunique() <6: ### TNMT have 5
            categorical_variable.append(i)
        else:
            continuous_variable.append(i)
    with open('./variable_selection/all_categorical_vars.txt', 'w') as file:
        file.write('\n'.join(categorical_variable))
    with open('./variable_selection/all_continuous_vars.txt', 'w') as file:
        file.write('\n'.join(continuous_variable))
        
    return categorical_variable, continuous_variable


def create_summary_table(df, categorical,mapvar):
    df = df.copy()
    rows = []
    other_cont = ["number_ln_exc", "number_posit_ln"]
    categorical = [var for var in categorical if var not in other_cont]
    measure = {'age': "in years",
               'bmi': "kg/m^2", 
                'tumor_size': "in centimeters", 
                'ln_ratio': "in proportion lymph/nodes positives", 
                'size_posit_ln': "in centimeters", 
                'raidose': "in units zzzzz",
                'tg_pre_rai': "in units zzzzz", 
                "number_ln_exc": "n",
                "number_posit_ln": "n",
                'follow_months': "in months",
                'tsh_follow': "in units zzzzz", 
                'tg_follow': "in units zzzzz"}
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

if __name__ == "__main__":
    args = parse_args()

    datalocated_fold = args.datalocated_fold
    time_followup = args.time_followup * 12
    test_size = args.test_size

    data_base1 = pd.read_csv(f"./{datalocated_fold}/Base_pos_limpeza_V9_with Record ID.csv", sep = ";", encoding='utf-8-sig')
    data_base1["RAI"].replace("0,00", 2, inplace=True) 

    data_base2 = pd.read_csv(f"./{datalocated_fold}/HEE_limpia 6.csv", sep = ";", encoding='utf-8-sig')

    data_base3 = pd.read_csv(f"./{datalocated_fold}/AMBATO_Base_limpia_3.csv", sep = ";")

    data_base4 = pd.read_csv(f"./{datalocated_fold}/BASE DE DATOS CDT HNERM FINAL 2018 Paola.csv", sep = ";", encoding='utf-8-sig', nrows=195, usecols=range(49))
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
        "THYROID DISEASE PREOP": {1: 0, 2: 1, 3: 2}, #ONE HOT ENCODE
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
            "thy_disease_preop": {0: "EUTHYROIDISM", 1: "HYPOTHYROIDISM", 2: "HYPERTHYROIDISM"},
            "sex": {0: "Female", 1: "Male"}, #1-> 0, 2->1
            "radiotherapy": {1: "Yes", 0: "No"}, #2->0
            "family_history": {1: "Yes", 0: "No"}, #2->0,
            "euthyroidism": {0: "No", 1: "Yes"}, #2->0
            "hypothyroidism": {0: "No", 1: "Yes"},
            "hyperthyroidism": {0: "No", 1: "Yes"},
            "thyrodectomy_approach": {0: "Total", 1: "Total + Lymphadenectomy"}, #1->0, 2->1
            "type_resection": {0: "R0", 1: "R1", 2: "R2"},#1->0, 2->1, 3->2   
            "histology": {0: "Papilar", 1: "Folicular", 2: "Hurtle Cells"}, #1->0, 2->1, 3->2
            "subtype": {0: "Minimally invasive", 1: "Encapsulated invasive", 2: "Widely invasive",
                3: "Classic", 4: "Follicular variant", 5: "Encapsulated", 6: "Diffuse sclerosant",
                7: "High cells", 8: "Colunar cells", 9: "Cribiform-morular", 10: "Hobnail", 
                11: "Warthin-like", 12: "Oncocytic", 13: "Trabecular/Solid", 14: "Classic and Follicular",
                15: "Follicular and oncocytic"}, #1->0, 2->1, 3->2, 15->3, 16->4, 17->5, 5->6, 6->7, 7->8, 8->9, 9->10, 10->11, 11->12, 12->13, 13->14, 14->15
            "extra_thy_exten": {0: "Absent", 1: "Microscopic", 2: "Macroscopic"}, #1->0, 2->1, 3->2
            "multicentric": {1: "Yes", 0: "No"}, #2->0
            "multicent_bilat": {1: "Yes", 0: "No"}, #2->0
            "vascular_inv": {1: "Yes", 0: "No"}, #2->0
            "perineural_inv": {1: "Yes", 0: "No"}, #2->0
            "positive_ln": {0: "No excision", 2: "Yes", 1: "No"}, #2->1, 1->2
            "extranod_exten": {1: "Yes", 0: "No"}, #2->0
            "tnm_t": {0: "Tx", 1: "T1", 2: "T2", 3: "T3", 4: "T4"},
            "hashimoto": {1: "Yes", 0: "No"}, #2->0
            "tnm_n": {1: "N0", 2: "N1a", 3: "N1b", 0: "Nx"}, #3->0, 0->1, 1->2, 2->3
            "tnm_m": {0: "M0", 1: "M1"}, 
            "stage": {0: "I", 1: "II", 2: "III", 3: "IV"}, #1->0, 2->1, 3->2, 4->3
            "ata_2015": {1: "Bajo", 2: "Intermedio", 3: "Alto"}, 
            "ata_2025": {1: "Bajo", 2: "Intermedio o Bajo", 3: "Intermedio o Alto", 4: "Alto"},
            "rai": {1: "Yes", 0: "No"}, #2->0
            "recurrence": {0: "No Recurrence", 1: "Recurrence"}
            }

    # df.describe(include="all").T
    df = df.rename(columns={"SEX": "sex", "RADIOTHERAPY EXPOSURE": "radiotherapy", "AGEATDIGNOSIS": "age",
        "THYROID DISEASE PREOP": "thy_disease_preop", "FAMILY HISTORY OF THYROID CANCER": "family_history",
        "THYROIDECTOMY APPROACH": "thyrodectomy_approach","TYPEOFRESECTION": "type_resection",
        "HISTOLOGY": "histology", "SUBTYPE_FOLLI_PAPIL": "subtype", "BMI": "bmi",
        "TUMORSIZE (cm)": "tumor_size", "SIZEOFPOSITIVELYMPHNODE(cm)": "size_posit_ln",
        "NUMBEROFLYMPHNODEEXCISION": "number_ln_exc", "NUMBEROFPOSITIVELYMPHNODEEXCISION": "number_posit_ln",
        "EXTRATHYROIDALEXTENSION": "extra_thy_exten", "MULTICENTRIC": "multicentric",
        "MULTICENTER_BILATERAL": "multicent_bilat", "VASCULARINVASION": "vascular_inv",
        "PERINEURALINVASION": "perineural_inv", "POSITIVELYMPHNODEN1": "positive_ln", "LN RATIO": "ln_ratio",
        "EXTRANODALEXTENSION": "extranod_exten", "TNMT": "tnm_t", "HASHIMOTO THYROIDITIS": "hashimoto",
        "TNMN": "tnm_n", "TNMM": "tnm_m", "STAGE": "stage", "ATA_2015_RISCO_INICIAL": "ata_2015",
        "ATA_2025_RISCO_INICIAL": "ata_2025", "RAI": "rai", "RAIDOSE": "raidose",
        "OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY": "follow_months", "TG PRE RAI": "tg_pre_rai",
        "TSH FOLLOW UP": "tsh_follow",
        "TG FOLLOW UP": "tg_follow", "RECURRENCE": "recurrence"})

    categorical_variable, continuous_variable = define_variables(df)
    from sklearn.model_selection import train_test_split
    
    if time_followup >= max(df["follow_months"]):
        df.to_csv(f"./{datalocated_fold}/database_complete.csv", index=False)
        # table_stats_intial = create_summary_table(df, categorical_variable, map_variables)
        # table_stats_intial.to_excel("./results/initial_summary_table.xlsx", index=False)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["recurrence"], random_state=0)
        train_df.to_csv(f"./{datalocated_fold}/train_data.csv", index=False)
        test_df.to_csv(f"./{datalocated_fold}/test_data.csv", index=False)

    else:
        df_ly = df[df["follow_months"] <= time_followup]  # Select patients within the last {time_followup} years of follow-up
        df_ly.to_csv(f"./{datalocated_fold}/database_{str(time_followup // 12)}ly.csv", index=False)

        train_df_ly, test_df_ly = train_test_split(df_ly, test_size=0.2, stratify=df_ly["recurrence"], random_state=0)
        train_df_ly.to_csv(f"./{datalocated_fold}/train_data{str(time_followup // 12)}ly.csv", index=False)
        test_df_ly.to_csv(f"./{datalocated_fold}/test_data{str(time_followup // 12)}ly.csv", index=False)

        #### INFORMATION TRAINING DATASET
        # it is equal to R
        # table_stats_train = create_summary_table(train_df, categorical_variable, map_variables)
        # table_stats_train.to_excel("./results/training_summary_table.xlsx", index=False)

        # ### INFORMATION TRAINING DATASET
        # table_stats_test = create_summary_table(test_df, categorical_variable, map_variables)
        # table_stats_test.to_excel("./results/test_summary_table.xlsx", index=False)
