class LOAD_AND_PREPROCESS_DATA:
    
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
        "THYROID DISEASE PREOP ": {1: "Euthyroidism", 2: "Hypothyroidism", 3: "Hyperthyroidism"},
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

        ##### Data wrangling
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

    total = df.isnull().sum().sort_values(ascending=False)
    percent_total = (df.isnull().sum()/len(df)).sort_values(ascending=False)*100
    missing = pd.concat([total, round(percent_total, 2)], axis=1, keys=['Total', 'Percent'])
    missing = missing[missing['Total']>0]
    missing

    # nan_columns_list = missing.index.tolist()

    # n_rows = 8

    # fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(15, 42))
    # # Calculamos cuántas columnas de datos van por cada fila
    # x = len(nan_columns_list) // n_rows

    # for i in range(n_rows):
    #     # Seleccionamos el subconjunto de columnas para esta fila
    #     start = i * x
    #     # Si es la última fila, tomamos hasta el final para no dejar columnas fuera
    #     end = (i + 1) * x if i < n_rows - 1 else len(nan_columns_list)
        
    #     subset = nan_columns_list[start:end]
        
    #     # Dibujamos el boxplot en el eje correspondiente (axes[i])
    #     df[subset].boxplot(ax=axes[i])
    #     axes[i].set_title(f" ")

    # # Ajustamos el layout para que no se encimen los textos
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

    #########################################
    ### IMPUTATION
    #########################################
    for i in df.columns: 
        if i in categorical_variable:
            df.fillna({i: df[i].mode()[0]}, inplace=True)
        else: 
            df.fillna({i: df[i].median()}, inplace=True)

    #########################################      
    ## CHARTS
    #########################################
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
    plt.show()

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
        plt.show(block=True)
        plt.close()
        
    # Correlations? 
    path_dis = "./distribution/df_continuous_distributions.png"
    if not os.path.exists(path_dis):
        sns.pairplot(df[continuous_variable], hue="RECURRENCE")
        plt.savefig(path_dis)
        plt.show(block=True)
        plt.close()
        
    ## Chart Correlation Heatmap
    plt.figure(figsize=(12,12))
    correlation = df.corr()
    sns.heatmap((correlation), annot=False, cmap=sns.color_palette("mako", as_cmap=True))
    plt.savefig("./variable_selection/heatmap_correlation.png")
    plt.show(block=True)
    plt.close()

    ###### Handling Outliers & Outlier treatments
    fig, axes = plt.subplots(5, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, continuous_variable):
        sns.boxplot(df[col], ax=ax)
        ax.set_title(col.title(), weight='bold')
    plt.tight_layout()
    plt.show()

    df[continuous_variable] = np.log(df[continuous_variable] + 1)  # Log-transform to handle skewness and outliers
    fig, axes = plt.subplots(5, 4, figsize=(15, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, continuous_variable):
        sns.boxplot(df[col], ax=ax)
        ax.set_title(col.title(), weight='bold')
    plt.tight_layout()
    plt.show()

    ### Categorical Encoding
    for col in df.columns:
        if col in categorical_variable:
            df[col] = df[col].astype('uint8') #For improve performance
    
    return df, categorical_variable, continuous_variable, map_variables    
