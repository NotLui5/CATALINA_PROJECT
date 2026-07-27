#https://www.youtube.com/watch?v=PH2o_KVF7Jw
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from step1_pre_data import define_variables

path_base = "./database/hee_brazil_ambato_peru_base.csv"

data = pd.read_csv(path_base)

data = data[['SEX', 'AGEATDIGNOSIS', 'RADIOTHERAPY EXPOSURE',
       'FAMILY HISTORY OF THYROID CANCER', 'BMI', 'THYROIDECTOMY APPROACH',
       'TYPEOFRESECTION', 'HISTOLOGY', 'TUMORSIZE (cm)',
       'EXTRATHYROIDALEXTENSION', 'MULTICENTRIC', 'MULTICENTER_BILATERAL',
       'VASCULARINVASION', 'PERINEURALINVASION', 'POSITIVELYMPHNODEN1',
       'NUMBEROFLYMPHNODEEXCISION', 'NUMBEROFPOSITIVELYMPHNODEEXCISION',
       'LN RATIO', 'SIZEOFPOSITIVELYMPHNODE(cm)', 'EXTRANODALEXTENSION',
       'HASHIMOTO THYROIDITIS', 'TNMT', 'TNMN', 'TNMM', 'STAGE',
       'RAI', 'RAIDOSE', 'TG PRE RAI', 'TSH FOLLOW UP', 'TG FOLLOW UP', 
       'SUBTYPE_FOLLI_PAPIL', 'EUTHYROIDISM', 'HYPOTHYROIDISM', 'HYPERTHYROIDISM',
       'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY', 'RECURRENCE']]

categorical_vars, continuous_vars = define_variables(data)

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

## See distribution after imputations
def distribution_column(column: str, data: pd.DataFrame,  
                        imputation: bool, i_style: str ="median", hue: str = "RECURRENCE",
                        other_variables: list = None, colum_mask: str = None):
    """ 
    See distributions of one col in a data frame after than for imputation choose False or True, type [mean, mode, median, drope] in i_style, 
    it shows according to RECURRENCE. other_variables apply if masking data is neccesary to let NaN values according to column_mask.
    """
    if imputation:
        if i_style == "median":  
            data[column].fillna(data[column].median(), inplace=True)
        if i_style == "mode":
            data[column].fillna(data[column].mode()[0], inplace=True)
        if i_style == "mean":
            data[column].fillna(data[column].mean(), inplace=True)  
        if i_style == "drop":
            data.dropna(subset=[column], inplace=True)
            
        ### Based on column = 0, we set to NaN the other vars
        if other_variables is not None:
            colum_mask = column if colum_mask is None else colum_mask
            mask = (data[colum_mask] == 0) #NaN 
            data[other_variables] = data[other_variables].mask(mask, np.nan)    
    else:
        pass
        
    sns.histplot(data=data, x=column, hue=hue, kde=False, 
                            element="step", common_norm=False, alpha=0.6)
    plt.title(f'Distribution of Recurrence by {column}')
    plt.legend(title="Recurrence", labels=["No", "Yes"], prop={'size': 10})
    plt.tight_layout()
    plt.show()

# # ((data.isnull().sum())/len(data)*100).sort_values(ascending=False)
# mask_data = {"nodes": ["NUMBEROFPOSITIVELYMPHNODEEXCISION", "LN RATIO", "SIZEOFPOSITIVELYMPHNODE(cm)"], #NaN -> 604, 614, 921
#             "rai": ["RAIDOSE", "TG PRE RAI"] #NaN -> 284, 520
# }
# for i in data.columns:
#     if ((data[i].isnull().sum())/len(data)) < 0.05:
#         print(i, "has less than 5%\ of missing data. \n Applying Simpple Imputation... \n")
    
#     elif not any(i in v for v in mask_data.values() if isinstance(v, list)):
#         print(i, "is not neccesary to aply masking in imputation on data")
        
#         if i in categorical_vars:
#             distribution_column(column=i, data=data, imputation=True, i_style="mode")
#             print("Mode imputation done in", i)
#         else:
#             if data[i].mean != data[i].median:
#                 distribution_column(column=i, data=data, imputation=True, i_style="median")
#                 print(i, f"imputed by median {data[i].median}")
#             if data[i].mean == data[i].median:
#                 distribution_column(column=i, data=data, imputation=True, i_style="mean")
        
# distribution_column(column="BMI", data=data, imputation=True, i_style="median", other_variables=None)

# distribution_column(column="TUMORSIZE (cm)", data=data, imputation=True, i_style="median", other_variables=None)

# nodes = ["NUMBEROFPOSITIVELYMPHNODEEXCISION", "LN RATIO", "SIZEOFPOSITIVELYMPHNODE(cm)"] #NaN -> 604, 614, 921
# ### Based on NUMBEROFLYMPHNODEEXCISION = 0, we set to NaN the nodes vars
# distribution_column(column="NUMBEROFLYMPHNODEEXCISION", data=data, imputation=True, i_style="mode", other_variables=nodes)

# distribution_column(column="NUMBEROFPOSITIVELYMPHNODEEXCISION", data=data, imputation=True, i_style="mode", other_variables=nodes,
#                     colum_mask="NUMBEROFLYMPHNODEEXCISION")

# distribution_column(column="LN RATIO", data=data, imputation=True, i_style="median", other_variables=nodes,
#                     colum_mask="NUMBEROFLYMPHNODEEXCISION")

# distribution_column(column="SIZEOFPOSITIVELYMPHNODE(cm)", data=data, imputation=True, i_style="median", other_variables=nodes,
#                     colum_mask="NUMBEROFLYMPHNODEEXCISION")


# rai = ["RAIDOSE", "TG PRE RAI"] #NaN -> 284, 520
# distribution_column(column="RAI", data=data, imputation=True, i_style="mode", other_variables=rai) #NaN -> 3

# distribution_column(column="RAIDOSE", data=data, imputation=True, i_style="median", other_variables=rai,
#                     colum_mask="RAI")

# distribution_column(column="TG PRE RAI", data=data, imputation=True, i_style="median", other_variables=rai,
#                     colum_mask="RAI")


# distribution_column(column="TSH FOLLOW UP", data=data, imputation=True, i_style="median")
# distribution_column(column="TG FOLLOW UP", data=data, imputation=True, i_style="median")

# data1 = data.copy()
# distribution_column(column="OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY", data=data, imputation=True, i_style="median")
# distribution_column(column="OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY", data=data1, imputation=True, i_style="drop")

T =data["OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY"]
E = data["RECURRENCE"]  
w = data["w"]
plt.hist(T, bins=30, alpha=0.7)
plt.show()



#==================================================
### Fitting a non-parametric model (Kaplan-Meier) 
#==================================================

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
kmf = KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E, weights=w)
kmf.plot_survival_function()
plt.show()

kmf.survival_function_.plot()
plt.show()

kmf.plot_cumulative_density()
plt.show()

kmf.median_survival_time_


kmf = KaplanMeierFitter()
kmf.fit(durations=T1, event_observed=E1)
kmf.plot_survival_function()
plt.show()
from lifelines.utils import median_survival_times
median_ = kmf.median_survival_time_
median_connfidence_interval = median_survival_times(kmf.confidence_interval_)
print(median_)
print(median_connfidence_interval)


from lifelines.utils import median_survival_times
median_ = kmf.median_survival_time_
median_connfidence_interval = median_survival_times(kmf.confidence_interval_)
print(median_)
print(median_connfidence_interval)


### Survival curves by sex
def kmf_fit_by(data, variable, names: list):
    """Take account to RECURRENCE and OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY
    - Input: variable or variables to fit
    + Output: plot a survival function"""
    T = data["OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY"]
    E = data["RECURRENCE"]  
    if len(names) == 2:
        ax = plt.subplot(111)

        m = (data[variable] == 1)
        
        kmf.fit(durations = T[m], event_observed = E[m], label = names[0])
        kmf.plot_survival_function(ax = ax)

        kmf.fit(durations = T[~m], event_observed = E[~m], label = names[1])
        kmf.plot_survival_function(ax = ax, at_risk_counts = True)

        plt.title(f"Recurrence of different {variable} group")
        plt.show()
    else:
        for i, var_type in enumerate(names):
            ax = plt.subplot(2, 3, i+1)
            ix = (data[variable] == i)
            kmf.fit(T[ix], E[ix], label = var_type)
            kmf.plot_survival_function(ax = ax, legend = False)
            ax.set_title(var_type)
            ax.set_xlim(0, 600)
        plt.tight_layout()
        # plt.axis('off')
        # plt.figure(frameon=False)   
        plt.show()

### Survival curves by sex
for i in categorical_vars:
    if i in ["POSITIVELYMPHNODEN1","NUMBEROFLYMPHNODEEXCISION", "NUMBEROFPOSITIVELYMPHNODEEXCISION", "SUBTYPE_FOLLI_PAPIL", "RECURRENCE"]:
        continue
    else:
        kmf_fit_by(data, variable=i, #!
                names = list(map_variables[i].values()))

#-----------------------------------
#####Fitting Cox Proportionl Hazard model (semiparametric)
#-----------------------------------
### Hazard and Hazard ratio
resection_type = pd.get_dummies(data["TYPEOFRESECTION"], prefix="TYPEOFRESECTION")
hystology_type = pd.get_dummies(data["HISTOLOGY"], prefix = "HISTOLOGY")
extrathy_extension = pd.get_dummies(data["EXTRATHYROIDALEXTENSION"], prefix = "EXTRATHYROIDALEXTENSION")
tnmt = pd.get_dummies(data["TNMT"], prefix="TNMT")
tnmn = pd.get_dummies(data["TNMN"], prefix = "TNMN")
stage = pd.get_dummies(data["STAGE"], prefix = "STAGE")

data = pd.concat([data, resection_type, hystology_type, extrathy_extension, tnmt, tnmn, stage], axis=1)
data = data.drop(["TYPEOFRESECTION", "HISTOLOGY", "EXTRATHYROIDALEXTENSION", "TNMT", "TNMN", "STAGE"], axis = 1) 

cph = CoxPHFitter()
cph.fit(data1, duration_col = 'OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY',  event_col = "RECURRENCE")

cph.plot() ## in log HR

# cph.plot_partial_effects_on_outcome(covariates="AGE", values = [50, 60, 70, 80], cmap = 'coolwarm')
cph.check_assumptions(data, p_value_threshold=0.05)

from lifelines.statistics import proportional_hazard_test
results = proportional_hazard_test(cph, data, time_transform="rank")
results.print_summary(decimals=3, model="untransformed variables")

#=========================================================
### Parametric [Accelerated Failure Time Model (AFT)]
#=========================================================

from lifelines import WeibullFitter,\
    ExponentialFitter,\
        LogNormalFitter,\
            LogLogisticFitter,\
                WeibullAFTFitter

wb = WeibullFitter()
ex = ExponentialFitter()
log = LogNormalFitter()
loglogis = LogLogisticFitter()

for model in [wb, ex, log, loglogis]:
    model.fit(durations=data["OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY"],
              events_observed = data["RECURRENCE"])
    print("The AIC value for", model.__class__.__name__, "is", model.AIC_)

wb_aft = WeibullAFTFitter()
wb_aft.fit(data, durations=data["OUTCOMEFOLLOWUP_MONTHSPOSTSURGERY"],
              events_observed = data["RECURRENCE"])
wb_aft.print_summary(3)

print(wb_aft.median_survival_time_)
print(wb_aft.mean_survival_time_)

plt.subplots(figsize=(10,6))
wb_aft.plot() #LOG COEF AFR
plt.show()

plt.subplots(figsize=(10,6))
wb_aft.plot_partial_effects_on_outcome("AGE". range(50,80,10), cmap='coolwarm')


