# https://www.youtube.com/watch?v=2tSQvxFArj4
library(VIM)
require(Hmisc)
require(data.table)
require(qreport)  # Define dataChk, missChk, maketabs, ...
library(dplyr)


data_model = read.csv("./database/database_summaried.csv", encoding = "UTF-8")
names(data_model) = c("sex", "age", "radiotherapy", "family_history", 
                      "thy_disease_preop", "bmi", 
                      "thyrodectomy_approach", "type_resection", "histology", 
                      "tumor_size", "extra_thy_exten", "multicentric", 
                      "multicent_bilat", "vascular_inv", "perineural_inv", 
                      "positive_ln", "number_ln_exc", "number_posit_ln",
                      "ln_ratio", "size_posit_ln", "extranod_exten", "hashimoto",
                      "tnm_t", "tnm_n", "tnm_m", "stage", "ata_2015", "ata_2025",
                      "rai", "raidose", "tg_pre_rai", "follow_months", "tsh_follow",
                      "tg_follow", "subtype", "recurrence")

cols_nodes_excision <- c("number_posit_ln", "ln_ratio", "size_posit_ln")
cols_rai <- c("raidose", "tg_pre_rai")
data_model_sub <- data_model %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))

data_model_sub <- data_model_sub %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, 0, .x)))

# data_model_subset <- data_model %>%
#   select(-c(number_posit_ln, ln_ratio, size_posit_ln, 
#             raidose, tg_pre_rai))

# [colSums(is.na(data_model_subset)) > 0]
aggr(data_model_sub[colSums(is.na(data_model_sub)) > 0], delimiter = NULL, sortVars = TRUE, 
     cex.axis = 0.45, las = 2, only.miss = FALSE)

Hmisc::na.delete(frame = data_model_sub) ### 48 observations without missing values

naclus <- Hmisc::naclus(df= data_model_sub,
                        method = "complete") #single
plot(naclus)

naclus_ward.D <- Hmisc::naclus(df=data_model_sub, 
                               method = "ward.D")
Hmisc::naplot(obj=naclus_ward.D,
              which = "na per obs") # na per var, mean na

# rms::lrm(formula, data = data_model_sub)

S1 <- rms::Surv(as.numeric(data_model_sub$follow_months), 
           as.numeric(data_model_sub$recurrence))
# Fit non-parametric survival
fit <- rms::npsurv(S1 ~ 1)

# Plot with risk table and confidence band
rms::survplot(fit, n.risk = TRUE, conf = "band", 
              ylab = "No-Recurrence ThyCanc probability",
              xlab = "Follow-up Time (months)",
              col.fill = 2, col = 1) 









data_model <- data_model %>%
  mutate(`thy_disease_preop` = recode(`thy_disease_preop`,
                                          "EUTHYROIDISM" = 0,
                                          "HYPOTHYROIDISM" = 1,
                                          "HYPERTHYROIDISM" = 2))

vars_to_factor <- c("sex", "radiotherapy", "family_history", "thy_disease_preop",
                    "thyrodectomy_approach", "type_resection", "histology",
                    "extra_thy_exten", "multicentric", "multicent_bilat",
                    "vascular_inv", "perineural_inv", "positive_ln",
                    "number_ln_exc", "number_posit_ln", "extranod_exten",
                    "hashimoto", "tnm_t", "tnm_n", "tnm_m", "stage",
                    "ata_2015", "ata_2025", "rai", "subtype", "recurrence")
data_raw <- data_model
data_model$tnm_t[data_model$tnm_t == 0] <- NA # tx solo 1, problems with bottstrap
data_model[vars_to_factor] <- lapply(data_model[vars_to_factor], as.factor)
data_raw[vars_to_factor] <- lapply(data_raw[vars_to_factor], as.factor)

#colinearity in positive_ln, ln_ratio, number_posit_ln 
# (for NaN we exclude ln_ratio, and change a number_posit_ln, number_ln_exc, 
# for categoterical ordinal since too few unique values)
# familiy_history, histology and tnm_n we exclude it in imputation for few unique values
data_model$number_posit_ln_cat <- cut(
  as.numeric(as.character(data_model$number_posit_ln)),
  breaks = c(-1, 0, 2, 5, Inf),
  labels = c("0", "1-2", "3-5", ">5")
)

data_model$number_ln_exc_cat <- cut(
  as.numeric(as.character(data_model$number_ln_exc)),
  breaks = c(-1, 10, 20, 50, Inf),
  labels = c("<=10", "11-20", "21-50", ">50")
)

data_m <- aregImpute(
  ~ recurrence + sex + age + thy_disease_preop +
    extra_thy_exten + multicentric + multicent_bilat + vascular_inv + 
    bmi + thyrodectomy_approach + type_resection + tumor_size + 
    perineural_inv + positive_ln + number_ln_exc_cat + number_posit_ln_cat + 
    size_posit_ln + extranod_exten + hashimoto + tnm_t + tnm_m + stage + 
    rai + raidose + tg_pre_rai + follow_months + tsh_follow + tg_follow,
  n.impute=5, data=data_model, group = data_model$recurrence, nk = 4)

data_model1 <- data_model
imputed<- impute.transcan(data_m, imputation=1, data=data_model)
data_model1[names(imputed)] <- imputed

# imputation with mice:
library(mice)
data_raw$number_ln_exc <- as.numeric(as.character(data_raw$number_ln_exc))
data_raw$number_posit_ln <- as.numeric(as.character(data_raw$number_posit_ln))
gm <- mice(data_raw, m=10, seed=1)
# densityplot(gm)

data_m1 <- complete(gm, 1) #### DATA CON MUJLTIPLE IMPUTATION 

# stacked set:
data_m10 <- complete(gm,1)
for (i in 2:10) { #2:m
  data_m10  <- rbind(data_m10, complete(gm,i))}
data_m10$w <- 1/10 # 1/m  #### DATA CON MULTIPLE IMPUTATION X10 