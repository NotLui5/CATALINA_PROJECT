# https://www.youtube.com/watch?v=2tSQvxFArj4
library(VIM)
require(Hmisc)
require(data.table)
require(qreport)  # Define dataChk, missChk, maketabs, ...
library(dplyr)
library(mice)
library(DescTools)
library(rms)

### Data summaried is train, develop and internal validation dataset.
## All data
# data_model = read.csv("./database/database_complete.csv", encoding = "UTF-8")
# data_raw= read.csv("./database/train_data.csv", encoding = "UTF-8")
# data_raw = read.csv("./database/test_data.csv", encoding = "UTF-8")
# path_save_mice1_train = "./database/data_train_mice1.csv"
# path_save_mice10_train = "./database/data_train_mice10.csv"
# path_save_mice_train = "./database/imputation_mice/data_train_imp_"
# path_save_mice1_test = "./database/data_internal_valid_mice1.csv"
# path_save_mice10_test = "./database/data_internal_valid_mice10.csv"
# path_save_mice_test = "./database/imputation_mice/data_internal_valid_imp_"


## Specific last follow up years data
## only in data_model name "15ly" is after '_' in the rest "15ly" is immediately after of 'data'
data_model = read.csv("./database/database_15ly.csv", encoding = "UTF-8")
data_raw_train = read.csv("./database/train_data15ly.csv", encoding = "UTF-8")
data_raw_test = read.csv("./database/test_data15ly.csv", encoding = "UTF-8")
path_save_mice1_train = "./database/data15ly_train_mice1.csv"
path_save_mice10_train = "./database/data15ly_train_mice10.csv"
path_save_mice_train = "./database/imputation_mice/data15ly_train_imp_"
path_save_mice1_test = "./database/data15ly_internal_valid_mice1.csv"
path_save_mice10_test = "./database/data15ly_internal_valid_mice10.csv"
path_save_mice_test = "./database/imputation_mice/data15ly_internal_valid_imp_"



######################################
### ALWAYS DO FOR MATCH WITH A LOGICAL RECOLECTION DATA.
#####################################
data_model <- data_model %>% 
  select(-"follow_months")
data_raw_train <- data_raw_train %>% 
  select(-"follow_months")
data_raw_test <- data_raw_test %>% 
  select(-"follow_months")

cols_nodes_excision <- c("number_posit_ln", "ln_ratio", "size_posit_ln")
cols_rai <- c("raidose", "tg_pre_rai")
data_model_sub <- data_model %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))  %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, 0, .x))) 


#################################################################################
### Visualizing NaN data: 
#################################################################################

# data_model_subset <- data_model %>%
#   select(-c(number_posit_ln, ln_ratio, size_posit_ln, 
#             raidose, tg_pre_rai))

# [colSums(is.na(data_model_subset)) > 0]
aggr(data_model_sub[colSums(is.na(data_model_sub)) > 0], delimiter = NULL, sortVars = TRUE, 
     cex.axis = 0.45, las = 2, only.miss = FALSE)

Hmisc::na.delete(frame = data_model_sub) ### 48 observations without missing values
print(paste0(length(data_model_sub), " observations without missing values"))

naclus <- Hmisc::naclus(df= data_model_sub,
                        method = "complete") #single
plot(naclus)

naclus_ward.D <- Hmisc::naclus(df=data_model_sub, 
                               method = "ward.D")
Hmisc::naplot(obj=naclus_ward.D,
              which = "na per obs") # na per var, mean na


# rms::lrm(formula, data = data_model_sub)


#################################################################################
### Imputation starts...
#################################################################################

######################################
## Single imputation 

vars_to_factor <- c("sex", "radiotherapy", "family_history", "thy_disease_preop",
                    "thyrodectomy_approach", "type_resection", "histology",
                    "extra_thy_exten", "multicentric", "multicent_bilat",
                    "vascular_inv", "perineural_inv", "positive_ln",
                    "number_ln_exc", "number_posit_ln", "extranod_exten",
                    "hashimoto", "tnm_t", "tnm_n", "tnm_m", "stage",
                    "ata_2015", "ata_2025", "rai", "subtype", "recurrence")
data_model$tnm_t[data_model$tnm_t == 0] <- NA # tx solo 1, problems with bottstrap
data_model[vars_to_factor] <- lapply(data_model[vars_to_factor], as.factor)

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
    rai + raidose + tg_pre_rai + tsh_follow + tg_follow, #follow_months,
  n.impute=5, data=data_model, group = data_model$recurrence, nk = 4)

data_model1 <- data_model
imputed<- impute.transcan(data_m, imputation=1, data=data_model)
data_model1[names(imputed)] <- imputed



######################################
# MICE imputation training dataset
######################################

data_raw_train[vars_to_factor] <- lapply(data_raw_train[vars_to_factor], as.factor)
data_raw_train$recurrence <- as.numeric(as.character(data_raw_train$recurrence))
data_raw_train$number_ln_exc <- as.numeric(as.character(data_raw_train$number_ln_exc))
data_raw_train$number_posit_ln <- as.numeric(as.character(data_raw_train$number_posit_ln))

cols_skewed = c("tumor_size", "ln_ratio", "tg_pre_rai", "tsh_follow", "tg_follow")
data_raw_train <- data_raw_train %>%
  mutate(across(cols_skewed, ~ DescTools::Winsorize(., val = c(0.01, 0.99))))%>%
  mutate(rai = ifelse(is.na(rai), names(which.max(table(rai))), as.character(rai))) %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, 0, .x))) %>%
  mutate(across(all_of(cols_skewed), ~ log(.)))

data_raw_train$rai <- as.factor(data_raw_train$rai)

gm <- mice(data_raw_train, m=10, seed=1)
# densityplot(gm)


data_m1 <- complete(gm, 1) #### DATASET 1 CON MUJLTIPLE IMPUTATION 

# stacked set:
data_m10 <- complete(gm,1)
for (i in 2:10) { #2:m
  data_m10  <- rbind(data_m10, complete(gm,i))}
data_m10$w <- 1/10 # 1/m  #### DATA CON MULTIPLE IMPUTATION X10 

data_m1 <- data_m1 %>%
  mutate(across(cols_skewed, ~exp(.))) %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))
  
write.csv(data_m1, path_save_mice1_train, row.names = FALSE)

data_m10 <- data_m10 %>%
  mutate(across(cols_skewed, ~exp(.))) %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))

write.csv(data_m10, path_save_mice10_train, row.names = FALSE)

for (i in 1:10) {
  df_i <- complete(gm, i)
  df_i <- df_i %>%
    mutate(across(cols_skewed, ~exp(.))) %>%
    mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))
  write.csv(df_i, paste0(path_save_mice_train, i, ".csv"), row.names = FALSE)
}

####
# #Check percentiles
# for (i in 1:5) {
#   cat("\nImputación:", i, "\n")
#   print(summary(complete(gm, i)[cont_vars]))
# }
# 
# #Winsorize
# data_raw_train <- data_raw_train %>%
#   mutate(across(c(tumor_size, ln_ratio, tg_pre_rai, tsh_follow, tg_follow),
#                 ~ DescTools::Winsorize(., val = c(0.01, 0.99))))%>%
#        mutate(recurrence = as.numeric(as.character(recurrence)))


######################################
# MICE imputation test dataset
######################################

data_raw_test[vars_to_factor] <- lapply(data_raw_test[vars_to_factor], as.factor)
data_raw_test$recurrence <- as.numeric(as.character(data_raw_test$recurrence))
data_raw_test$number_ln_exc <- as.numeric(as.character(data_raw_test$number_ln_exc))
data_raw_test$number_posit_ln <- as.numeric(as.character(data_raw_test$number_posit_ln))

data_raw_test <- data_raw_test %>%
  mutate(across(cols_skewed, ~ DescTools::Winsorize(., val = c(0.01, 0.99))))%>%
  mutate(rai = ifelse(is.na(rai), names(which.max(table(rai))), as.character(rai))) %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, 0, .x))) %>%
  mutate(across(all_of(cols_skewed), ~ log(.)))

data_raw_test$rai <- as.factor(data_raw_test$rai)

gm <- mice(data_raw_test, m=10, seed=1)
# densityplot(gm)

data_v1 <- complete(gm, 1) #### DATASET 1 CON MUJLTIPLE IMPUTATION 

# stacked set:
data_v10 <- complete(gm,1)
for (i in 2:10) { #2:m
  data_v10  <- rbind(data_v10, complete(gm,i))}
data_v10$w <- 1/10 # 1/m  #### DATA CON MULTIPLE IMPUTATION X10 

data_v1 <- data_v1 %>%
  mutate(across(cols_skewed, ~exp(.))) %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))
  
write.csv(data_v1, path_save_mice1_test, row.names = FALSE)

data_v10 <- data_v10 %>%
  mutate(across(cols_skewed, ~exp(.))) %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))
write.csv(data_v10, path_save_mice10_test, row.names = FALSE)

for (i in 1:10) {
  df_i <- complete(gm, i)
  df_i <- df_i %>%
    mutate(across(cols_skewed, ~exp(.))) %>%
    mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))
  write.csv(df_i, paste0(path_save_mice_test, i, ".csv"), row.names = FALSE)
}


####
# #Check percentiles
# for (i in 1:5) {
#   cat("\nImputación:", i, "\n")
#   print(summary(complete(gm, i)[cont_vars]))
# }
# 
# #Winsorize
# data_raw_test <- data_raw_test %>%
#   mutate(across(c(tumor_size, ln_ratio, tg_pre_rai, tsh_follow, tg_follow),
#                 ~ DescTools::Winsorize(., val = c(0.01, 0.99))))%>%
#        mutate(recurrence = as.numeric(as.character(recurrence)))



# 
# # #################################################################################
# # # Survival curve according to follow time.
# # #################################################################################
# # 
# # S1 <- rms::Surv(as.numeric(data_model_sub$follow_months), 
# #            as.numeric(data_model_sub$recurrence))
# # # Fit non-parametric survival
# # fit <- rms::npsurv(S1 ~ 1)
# # 
# # # Plot with risk table and confidence band
# # rms::survplot(fit, n.risk = TRUE, conf = "band", 
# #               ylab = "No-Recurrence ThyCanc probability",
# #               xlab = "Follow-up Time (months)",
# #               col.fill = 2, col = 1) 
# 
# #Cox Regression
# # cph_with_xy <- function(...) {cph(..., x = TRUE, y = TRUE, surv = TRUE)}
# # 
# # fit <- fit.mult.impute(
# #   Surv(follow_months, recurrence) ~ age + thyrodectomy_approach + type_resection + 
# #     tumor_size + extra_thy_exten + multicentric + vascular_inv + number_ln_exc + 
# #     number_posit_ln + stage + rai + tsh_follow + tg_follow + subtype, 
# #   fitter = cph, 
# #   xtrans = gm, # This is your mice object
# #   data = data_raw
# # )
# 
# formula_lasso = (Surv(follow_months, recurrence) ~ sex + family_history +
#                    bmi + thyrodectomy_approach + type_resection + tumor_size + 
#                    extra_thy_exten + multicentric + multicent_bilat + vascular_inv + 
#                    perineural_inv + positive_ln + number_ln_exc + extranod_exten +
#                    hashimoto + tnm_t + tnm_n + tnm_m + stage + rai + tg_follow) # + subtype, 
# 
# formula_elastic = (Surv(follow_months, recurrence) ~ sex + age + radiotherapy + family_history +
#   thy_disease_preop + bmi + thyrodectomy_approach + type_resection + histology +
#   tumor_size + extra_thy_exten + multicentric + multicent_bilat + vascular_inv + 
#   perineural_inv + positive_ln + number_ln_exc + extranod_exten + hashimoto +
#   tnm_t + tnm_n + tnm_m + stage + rai + tsh_follow + tg_follow) # + subtype, 
# 
# 
# cph(formula_elastic, data = data_m10, weights = w)
# cph(formula_lasso, data = data_m10, weights = w)
# 
# cox01.m = fit.mult.impute(formula = formula_elastic,fitter = cph, xtrans = gm)
# 
# # fit <- lapply(1:10, function(i) {
# #   d <- complete(gm, i)
# #   d_win <- d %>%
# #     mutate(across(c("tumor_size", "ln_ratio", "tg_pre_rai", "tsh_follow", "tg_follow"),
# #                   ~ DescTools::Winsorize(., val = c(0.01, 0.99)))) %>%
# #     mutate(recurrence = as.numeric(as.character(recurrence)))
# #   
# #   coxph(Surv(follow_months, recurrence) ~ age + thyrodectomy_approach + type_resection +
# #           tumor_size + extra_thy_exten + multicentric + vascular_inv + number_ln_exc + 
# #           number_posit_ln + stage + rai + tsh_follow + tg_follow + subtype,
# #         data = d_win)
# # })
# # pooled <- pool(fit)
# # summary(pooled)
# 
# # Relative contribution of predictors
# plot(anova(fit), what = "proportion chisq",
#      main = "Relative contribution of predictors",
#      cex.names = 0.7, las = 2)
# 
# #Kaplan-Meier curve
# d_first <- complete(gm, 1) %>%
#   mutate(across(c("tumor_size", "ln_ratio", "tg_pre_rai", "tsh_follow", "tg_follow"),
#                 ~ DescTools::Winsorize(., val = c(0.01, 0.99)))) %>%
#   mutate(recurrence = as.numeric(as.character(recurrence)))
# 
# d_first$lp <- predict(fit, newdata = d_first, type = "lp")
# 
# d_first$risk_group <- cut2(d_first$lp, g = 4)
# levels(d_first$risk_group) <- 1:4   # etiquetas 1-4
# 
# fit_groups <- npsurv(Surv(follow_months, recurrence) ~ risk_group, data = d_first)
# 
# survplot(fit_groups,
#          n.risk = TRUE,
#          conf = "bars",
#          xlab = "Follow-up (months)",
#          ylab = "Recurrence-free survival",
#          label.curves = list(keys = c("Group 1 (lowest risk)", "Group 2", 
#                                       "Group 3", "Group 4 (highest risk)")),
#          col = 1:4, lwd = 2)
# 
# #Normogram
# dd <- datadist(complete(gm, 1))
# options(datadist = "dd")
# 
# surv_fn <- Survival(fit)
# 
# surv_3y <- function(lp) surv_fn(36, lp = lp)
# surv_5y <- function(lp) surv_fn(60, lp = lp)
# 
# nom <- nomogram(fit,
#                 fun = list(surv_3y, surv_5y),
#                 funlabel = c("3-year survival", "5-year survival"),
#                 lp = FALSE,
#                 maxscale = 100)
# 
# plot(nom, col.grid = gray(c(0.8, 0.95)),
#      main = "Nomogram for recurrence-free survival")
