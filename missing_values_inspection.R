# https://www.youtube.com/watch?v=2tSQvxFArj4
library(VIM)
require(Hmisc)
require(data.table)
require(qreport)  # Define dataChk, missChk, maketabs, ...
library(dplyr)

library(DescTools)
library (rms)


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
mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, 0, .x)))  %>%
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
data_raw$recurrence <- as.numeric(as.character(data_raw$recurrence))

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

data_m1 <- data_m1 %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, NA, .x)))  %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, NA, .x)))
# write.csv(data_m1,"./database/data_mice1.csv", row.names = FALSE)

data_m10 <- data_m10 %>%
  mutate(across(all_of(cols_nodes_excision), ~ if_else(number_ln_exc == 0, NA, .x)))  %>%
  mutate(across(all_of(cols_rai), ~ if_else(rai == 0, NA, .x)))

# write.csv(data_m10,"./database/data_mice10.csv", row.names = FALSE)

for (i in 1:10) {
  df_i <- complete(gm, i)
  write.csv(df_i, paste0("./database/imputation_mice/data_imp_", i, ".csv"), row.names = FALSE)
}

# #Check percentiles
# for (i in 1:5) {
#   cat("\nImputación:", i, "\n")
#   print(summary(complete(gm, i)[cont_vars]))
# }
# 
# #Winsorize
# data_raw <- data_raw %>%
#   mutate(across(c(tumor_size, ln_ratio, tg_pre_rai, tsh_follow, tg_follow),
#                 ~ DescTools::Winsorize(., val = c(0.01, 0.99))))%>%
#        mutate(recurrence = as.numeric(as.character(recurrence)))

#Cox Regression
cph_with_xy <- function(...) {cph(..., x = TRUE, y = TRUE, surv = TRUE)}

fit <- fit.mult.impute(
  Surv(follow_months, recurrence) ~ age + thyrodectomy_approach + type_resection +
    tumor_size + extra_thy_exten + multicentric + vascular_inv + number_ln_exc + 
    number_posit_ln + stage + rai + tsh_follow + tg_follow + subtype,
  fitter = cph_with_xy,
  xtrans = gm,
  data = data_raw
)
# fit <- lapply(1:10, function(i) {
#   d <- complete(gm, i)
#   d_win <- d %>%
#     mutate(across(c("tumor_size", "ln_ratio", "tg_pre_rai", "tsh_follow", "tg_follow"),
#                   ~ DescTools::Winsorize(., val = c(0.01, 0.99)))) %>%
#     mutate(recurrence = as.numeric(as.character(recurrence)))
#   
#   coxph(Surv(follow_months, recurrence) ~ age + thyrodectomy_approach + type_resection +
#           tumor_size + extra_thy_exten + multicentric + vascular_inv + number_ln_exc + 
#           number_posit_ln + stage + rai + tsh_follow + tg_follow + subtype,
#         data = d_win)
# })
# pooled <- pool(fit)
# summary(pooled)

# Relative contribution of predictors
plot(anova(fit), what = "proportion chisq",
     main = "Relative contribution of predictors",
     cex.names = 0.7, las = 2)

#Kaplan-Meier curve
d_first <- complete(gm, 1) %>%
  mutate(across(c("tumor_size", "ln_ratio", "tg_pre_rai", "tsh_follow", "tg_follow"),
                ~ DescTools::Winsorize(., val = c(0.01, 0.99)))) %>%
  mutate(recurrence = as.numeric(as.character(recurrence)))

d_first$lp <- predict(fit, newdata = d_first, type = "lp")

d_first$risk_group <- cut2(d_first$lp, g = 4)
levels(d_first$risk_group) <- 1:4   # etiquetas 1-4

fit_groups <- npsurv(Surv(follow_months, recurrence) ~ risk_group, data = d_first)

survplot(fit_groups,
         n.risk = TRUE,
         conf = "bars",
         xlab = "Follow-up (months)",
         ylab = "Recurrence-free survival",
         label.curves = list(keys = c("Group 1 (lowest risk)", "Group 2", 
                                      "Group 3", "Group 4 (highest risk)")),
         col = 1:4, lwd = 2)

#Normogram
dd <- datadist(complete(gm, 1))
options(datadist = "dd")

surv_fn <- Survival(fit)

surv_3y <- function(lp) surv_fn(36, lp = lp)
surv_5y <- function(lp) surv_fn(60, lp = lp)

nom <- nomogram(fit,
                fun = list(surv_3y, surv_5y),
                funlabel = c("3-year survival", "5-year survival"),
                lp = FALSE,
                maxscale = 100)

plot(nom, col.grid = gray(c(0.8, 0.95)),
     main = "Nomogram for recurrence-free survival")
