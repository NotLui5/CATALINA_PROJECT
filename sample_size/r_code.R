# Calc sample size 
install.packages("pmsampsize")

# https://cran.r-project.org/web/packages/pmvalsampsize/pmvalsampsize.pdf
# https://cran.r-project.org/web/packages/pmsampsize/pmsampsize.pdf
library(pmsampsize) 
## Size to train model # results: 291 -> 1004 sample size
# parameters - 13 - 18 - 31 - 38
# Pak, et al. c-index 0.910
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.91, parameters = 13)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.91, parameters = 18)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.91, parameters = 31)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.91, parameters = 38)

# Borzooei, et al. AUC 0.99
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.99, parameters = 13)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.99, parameters = 18)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.99, parameters = 31)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.99, parameters = 38)

# Aida, et al. Prevalence = 0.20 , AUC 1.00 best model ELM
# pmsampsize(type = "b", prevalence = 0.25, cstatistic = 1.00, parameters = 13)
# pmsampsize(type = "b", prevalence = 0.25, cstatistic = 1.00, parameters = 18)
# pmsampsize(type = "b", prevalence = 0.25, cstatistic = 1.00, parameters = 31)
# pmsampsize(type = "b", prevalence = 0.25, cstatistic = 1.00, parameters = 38)

# Lee, et al. Prevalence = 0.08 - 0.28 , AUROC 0.9622
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.9622, parameters = 13)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.9622, parameters = 18)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.9622, parameters = 31)
pmsampsize(type = "b", prevalence = 0.25, cstatistic = 0.9622, parameters = 38)
