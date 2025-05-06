library(dplyr)
library(tibble)

cd <- read.csv("Libro de codigos - Hoja 1.csv")
df <- read.csv("CaTaLiNA_DATA_2025-03-20_1704.csv")
str(cd)
length(cd)
summary(cd)
codes <- cd
codes$codigo_id <- gsub("[", "", sub("]", "",cd$codigo_id,fixed=T), fixed = T)
codes$codigo_id <- gsub(" ", "", codes$codigo_id)
codes <- codes[!apply(codes == "",1,all),]

variables_list <- read.table("codes.txt", sep = ",", header = F, dec = ".")



df %>% 
  rename_with(~deframe(codes)[.x], .cols = codes$codigo_id) %>% 
  select(codigo_id, redcap_event_name, any_of(codes$CODIGO.ID))
