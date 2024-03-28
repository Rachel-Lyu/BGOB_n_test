library(JM)
library(nlme)
library(dplyr)
library(tidyverse)
library(magrittr)
results_df <- data.frame(Species = character(),
                         Coef = numeric(),
                         StdErr = numeric(),
                         z_value = numeric(),
                         Pr_z = numeric(),
                         stringsAsFactors = FALSE)

survival_data <- read.csv('ID_age_cavity.csv')
longitudinal_data <- read.csv('subject_species.csv')

species_list <- unique(longitudinal_data$species_name)
for(species in species_list){
  species_data <- longitudinal_data[longitudinal_data$species_name == species,]
  extended_data <- data.frame()
  for(id in unique(species_data$ID)){
    long_data <- species_data[species_data$ID == id, ]
    long_data <- long_data[!is.na(long_data$data), ]
    surv_data <- survival_data[survival_data$ID == id, ]
    if (long_data$time[1] == 0){
      for(i in 2:nrow(long_data)) {
        start_time <- long_data$time[i - 1]
        stop_time <- long_data$time[i]
        status <- ifelse(stop_time >= surv_data$time, surv_data$status, 0)
        data_val <- long_data$data[i]
        temp_data <- data.frame(ID = id, start_time = start_time, stop_time = stop_time, data = data_val, status = status)
        extended_data <- rbind(extended_data, temp_data)
      }
    } else {
      for(i in 1:nrow(long_data)) {
        start_time <- ifelse(i == 1, 0, long_data$time[i - 1])
        stop_time <- long_data$time[i]
        status <- ifelse(stop_time >= surv_data$time, surv_data$status, 0)
        data_val <- long_data$data[i]
        temp_data <- data.frame(ID = id, start_time = start_time, stop_time = stop_time, data = data_val, status = status)
        extended_data <- rbind(extended_data, temp_data)
      }
    }
  }
  
  cox_model <- coxph(Surv(start_time, stop_time, status) ~ data, data = extended_data)
  coef_table <- summary(cox_model)$coefficients
  temp_df <- data.frame(Species = species,
                        Coefficient = coef_table[, "coef"],
                        StdError = coef_table[, "se(coef)"],
                        zValue = coef_table[, "z"],
                        PValue = coef_table[, "Pr(>|z|)"])
  results_df <- rbind(results_df, temp_df)
}
results_df %<>%
  mutate(QValue = p.adjust(PValue, method = 'BH'))
write.csv(results_df, "coxph_data.csv", row.names = FALSE)
