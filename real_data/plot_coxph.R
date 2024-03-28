library(ggplot2)
library(tidyr)
library(ggrepel)
library(dplyr)
data <- read.csv("coxph_data.csv")
real <- read.csv("coxph_real.csv")
data <- merge(data, real, by = "Species", all = TRUE)
data$neg_log_QValue_data <- -log10(data$QValue.x)
data$neg_log_QValue_real <- -log10(data$QValue.y)

selected_species <- data %>% filter(QValue.x < 0.001 | QValue.y < 0.001) %>% pull(Species)
data$group <- NA

# Assign groups based on conditions
# data$group[data$QValue.x < 0.05 & data$QValue.y < 0.05] <- "A"
# data$group[data$QValue.x < 0.05 & data$QValue.y >= 0.05] <- "B"
# data$group[data$QValue.x >= 0.05 & data$QValue.y < 0.05] <- "C"
# data$group[data$QValue.x >= 0.05 & data$QValue.y >= 0.05] <- "D"
data$group[data$Coefficient.x < 0 & data$Coefficient.y < 0] <- "Both Negative"
data$group[data$Coefficient.x < 0 & data$Coefficient.y >= 0] <- "Original Postive; Interpolated Negative"
data$group[data$Coefficient.x >= 0 & data$Coefficient.y < 0] <- "Original Negative; Interpolated Postive"
data$group[data$Coefficient.x >= 0 & data$Coefficient.y >= 0] <- "Both Positive"

library(ggplot2)
library(ggrepel)

# Your ggplot code
p <- ggplot(data, aes(x=neg_log_QValue_real, y=neg_log_QValue_data, color=group)) +
  geom_point() +
  geom_abline(intercept=0, slope=1, color="red") +
  geom_hline(yintercept=-log10(0.05), linetype="dashed", color="blue") +
  geom_vline(xintercept=-log10(0.05), linetype="dashed", color="blue") +
  geom_text_repel(data=subset(data, Species %in% selected_species), 
                  aes(label=Species), size=3) +
  annotate("text", x=Inf, y=7.5, label="significant for both", 
           hjust=1, vjust=1, color="black") +
  annotate("text", x=-Inf, y=7.5, label="significant for interpolated data only", 
           hjust=0, vjust=1, color="black") +
  annotate("text", x=Inf, y=0, label="significant for original data only", 
           hjust=1, vjust=0, color="black") +
  theme_minimal() +
  xlab("-log10(p value) - using original data") +
  ylab("-log10(p value) - using interpolated data") +
  xlim(0, 2.5) +
  ylim(0, 7.5) +
  # theme(legend.position = "none")
  labs(color = "Sign of coefs")

# Use ggsave to save your plot
# ggsave("coxph_data.png", plot = p, width = 10, height = 8, dpi = 300)
ggsave("coxph_sign.png", plot = p, width = 10, height = 8, dpi = 300)
