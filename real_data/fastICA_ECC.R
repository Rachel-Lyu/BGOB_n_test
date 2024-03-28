## fastICA + hclust
library(fastICA)
library(reshape2)
library(sparcl)
ColorDendrogram_axes <- function(hc, y, main = "", branchlength = 0.7, labels = NULL, 
                             xlab = NULL, sub = NULL, ylab = "", cex.main = NULL) {
  if (is.null(labels)) 
    labels <- rep("", length(y))
  plot(hc, hang = 0.2, main = main, labels = labels, xlab = xlab, 
       sub = sub, ylab = ylab, cex.main = cex.main, axes = FALSE) # Suppress the axes
  y <- y[hc$ord]
  if (is.numeric(y)) {
    y <- y + 1
    y[y == 7] <- "orange"
  }
  for (i in 1:length(hc$ord)) {
    o = hc$merge[, 1] == -hc$ord[i] | hc$merge[, 2] == -hc$ord[i]
    segments(i, hc$he[o] - branchlength, i, hc$he[o], col = y[i])
  }
}

ECC <- read.csv("ECC_group_80_before_truncate.csv")
idx <- which(ECC$species_name == "Other")
ECC = ECC[-idx, ]
pre_ICA = NULL
sbjName = unique(ECC$subject)
for (subject in sbjName) {
  subject_idx = which(ECC$subject == subject)
  df = data.frame(species = ECC$species[subject_idx],
                  time = ECC$time[subject_idx],
                  data = ECC$data[subject_idx])
  df_dcast = dcast(df, species ~ time, value.var = "data")[, -1]
  if (is.null(pre_ICA)){
    pre_ICA = df_dcast
  }else{
    pre_ICA = cbind(pre_ICA, df_dcast)
  }
}

ICA_part <- fastICA(pre_ICA, n.comp=15, alg.typ = "parallel", fun = "exp", alpha = 1.0,
                    method = "C", row.norm = FALSE, maxit = 5000, tol = 1e-03, verbose = TRUE)
ICA_df = ICA_part$S
all_species_nm = unique(ECC$species_name)
row.names(ICA_df) = all_species_nm

## hclust
pdf("fastICA_ECC.pdf", width=20, height=10)
dist = dist(ICA_df)
hc = hclust(dist, method  = "ward.D",  members=NULL)
y=cutree(hc, 2)
ColorDendrogram_axes(hc, y, labels=names(y), main="ECC Data", 
                branchlength = 0.7, xlab = "", sub = "", ylab = "", cex.main = 2)
dev.off()
