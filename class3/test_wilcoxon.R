library(idarps) 
data(diet)

# Compute weight loss
diet$weight.loss = diet$initial.weight - diet$final.weight

# Variables of interest
dietA = diet$weight.loss[diet$diet.type=="A"]
dietC = diet$weight.loss[diet$diet.type=="C"]

# Perfom test
wilcox.test(dietA, dietC, alternative = "less", mu=-1)


########################################################################3
# Import data
library(idarps) 
data(diet)

# Compute weight loss
diet$weight.loss = diet$initial.weight - diet$final.weight

# Variables of interest
dietA = diet$weight.loss[diet$diet.type=="A"]
dietC = diet$weight.loss[diet$diet.type=="C"]

# Perfom test
wilcox.test(dietA, dietC, alternative = "less", mu = -0.95)