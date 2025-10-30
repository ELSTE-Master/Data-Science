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








###############################
library(lterdatasampler)

index1 = and_vertebrates$year > 2018
index2 = and_vertebrates$species == "Coastal giant salamander"
index3 = and_vertebrates$section == "CC"
index4 = and_vertebrates$section == "OG"

X = and_vertebrates[index1 & index2 & index3,]
Y = and_vertebrates[index1 & index2 & index4,]

boxplot(X$weight_g, Y$weight_g)
t.test(X$weight_g, Y$weight_g)
wilcox.test(X$weight_g, Y$weight_g)