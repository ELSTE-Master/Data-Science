install.packages("lterdatasampler")
library(lterdatasampler)
index1 = and_vertebrates$year > 2018
index2 = and_vertebrates$species == "Coastal giant salamander"
index3 = and_vertebrates$section == "CC"
index4 = and_vertebrates$section == "OG"

X = and_vertebrates[index1 & index2 & index3,]
Y = and_vertebrates[index1 & index2 & index4,]


library(dplyr)
df = bind_rows(
  X,
  Y
) %>% select(section, species, weight_g)
head(df)
write.csv(df, "Data/salamander_weights.csv")


vec_CC = df %>% filter(section == "CC") %>% pull(weight_g)
vec_OG = df %>% filter(section == "OG") %>% pull(weight_g)
boxplot(vec_CC, vec_OG)
