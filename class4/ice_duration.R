library(lterdatasampler)
data("ntl_icecover")
ntl_icecover
plot(ice_duration ~ year, data = ntl_icecover, col = lakeid)
unique(ntl_icecover$lakeid)
table(ntl_icecover$lakeid)
# use ggplot2 do a plot of ice_duration vs year colored by lakeid
library(ggplot2)
ggplot(ntl_icecover, aes(x = year, y = ice_duration, color = lakeid)) +
  geom_line() +
  labs(title = "Ice Duration vs Year by Lake ID",
       x = "Year",
       y = "Ice Duration (days)") +
  theme_minimal()


# proche du deuxieme exercice sur la regression
# tu leur fait essayer different model et comparer
# un intercept pour chaque lac
# pente pour chaque lac
# est ce que ca baisse dans le temps de maniere globale
# comment interpreter lintercepete
# est ce qu on peut conclure a une difference entre les lacs
# l epaisseur de glace au temps zero 
# quantite de glace en lan du depart de letude
# preidction de glace en lan avec calcul en chiffre negatif
# prend linteraction difference de pente entre les deux, tester linteraction si significativement differente de zero
# en fait les donnes sont dependante, est ce que cest un pbroelme
# 
# )



mod = lm(ice_duration ~ year*lakeid, data = ntl_icecover)
summary(mod)
acf(mod$resid)

