library(lterdatasampler)
data("ntl_icecover")
ntl_icecover
unique(ntl_icecover$lakeid)
plot(ice_duration ~ year, data = ntl_icecover, col = lakeid)
unique(ntl_icecover$lakeid)
table(ntl_icecover$lakeid)

sum(is.na(ntl_icecover))
df = na.omit(ntl_icecover)
head(df)
df_lake_mendota = df[df$lakeid == "Lake Mendota", ]
df_lake_monona = df[df$lakeid == "Lake Monona", ]

mycol = c("#6a5acd", "#e64173")

# plot
plot(df_lake_mendota$year, df_lake_mendota$ice_duration, col=mycol[1], las=1, xlab="Year", ylab="Ice duration (days)", main="Ice duration over years")
points(df_lake_monona$year, df_lake_monona$ice_duration, col=mycol[2])
legend("topright", legend = c("Mendota", "Monona"), col = mycol, lty=1, bty="n")


# add monona

# add legend
legend("topright", legend = c("Mendota", "Monona"), col = c("red", "blue"), pch = 16)
# save in csv
# write.csv(df, "Data/df_icecover.csv", row.names = FALSE)




fit1 = lm(ice_duration ~ year, data = df)
summary(fit1)


# plot
plot(ice_duration ~ year, data = df)
abline(fit1, col = "red")

# now model with two intercepts
fit2 = lm(ice_duration ~ year + lakeid, data = df)
summary(fit2)

# plot with color code for obs and the two regression lines
plot(ice_duration ~ year, data = df, col = df$lakeid)
# add legend

abline(coef(fit2)[1], coef(fit2)[2], col = "red") # mendota
abline(coef(fit2)[1] + coef(fit2)[3], coef(fit2)[2], col = "blue")  # monona

# model with different intercepts and slopes
fit3 = lm(ice_duration ~ year * lakeid, data = df)
summary(fit3)

# plot with color code for obs and the regression lines
plot(ice_duration ~ year, data = df, col = df$lakeid)
abline(coef(fit3)[1], coef(fit3)[2], col = "red") # mendota
abline(coef(fit3)[1] + coef(fit3)[3], coef(fit3)[2] + coef(fit3)[4], col = "blue")  # monona

AIC(fit1, fit2, fit3)



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

