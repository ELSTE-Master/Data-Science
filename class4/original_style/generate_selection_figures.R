# Sourced by generate_figures.R; reuse its theme, palette and training models.

# Count candidate subsets; illustrate the search space without fitting or selecting.
subset_counts <- data.frame(Terms=0:9,Subsets=choose(9,0:9))
stopifnot(sum(subset_counts$Subsets)==2^9)
write.csv(subset_counts,file.path(root,'candidate_subset_counts.csv'),row.names=FALSE)
save_plot('candidate_subsets',ggplot(subset_counts,aes(Terms,Subsets)) +
 geom_col(fill=purple,width=.7) +
 geom_text(aes(label=Subsets),vjust=-.5,size=5.5,colour='grey35') +
 scale_x_continuous(breaks=0:9) +
 scale_y_continuous(limits=c(0,145),breaks=c(0,50,100)) +
 labs(x='Number of included terms',y='Number of subsets',
 caption='Unrestricted subsets · intercept always included') +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=7.8,h=4.5)

selection_python <- python_exec
stopifnot(system2(selection_python,shQuote(file.path(root,'prepare_selection.py')))==0)
stopifnot(system2(python_exec,shQuote(file.path(root,'generate_confounding_svg.py')))==0)
aic_data <- read.csv(file.path(root,'aic_scores.csv'),check.names=FALSE)
cv_data <- read.csv(file.path(root,'cv_scores.csv'),check.names=FALSE)
cv_detail <- read.csv(file.path(root,'cv_fold_scores.csv'),check.names=FALSE)
cv_manifest <- read.csv(file.path(root,'model_selection_cv_folds.csv'),colClasses=c(validation_group='character'))
selection_models <- list(MgO=simple,'3 oxides'=multiple,Quadratic=quadratic_three)
stopifnot(identical(aic_data$Model,names(selection_models)))
# Match statsmodels' OLS coefficient-count convention (R's AIC(lm) adds a common 2).
r_aic <- vapply(selection_models,function(m) -2*as.numeric(logLik(m))+2*length(coef(m)),numeric(1))
stopifnot(max(abs(r_aic-aic_data$AIC))<1e-8)
fold_id <- cv_manifest$Fold[match(s$row_id[s$split=='train'],cv_manifest$row_id)]
stopifnot(!anyNA(fold_id),length(fold_id)==1402,
 all(vapply(split(fold_id,s$validation_group[s$split=='train']),function(v)length(unique(v))==1,logical(1))))
r_cv <- vapply(selection_models,function(m) {
 err <- rep(NA_real_,nrow(tr))
 for(f in 0:4) {
  fitting <- tr[fold_id!=f,]; validation <- tr[fold_id==f,]
  refit <- lm(formula(m),data=fitting)
  err[fold_id==f] <- validation$T_C-predict(refit,validation)
 }
 stopifnot(all(is.finite(err)))
 sqrt(mean(err^2))
},numeric(1))
stopifnot(max(abs(r_cv-cv_data$CV_RMSE_C))<1e-8)
order_models <- c('Quadratic','3 oxides','MgO')
selection_colours <- c(MgO=purple,'3 oxides'=turquoise,Quadratic=pink)
for(obj in c('aic_data','cv_data','cv_detail')) {
 value <- get(obj); value$Model <- factor(value$Model,levels=order_models);assign(obj,value)
}
save_plot('aic_penalty',ggplot(aic_data,aes(2*Coefficients,Model,fill=Model)) +
 geom_col(width=.5) + geom_text(aes(label=paste0('2 × ',Coefficients,' = ',2*Coefficients)),
  hjust=-.15,size=6,colour='grey30') + scale_fill_manual(values=selection_colours) +
 scale_x_continuous(limits=c(0,30),breaks=c(0,10,20,30)) +
 labs(x='Complexity penalty (2k)',y=NULL) + theme(legend.position='none'),w=8,h=4.8)
save_plot('aic_comparison',ggplot(aic_data,aes(Delta_AIC,Model,fill=Model)) +
 geom_col(width=.5) + geom_point(data=subset(aic_data,Delta_AIC<1e-8),size=3,colour=pink) +
 geom_text(aes(label=sprintf('%.1f',Delta_AIC)),hjust=-.2,size=6,colour='grey30') +
 scale_fill_manual(values=selection_colours) + scale_x_continuous(limits=c(0,1050),breaks=c(0,400,800)) +
 labs(x='ΔAIC (lower is preferred)',y=NULL) + theme(legend.position='none'),w=7,h=4.8)

fold_grid <- expand.grid(Part=1:5,Round=1:5)
fold_grid$Use <- ifelse(fold_grid$Part==fold_grid$Round,'Validate','Fit')
save_plot('cv_rotation',ggplot(fold_grid,aes(Part,6-Round,fill=Use)) +
 geom_tile(width=.94,height=.85,colour='white',linewidth=1) +
 geom_text(aes(label=Use,colour=Use),size=5.5,show.legend=FALSE) +
 annotate('rect',xmin=6,xmax=7.7,ymin=.55,ymax=5.45,fill='grey96',colour='grey75') +
 annotate('text',x=6.85,y=3,label='Test set\nkept aside',colour='grey40',size=6) +
 scale_fill_manual(values=c(Fit='#e4e0f6',Validate=pink)) +
 scale_colour_manual(values=c(Fit=purple,Validate='white')) +
 scale_x_continuous(breaks=1:5,labels=paste('Part',1:5),limits=c(.45,7.8),position='top') +
 scale_y_continuous(breaks=1:5,labels=paste('Round',5:1),limits=c(.45,5.55)) +
 labs(x=NULL,y=NULL) + theme(legend.position='none',panel.border=element_blank(),
 panel.grid=element_blank(),axis.ticks=element_blank(),axis.text=element_text(size=19)),w=11,h=5.3)
save_plot('cv_comparison',ggplot(cv_detail,aes(RMSE_C,Model,colour=Model)) +
 geom_point(size=2.6,alpha=.35) +
 geom_point(data=cv_data,aes(x=CV_RMSE_C),size=4.5) +
 geom_text(data=cv_data,aes(x=CV_RMSE_C,label=sprintf('%.1f',CV_RMSE_C)),
  nudge_y=.28,size=6,show.legend=FALSE) + scale_colour_manual(values=selection_colours) +
 scale_x_continuous(limits=c(60,145),breaks=c(75,100,125)) +
 labs(x='Cross-validated RMSE (°C)',y=NULL,
 caption='Small points: five validation folds\nLarge points: pooled CV RMSE') +
 theme(legend.position='none',plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=7,h=5.3)

# A structural teaching simulation: weather drives both variables, with no
# causal arrow from ice-cream sales to shark incidents. Not observed records.
set.seed(3642)
confounding <- data.frame(Weather=factor(rep(c('Cooler','Mild','Warmer'),each=40),
 levels=c('Cooler','Mild','Warmer')))
weather_id <- as.integer(confounding$Weather)
confounding$Ice_cream <- 20*weather_id+rnorm(120,sd=7)
confounding$Shark_incidents <- rpois(120,lambda=c(1,3,6)[weather_id])
write.csv(confounding,file.path(root,'confounding_simulated.csv'),row.names=FALSE)
save_plot('icecream_sharks',ggplot(confounding,aes(Ice_cream,Shark_incidents)) +
 geom_point(shape=21,colour=purple,fill=paste0(purple,'55'),size=3,stroke=.4) +
 geom_smooth(method='lm',formula=y~x,se=FALSE,colour=pink,linewidth=1.2) +
 scale_y_continuous(breaks=seq(0,10,2)) +
 labs(x='Ice-cream sales (arbitrary units)',y='Shark incidents (simulated count)',
 caption='Simulated teaching example · not observed shark records') +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=8.5,h=5.2)

# Separate, explicitly simulated seasonal illustration, not empirical shark data.
# A Northern Hemisphere summer peak is a teaching premise, not a universal law.
seasonal_mean <- function(month) 1+9*exp(-.5*((month-7)/1.8)^2)
set.seed(3643)
seasonal <- data.frame(Month=1:12)
seasonal$Shark_incidents <- rpois(12,lambda=seasonal_mean(seasonal$Month))
stopifnot(all(seasonal$Shark_incidents>=0),
 seasonal$Month[which.max(seasonal$Shark_incidents)] %in% 6:8)
write.csv(seasonal,file.path(root,'sharks_seasonal_simulated.csv'),row.names=FALSE)
seasonal_grid <- data.frame(Month=seq(1,12,length.out=400))
seasonal_grid$Mean <- seasonal_mean(seasonal_grid$Month)
seasonal_top <- max(12,max(seasonal$Shark_incidents)+2)
save_plot('sharks_seasonal',ggplot(seasonal,aes(Month,Shark_incidents)) +
 annotate('rect',xmin=5.5,xmax=8.5,ymin=-Inf,ymax=Inf,
  fill=turquoise,alpha=.09) +
 annotate('text',x=7,y=seasonal_top-.4,label='Summer',colour='#167c77',size=6) +
 geom_line(data=seasonal_grid,aes(y=Mean),colour=pink,linewidth=1.2) +
 geom_point(shape=21,colour=purple,fill=purple,size=3.5,stroke=.4) +
 scale_x_continuous(breaks=1:12,labels=month.abb,expand=expansion(mult=.03)) +
 scale_y_continuous(breaks=seq(0,seasonal_top,2),limits=c(0,seasonal_top)) +
 labs(x='Month (Northern Hemisphere)',y='Shark incidents (simulated count)',
 caption='Simulated teaching example · not observed shark records') +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=10,h=5.8)
