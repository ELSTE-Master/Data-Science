# See README.md and ../../Data/cpx_README.md for attribution and scientific limitations.
# Run from any directory: Rscript /path/to/original_style/generate_figures.R
library(ggplot2)
library(svglite)
library(digest)
args <- commandArgs(FALSE)
script <- normalizePath(sub('^--file=', '', args[grepl('^--file=', args)]))
root <- dirname(script)
data_dir <- file.path(root, '../../Data')
# Use one interpreter throughout, including on systems without a python3 alias.
python_exec <- Sys.getenv('CPX_PYTHON', '')
if (!nzchar(python_exec)) {
 candidates <- Sys.which(c('python3','python'))
 candidates <- candidates[nzchar(candidates)]
 if (!length(candidates)) stop('Install Python 3 or set CPX_PYTHON to its executable.')
 python_exec <- unname(candidates[1])
}
stopifnot(digest(file=file.path(data_dir,'cpx_teaching.csv'), algo='sha256') ==
 '6a3ff5ebe8c6371a4f7d453c8380615c2640080ec813cfd9c59cb10454c9ab96')
# Verify the existing allocation, without rewriting it.
status <- system2(python_exec, c(shQuote(file.path(root,'../prepare_cpx_split.py')), '--verify'))
stopifnot(status == 0)
d <- read.csv(file.path(data_dir,'cpx_teaching.csv'), colClasses=c(Ref='character',Run='character'))
s <- read.csv(file.path(data_dir,'cpx_teaching_split.csv'), colClasses=c(Ref='character'))
stopifnot(identical(d$Ref,s$Ref), identical(s$row_id,seq_len(nrow(d))))
tr <- d[s$split=='train',]; ev <- d[s$split=='evaluation',]
stopifnot(nrow(tr)==1402,nrow(ev)==294)
simple <- lm(T_C ~ MgO_Cpx, tr)
multiple <- lm(T_C ~ MgO_Cpx + Al2O3_Cpx + TiO2_Cpx, tr)
pink <- '#eb078e'; purple <- '#6A5ACD'; turquoise <- '#20B2AA'; blue <- '#2b59c3'
# Lecture3's uncluttered themes + the dashed grid/serif axes in its reading plots.
theme_original <- theme_bw(base_size=19,base_family='Liberation Serif') + theme(
 panel.grid.minor=element_blank(), panel.grid.major=element_line(colour='grey86',linetype='dashed',linewidth=.35),
 panel.border=element_rect(colour='grey25',linewidth=.5),
 axis.text=element_text(colour='grey35'),axis.title=element_text(size=21,colour='grey35'),
 plot.title=element_text(size=22,face='bold',colour='grey35'),
 strip.background=element_blank(),strip.text=element_text(size=20),
 legend.position='bottom',legend.title=element_blank(),plot.margin=margin(12,16,12,12))
theme_set(theme_original)
save_plot <- function(name,p,w=9,h=5.5) {
 ggsave(file.path(root,'figures',paste0(name,'.svg')),p,width=w,height=h,bg='white')
 ggsave(file.path(root,'figures',paste0(name,'.png')),p,width=w,height=h,dpi=240,bg='white')
}
base <- ggplot(tr,aes(MgO_Cpx,T_C)) + geom_point(shape=21,colour=purple,fill=paste0(purple,'55'),size=2.3,stroke=.35) +
 labs(x='MgO (reported concentration units)',y='Experimental temperature (°C)')
save_plot('scatter',base)
# Fixed-width bins illustrate local averages, not a fitted alternative model.
tr$bin <- cut(tr$MgO_Cpx,breaks=seq(0,32,2),include.lowest=TRUE)
bins <- aggregate(cbind(MgO_Cpx,T_C)~bin,tr,mean)
counts <- table(tr$bin); bins <- bins[counts[as.character(bins$bin)]>=15,]
save_plot('conditional',base + geom_point(data=bins,size=4.5,shape=23,fill='#EB811B',colour='white',stroke=.7))
line <- geom_abline(intercept=coef(simple)[1],slope=coef(simple)[2],colour=pink,linewidth=1.2)
save_plot('linear',base + line)

# Classical pointwise intervals for the existing simple training regression.
# These working-model intervals do not establish coverage in other settings.
interval_grid <- data.frame(MgO_Cpx=seq(min(tr$MgO_Cpx),max(tr$MgO_Cpx),length.out=400))
mean_interval <- predict(simple,interval_grid,interval='confidence',level=.95)
new_interval <- predict(simple,interval_grid,interval='prediction',level=.95)
stopifnot(max(abs(mean_interval[,'fit']-new_interval[,'fit']))<1e-10,
 all(new_interval[,'lwr']<mean_interval[,'lwr']),
 all(new_interval[,'upr']>mean_interval[,'upr']))
interval_grid$fit <- mean_interval[,'fit']
interval_grid$CI_lower <- mean_interval[,'lwr']
interval_grid$CI_upper <- mean_interval[,'upr']
interval_grid$PI_lower <- new_interval[,'lwr']
interval_grid$PI_upper <- new_interval[,'upr']
write.csv(interval_grid,file.path(root,'simple_regression_intervals.csv'),row.names=FALSE)
prediction_mgo <- unname((1200-coef(simple)[1])/coef(simple)[2])
prediction_row <- data.frame(MgO_Cpx=prediction_mgo)
prediction_ci <- predict(simple,prediction_row,interval='confidence',level=.95)
prediction_pi <- predict(simple,prediction_row,interval='prediction',level=.95)
stopifnot(prediction_mgo>min(tr$MgO_Cpx),prediction_mgo<max(tr$MgO_Cpx),
 abs(prediction_ci[,'fit']-1200)<1e-10)
write.csv(data.frame(MgO_Cpx=prediction_mgo,fit=1200,
 CI_lower=prediction_ci[,'lwr'],CI_upper=prediction_ci[,'upr'],
 PI_lower=prediction_pi[,'lwr'],PI_upper=prediction_pi[,'upr']),
 file.path(root,'simple_regression_prediction_1200.csv'),row.names=FALSE)
save_plot('simple_regression_intervals',ggplot(interval_grid,aes(MgO_Cpx,fit)) +
 geom_ribbon(aes(ymin=PI_lower,ymax=PI_upper,fill='95% PI: new individual'),alpha=.22) +
 geom_ribbon(aes(ymin=CI_lower,ymax=CI_upper,fill='95% CI: mean response'),alpha=.6) +
 geom_point(data=tr,aes(y=T_C),colour=purple,alpha=.26,size=1.8) +
 geom_line(aes(y=CI_lower),colour=purple,linewidth=.45) +
 geom_line(aes(y=CI_upper),colour=purple,linewidth=.45) +
 geom_line(colour=pink,linewidth=1.1) +
 annotate('point',x=prediction_mgo,y=1200,shape=21,fill=pink,colour='white',size=4,stroke=.8) +
 annotate('label',x=prediction_mgo+.8,y=1270,label='1,200 °C',hjust=0,
  colour=pink,fill='white',size=5.5) +
 scale_fill_manual(values=c('95% CI: mean response'=purple,'95% PI: new individual'=turquoise),
  breaks=c('95% CI: mean response','95% PI: new individual')) +
 labs(x='MgO (reported concentration units)',y='Temperature (°C)') +
 theme(legend.text=element_text(size=18),legend.key.width=grid::unit(1.2,'cm')),
 w=9.5,h=4.8)
# Display 12 evenly spaced records in MgO rank; the fit always uses all training rows.
z <- tr[order(tr$MgO_Cpx),][round(seq(1,nrow(tr),length.out=12)),]
z$fit <- predict(simple,z)
save_plot('least_squares',ggplot(z,aes(MgO_Cpx,T_C)) + line +
 geom_segment(aes(xend=MgO_Cpx,yend=fit),colour=turquoise,linewidth=1.1) +
 geom_point(shape=21,fill=paste0(purple,'55'),colour=purple,size=3.7) +
 labs(x='MgO (reported concentration units)',y='Experimental temperature (°C)'))
x0 <- 15; y0 <- unname(coef(simple)[1]+coef(simple)[2]*x0); slope <- unname(coef(simple)[2])
save_plot('coefficient',ggplot(data.frame(x=c(14.5,16.6)),aes(x)) +
 stat_function(fun=function(x)coef(simple)[1]+coef(simple)[2]*x,colour=pink,linewidth=1.3) +
 annotate('segment',x=15,xend=16,y=y0,yend=y0,colour=blue,linewidth=1.2) +
 annotate('segment',x=16,xend=16,y=y0,yend=y0+slope,colour=turquoise,linewidth=1.2) +
 annotate('text',x=15.5,y=y0-8,label='+1 unit MgO',colour=blue,size=6) +
 annotate('text',x=16.05,y=y0+slope/2,label=sprintf('+%.2f °C',slope),hjust=0,colour='#167c77',size=6) +
 labs(x='MgO (reported concentration units)',y='Predicted temperature (°C)'),w=8,h=5)
# Conditional coefficient illustration: hold the other oxides at values from a
# training record near MgO=15. These are model comparisons, not interventions.
partial_anchor <- tr[which.min(abs(tr$MgO_Cpx-15)),]
partial_grid <- data.frame(MgO_Cpx=seq(14.5,16.6,length.out=100),
 Al2O3_Cpx=partial_anchor$Al2O3_Cpx,TiO2_Cpx=partial_anchor$TiO2_Cpx)
partial_grid$fit <- predict(multiple,partial_grid)
partial_steps <- data.frame(MgO_Cpx=c(15,16),
 Al2O3_Cpx=partial_anchor$Al2O3_Cpx,TiO2_Cpx=partial_anchor$TiO2_Cpx)
partial_y <- predict(multiple,partial_steps)
stopifnot(abs(diff(partial_y)-coef(multiple)['MgO_Cpx'])<1e-8)
save_plot('multiple_coefficient',ggplot(partial_grid,aes(MgO_Cpx,fit)) +
 geom_line(colour=pink,linewidth=1.3) +
 annotate('segment',x=15,xend=16,y=partial_y[1],yend=partial_y[1],colour=blue,linewidth=1.2) +
 annotate('segment',x=16,xend=16,y=partial_y[1],yend=partial_y[2],colour=turquoise,linewidth=1.2) +
 annotate('text',x=15.5,y=partial_y[1]-8,label='+1 unit MgO',colour=blue,size=6) +
 annotate('text',x=16.05,y=mean(partial_y),label=sprintf('+%.2f °C',diff(partial_y)),
 hjust=0,colour='#167c77',size=6) +
 labs(x='MgO (reported concentration units)',y='Predicted temperature (°C)',
 caption=sprintf('Held fixed: Al₂O₃ = %.2f; TiO₂ = %.2f (reported units)',
 partial_anchor$Al2O3_Cpx,partial_anchor$TiO2_Cpx)) +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=8,h=5.3)
fitted <- rbind(data.frame(Model='MgO',observed=tr$T_C,fit=fitted(simple)),
 data.frame(Model='MgO + Al₂O₃ + TiO₂',observed=tr$T_C,fit=fitted(multiple)))
save_plot('fit',ggplot(fitted,aes(fit,observed)) + geom_abline(slope=1,intercept=0,colour=pink,linewidth=.8) +
 geom_point(colour=purple,alpha=.35,size=1.7) + facet_wrap(~Model,nrow=1) + coord_equal() +
 labs(x='Fitted temperature (°C)',y='Observed temperature (°C)'),w=11,h=4.6)
fitted$residual <- fitted$observed-fitted$fit
save_plot('residuals',ggplot(fitted,aes(fit,residual)) + geom_hline(yintercept=0,colour=pink,linewidth=.9) +
 geom_point(colour=purple,alpha=.35,size=1.7) + facet_wrap(~Model,nrow=1) +
 labs(x='Fitted temperature (°C)',y='Training residual (°C)'),w=11,h=5)
# Training R-squared: compare fitted squared error with the training-mean baseline.
r2_simple <- summary(simple)$r.squared
r2_errors <- data.frame(Model=factor(c('Training mean','MgO regression'),
 levels=c('MgO regression','Training mean')), Relative_error=c(100,100*(1-r2_simple)))
save_plot('r_squared',ggplot(r2_errors,aes(Relative_error,Model,fill=Model)) +
 geom_col(width=.5) + geom_text(aes(label=sprintf('%.1f%%',Relative_error)),hjust=-.25,size=6,colour='grey30') +
 scale_fill_manual(values=c('Training mean'='grey70','MgO regression'=purple)) +
 scale_x_continuous(limits=c(0,120),breaks=c(0,25,50,75,100)) +
 labs(x='Squared error (% of mean-only baseline)',y=NULL) +
 theme(legend.position='none'),w=9,h=3.8)

# Pedagogical extension: functions, a training-only quadratic, and simulations.
# The MgO-only polynomials and simulations do not enter the final comparison.
function_grid <- data.frame(x=seq(-2,2,length.out=301))
function_curves <- rbind(data.frame(x=function_grid$x,y=function_grid$x,Function='f(x) = x'),
                        data.frame(x=function_grid$x,y=function_grid$x^2,Function='f(x) = x²'))
save_plot('functions',ggplot(function_curves,aes(x,y,colour=Function)) +
 geom_hline(yintercept=0,colour='grey75',linewidth=.4) +
 geom_vline(xintercept=0,colour='grey75',linewidth=.4) + geom_line(linewidth=1.3) +
 scale_colour_manual(values=c('f(x) = x'=purple,'f(x) = x²'=pink)) +
 labs(x='x',y='f(x)'),w=8,h=5)
quadratic <- lm(T_C ~ MgO_Cpx + I(MgO_Cpx^2),data=tr)
mg_grid <- data.frame(MgO_Cpx=seq(min(tr$MgO_Cpx),max(tr$MgO_Cpx),length.out=400))
curves <- rbind(data.frame(mg_grid,fit=predict(simple,mg_grid),Model='Linear'),
                data.frame(mg_grid,fit=predict(quadratic,mg_grid),Model='Quadratic'))
poly_scores <- data.frame(Model=c('Linear','Quadratic'),
 Training_RMSE=c(sqrt(mean(residuals(simple)^2)),sqrt(mean(residuals(quadratic)^2))))
write.csv(poly_scores,file.path(root,'training_polynomial_comparison.csv'),row.names=FALSE)
save_plot('polynomial',base + geom_line(data=curves,aes(y=fit,colour=Model),linewidth=1.15) +
 scale_colour_manual(values=c(Linear=pink,Quadratic=turquoise)) +
 labs(y='Temperature (°C)',caption=sprintf('Training RMSE: linear %.1f °C; quadratic %.1f °C',
                     poly_scores$Training_RMSE[1],poly_scores$Training_RMSE[2])) +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=9,h=4.6)
# A full quadratic in three oxide predictors: illustrative TRAINING fit only.
# Its additional frozen-test evaluation below was subsequently requested by the instructor.
quadratic_three <- lm(T_C ~ (MgO_Cpx + Al2O3_Cpx + TiO2_Cpx)^2 +
 I(MgO_Cpx^2) + I(Al2O3_Cpx^2) + I(TiO2_Cpx^2),data=tr)
stopifnot(length(coef(quadratic_three))==10,quadratic_three$rank==10)
complex_scores <- data.frame(Model=c('Additive','Full quadratic'),
 Coefficients=c(length(coef(multiple)),length(coef(quadratic_three))),
 Training_R2=c(summary(multiple)$r.squared,summary(quadratic_three)$r.squared),
 Training_RMSE=c(sqrt(mean(residuals(multiple)^2)),sqrt(mean(residuals(quadratic_three)^2))))
write.csv(complex_scores,file.path(root,'three_oxide_polynomial_training.csv'),row.names=FALSE)
write.csv(data.frame(Term=names(coef(quadratic_three)),Estimate=coef(quadratic_three)),
 file.path(root,'three_oxide_polynomial_coefficients.csv'),row.names=FALSE)
complex_fit <- rbind(data.frame(Model='Additive',observed=tr$T_C,fit=fitted(multiple)),
 data.frame(Model='Full quadratic',observed=tr$T_C,fit=fitted(quadratic_three)))
complex_fit$Model <- factor(complex_fit$Model,levels=c('Additive','Full quadratic'))
save_plot('three_oxide_polynomial',ggplot(complex_fit,aes(fit,observed)) +
 geom_abline(slope=1,intercept=0,colour=pink,linewidth=.8) +
 geom_point(colour=purple,alpha=.35,size=1.7) + facet_wrap(~Model,nrow=1) + coord_equal() +
 labs(x='Fitted temperature (°C)',y='Observed temperature (°C)',
 caption=sprintf('Training R²: additive %.3f; quadratic %.3f',complex_scores$Training_R2[1],complex_scores$Training_R2[2])) +
 theme(strip.text=element_text(size=17),plot.caption=element_text(size=17,hjust=0)),w=11,h=4.4)
print(complex_scores)

# Revisit residual diagnostics after the full quadratic model's fitted summary.
# Same training observations and common axes; no test predictions are involved.
complex_residuals <- transform(complex_fit,residual=observed-fit)
stopifnot(nrow(complex_residuals)==2*nrow(tr),
 all(is.finite(complex_residuals$residual)))
save_plot('quadratic_residuals',ggplot(complex_residuals,aes(fit,residual)) +
 geom_hline(yintercept=0,colour=pink,linewidth=.9) +
 geom_point(colour=purple,alpha=.35,size=1.7) +
 facet_wrap(~Model,nrow=1) +
 labs(x='Fitted temperature (°C)',y='Training residual (°C)'),w=11,h=5)

# Ordinary training R-squared for four nested models, including the full quadratic.
# Same response/rows and an intercept throughout; no evaluation rows are used.
r2_models <- data.frame(Model=c('Baseline','Simple','Multiple','Quadratic'),
 Coefficients=c(1,2,4,10),R2=c(0,summary(simple)$r.squared,summary(multiple)$r.squared,
 summary(quadratic_three)$r.squared))
stopifnot(nobs(simple)==nrow(tr),nobs(multiple)==nrow(tr),nobs(quadratic_three)==nrow(tr),
 all(diff(r2_models$R2)>=0),
 abs(summary(lm(T_C ~ 1,data=tr))$r.squared)<1e-12)
write.csv(r2_models,file.path(root,'r_squared_models.csv'),row.names=FALSE)
r2_models$Label <- factor(r2_models$Model,levels=c('Quadratic','Multiple','Simple','Baseline'),
 labels=c('Three-oxide quadratic','MgO + Al₂O₃ + TiO₂','MgO','Mean-only baseline'))
save_plot('r_squared_models',ggplot(r2_models,aes(R2,Label,fill=Model)) +
 geom_col(width=.5) +
 geom_point(data=subset(r2_models,Model=='Baseline'),colour='grey55',size=3) +
 geom_text(aes(label=sprintf('%.3f',R2)),hjust=-.25,size=6,colour='grey30') +
 scale_fill_manual(values=c(Baseline='grey70',Simple=purple,Multiple=turquoise,Quadratic=pink)) +
 scale_x_continuous(limits=c(0,1),breaks=c(0,.25,.5,.75,1)) +
 labs(x='Training R² (ordinary)',y=NULL,
 caption='Same 1,402 training observations; each model includes an intercept') +
 theme(legend.position='none',plot.caption=element_text(size=16,colour='grey35',hjust=0)),
 w=10,h=5.2)

# A degree-10 MgO polynomial, fitted on training only with a stable basis.
# Orthogonal polynomial terms span the same model as powers of rescaled MgO.
mg_poly10 <- lm(T_C ~ poly(MgO_Cpx,10),data=tr)
stopifnot(length(coef(mg_poly10))==11,mg_poly10$rank==11)
mg10_grid <- data.frame(MgO_Cpx=seq(min(tr$MgO_Cpx),max(tr$MgO_Cpx),length.out=1200))
mg10_curves <- rbind(data.frame(mg10_grid,fit=predict(quadratic,mg10_grid),Model='Degree 2'),
 data.frame(mg10_grid,fit=predict(mg_poly10,mg10_grid),Model='Degree 10'))
mg10_scores <- data.frame(Model=c('Degree 2','Degree 10'),Coefficients=c(3,11),
 Training_R2=c(summary(quadratic)$r.squared,summary(mg_poly10)$r.squared),
 Training_RMSE=c(sqrt(mean(residuals(quadratic)^2)),sqrt(mean(residuals(mg_poly10)^2))))
write.csv(mg10_scores,file.path(root,'mgo_degree10_training.csv'),row.names=FALSE)
write.csv(data.frame(mg10_grid,fit=predict(mg_poly10,mg10_grid)),
 file.path(root,'mgo_degree10_curve.csv'),row.names=FALSE)
save_plot('mgo_degree10',base +
 geom_line(data=mg10_curves,aes(y=fit,colour=Model),linewidth=1.2) +
 scale_colour_manual(values=c('Degree 2'=turquoise,'Degree 10'=pink)) +
 labs(y='Temperature (°C)',caption=sprintf('Training RMSE: degree 2 %.1f °C; degree 10 %.1f °C',
 mg10_scores$Training_RMSE[1],mg10_scores$Training_RMSE[2])) +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=9,h=5)

# A fixed, transparent toy example; unrelated to CPX temperatures or chemistry.
set.seed(2026)
sim <- data.frame(x=seq(-1,1,length.out=18))
truth <- function(x) 1+2*x-2*x^2
sim$y <- truth(sim$x)+rnorm(nrow(sim),sd=.3)
sim_grid <- data.frame(x=seq(-1,1,length.out=500))
sim_test <- data.frame(x=runif(2000,-1,1));sim_test$y <- truth(sim_test$x)+rnorm(2000,sd=.3)
degrees <- c(1,2,10)
sim_models <- vector('list',length(degrees))
# Freeze each formula's degree in its prediction environment (lm stores the call).
for(j in seq_along(degrees)) {
 k_fixed <- degrees[j]
 sim_models[[j]] <- lm(as.formula(sprintf('y ~ poly(x, %d)',k_fixed)),data=sim)
}
panel_names <- c('Too simple · degree 1','Reasonably flexible · degree 2','Follows noise · degree 10')
sim_lines <- do.call(rbind,lapply(seq_along(degrees),function(j)
 data.frame(sim_grid,fit=predict(sim_models[[j]],sim_grid),Model=panel_names[j])))
sim_points <- do.call(rbind,lapply(panel_names,function(name)data.frame(sim,Model=name)))
sim_truth <- do.call(rbind,lapply(panel_names,function(name)data.frame(sim_grid,y=truth(sim_grid$x),Model=name)))
for(obj in c('sim_lines','sim_points','sim_truth')) {
 value <- get(obj); value$Model <- factor(value$Model,levels=panel_names);assign(obj,value)
}
sim_scores <- data.frame(Degree=degrees,
 Training_RMSE=sapply(sim_models,function(model)sqrt(mean(residuals(model)^2))),
 Independent_simulated_RMSE=sapply(sim_models,function(model)sqrt(mean((sim_test$y-predict(model,sim_test))^2))))
write.csv(sim_scores,file.path(root,'simulation_metrics.csv'),row.names=FALSE)
save_plot('overfitting',ggplot(sim_points,aes(x,y)) +
 geom_line(data=sim_truth,colour='grey55',linetype='dashed',linewidth=.7) +
 geom_point(colour=purple,size=2.3) +
 geom_line(data=sim_lines,aes(y=fit),colour=pink,linewidth=1) +
 facet_wrap(~Model,nrow=1) +
 labs(x='Simulated predictor',y='Simulated response',
      caption='Simulated data · pink: fitted model · dashed grey: generating mean') +
 theme(strip.text=element_text(size=16),plot.caption=element_text(size=17,hjust=0)),w=12,h=4.4)
# A schematic, not calculated CPX or simulation errors and not a selection rule.
t <- seq(0,1,length.out=300)
complexity <- rbind(data.frame(Complexity=t,Error=.4*exp(-4*t)+.05,Curve='Training'),
                    data.frame(Complexity=t,Error=.7*(t-.45)^2+.38,Curve='Prediction'))
save_plot('complexity',ggplot(complexity,aes(Complexity,Error,colour=Curve)) + geom_line(linewidth=1.4) +
 scale_colour_manual(values=c(Training=purple,Prediction=pink)) +
 scale_x_continuous(breaks=c(0,1),labels=c('Lower','Higher')) +
 labs(x='Model complexity',y='Error (arbitrary units)'),w=9,h=5)

# Empirical TiO2 complexity illustration: fixed validation subset WITHIN training.
# Preserve linked source groups internally, and never consult the final test rows.
# The hash rule and 20% group fraction are fixed before inspecting these errors.
training_groups <- s$validation_group[s$split=='train']
group_keys <- unique(training_groups)
validation_salt <- 'ELSTE-Class4-polynomial-validation-v1'
group_hashes <- vapply(group_keys,function(g)
 digest(paste0(validation_salt,'\n',g),algo='sha256',serialize=FALSE),character(1))
validation_keys <- group_keys[order(group_hashes)][seq_len(ceiling(.2*length(group_keys)))]
is_validation <- training_groups %in% validation_keys
fit_part <- tr[!is_validation,]; validation_part <- tr[is_validation,]
stopifnot(nrow(fit_part)+nrow(validation_part)==1402,
 !any(training_groups[!is_validation] %in% training_groups[is_validation]))
fit_lo <- min(fit_part$TiO2_Cpx); fit_hi <- max(fit_part$TiO2_Cpx)
fit_part$z <- 2*(fit_part$TiO2_Cpx-fit_lo)/(fit_hi-fit_lo)-1
validation_part$z <- 2*(validation_part$TiO2_Cpx-fit_lo)/(fit_hi-fit_lo)-1
stopifnot(all(validation_part$z>=-1 & validation_part$z<=1))
# Degree 0 is the intercept-only baseline; higher degrees use the same rows.
degree_errors <- do.call(rbind,lapply(0:10,function(degree) {
 model <- if(degree==0) lm(T_C ~ 1,data=fit_part) else
  lm(as.formula(sprintf('T_C ~ poly(z,%d)',degree)),data=fit_part)
 stopifnot(model$rank==degree+1)
 data.frame(Degree=degree,Training_RMSE=sqrt(mean(residuals(model)^2)),
  Validation_RMSE=sqrt(mean((validation_part$T_C-predict(model,validation_part))^2)))
}))
stopifnot(all(diff(degree_errors$Training_RMSE)<=1e-8),
 all(is.finite(as.matrix(degree_errors))))
write.csv(degree_errors,file.path(root,'tio2_complexity_errors.csv'),row.names=FALSE)
write.csv(data.frame(row_id=s$row_id[s$split=='train'],validation_group=training_groups,
 Use=ifelse(is_validation,'validation','fitting')),
 file.path(root,'polynomial_complexity_validation_split.csv'),row.names=FALSE)
degree_long <- rbind(data.frame(Degree=degree_errors$Degree,RMSE=degree_errors$Training_RMSE,
  Curve='Training'),data.frame(Degree=degree_errors$Degree,RMSE=degree_errors$Validation_RMSE,
  Curve='Validation prediction'))
degree_long$Curve <- factor(degree_long$Curve,levels=c('Training','Validation prediction'))
# Display degrees 0–9; retain the complete computed table through degree 10.
save_plot('tio2_complexity_errors',ggplot(subset(degree_long,Degree<=9),aes(Degree,RMSE,colour=Curve)) +
 geom_line(linewidth=1.2) + geom_point(size=3) +
 scale_colour_manual(values=c(Training=purple,'Validation prediction'=pink)) +
 scale_x_continuous(breaks=0:9) +
 scale_y_continuous(limits=c(100,300),breaks=seq(100,300,50)) +
 labs(x='Polynomial degree (TiO₂ only)',y='RMSE (°C)',
  caption='Actual CPX data · fixed validation subset; final test unused') +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0)),w=9,h=5.3)
print(poly_scores);print(sim_scores)

# Only these evaluation assets are used in the concluding comparison slide.
scores <- do.call(rbind,lapply(c('Training','Evaluation'),function(part) {
 frame <- if(part=='Training') tr else ev
 preds <- list(Baseline=rep(mean(tr$T_C),nrow(frame)),Simple=predict(simple,frame),Multiple=predict(multiple,frame))
 do.call(rbind,lapply(names(preds),function(n) {
 e <- frame$T_C-preds[[n]]
 data.frame(Partition=part,Model=n,RMSE=sqrt(mean(e^2)),MAE=mean(abs(e)),
 Publication_RMSE=sqrt(mean(tapply(e^2,frame$Ref,mean))),Publication_MAE=mean(tapply(abs(e),frame$Ref,mean)))
 }))
}))
write.csv(scores,file.path(root,'scores.csv'),row.names=FALSE)
write.csv(data.frame(Model=c('Simple','Multiple'),R2=c(summary(simple)$r.squared,summary(multiple)$r.squared)),file.path(root,'fit_summary.csv'),row.names=FALSE)
write.csv(rbind(data.frame(Model='Simple',Term=names(coef(simple)),Estimate=coef(simple)),data.frame(Model='Multiple',Term=names(coef(multiple)),Estimate=coef(multiple))),file.path(root,'coefficients.csv'),row.names=FALSE)
eval <- scores[scores$Partition=='Evaluation',]
# Instructor-requested evaluation of the already-defined three-oxide quadratic.
# Fit remains training-only; retain the original three-model score file intact.
quadratic_error <- ev$T_C-predict(quadratic_three,ev)
quadratic_eval <- data.frame(Partition='Evaluation',Model='Quadratic',
 RMSE=sqrt(mean(quadratic_error^2)),MAE=mean(abs(quadratic_error)),
 Publication_RMSE=sqrt(mean(tapply(quadratic_error^2,ev$Ref,mean))),
 Publication_MAE=mean(tapply(abs(quadratic_error),ev$Ref,mean)))
write.csv(quadratic_eval,file.path(root,'quadratic_evaluation.csv'),row.names=FALSE)
eval <- rbind(eval,quadratic_eval)
write.csv(eval,file.path(root,'evaluation_models.csv'),row.names=FALSE)
# Display the same frozen split; the regression line is fitted on training only.
train_test <- ggplot(tr,aes(MgO_Cpx,T_C)) +
 geom_point(aes(colour='Training',shape='Training'),alpha=.35,size=2.3) +
 geom_abline(intercept=coef(simple)[1],slope=coef(simple)[2],colour=blue,linewidth=1) +
 geom_point(data=ev,aes(colour='Test',shape='Test'),alpha=.75,size=2.7) +
 scale_colour_manual(name=NULL,values=c(Training=purple,Test=pink),breaks=c('Training','Test')) +
 scale_shape_manual(name=NULL,values=c(Training=16,Test=17),breaks=c('Training','Test')) +
 guides(colour=guide_legend(override.aes=list(alpha=1,size=3))) +
 labs(x='MgO (reported concentration units)',y='Temperature (°C)',
      caption='Blue line: MgO model fitted on training observations') +
 theme(plot.caption=element_text(size=16,colour='grey35',hjust=0))
save_plot('train_test',train_test,w=9,h=5.2)
# Four-model test comparison; the original three sets of values are unchanged.
error_long <- rbind(data.frame(Model=eval$Model,Metric='RMSE',Error=eval$RMSE),
                    data.frame(Model=eval$Model,Metric='MAE',Error=eval$MAE))
error_long$Model <- factor(error_long$Model,levels=c('Quadratic','Multiple','Simple','Baseline'),
 labels=c('Three-oxide quadratic','MgO + Al₂O₃ + TiO₂','MgO','Training-mean baseline'))
save_plot('evaluation',ggplot(error_long,aes(Error,Model,colour=Metric)) +
 geom_point(size=4.5,position=position_dodge(width=.45)) +
 geom_text(aes(label=sprintf('%.1f',Error)),position=position_dodge(width=.45),hjust=-.5,size=5.5,show.legend=FALSE) +
 scale_colour_manual(values=c(MAE=turquoise,RMSE=blue)) +
 scale_x_continuous(limits=c(0,190)) + labs(x='Test prediction error (°C)',y=NULL),w=10,h=5.2)
capture.output(sessionInfo(),file=file.path(root,'R-session.txt'))
print(scores);print(coef(simple));print(coef(multiple));print(summary(simple)$r.squared);print(summary(multiple)$r.squared)
source(file.path(root,'generate_selection_figures.R'),local=TRUE)
