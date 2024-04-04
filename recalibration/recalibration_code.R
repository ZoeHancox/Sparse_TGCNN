library(ggplot2)
library(caret)
library(Hmisc)
library(boot)
library(predtools)
library(moments)
library(dplyr)
library(meta)
library(pROC)
library(rms)
library(PRROC)

setwd('./')

col_names = c('Accuracy', '95% CI', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Prevalence', 'Detection Rate', 'Detection Prevalence', 'Balanced Accuracy', 'O/E mean', 'O/E 95% CI', 'O/E SD',	'C-slope mean',	'C-slope 95% CI',	'C-slope SD', 'CITL mean',	'CITL 95% CI',	'CITL SD',	'AUROC mean', 'AUROC 95% CI',	'AUROC SD', 'AUPRC mean',	'AUPRC SD',	'AUPRC CI')
empty_df <- data.frame(matrix(nrow = 0, ncol = length(col_names)))
colnames(empty_df) <- col_names

compute_metrics <- function(df, linpreds, outcomes, bootstrapnum, test_type) {
  print(c('N =', nrow(df)))
  
  print('Linear predictor distribution')
  print(c('Mean =', mean(linpreds)))
  print(c('Var =', var(linpreds)))
  print(c('Skew =', skewness(linpreds)))
  print(c('Kurtosis =', kurtosis(linpreds)))
  print(c('Median =', median(linpreds)))
  print(c('IQR =', IQR(linpreds)))
  
  # OE ratio
  probs <- 1/(1+exp(-linpreds))
  oe <- sum(outcomes) / sum(probs)
  #print(probs)
  pred_vals <- ifelse(probs > 0.5, 1, 0)
  #print(pred_vals)
  #print(outcomes)
  
  pred_vals <- factor(pred_vals, levels = c(0, 1))
  actual_outcomes <- factor(outcomes, levels = c(0, 1))
  conf_matrix <- confusionMatrix(pred_vals, actual_outcomes, positive='1')
  print(conf_matrix)
  
  # Calibration slope
  mtemp <- glm(outcomes ~ linpreds, family="binomial")
  mcslope <- as.numeric(mtemp$coefficients[2])
  
  # CITL
  mtemp <- glm(outcomes ~ offset(linpreds), family="binomial")
  citl <- as.numeric(mtemp$coefficients[1])
  
  # C-statistic
  preds <- 1/(1+exp(-linpreds))
  #auc <- mAUC(mROC(preds))
  auc <- auc(roc(outcomes, preds, plot=TRUE))#, direction="<"))
  
  
  # Print original data results
  print('Metrics on original data')
  print(c('O/E = ', oe))
  print(c('C-slope =', mcslope))
  print(c('CITL =', citl))
  print(c('AUC =', auc))
  
  
  # bootstrap
  allidx <- seq(nrow(df))
  oelist <- c()
  cslopelist <- c()
  citllist <- c()
  auclist <- c()
  auprc_list <- c()
  
  for (i in seq(bootstrapnum)) {
    myidx <- sample(seq(nrow(df)), nrow(df), replace=TRUE)
    myoutcomes <- outcomes[myidx]
    mylinpreds <- linpreds[myidx]
    
    # OE ratio
    myprobs <- 1/(1+exp(-mylinpreds))
    oe <- sum(myoutcomes) / sum(myprobs)
    oelist <- c(oelist, oe)
    
    # Calibration slope
    mtemp <- glm(myoutcomes ~ mylinpreds, family="binomial")
    mcslope <- as.numeric(mtemp$coefficients[2])
    cslopelist <- c(cslopelist, mcslope)
    
    # CITL
    mtemp <- glm(myoutcomes ~ offset(mylinpreds), family="binomial")
    citl <- as.numeric(mtemp$coefficients[1])
    citllist <- c(citllist, citl)
    
    # C-statistic
    preds <- 1/(1+exp(-mylinpreds))
    auc <- auc(roc(myoutcomes, mylinpreds))
    auclist <- c(auclist, auc)
    
    # AUPRC
    pr_values <- pr.curve(scores.class0 = mylinpreds, weights.class0 = myoutcomes)
    auprc <- pr_values$auc.integral
    auprc_list <- c(auprc_list, auprc)
    
  }
  fac = 1/sqrt(bootstrapnum)
  print(c('Metrics - bootstrap n =', bootstrapnum))
  L <- mean(oelist) - 1.96*sd(oelist)*fac
  U <- mean(oelist) + 1.96*sd(oelist)*fac
  
  print(c('OE mean =', mean(oelist), 'SD =', sd(oelist), 'L =', L, 'U =', U))
  oe_CI <- paste(c(L,U))
  print(oe_CI)
  L <- mean(cslopelist) - 1.96*sd(cslopelist)*fac
  U <- mean(cslopelist) + 1.96*sd(cslopelist)*fac
  print(c('C-slope mean =', mean(cslopelist), 'SD =', sd(cslopelist), 'L =', L, 'U =', U))
  cslope_CI <- c(L,U)
  print(cslope_CI)
  L <- mean(citllist) - 1.96*sd(citllist)*fac
  U <- mean(citllist) + 1.96*sd(citllist)*fac
  print(c('CITL mean =', mean(citllist), 'SD =', sd(citllist), 'L =', L, 'U =', U))
  CITL_CI <- c(L,U)
  print(CITL_CI)
  L <- mean(auclist) - 1.96*sd(auclist)*fac
  U <- mean(auclist) + 1.96*sd(auclist)*fac
  print(c('AUC mean =', mean(auclist), 'SD =', sd(auclist), 'L =', L, 'U =', U))
  AUROC_CI <- c(L,U)
  print(AUROC_CI)
  
  mean_auprc <- mean(auprc_list)
  sd_auprc <- sd(auprc_list)
  fac <- 1.96  # 95% confidence interval
  L <- mean_auprc - fac * sd_auprc
  U <- mean_auprc + fac * sd_auprc
  cat("AUPRC mean =", mean_auprc, "SD =", sd_auprc, "L =", L, "U =", U, "\n")
  AUPRC_CI <- c(L,U)
  print(AUPRC_CI)
  
  
  list_for_results <<- list(conf_matrix$overall['Accuracy'], conf_matrix$table['95% CI'], conf_matrix$byClass['Sensitivity'], conf_matrix$byClass['Specificity'], conf_matrix$byClass['Pos Pred Value'], conf_matrix$byClass['Neg Pred Value'], conf_matrix$byClass['Prevalence'], conf_matrix$byClass['Detection Rate'], conf_matrix$byClass['Detection Prevalence'], conf_matrix$byClass['Balanced Accuracy'], mean(oelist), list(oe_CI), sd(oelist), mean(cslopelist), list(cslope_CI), sd(cslopelist), mean(citllist), list(CITL_CI), sd(citllist), mean(auclist), list(AUROC_CI), sd(auclist), mean(auprc_list), list(AUPRC_CI), sd(auprc_list))
  #print(list_for_results)
  new_row_df <- as.data.frame(t(unlist(list_for_results, recursive=FALSE)))
  
  colnames(new_row_df) <- colnames(empty_df)
  
  
  
  if (test_type=='before'){
  results_df_before <<- rbind(empty_df, new_row_df)
  } else if (test_type=='after'){results_df_after <<- rbind(empty_df, new_row_df)
  } else {results_df_unseen <<- rbind(empty_df, new_row_df)
  }
}


# Probability distplot and calibration curves and decision curves
mainplots <- function (df, linpreds, outcomes, plotprefix) {
  lpmean <- mean(linpreds)
  lpmedian <- median(linpreds)
  lpsd <- sd(linpreds)
  lpiqr <- IQR(linpreds)
  
  preds <- 1/(1+exp(-linpreds))
  y <- outcomes
  
  # Linear predictor distribution
  legendstring <- paste('Mean = ', format(lpmean, digits=3), '\nSD = ', format(lpsd, digits=3),
                        '\nMedian = ', format(lpmedian, digits=3), '\nIQR = ', format(lpiqr, digits=3))
  png(filename = paste(plotprefix, '_linpreddist.png', sep=''))
  hist(linpreds, prob=TRUE, breaks=20, xlab='Linear Predictor', ylim=c(0,1), main='')
  text(-1, 0.9, legendstring)
  dev.off()
  
  # Probability distribution
  png(filename = paste(plotprefix, '_probdist.png', sep=''))
  hist(preds, prob=TRUE, breaks=20, xlab='Predicted probability', xlim=c(0, 1), ylim=c(0, 30), main='')
  dev.off()
  
  # Calibration curve
  mval <- data.frame(preds, y)
  png(filename = paste(plotprefix, '_calibrationcurve.png', sep=''))
  calplot <- calibration_plot(data=mval, obs="y", pred="preds", nTiles=100, x_lim=c(0,1), y_lim=c(0,1), title='', data_summary = TRUE)
  xplot <- calplot$data_summary$predRate
  yplot <- calplot$data_summary$obsRate
  calplot <- calibration_plot(data=mval, obs="y", pred="preds", nTiles=100, x_lim=c(0,1), y_lim=c(0,1), title='', data_summary = TRUE)
  xdots <- calplot$data_summary$predRate
  ydots <- calplot$data_summary$obsRate
  yupper <- calplot$data_summary$obsRate_UCL
  ylower <- calplot$data_summary$obsRate_LCL
  plot(c(-1), c(-1), xlim=c(0,1), ylim=c(0,1), xlab='Expected', ylab='Observed')
  plot(xdots, ydots, pch=1, col='darkgreen', xlim=c(0,1), ylim=c(0,1), xlab='Expected', ylab='Observed')
  arrows(x0=xdots, y0=ylower, x1=xdots, y1=yupper, length=0.02, code=3, angle=90, col="darkgreen")
  m <- loess(yplot~xplot)
  lines(m$x, m$fitted, col='deepskyblue3')
  lines(c(0,1), c(0,1), lty='dashed', col='darkblue')
  dev.off()
  
  # Create deciles based on predicted probabilities
  mval$decile <- cut(mval$preds, breaks = quantile(mval$preds, seq(0, 1, by = 0.1), na.rm = TRUE), labels = FALSE)
  
  # Initialise data frame to store results
  result_table <- data.frame(Decile = numeric(0), Observed_Probability = numeric(0), Predicted_Probability = numeric(0))
  
  # Loop through each decile
  for (i in unique(mval$decile)) {
    # Subset data for the current decile
    subset_data <- subset(mval, decile == i)
    
    observed_prob <- sum(subset_data$y) / nrow(subset_data)
    predicted_prob <- mean(subset_data$preds)
    
    result_table <- rbind(result_table, data.frame(Decile = i, Observed_Probability = observed_prob, Predicted_Probability = predicted_prob))
  }
  
  # Order the result table by the Decile column
  result_table <- result_table[order(result_table$Decile), ]
  
  # print(result_table)
  write.csv(result_table, file = paste(plotprefix, '_cal_decile.csv', sep=''), row.names = FALSE)
  
  
}

#model_name <- 'hip_1999_to_one_year_advance_full_model1'
# model_name <- 'RF_demo_only'
model_name <- readline(prompt = "Enter the name of the model: ")
include_subgroup <- readline(prompt = "Do you want to perform subgroup analysis with this model?")
#print(model_name)

# Read in the logits and outcomes
df <- read.csv(paste('logits_and_outcome_csvs/', 'logits_', model_name,'_holdout_1.csv', sep=''))
#head(df)

# Before recal
compute_metrics(df, df$logit, df$outcome, 10, 'before')
mainplots(df, df$logit, df$outcome, paste('plots/before', model_name, sep='_'))

# Recal
m <- glm(outcome ~ logit, data=df, family='binomial')
summary(m)

# if we want to predict on the model and get the new logits or probabilities

newlogit <- predict(m, type='link')
newprobs <- predict(m, type='response')

compute_metrics(df, newlogit, df$outcome, 10, 'after')
mainplots(df, newlogit, df$outcome, paste('plots/after', model_name, sep='_'))



#Test the recalibrated model on a completely unseen dataset

# Load the unseen/test set 2 dataset
unseen_df <- read.csv(paste('logits_and_outcome_csvs/', 'logits_', model_name,'_holdout_2.csv', sep=''))

# Predict using the recalibrated model
unseen_logit <- predict(m, newdata = unseen_df, type = 'link')

# Evaluate performance on unseen data
compute_metrics(unseen_df, unseen_logit, unseen_df$outcome, 10, 'unseen')
mainplots(unseen_df, unseen_logit, unseen_df$outcome, paste('plots/unseen_data', model_name, sep='_'))

# Add the new logits to the unseen dataframe
unseen_df$lpnew <- unseen_logit

if (include_subgroup=='y'){
  bootstrapnum = 10
  
  
  # NEED TO GET THE NEW MODEL AFTER RECALIBRATION TO GET NEW LINEAR PREDICTORS (lp)
  
  # # ORIGINAL DATA
  # print('ORIGINAL DATA')
  # compute_metrics(unseen_df, unseen_df$lpnew, unseen_df$outcome, bootstrapnum, 'unseen')
  # mainplots(unseen_df, unseen_df$lpnew, unseen_df$outcome, paste('plots/subgroups/orig', model_name, sep='_'))
  
  # SEX SPLIT
  print('SEX SPLIT - MALE')
  curdf <- unseen_df[unseen_df$sex == 1,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/sex_male', model_name, sep='_'))
  
  print('SEX SPLIT - FEMALE')
  curdf <- unseen_df[unseen_df$sex == 0,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/sex_female', model_name, sep='_'))
  
  
  # IMD GROUP
  curdf <- unseen_df[unseen_df$imd_quint == 1,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/imd_quint1', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$imd_quint == 2,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/imd_quint2', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$imd_quint == 3,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/imd_quint3', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$imd_quint == 4,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/imd_quint4', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$imd_quint == 5,]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/imd_quint5', model_name, sep='_'))
  
  # AGE AT HIP REPLACEMENT GROUP
  
  curdf <- unseen_df[unseen_df$age_at_label >= 40 & unseen_df$age_at_label <= 60, ]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/age_40_to_60', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$age_at_label > 60 & unseen_df$age_at_label <= 70, ]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/age_60_to_70', model_name, sep='_'))
  
  curdf <- unseen_df[unseen_df$age_at_label > 70, ]
  compute_metrics(curdf, curdf$lpnew, curdf$outcome, bootstrapnum, 'unseen')
  mainplots(curdf, curdf$lpnew, curdf$outcome, paste('plots/subgroups/age_above_70', model_name, sep='_'))
  
}



