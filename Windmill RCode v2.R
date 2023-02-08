#Project Windmill power prediction
#install.packages("htmltools")


# detach(package:readr)          # Data Read & Manipulation
# detach(package:dplyr)          # Data Manipulation
# detach(package:superml)        # Label Encoding
# detach(package:imputeTS)       # Mean Imputing
# detach(package:outliers)       # Remove Outliers
# detach(package:funModeling)    # Analysing Numeric Features
# detach(package:naniar)         # Analysis Missing Values
# detach(package:ggplot2)        # Correlation Matrix 
# #detach(package:GGally)         # Correlation Matrix 
# #detach(package:MASS)           # Forward Backward Selection
# detach(package:h2o)            # AutoML
# detach(package:htmltools)      # HTML/JS Embed in R Markdown


install.packages("devtools")
library(devtools)
devtools::install_github('araastat/reprtree')
library(reprtree)

#Importing Required Libraries
library(readr)          # Data Read & Manipulation
library(dplyr)          # Data Manipulation
library(superml)        # Label Encoding
library(imputeTS)       # Mean Imputing
library(outliers)       # Remove Outliers
library(funModeling)    # Analysing Numeric Features
library(naniar)         # Analysis Missing Values
library(ggplot2)        # Correlation Matrix
library(GGally)         # Correlation Matrix
library(caret)
library(relaimpo)
library(randomForest)
library(gbm)
library(rpart)
library(rpart.plot) #for plotting decision trees
library(car)
library(hrbrthemes)
library(modelr)
#install.packages("reptree")
#ibrary(reprtree)
library(ggRandomForests)
#library(MASS)           # Forward Backward Selection
# library(h2o)            # AutoML
# library(htmltools)      # HTML/JS Embed in R Markdown
#options(warn=-1)

# Importing the data
data = read_csv("E:/Aakarshan/Project Windies/Windmilla Data.csv")
head(data)
data=data.frame(data)
str(data)
View(data)

# Changing Column Names 
rename = function(df, name){
  for(i in 1:ncol(df))
    colnames(df)[i] = name[i]
  return(df)
}

names = c("tracking_id","datetime","wind_speed","atmospheric_temperature","shaft_temperature","blades_angle","gearbox_temperature",
          "engine_temperature","motor_torque","generator_temperature","atmospheric_pressure","area_temperature",
          "windmill_body_temperature","wind_direction","resistance","rotor_torque","turbine_status","cloud_level",
          "blade_length","blade_breadth","windmill_height","windmill_generated_power")

data = rename(data, names)

splitting_date_time = function(df)
{
  df$Date = as.Date(df$datetime, format = "%d-%m-%Y")
  datetxt = as.Date(df$Date)
  df_date = data.frame(year = as.numeric(format(datetxt, format = "%Y")),
                       month = as.numeric(format(datetxt, format = "%m")),
                       day = as.numeric(format(datetxt, format = "%d")))
  df = cbind(df_date, df)
  return(df)
}

data = splitting_date_time(data)

#--------------------------------------------------------------Data Analysis-------------------------------------------------------

#Select all Y rows except the rows that have NA
data = data[!(is.na(data$windmill_generated_power)),]

data = dplyr::select(data, -tracking_id, -datetime, -Date)

options(repr.plot.width = 20, repr.plot.height = 12)

head(data)

str(data)

data$turbine_status = as.factor(data$turbine_status)
data$cloud_level = as.factor(data$cloud_level)

#-------------------------------------------------------Plot percentage of missing data-------------------------------------------
gg_miss_var(data, show_pct = TRUE)

#-------------------------------------------------------Variables missing values--------------------------------------------------
gg_miss_upset(data)

#-------------------------------------------------------Monthwise Generated Power w.r.t Day---------------------------------------
#Monthwise Generated Power w.r.t Day
options(repr.plot.width = 20, repr.plot.height = 18)

# na.omit(data) %>%
#   ggplot(aes(x = month, y = windmill_generated_power)) +
#   geom_point(color="blue") +
#   labs(title = "Monthwise Generated Power",
#        y = "Power Genereraed(Kw/h)",
#        x = "Days") + theme_bw(base_size = 8) + facet_wrap(~day)

#Display the power generated in across each month 
na.omit(data) %>%
  ggplot(aes(x = day, y = windmill_generated_power)) +
  geom_point(color="blue") +
  labs(title = "Monthwise Generated Power",
       y = "Power Genereraed(Kw/h)",
       x = "Month") + theme_bw(base_size = 8) + facet_wrap(~month)

#Analysing all the Numeric Features
#Check variance of each numeric variable
plot_num(data[,c(-1,-2,-3)])

#Analysing all the Categorical Features
options(repr.plot.width = 20, repr.plot.height = 8)

freq(data$turbine_status)

freq(data$cloud_level)

head(data)

#----------------------------------------------------------------------Data Cleaning-----------------------------------------------

# Data Imputation on Categorical & Numeric Features
imputer = function(df){
  df$cloud_level[is.na(df$cloud_level)] = "Low"
  df$turbine_status[is.na(df$turbine_status)] = "BB"
  df$cloud_level = as.numeric(factor(df$cloud_level, levels = c("Extremely Low", "Low", "Medium")))
  label = LabelEncoder$new()
  df$turbine_status = label$fit_transform(df$turbine_status)
  df = na_mean(df)
  return(df)
}

imputed_data = imputer(data)

# Box Plot Before Removing Outliers
dev.off()

#windows.options(width = 20, height = 10, reset = TRUE)

#dev.new(width=20, height=10, unit="in")

#par(mai = c(1, 1, 1, 1))

par(mfrow = c(1,5))


for (i in c(4:8)) {
  boxplot(imputed_data[,i], main=names(imputed_data[i]), type="l")
}

for (i in c(10:14)) {
  boxplot(imputed_data[,i], main=names(imputed_data[i]), type="l")
}

for (i in c(15:19)) {
  boxplot(imputed_data[,i], main=names(imputed_data[i]), type="l")
}

for (i in c(20:23)) {
  boxplot(imputed_data[,i], main=names(imputed_data[i]), type="l")
}

head(imputed_data)

#Remove year column
year_data = imputed_data[,1]
unique(year_data)

#Removing Outliers
outlier_data = rm.outlier(imputed_data[,-1], fill = TRUE, median = TRUE)

final_data = cbind(year_data,outlier_data)

#Check for NA
foundna = "no"
for(i in 1:ncol(final_data)){
  # print(names(final_data)[i])
  for(j in 1:nrow(final_data))
  {
    #print(names(outlier_data[i]))
    if(is.na(final_data[j,i]))
    { foundna = "yes"
    print(names(final_data[i]))
    print(paste("j=",j,"i=",i))
    }
  }
}
foundna

head(final_data)
View(final_data)


#--------------------------------------------------------------------Check Multicollinearity----------------------------------------
LR.model = lm(final_data$windmill_generated_power~., data = final_data)
summary(LR.model)
vif(LR.model)
vif = data.frame(vif(LR.model))
vif

#motor_torque and generator_temperature have vif>5
df=data.frame(names(final_data[-22]),vif$vif.LR.model.)
names(df)[1] = "Features"
names(df)[2] = "VIFValues"

#par("mar")

dev.off()

par(mar = c(6, 12, 2, 4) + 0.1) 

barplot(names.arg = df$Features, 
        height =df$VIFValues, 
        main = "VIF Values",
        xlab = "VIF Values",
        cex.names = 0.8,
        cex.axis  = 0.8,
        #ylab = "Features",
        col = ifelse(df$VIFValues>5,"red","blue"),
        las=1,
        width=2,
        horiz = TRUE)
        
#Run lin regression without motor torque
LR.model1 = lm(final_data$windmill_generated_power~., data = final_data[,-10])
summary(LR.model1)
vif(LR.model1)

#Run lin regression without generator_temperature
LR.model2 = lm(final_data$windmill_generated_power~., data = final_data[,-11])
summary(LR.model2)
vif(LR.model2)

#create a new variable by multiplying motor_torque and generator temperature
data_new=cbind(final_data,mottor_gentemp=final_data$motor_torque*final_data$generator_temperature)
head(data_new)

#Run Regressioin by adding new variable
LR.model4 = lm(final_data$windmill_generated_power~., data = data_new)
summary(LR.model4)
vif(LR.model4)

#run regression with the new variable and by removing motor torque
LR.model5 = lm(final_data$windmill_generated_power~., data = data_new[-10])
summary(LR.model5)
vif(LR.model5)

#run regression with the new variable and by removing generator temperature
LR.model6 = lm(final_data$windmill_generated_power~., data = data_new[-11])
summary(LR.model6)
vif(LR.model6)



#Choose either model1 (without motor torque) or model2 (without generator temperature)
corpowertorque = round(cor(final_data$windmill_generated_power,final_data$motor_torque),2)
corpowertorque=paste("Correlation: ",corpowertorque)
corpowertorque

corrpowertemp = round(cor(final_data$windmill_generated_power,final_data$generator_temperature),2)
corrpowertemp=paste("Correlation: ",corrpowertemp)
corrpowertemp


par(mfrow = c(2,1))


ggplot(final_data, aes(x=final_data$motor_torque, y=final_data$windmill_generated_power,color=motor_torque)) + 
geom_point(size = 2,alpha = 0.5) +
  theme_classic() +
  xlab("motor torque")+
  ylab("Windmill generated power") +
  stat_smooth(method = "lm",
              col = "red", se = FALSE, size = 1) +
  ggtitle(corpowertorque)



ggplot(final_data, aes(x=final_data$generator_temperature, y=final_data$windmill_generated_power, color= generator_temperature)) + 
  geom_point(size = 2, alpha = 0.5 ) +
  theme_classic() +
  xlab("Generator Temperature") +
  ylab("Windmill generated power") +
  stat_smooth(method = "lm",
              col = "red", se = FALSE, size = 1) +
  ggtitle(corrpowertemp)


#scatterplot(final_data$windmill_generated_power,y=final_data$motor_torque)
#scatterplot(final_data$windmill_generated_power,y=final_data$generator_temperature)

par(mfrow = c(5,5))
#correlation between windmill_generated_power and motor_torque is higher, hence retaining motor torque
final_data = final_data[,-11]
head(final_data)

LR.modelfinal = lm(final_data$windmill_generated_power~., data = final_data)
summary(LR.modelfinal)
vif(LR.modelfinal)

vif(LR.modelfinal)
vif = data.frame(vif(LR.modelfinal))
vif

df=data.frame(names(final_data[-21]),vif$vif.LR.modelfinal.)
names(df)[1] = "Features"
names(df)[2] = "VIFValues"

dev.off()

par(mar = c(6, 12, 2, 4) + 0.1)  

barplot(names.arg = df$Features, 
        height =df$VIFValues, 
        main = "VIF Values",
        xlab = "VIF Values",
        cex.names = 0.8,
        cex.axis = 0.8,
        #ylab = "Features",
        col = ifelse(df$VIFValues>5,"red","blue"),
        las=1,
        width=2,
        horiz = TRUE)

#-----------------------------------------------------------------------Feature Selection-------------------------------------------

options(repr.plot.width = 20, repr.plot.height = 20)

cor(final_data)

ggcorr(final_data, nbreaks = 5, label = TRUE, label_color = "black", low = "lightblue", mid = "white", high = "red", hjust = 0.9)

#Split into train and test data
paste("Count of data:",count(final_data))

set.seed(100)

final_data$id <- 1:nrow(final_data)

#train_test_data_split = createDataPartition(final_data$windmill_generated_power,p=0.7,list=FALSE)

#train_data = final_data[train_test_data_split,]
#test_data = final_data[-train_test_data_split,]

#train_data = data.frame(train_data)
#test_data = data.frame(test_data)

train_data <- final_data %>% dplyr::sample_frac(0.7)
test_data <- dplyr::anti_join(final_data, train_data, by='id')

paste("Count of train data:", count(train_data))
paste("Count of test data:", count(test_data))

Sys.setenv(JAVA_HOME="")
options(java.parameters = c("-XX:+UseConcMarkSweepGC", "-Xmx8192m"))
gc()
library(xlsx)
write.xlsx2(final_data, "E:/Aakarshan/Project Windies/Clean Data.xlsx")
write.xlsx2(train_data, "E:/Aakarshan/Project Windies/Train Data.xlsx")
write.xlsx2(test_data, "E:/Aakarshan/Project Windies/Test Data.xlsx")

#Run linear regression
#----------------------------------------------------Linear Regression on all x variables------------------------------------------
full.model = lm(windmill_generated_power ~. , data = train_data)

summary(full.model)
#------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------Running stepwise regression------------------------------------------
#Run stepwise regression to remove the unwanted features

step.model = step(full.model, direction="both", trace = FALSE)

summary(step.model)
step.model$model

#Features selected by the step function
selectedfeatures=as.data.frame(names(step.model$model))
rename(selectedfeatures, "Selected Features")
selectedfeatures

#-----------------------------------------------------Selected Features-----------------------------------------------------------
# selFeatures = data.frame(selectedfeatures)
# selFeatures
#----------------------------------------------------------------------------------------------------------------------------------

#enter the final model after step function
feat_step = windmill_generated_power	~
  year_data	+
  month	+
  day	+
  wind_speed	+
  atmospheric_temperature	+
  blades_angle	+
  engine_temperature	+
  motor_torque	+
  atmospheric_pressure	+
  area_temperature	+
  windmill_body_temperature	+
  wind_direction	+
  resistance	+
  cloud_level	+
  blade_length	+
  blade_breadth	

View(feat_step)


#-------------------------------------------------Linear Regression after removing unwanted features--------------------------------
#Model after stepwise regression
poststep.model=lm(feat_step, data = train_data)

summary(poststep.model)
#-----------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------Running RFE---------------------------------------------------------------
#Run RFE
set.seed(100)
train_data1 = train_data[1:50,]

# x_train_data1=train_data1[-22]
# y_train_data1=train_data1[22]
# 
# nrow(x_train_data1)
# nrow(y_train_data1)

#subsets=c(15:22)

ctrl = rfeControl(
  functions = rfFuncs,
  method = "repeatedcv",
  repeats = 10,
  verbose = FALSE
)

rfe.model = rfe(x=train_data1[1:21],
                y=train_data1$windmill_generated_power,
                sizes=c(15:21),
                rfeControl = ctrl)

rfe.model
predictors = predictors(rfe.model)
varImp(rfe.model)

summary(rfe.model)


ggplot(data = rfe.model, metric = "Rsquared") + theme_classic()

ggplot(data = rfe.model, metric = "RMSE") + theme_classic()

varimp_data <- data.frame(feature = row.names(varImp(rfe.model))[1:17],
                          importance = varImp(rfe.model)[1:17, 1])

ggplot(data = varimp_data, 
       aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat="identity", fill=ifelse(varimp_data$importance>3,"red","lightblue")) + labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust=0.5, color="black", size=3) + 
  coord_flip() +
  scale_x_discrete(guide = guide_axis(angle = 0)) +
  theme_bw() + theme(legend.position = "none")


feat_rfe = windmill_generated_power ~ motor_torque + 
  engine_temperature + wind_direction + 
  blades_angle + resistance + 
  wind_speed + gearbox_temperature + 
  month + area_temperature

# new_model = lm(windmill_generated_power ~ motor_torque	+
#   blades_angle	+
#   resistance	+
#   engine_temperature	+
#   wind_speed	+
#   rotor_torque	+
#   month	+
#   gearbox_temperature,
#   data = test_data
# )
#summary(new_model)

#-------------------------------------------------------------------Running Relative Importance-------------------------------------

#Run Relative Importance Regression
train_data1 = train_data[1:50,]

postrelimp.model = calc.relimp(poststep.model, data = train_data1, type = "lmg", rela = FALSE)

postrelimp.model

summary(postrelimp.model)


postrelimp.model = boot.relimp(poststep.model, data = train_data1, type = "lmg", b=10, rank = TRUE, diff = TRUE, rela = FALSE)

booteval = booteval.relimp(postrelimp.model)

#boosting.modelrel

summary(booteval)


features = c(
  "year_data"	,
  "month"	,
  "day"	,
  "wind_speed"	,
  "atmospheric_temperature"	,
  "blades_angle"	,
  "engine_temperature"	,
  "motor_torque"	,
  "atmospheric_pressure"	,
  "area_temperature"	,
  "windmill_body_temperature"	,
  "wind_direction"	,
  "resistance"	,
  "cloud_level"	,
  "blade_length"	,
  "blade_breadth"	
)

lmg = c(0.012733249	,
       0.06982522	,
       0.007045116	,
       0.007388437	,
       0.015404649	,
       0.010879653	,
       0.023698718	,
       0.153765502	,
       0.003500949	,
       0.077616979	,
       0.000103271	,
       0.049917432	,
       0.017890898	,
       0.000816547	,
       0.000434747	,
       0.004194774	
)

featureslmg=data.frame(features,lmg)

ggplot(data = featureslmg, 
       aes(x = reorder(features, lmg), y = lmg)) +
  geom_bar(stat="identity", fill=ifelse(lmg>0.01,"red","lightblue")) + labs(x = "Features", y = "lmg") + 
  geom_text(aes(label = round(lmg, 3)), vjust=0.5, color="black", size=3) + 
  coord_flip() +
  scale_x_discrete(guide = guide_axis(angle = 0)) +
  theme_bw() + theme(legend.position = "none")

feat_rel = windmill_generated_power ~ motor_torque +
  area_temperature +
  month +
  wind_direction +
  engine_temperature +
  resistance +
  atmospheric_temperature +
  year_data +
  blades_angle


# -------------------------------------------------------------------------------------------------------------------------------------
#Feature Elimination

#We Select features suggested by FBE & Correlation Matrix i.e. excluding Generator Temperator + FBE eliminated features.
# 
# final_data = dplyr::select(final_data, month, day, wind_speed, atmospheric_temperature, shaft_temperature, blades_angle,
#                       gearbox_temperature, engine_temperature, motor_torque, atmospheric_pressure,
#                       area_temperature, windmill_body_temperature, wind_direction, resistance, turbine_status, cloud_level,
#                       blade_length, blade_breadth, windmill_generated_power)
# 
# test = dplyr::select(test, month, day, wind_speed, atmospheric_temperature, shaft_temperature, blades_angle,
#                      gearbox_temperature, engine_temperature, motor_torque, atmospheric_pressure,
#                      area_temperature, windmill_body_temperature, wind_direction, resistance, turbine_status, cloud_level,
#                      blade_length, blade_breadth)

#------------------------------------------------------------------------Build Model and do Predictions---------------------------------
#Predict the values using the Linear Regression model

LR.modelstep = lm(feat_step, data = train_data)
LR.modelrfe = lm(feat_rfe, data = train_data)
LR.modelrel = lm(feat_rel, data = train_data)



#predictions

AIC(LR.modelstep)
data.frame(
  R2 = rsquare(LR.modelstep, data = train_data),
  RMSE = rmse(LR.modelstep, data = train_data),
  MAE = mae(LR.modelstep, data = train_data)
)


AIC(LR.modelrfe)
data.frame(
  R2 = rsquare(LR.modelrfe, data = train_data),
  RMSE = rmse(LR.modelrfe, data = train_data),
  MAE = mae(LR.modelrfe, data = train_data)
)

AIC(LR.modelrel)
data.frame(
  R2 = rsquare(LR.modelrel, data = train_data),
  RMSE = rmse(LR.modelrel, data = train_data),
  MAE = mae(LR.modelrel, data = train_data)
)

LR.modelfinal = LR.modelstep

#R2 of Step model is better, hence using rel model for predictions
predictions = predict(LR.modelfinal, newdata=test_data)

data.frame(
  R2 = rsquare(LR.modelfinal, data = test_data),
  RMSE = rmse(LR.modelfinal, data = test_data),
  MAE = mae(LR.modelfinal, data = test_data)
)

comparisons = data.frame(test_data$windmill_generated_power)
names(comparisons)[1] = "ActualValues"

#nrow(comparisons)

LRcomparisons = data.frame(comparisons$ActualValues, predictions) 
names(LRcomparisons)[1] = "Actual Values"
names(LRcomparisons)[2] = "Lin Reg Predictions"

#nrow(LRcomparisons)
#head(LRcomparisons)

head(LRcomparisons)


#--------------------------------------------------------Predict using DT--------------------------------------------------------
#Predict using Decision Trees

train_data1 = train_data[1:5000,]


dectree.modelstep = rpart(feat_step,
                      data = train_data1, 
                      control=rpart.control(cp=0.01)
                      )

dectree.modelrfe = rpart(feat_rfe,
                          data = train_data1, 
                          control=rpart.control(cp=0.01)
)

dectree.modelrel = rpart(feat_rel,
                         data = train_data1, 
                         control=rpart.control(cp=0.01)
)



data.frame(
  R2 = rsquare(dectree.modelstep, data = train_data1),
  RMSE = rmse(dectree.modelstep, data = train_data1),
  MAE = mae(dectree.modelstep, data = train_data1)
)

data.frame(
  R2 = rsquare(dectree.modelrfe, data = train_data1),
  RMSE = rmse(dectree.modelrfe, data = train_data1),
  MAE = mae(dectree.modelrfe, data = train_data1)
)

data.frame(
  R2 = rsquare(dectree.modelrel, data = train_data1),
  RMSE = rmse(dectree.modelrel, data = train_data1),
  MAE = mae(dectree.modelrel, data = train_data1)
)

dectree.modelfinal = dectree.modelrel

best <- dectree.modelfinal$cptable[which.min(dectree.modelfinal$cptable[,"xerror"]),"CP"]

pruned_tree <- prune(dectree.modelfinal, cp=best)

rpart.plot(pruned_tree)

# prp(pruned_tree,
#     faclen=0, #use full names for factor labels
#     extra=1, #display number of obs. for each terminal node
#     roundint=F, #don't round to integers in output
#     digits=5, #display 5 decimal places in output
#     branch = 0,
#     #box.palette = "blue"
#     border.col = "black"
#     ) 

#summary(pruned_tree)

#AIC(pruned_tree)
data.frame(
  R2 = rsquare(pruned_tree, data = test_data),
  RMSE = rmse(pruned_tree, data = test_data),
  MAE = mae(pruned_tree, data = test_data)
)

DTpredictions = predict(pruned_tree, newdata = test_data)
DTpredictions = data.frame(DTpredictions)

nrow(comparisons$ActualValues)
nrow(DTpredictions)

DTcomparisons = data.frame(comparisons$ActualValues, DTpredictions)
names(DTcomparisons)[1] = "Actual Values"
names(DTcomparisons)[2] = "Decision Tree Predictions"

head(DTcomparisons)

#---------------------------------------------------------------Predict using RF-------------------------------------------------
#Predict using Random Forests

#ncol(train_data)

# mtry = tuneRF(x = data.frame(train_data[,-23]), 
#        y = train_data$windmill_generated_power, 
#        stepFactor = 0.5, 
#        ntreeTry = 50, 
#        trace = TRUE, 
#        improve = 0.05,
#        plot = FALSE)
# 
# mtry

set.seed(1)
train_data1 = train_data[1:5000,]

randomforest.modelstep = randomForest(feat_step, 
                                  data = train_data1, 
                                  proximity = TRUE, 
                                  ntree = 150,
                                  trace = TRUE)
                                  # mtry = 14)

randomforest.modelrfe = randomForest(feat_rfe, 
                                      data = train_data1, 
                                      proximity = TRUE, 
                                      ntree = 150,
                                      trace = TRUE)
# mtry = 14)

randomforest.modelrel = randomForest(feat_rel, 
                                     data = train_data1, 
                                     proximity = TRUE, 
                                     ntree = 150,
                                     trace = TRUE)


data.frame(
  R2 = rsquare(randomforest.modelstep, data = train_data1),
  RMSE = rmse(randomforest.modelstep, data = train_data1),
  MAE = mae(randomforest.modelstep, data = train_data1)
)

data.frame(
  R2 = rsquare(randomforest.modelrfe, data = train_data1),
  RMSE = rmse(randomforest.modelrfe, data = train_data1),
  MAE = mae(randomforest.modelrfe, data = train_data1)
)

data.frame(
  R2 = rsquare(randomforest.modelrel, data = train_data1),
  RMSE = rmse(randomforest.modelrel, data = train_data1),
  MAE = mae(randomforest.modelrel, data = train_data1)
)

randomforest.modelfinal = randomforest.modelrel

#------------test data---------------------
data.frame(
  R2 = rsquare(randomforest.modelfinal, data = test_data),
  RMSE = rmse(randomforest.modelfinal, data = test_data),
  MAE = mae(randomforest.modelfinal, data = test_data)
)

randomforest.modelfinal

summary(randomforest.modelfinal)

vimp= ggRandomForests::gg_vimp(randomforest.modelfinal)
vimp

plot(vimp)

# tree = ctree(randomforest.model, data = test_data)
# plot(tree, type = "simple")

randForestpred = predict(randomforest.modelfinal, newdata=test_data)
randForestpred = data.frame(randForestpred)



#nrow(randForestpred)

RFcomparisons = data.frame(comparisons$ActualValues, randForestpred)
names(RFcomparisons)[1] = "Actual Values"
names(RFcomparisons)[2] = "RF Predictions"

head(RFcomparisons)
#---------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------Predict using Boosting------------------------------------------

train_data1 = train_data[1:5000,]


#Predict using Boosting
boosting.modelstep = gbm(feat_step,
                     data = train_data1,
                     n.trees = 100,
                     shrinkage = 0.01,
                     interaction.depth = 2)

boosting.modelrfe = gbm(feat_rfe,
                     data = train_data1,
                     n.trees = 100,
                     shrinkage = 0.01,
                     interaction.depth = 2)

boosting.modelrel = gbm(feat_rel,
                        data = train_data1,
                        n.trees = 100,
                        shrinkage = 0.01,
                        interaction.depth = 2)

data.frame(
  R2 = rsquare(boosting.modelstep, data = train_data1),
  RMSE = rmse(boosting.modelstep, data = train_data1),
  MAE = mae(boosting.modelstep, data = train_data1)
)

data.frame(
  R2 = rsquare(boosting.modelrfe, data = train_data1),
  RMSE = rmse(boosting.modelrfe, data = train_data1),
  MAE = mae(boosting.modelrfe, data = train_data1)
)

data.frame(
  R2 = rsquare(boosting.modelrel, data = train_data1),
  RMSE = rmse(boosting.modelrel, data = train_data1),
  MAE = mae(boosting.modelrel, data = train_data1)
)

boosting.modelfinal = boosting.modelrel

#Check accuracy on test data
data.frame(
  R2 = rsquare(boosting.modelfinal, data = test_data),
  RMSE = rmse(boosting.modelfinal, data = test_data),
  MAE = mae(boosting.modelfinal, data = test_data)
)

boosting.modelfinal
gbm.perf(boosting.modelfinal)

summary(boosting.modelfinal)



Boostingpred = predict(boosting.modelfinal, newdata=test_data)

Boostingcomparisons = data.frame(comparisons$ActualValues,Boostingpred)
names(Boostingcomparisons)[1] = "Actual Values"
names(Boostingcomparisons)[2] = "Boosting Predictions"

head(Boostingcomparisons)


#---------------------------------------------------------------------------Compare all models------------------------------------------

LR = data.frame(
  LR_R2 = rsquare(LR.modelfinal, data = test_data),
  LR_RMSE = rmse(LR.modelfinal, data = test_data),
  LR_MAE = mae(LR.modelfinal, data = test_data)
)

DT = data.frame(
  DT_R2 = rsquare(pruned_tree, data = test_data),
  DT_RMSE = rmse(pruned_tree, data = test_data),
  DT_MAE = mae(pruned_tree, data = test_data)
)

RF = data.frame(
  RF_R2 = rsquare(randomforest.modelfinal, data = test_data),
  RF_RMSE = rmse(randomforest.modelfinal, data = test_data),
  RF_MAE = mae(randomforest.modelfinal, data = test_data)
)

BM = data.frame(
  BM_R2 = rsquare(boosting.modelfinal, data = test_data),
  BM_RMSE = rmse(boosting.modelfinal, data = test_data),
  BM_MAE = mae(boosting.modelfinal, data = test_data)
)

allmodels = data.frame(LR,DT,RF,BM)

allmodels




compareAllPredictions=data.frame(comparisons$ActualValues,LRcomparisons$`Lin Reg Predictions`,DTcomparisons$`Decision Tree Predictions`, RFcomparisons$`RF Predictions`, Boostingcomparisons$`Boosting Predictions`)
names(compareAllPredictions)[1] = "Actual Values"
names(compareAllPredictions)[2] = "Lin Reg Predictions"
names(compareAllPredictions)[3] = "Decision Tree Predictions"
names(compareAllPredictions)[4] = "RF Predictions"
names(compareAllPredictions)[5] = "Boosting Predictions"

head(compareAllPredictions)



#=====================================================================END=============================================================

#------------------------------------------------------------------Plot Tree----------------------------------------------------------
# library(dplyr)
# library(ggraph)
# library(igraph)
# tree_func <- function(final_model, 
#                       tree_num) {
#   
#   # get tree by index
#   tree <- randomForest::getTree(final_model, 
#                                 k = tree_num, 
#                                 labelVar = TRUE) %>%
#     tibble::rownames_to_column() %>%
#     # make leaf split points to NA, so the 0s won't get plotted
#     mutate(`split point` = ifelse(is.na(prediction), `split point`, NA))
#   
#   # prepare data frame for graph
#   graph_frame <- data.frame(from = rep(tree$rowname, 2),
#                             to = c(tree$`left daughter`, tree$`right daughter`))
#   
#   # convert to graph and delete the last node that we don't want to plot
#   graph <- graph_from_data_frame(graph_frame) %>%
#     delete_vertices("0")
#   
#   # set node labels
#   V(graph)$node_label <- gsub("_", " ", as.character(tree$`split var`))
#   V(graph)$leaf_label <- as.character(tree$prediction)
#   V(graph)$split <- as.character(round(tree$`split point`, digits = 2))
#   
#   # plot
#   plot <- ggraph(graph, 'dendrogram') + 
#     theme_bw() +
#     geom_edge_link() +
#     geom_node_point() +
#     geom_node_text(aes(label = node_label), na.rm = TRUE, repel = TRUE) +
#     geom_node_label(aes(label = split), vjust = 2.5, na.rm = TRUE, fill = "white") +
#     geom_node_label(aes(label = leaf_label, fill = leaf_label), na.rm = TRUE, 
#                     repel = TRUE, colour = "white", fontface = "bold", show.legend = FALSE) +
#     theme(panel.grid.minor = element_blank(),
#           panel.grid.major = element_blank(),
#           panel.background = element_blank(),
#           plot.background = element_rect(fill = "white"),
#           panel.border = element_blank(),
#           axis.line = element_blank(),
#           axis.text.x = element_blank(),
#           axis.text.y = element_blank(),
#           axis.ticks = element_blank(),
#           axis.title.x = element_blank(),
#           axis.title.y = element_blank(),
#           plot.title = element_text(size = 18))
#   
#   print(plot)
# }
# 
# tree_num <- which(randomforest.model$finalModel$forest$ndbigtree == min(randomforest.model$finalModel$forest$ndbigtree))
# tree_func(final_model = randomforest.model$finalModel, tree_num)
# 
# randomforest.model$forest
# 
# tree_num <- which(model_rf$finalModel$forest$ndbigtree == min(model_rf$finalModel$forest$ndbigtree))
# tree_func(final_model = model_rf$finalModel, tree_num)

