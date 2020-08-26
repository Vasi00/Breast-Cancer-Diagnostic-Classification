# Load required libraries
library(dplyr)
library(caret)
library(caretEnsemble)
library(reshape2)
library(ggplot2)
library(ggthemes)
library(psych)
library(party)
library(rpart)
library(rpart.plot)
library(GGally)

# Loading dataset and looking at its structure
setwd("C:/Users/Faiz/Desktop/Project")
rawData <- read.csv('data.csv')
str(rawData)
head(rawData)
describe(rawData[,3:ncol(rawData)])   #OR
# summary(rawData)

# Dropping id column as it has no use in predicting/classification
data<- rawData[2:ncol(rawData)]
# OR  data <- subset(rawData, select = -id)

# count of benign and malign datapoints
table(data$diagnosis=="M")
# proportion of B's and M's 
prop.table(table(data$diagnosis))   
# Histogram of count of B's and M's
ggplot(data, aes(x = diagnosis)) +
  geom_bar(aes(fill = "yellow")) +
  ggtitle("Distribution of diagnosis for the entire dataset") +
  theme(legend.position="none")

# Labeling categorical values of diagnosis column as 0 and 1 for benign and malign respectively
data["diagnosis"] <- factor(data$diagnosis,
                            levels = c("B","M"), labels = c(0,1))
# missing values
table(complete.cases(data))  # counting all values that are filled(complete) as True and NA as False
table(is.na(data))           # counting total NA values

normalizer <- function(x)
{(x-mean(x))/sd(x)
}
# Normalizing data
(len <- length(data))
data[, 2:len] <- lapply(data[, 2:len], normalizer)  # OR 
# data[,-1]<-scale(data[,-1])                       # using built-in R function scale()
head(data)

# CORRELATION ANALYSIS

## Correlation between diagnosis and all other predictor variables along with their p-values
a <- corr.test(data[,-1], as.numeric(data$diagnosis))
a

# Correlation between independent features
M<-cor(data[,-1])
cor2(data[,-1])     # lower panel of correlation matrix with values rounded off to 2 decimal places

## Plotting correlation matrix using corrplot()
library(corrplot)
library(RColorBrewer)
dev.off()
corrplot(M,tl.cex = 0.7)  
corrplot(M, type="upper", order="hclust",
         tl.cex = 0.7, col=brewer.pal(n=9, name="RdYlBu"))

## Finding features with the correlation >= 0.9 
highlyCor <- colnames(data)[findCorrelation(M, cutoff = 0.9, verbose = TRUE)]
highlyCor

## Removing all features with a correlation higher than 0.9
data_cor <- data[, which(!colnames(data) %in% highlyCor)]
ncol(data_cor)  # So our new dataframe data_cor has 21 variables with target variable included 

# new_data will be used for logistic regression and neural nets
new_data <- cbind(diagnosis = data$diagnosis, data_cor)

# Training-testing split for Logistic Regression and Neural nets
library(caTools)
set.seed(0303)
split_new = sample.split(new_data, SplitRatio = 0.7)
training_new = subset(new_data, split_new == TRUE)
test_new = subset(new_data, split_new == FALSE)

## Proportions of the data split (useful for checking Imbalance in the dataset)
prop.table(table(training_new$diagnosis))
prop.table(table(test_new$diagnosis))

# MODELING AND CLASSIFICATION USING MACHINE LEARNING ALGORITHMS

# LOGISTIC REGRESSION
logistic <- glm(diagnosis~., data = training_new, family = binomial(link="logit"),
                control = list(maxit = 100))
summary(logistic)
par1 <- par(mfrow=c(2,4))
plot(logistic)

pred <- plogis(predict(logistic, test_new[,-1]))  # predicted log odds values
### ( using plogis() to obtain probability values from odds ratio), OR use response
### inside like: pred <- predict(logistic, test[,-1], type="response")  # another way 

logpred <- ifelse(pred >= 0.5, "1", "0")  # converting to 0's and 1's

## Confusion matrix, accuracy and other statistics
(c1 <- confusionMatrix(as.factor(logpred), as.factor(test_new[,1])))
table(logpred,test_new[,1])        # shows confusion matrix without other statistics  ## OR

# Training-testing split for other algorithms (Decision tree, naive bayes, SVM, ANN)
set.seed(909)
split = sample.split(data, SplitRatio = 0.7)
training = subset(data, split == TRUE)
test = subset(data, split == FALSE)

## not necessary to make predictors dataframe, just to simplify things
test.X <- test[,-1]
## Separating the dependent variable i.e. diagonsis from test dataframe
test.Y <- as.factor(test$diagnosis)

# ARTIFICIAL NEURAL NETWORKS (ANN) CLASSIFICATION
library(neuralnet)
library(nnet)
nn <- neuralnet(diagnosis~., data = training, hidden = c(16,12),
                act.fct = "tanh", linear.output = FALSE)
# plot(nn)

nnpred0 <- predict(nn, test[,-1])     # Prediction
nn.pred <- max.col(nnpred0)           # choosing col.no. with greater value for every row

nnpred <- ifelse(nn.pred == 2, "1", "0") # converting them to 0 and 1

(c2 <- confusionMatrix(as.factor(nnpred),as.factor(test[,1])))   

# DECISION TREE AND RANDOM FOREST ANALYSIS
## Many functions available like ctree(),rpart(),....
library(party)
library(rpart)
library(caret)

## Conditional inference based tree
ctreemodel <- ctree(training$diagnosis~.,data = training)
ctreemodel
summary(ctreemodel)
dev.off()
plot(ctreemodel)  # plotting the tree
ctreepred <- predict(ctreemodel,test[,-1])   # Prediction on test set

c3 <- confusionMatrix(ctreepred,test$diagnosis) 
c3

# DECISION TREE using CART (rpart)
cartmodel <- rpart(diagnosis~.,data=training)
summary(cartmodel)
library(rpart.plot)
rpart.plot(cartmodel)  # plot

## CART using cross-validation
ctrl <- trainControl(method = 'repeatedcv',number = 10, 
                     repeats = 3,savePredictions = T)  # Cross-validation parameters

cartmodelcv <- train(diagnosis~.,data=training, method='rpart', 
                     parms=list(split='Information'), trControl = ctrl)
cartmodelcv

cartpred <- predict(cartmodelcv,test[,-1])    # Prediction

(c4 <- confusionMatrix(cartpred,test.Y))   # Confusion matrix, accuracy and statistics

# RANDOM FOREST MODEL
library(randomForest)
rf <- randomForest(diagnosis~.,data = training,keep.Forest=F)
rf
plot(rf, log="y")
print(importance(rf,type = 2))  # Importance of each predictor

rfpred0 <- predict(rf, test[,-1])
confusionMatrix(rfpred0, test.Y)

## Random Forest with preprocess and cross-validation
rfmodel <- train(y = training[,1], x = training[,-1],
                   method='rf', preProcess=c("center","scale"), trControl = ctrl)
rfmodel
plot(rfmodel)                       # Accuracy vs no. of randomly selected predictors plot
rfpred<-predict(rfmodel,test[,-1])  # Predictions on test set

(c5 <- confusionMatrix(rfpred,test.Y))  # Confusion matrix, accuracy and statistics

# NAIVE BAYES CLASSIFICATION 
library(e1071) # for naive bayes and support vector machine
nbmodel<-naiveBayes(diagnosis~., data=training)
nbmodel

nbpred<-predict(nbmodel, test[,-1])

(c6 <- confusionMatrix(nbpred, test.Y))   # Confusion matrix, accuracy and statistics

# SUPPORT VECTOR MACHINES (SVM Classification)
svmodel <- svm(diagnosis~.,data = training)
svmodel

svmpred <- predict(svmodel, test[,-1])             # Prediction
(c7 <- confusionMatrix(svmpred, test.Y))    # Confusion matrix, accuracy and statistics

# VISUALIZATION 
## Receiver Operating Characteristic (ROC) curve
library(ROCR) 
library(Metrics)

pr1 <- prediction(as.numeric(logpred), test_new$diagnosis)
pr2 <- prediction(as.numeric(nnpred), test_new$diagnosis)
pr3 <- prediction(as.numeric(ctreepred), test[,1])
pr4 <- prediction(as.numeric(cartpred), test$diagnosis)
pr5 <- prediction(as.numeric(rfpred), test$diagnosis)
pr6 <- prediction(as.numeric(nbpred), test$diagnosis)
pr7 <- prediction(as.numeric(svmpred), test$diagnosis)


perf1 <- performance(pr1,measure = "tpr",x.measure = "fpr")
perf2 <- performance(pr2, "tpr", "fpr")
perf3 <- performance(pr3, "tpr", "fpr")
perf4 <- performance(pr4, "tpr", "fpr")
perf5 <- performance(pr5, "tpr", "fpr")
perf6 <- performance(pr6, "tpr", "fpr")
perf7 <- performance(pr7, "tpr", "fpr")
dev.off()
 
plot(perf1, main = "ROC Curve for different classification models", col="blue")
plot(perf2, add = TRUE, col="red" )
plot(perf3, add = TRUE, col="yellow")
plot(perf4, add = TRUE, col="green")
plot(perf5, add = TRUE, col="black")
plot(perf6, add = TRUE, col="purple")
plot(perf7, add = TRUE, col="orange")
abline(a=0, b= 1)
legend(0.75,0.75, legend = c("Logistic","Neural net","Ctree","CART tree","Random forest",
                    "Naive Bayes","SVM"),fill=c("blue","red","yellow","green",
                    "black","purple","orange"), title = "Models", text.width = strwidth("100000"),
                     cex = 0.5, lwd = 1, merge = TRUE)

(auc1 <- auc(test$diagnosis,logpred))
(auc2 <- auc(test$diagnosis,nnpred))
(auc3 <- auc(test$diagnosis,ctreepred))
(auc4 <- auc(test$diagnosis,cartpred))
(auc5 <- auc(test$diagnosis,rfpred))
(auc6 <- auc(test$diagnosis,nbpred))
(auc7 <- auc(test$diagnosis,svmpred))

# Confusion matrix visualization
ctable1 <- as.table(matrix(c(c1$table), nrow = 2, byrow = TRUE)) 
ctable2 <- as.table(matrix(c(c2$table), nrow = 2, byrow = TRUE))
ctable3 <- as.table(matrix(c(c3$table), nrow = 2, byrow = TRUE))
ctable4 <- as.table(matrix(c(c4$table), nrow = 2, byrow = TRUE)) 
ctable5 <- as.table(matrix(c(c5$table), nrow = 2, byrow = TRUE)) 
ctable6 <- as.table(matrix(c(c6$table), nrow = 2, byrow = TRUE)) 
ctable7 <- as.table(matrix(c(c7$table), nrow = 2, byrow = TRUE)) 

new.par <- par(mfrow=c(2,4))
fourfoldplot(ctable1, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "Logistic regression")
 
fourfoldplot(ctable2, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "Neural networks")
 
fourfoldplot(ctable3, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "Conditional decision tree")

fourfoldplot(ctable4, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "CART tree")

fourfoldplot(ctable5, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "Random Forest")

fourfoldplot(ctable6, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1,main = "Naive Bayes")

fourfoldplot(ctable7, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "SVM")







