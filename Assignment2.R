#Setup
library(rpart)
library(rpart.plot)
library(plyr)
library(pROC)
library(caret)

set.seed(369)

#Data Preparation
carData <- read.csv("C:/Users/princ/Downloads/car.data.txt", stringsAsFactors=TRUE)

#Feature Engineering
carData$capacity_indicator <- as.numeric(ifelse(carData$doors == "5more", 5, as.character(carData$doors))) / as.numeric(ifelse(carData$seats == "more", 5, as.character(carData$seats)))

head(carData)
nrow(carData)

#Data Cleaning
sum(is.na(carData))

#Data Split
trainIndex <- createDataPartition(carData$shouldBuy, p = .8, 
                                  list = FALSE, 
                                  times = 1)
carTrain <- carData[trainIndex,]
carTest <- carData[-trainIndex,]

nrow(carTrain)
nrow(carTest)

#----------Decision Tree with Rpart-----------------
#Training the model: Decision Tree with Rpart
treeCar = rpart(shouldBuy~.,data=carTrain, method="class",control = rpart.control(minsplit = 10))
treeCar

#Testing the model
predCar = predict(treeCar,newdata = carTest,type = "class")
head(predCar)
head(predCar, n = 20)
predCar

#Generate the Confusion Matrix
treeCM = table(carTest[,"shouldBuy"], predCar)
treeCM
sum(diag(treeCM)/sum(treeCM))
confusionMatrix(predCar, carTest[,"shouldBuy"])

#Plotting the Decision Tree
rpart.plot(treeCar, cex = 0.6,tweak = 1) 

#----------ROC for Decision Tree with Rpart-----------------
predCarProb = predict(treeCar,newdata = carTest, type="prob")
predCarProb

roc_multiclass = multiclass.roc(carTest[,"shouldBuy"],predCarProb[,2])
roc_multiclass

true_unacc_labels <- ifelse(carTest$shouldBuy == "unacc", 1, 0)
predicted_unacc_probs <- predCarProb[, "unacc"]
roc_unacc_obj <- roc(true_unacc_labels, predicted_unacc_probs)

true_acc_labels <- ifelse(carTest$shouldBuy == "acc", 1, 0)
predicted_acc_probs <- predCarProb[, "acc"]
roc_acc_obj <- roc(true_acc_labels, predicted_acc_probs)

true_good_labels <- ifelse(carTest$shouldBuy == "good", 1, 0)
predicted_good_probs <- predCarProb[, "good"]
roc_good_obj <- roc(true_good_labels, predicted_good_probs)

true_vgood_labels <- ifelse(carTest$shouldBuy == "vgood", 1, 0)
predicted_vgood_probs <- predCarProb[, "vgood"]
roc_vgood_obj <- roc(true_vgood_labels, predicted_vgood_probs)

plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "Specificity", ylab = "Sensitivity",
     main = "ROC Curves for Different Classes")

lines(1 - roc_unacc_obj$specificities, roc_unacc_obj$sensitivities, col = "blue", type = "l", lwd = 2)
lines(1 - roc_acc_obj$specificities, roc_acc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - roc_good_obj$specificities, roc_good_obj$sensitivities, col = "green", lwd = 2)
lines(1 - roc_vgood_obj$specificities, roc_vgood_obj$sensitivities, col = "purple", lwd = 2)

abline(0, 1, lty = 1)

legend("bottomright", 
       legend = c(
         paste("unacc (AUC =", round(auc(roc_unacc_obj), 3), ")", sep = ""),
         paste("acc (AUC =", round(auc(roc_acc_obj), 3), ")", sep = ""),
         paste("good (AUC =", round(auc(roc_good_obj), 3), ")", sep = ""),
         paste("vgood (AUC =", round(auc(roc_vgood_obj), 3), ")", sep = "")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)

#----------Decision Tree with Rpart (Fine-Tuning)-----------------

#Training the Model: Set complexity as 0.0001
treeCarFT = rpart(shouldBuy~.,data=carTrain, method="class",control = rpart.control(minsplit = 10, cp=0.0001)) 
treeCarFT

#Testing the Fine Tuned Model
predCarFT = predict(treeCarFT,newdata = carTest,type = "class")
predCarFT

#Generate Confusion Matrix
treeCMFT = table(carTest[,"shouldBuy"], predCarFT)
sum(diag(treeCMFT)/sum(treeCMFT))
confusionMatrix(predCarFT, carTest[,"shouldBuy"])

#Plot the Decision Tree
rpart.plot(treeCarFT,tweak = 1.3)

#----------ROC for Decision Tree with Rpart (Fine-Tuning)-----------------
predCarProbFT = predict(treeCar,newdata = carTest, type="prob")
predCarProbFT

roc_multiclassFT = multiclass.roc(carTest[,"shouldBuy"],predCarProbFT[,2])
roc_multiclassFT

true_unacc_labels <- ifelse(carTest$shouldBuy == "unacc", 1, 0)
predicted_unacc_probs <- predCarProbFT[, "unacc"]
roc_unacc_obj <- roc(true_unacc_labels, predicted_unacc_probs)

true_acc_labels <- ifelse(carTest$shouldBuy == "acc", 1, 0)
predicted_acc_probs <- predCarProbFT[, "acc"]
roc_acc_obj <- roc(true_acc_labels, predicted_acc_probs)

true_good_labels <- ifelse(carTest$shouldBuy == "good", 1, 0)
predicted_good_probs <- predCarProbFT[, "good"]
roc_good_obj <- roc(true_good_labels, predicted_good_probs)

true_vgood_labels <- ifelse(carTest$shouldBuy == "vgood", 1, 0)
predicted_vgood_probs <- predCarProbFT[, "vgood"]
roc_vgood_obj <- roc(true_vgood_labels, predicted_vgood_probs)

plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "Specificity", ylab = "Sensitivity",
     main = "ROC Curves for Different Classes")

lines(1 - roc_unacc_obj$specificities, roc_unacc_obj$sensitivities, col = "blue", type = "l", lwd = 2)
lines(1 - roc_acc_obj$specificities, roc_acc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - roc_good_obj$specificities, roc_good_obj$sensitivities, col = "green", lwd = 2)
lines(1 - roc_vgood_obj$specificities, roc_vgood_obj$sensitivities, col = "purple", lwd = 2)

abline(0, 1, lty = 1)

legend("bottomright", 
       legend = c(
         paste("unacc (AUC =", round(auc(roc_unacc_obj), 3), ")", sep = ""),
         paste("acc (AUC =", round(auc(roc_acc_obj), 3), ")", sep = ""),
         paste("good (AUC =", round(auc(roc_good_obj), 3), ")", sep = ""),
         paste("vgood (AUC =", round(auc(roc_vgood_obj), 3), ")", sep = "")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)

#----------Decision Tree (Fine-Tuned) with K-Fold----------------

# Define the train control
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

tuneGrid <- expand.grid(
  cp = c(0.0001)
)

# Train the decision tree model
treeCarKF <- train(carTrain[,c(1:6,8)],carTrain[,"shouldBuy"], 
  method = "rpart", 
  metric = "Accuracy", 
  trControl = control,
  tuneGrid = tuneGrid
)

# Print the results
print(treeCarKF)

#Testing the Fine Tuned Model
predCarKF = predict(treeCarKF,newdata = carTest)

#Generate Confusion Matrix
treeCMKF = table(carTest[,"shouldBuy"], predCarKF)
sum(diag(treeCMKF)/sum(treeCMKF))
confusionMatrix(predCarKF, carTest[,"shouldBuy"])

#Plot the Decision Tree
rpart.plot(treeCarKF$finalModel, yesno=2,tweak = 1.3)

#----------ROC-----------------
predCarProbKF = predict(treeCarKF,newdata = carTest, type="prob")
predCarProbKF

roc_multiclass = multiclass.roc(carTest[,"shouldBuy"],predCarProbKF[,2])
roc_multiclass

true_unacc_labels <- ifelse(carTest$shouldBuy == "unacc", 1, 0)
predicted_unacc_probs <- predCarProbKF[, "unacc"]
roc_unacc_obj <- roc(true_unacc_labels, predicted_unacc_probs)

true_acc_labels <- ifelse(carTest$shouldBuy == "acc", 1, 0)
predicted_acc_probs <- predCarProbKF[, "acc"]
roc_acc_obj <- roc(true_acc_labels, predicted_acc_probs)

true_good_labels <- ifelse(carTest$shouldBuy == "good", 1, 0)
predicted_good_probs <- predCarProbKF[, "good"]
roc_good_obj <- roc(true_good_labels, predicted_good_probs)

true_vgood_labels <- ifelse(carTest$shouldBuy == "vgood", 1, 0)
predicted_vgood_probs <- predCarProbKF[, "vgood"]
roc_vgood_obj <- roc(true_vgood_labels, predicted_vgood_probs)

plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "Specificity", ylab = "Sensitivity",
     main = "ROC Curves for Different Classes")

lines(1 - roc_unacc_obj$specificities, roc_unacc_obj$sensitivities, col = "blue", type = "l", lwd = 2)
lines(1 - roc_acc_obj$specificities, roc_acc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - roc_good_obj$specificities, roc_good_obj$sensitivities, col = "green", lwd = 2)
lines(1 - roc_vgood_obj$specificities, roc_vgood_obj$sensitivities, col = "purple", lwd = 2)

abline(0, 1, lty = 1)

legend("bottomright", 
       legend = c(
         paste("unacc (AUC =", round(auc(roc_unacc_obj), 3), ")", sep = ""),
         paste("acc (AUC =", round(auc(roc_acc_obj), 3), ")", sep = ""),
         paste("good (AUC =", round(auc(roc_good_obj), 3), ")", sep = ""),
         paste("vgood (AUC =", round(auc(roc_vgood_obj), 3), ")", sep = "")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)

#----Random Forest--------------------------------
#Training the model: Random Forest

#Model Configuration
randomForest_default = train(carTrain[,c(1:6,8)],carTrain[,"shouldBuy"],method = "rf",metric = "Accuracy")
print(randomForest_default)

#Get the Importance Variables for Random Forest
varImp(randomForest_default)
ggplot(varImp(randomForest_default))

#Testing the model
predictmodel <- predict(randomForest_default, carTest)

RandomForestCM = table(carTest[,"shouldBuy"], predictmodel)
RandomForestCM
sum(diag(RandomForestCM)/sum(RandomForestCM))
confusionMatrix(RandomForestCM)

#----------ROC for Random Forest-----------------
predCarProbRF = predict(randomForest_default,newdata = carTest, type="prob")
predCarProbRF

roc_multiclassRF = multiclass.roc(carTest[,"shouldBuy"],predCarProbRF[,2])
roc_multiclassRF

true_unacc_labels <- ifelse(carTest$shouldBuy == "unacc", 1, 0)
predicted_unacc_probs <- predCarProbRF[, "unacc"]
roc_unacc_obj <- roc(true_unacc_labels, predicted_unacc_probs)

true_acc_labels <- ifelse(carTest$shouldBuy == "acc", 1, 0)
predicted_acc_probs <- predCarProbRF[, "acc"]
roc_acc_obj <- roc(true_acc_labels, predicted_acc_probs)

true_good_labels <- ifelse(carTest$shouldBuy == "good", 1, 0)
predicted_good_probs <- predCarProbRF[, "good"]
roc_good_obj <- roc(true_good_labels, predicted_good_probs)

true_vgood_labels <- ifelse(carTest$shouldBuy == "vgood", 1, 0)
predicted_vgood_probs <- predCarProbRF[, "vgood"]
roc_vgood_obj <- roc(true_vgood_labels, predicted_vgood_probs)

plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "Specificity", ylab = "Sensitivity",
     main = "ROC Curves for Different Classes")

lines(1 - roc_unacc_obj$specificities, roc_unacc_obj$sensitivities, col = "blue", type = "l", lwd = 2)
lines(1 - roc_acc_obj$specificities, roc_acc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - roc_good_obj$specificities, roc_good_obj$sensitivities, col = "green", lwd = 2)
lines(1 - roc_vgood_obj$specificities, roc_vgood_obj$sensitivities, col = "purple", lwd = 2)

abline(0, 1, lty = 1)

legend("bottomright", 
       legend = c(
         paste("unacc (AUC =", round(auc(roc_unacc_obj), 3), ")", sep = ""),
         paste("acc (AUC =", round(auc(roc_acc_obj), 3), ")", sep = ""),
         paste("good (AUC =", round(auc(roc_good_obj), 3), ")", sep = ""),
         paste("vgood (AUC =", round(auc(roc_vgood_obj), 3), ")", sep = "")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)

#----Random Forest with K-folds & Fine Tuning--------------------------------
#Training the model: Random Forest
control = trainControl(method = "repeatedcv", number = 3, repeats = 2)
randomForestKFFT = train(carTrain[,c(1:6,8)],carTrain[,"shouldBuy"],method = "rf",metric = "Accuracy", trControl = control,ntree = 1000, nodesize = 1)
print(randomForestKFFT)
varImp(randomForestKFFT)
ggplot(varImp(randomForestKFFT))

#Testing the model
predictmodelKFFT <- predict(randomForestKFFT, carTest)

RandomForestKFFTCM = table(carTest[,"shouldBuy"], predictmodelKFFT)
RandomForestKFFTCM
sum(diag(RandomForestKFFTCM)/sum(RandomForestKFFTCM))
confusionMatrix(RandomForestKFFTCM)

#----------ROC for Random-----------------
predCarProbKFFT = predict(randomForestKFFT,newdata = carTest, type="prob")
predCarProbKFFT

roc_multiclassKFFT = multiclass.roc(carTest[,"shouldBuy"],predCarProbKFFT[,2])
roc_multiclassKFFT

true_unacc_labels <- ifelse(carTest$shouldBuy == "unacc", 1, 0)
predicted_unacc_probs <- predCarProbKFFT[, "unacc"]
roc_unacc_obj <- roc(true_unacc_labels, predicted_unacc_probs)

true_acc_labels <- ifelse(carTest$shouldBuy == "acc", 1, 0)
predicted_acc_probs <- predCarProbKFFT[, "acc"]
roc_acc_obj <- roc(true_acc_labels, predicted_acc_probs)

true_good_labels <- ifelse(carTest$shouldBuy == "good", 1, 0)
predicted_good_probs <- predCarProbKFFT[, "good"]
roc_good_obj <- roc(true_good_labels, predicted_good_probs)

true_vgood_labels <- ifelse(carTest$shouldBuy == "vgood", 1, 0)
predicted_vgood_probs <- predCarProbKFFT[, "vgood"]
roc_vgood_obj <- roc(true_vgood_labels, predicted_vgood_probs)

plot(NULL, xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "Specificity", ylab = "Sensitivity",
     main = "ROC Curves for Different Classes")

lines(1 - roc_unacc_obj$specificities, roc_unacc_obj$sensitivities, col = "blue", type = "l", lwd = 2)
lines(1 - roc_acc_obj$specificities, roc_acc_obj$sensitivities, col = "red", lwd = 2)
lines(1 - roc_good_obj$specificities, roc_good_obj$sensitivities, col = "green", lwd = 2)
lines(1 - roc_vgood_obj$specificities, roc_vgood_obj$sensitivities, col = "purple", lwd = 2)

abline(0, 1, lty = 1)

legend("bottomright", 
       legend = c(
         paste("unacc (AUC =", round(auc(roc_unacc_obj), 3), ")", sep = ""),
         paste("acc (AUC =", round(auc(roc_acc_obj), 3), ")", sep = ""),
         paste("good (AUC =", round(auc(roc_good_obj), 3), ")", sep = ""),
         paste("vgood (AUC =", round(auc(roc_vgood_obj), 3), ")", sep = "")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2)
