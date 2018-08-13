library(h2o)

h2o.init()

setwd("C:/Users/along/Documents/AJ Local/ArchiveDS/Week5")

# Import data for h20
trainPath <- "CreditCardLoanTrain_h2o.csv"
train <- h2o.uploadFile(path = trainPath, destination_frame = "train")

# Update variables to factors
train[,2] <- as.factor(train[,2]) 
train[,7] <- as.factor(train[,7]) 


# Review Variables and their types
h2o.getTypes(train)


testPath <- "CreditCardLoanValid.csv"
test <- h2o.uploadFile(path = testPath, destination_frame = "test")

# Update variables to factors
test[,1] <- as.factor(test[,1])
h2o.getTypes(test)


# Identify predictors and response

y <- c("Status_rc","Status")
x <- setdiff(names(train), y)
y <- "Status_rc"


# For binary classification, response should be a factor
train[,y] <- as.factor(train[,y])
#test[,y] <- as.factor(test[,y])

aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_runtime_secs = 90,
                  seed=123,
                  stopping_rounds = 5,
                  balance_classes = TRUE, #Used when observations are unbalanced
                  max_after_balance_size = 3
                  
)

# View the AutoML Leaderboard
lb <- aml@leaderboard
lb

aml@leader

# Gains and Lift Table
h2o.gainsLift(aml@leader, xval = TRUE)

pred <- h2o.predict(aml, test)  # predict(aml, test) also works
predtrain <- h2o.predict(aml, train)

train_predictions <- h2o.cbind(train, predtrain)
write.csv(as.data.frame(train_predictions), file = 'h2o.trainpredictions.csv')

test_predictions <- h2o.cbind(test, pred)
write.csv(as.data.frame(test_predictions), file = 'h2o.testpredictions.csv')