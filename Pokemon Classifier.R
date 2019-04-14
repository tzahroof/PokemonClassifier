
###
#  TODO:
#  Edit the getScore() function to change the way placements are evaluated (option 2 seems better)
#  Determine a better way to visualize the data
#  Drop this bad boy on github
#  Possibly show and evaluate the critical model parameters
#  Turn it into R code??
###


cat("\014") # clears the console
rm(list=ls()) #clears all objects from the environment

#laptop workspace
setwd("C:/Users/tariq/OneDrive/Projects and Memes/Pokemon classifier")

pokedata<- read.csv("./smogon.csv", as.is=T)

View(pokedata)

#Fill out all the empty columns in Type with "N/A"
pokedata[pokedata$Type.2 == "",]$Type.2 <- "N/A"

###
#Replace the tiers with scaling points 
###
tierCategories = unique(pokedata$Tier)

tierScore <- seq(0, length(unique(pokedata$Tier)) - 1)
tierScore <- rev(tierScore)
#tiers, in order: PU, BL4, NU, BL3, RU, BL2, UU, BL, OU, Uber, AG
#tiers, reversed: AG, Uber, OU, BL, UU, BL2, RU, BL3, NU, BL4, PU

pokedata <- data.frame(pokedata, rep(0, times=nrow(pokedata)))
colnames(pokedata)[ncol(pokedata)] <- "TierScore"

for(i in seq(1, length(tierCategories)))
{
  pokedata[pokedata$Tier == tierCategories[i], ]$TierScore <- tierScore[i]
}

###
#Replace Legendary boolean with 1/0 for true/false
###
pokedata$Legendary <- as.integer(as.logical(pokedata$Legendary))
pokedata$Mega      <- as.integer(as.logical(pokedata$Mega))

###
#Typing datasheet
###

nPokTypes <- length(unique(pokedata$Type.1))
typeMatrix <- data.frame(matrix(rep(0, (nPokTypes) * nrow(pokedata)), 
       nrow=nrow(pokedata), ncol=nPokTypes))
colnames(typeMatrix) <- unique(pokedata$Type.1)

for(i in 1:nrow(pokedata))
{
  typeMatrix[i, pokedata$Type.1[i]] = 1
  if(pokedata$Type.2[i] != "N/A")
  {
    typeMatrix[i, pokedata$Type.2[i]] = 1
  }
}

pokedata <- data.frame(pokedata, typeMatrix)

###
#Create training and test data
###
set.seed(2019)
maxfolds <- 7
folds <- sample(rep(1:maxfolds, length=nrow(pokedata)))

#set the last fold to be a test set
test.indices  <- which(folds == maxfolds)
train.indices <- which(folds != maxfolds)

###
# Helper Functions to display data
###
getRMSE <- function(preddata, testdata)
{
  this.rmse <- sqrt(mean((preddata-testdata)^2))
  this.rmse
}

getScore <- function(preddata, testdata)
{
  #Round the scores to the appropriate category (we want to see if the overall categories are correct)
  preddata[which(preddata > 10)] = 10
  preddata[which(preddata< 0)] = 0
  preddata = round(preddata)
  
  #Option 1: just sums the difference
  #score <- sum(abs(preddata - testdata))
  
  #Option 2: squares the difference (severely punishes bad placements)
  score <- sum((preddata - testdata)^2)
  score
}

getTable <- function(indices, predVals)
{
  predCat <- rep("N/A", length(predVals))
  predCat[which(predVals < 0.5)] = "PU"
  predCat[which(predVals >= 9.5)] = "AG"
  
  for(i in rev(2:length(tierCategories)-1))
  {
    predCat[which(predVals >= (tierScore[i] + tierScore[i+1])/2 & 
                    predVals < (tierScore[i] + tierScore[i-1])/2)] <- tierCategories[i]
  }
  
  newFrame <- data.frame(pokedata$Name[indices], pokedata$TierScore[indices], predVals, pokedata$Tier[indices], predCat,
                         pokedata[,3:14][indices,])
  colnames(newFrame) <- c("Pokemon", "Actual.Score", "Predicted.Score", "Real.Tier", "Predicted.Tier",colnames(pokedata)[3:14])
  newFrame <- newFrame[order(-newFrame$Actual.Score),]
  newFrame
}

##############################################################################################
#*********************************************************************************************
##############################################################################################

#PRINT and hold the model errors
model.test.errors <- c()

###
#Build Linear Model model  (no typing)
###
lm.notyping.fit <- lm(formula=TierScore~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary+Mega,
             data = pokedata, subset=train.indices)

lm.notyping.pred <- predict(lm.notyping.fit, newdata = pokedata[test.indices, ])

model.test.errors <- c(model.test.errors, getScore(lm.notyping.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "Lin.No.Type"

lm.notyping.table <- getTable(test.indices, lm.notyping.pred)

#Show results
summary(lm.notyping.fit)
coef(lm.notyping.fit)
View(lm.notyping.table)


#########################################################

###
#Build Linear Model model  (TYPING)
###
lm.typing.fit <- lm(formula=TierScore~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary+Mega+
                      Dragon+Ghost+Normal+Psychic+Fire+Dark+Steel+Water+Ground+Fighting+Bug+Fairy+Grass+Electric+
                      Rock+Flying+Poison+Ice,
                      data = pokedata, subset=train.indices)

lm.typing.pred <- predict(lm.typing.fit, newdata = pokedata[test.indices, ])

model.test.errors <- c(model.test.errors, getScore(lm.typing.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "Lin.TYPE"


lm.typing.table <- getTable(test.indices, lm.typing.pred)

#Show results
summary(lm.typing.fit)
coef(lm.typing.fit)
View(lm.typing.table)

#########################################################

require(glmnet)

###
# LASSO
###


#make y and x matrices because the LASSO package is hot garbage and you all know it
y.vals <- pokedata$TierScore
x.vals <- data.matrix(pokedata)
x.vals <- x.vals[,6:ncol(x.vals)]
x.vals <- cbind(x.vals[,1:9],x.vals[,12:ncol(x.vals)])

#make train and test data
y.test <- y.vals[test.indices]
x.test <- x.vals[test.indices, ]

y.train <- y.vals[train.indices]
x.train <- x.vals[train.indices, ]

#Build model
cv.lin.lasso <- cv.glmnet(x=x.train,y=y.train)
best.cv.lin.lasso.lam <- cv.lin.lasso$lambda.min
lin.lasso.fit <- glmnet(x=x.train, y=y.train)

#Predict and check
lin.lasso.pred <- predict(lin.lasso.fit, newx=data.matrix(x.test),
                      s=best.cv.lin.lasso.lam)


#Display data
model.test.errors <- c(model.test.errors, getScore(lin.lasso.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "LASSO"

lin.lasso.table <- getTable(test.indices, lin.lasso.pred)

coef(lin.lasso.fit, s=best.cv.lin.lasso.lam)
View(lin.lasso.table)

######################################################################

require(tree)

###
# Trees
###

#fit a tree
full.tree.fit <- tree(formula=TierScore~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary+Mega+
                      Dragon+Ghost+Normal+Psychic+Fire+Dark+Steel+Water+Ground+Fighting+Bug+Fairy+Grass+Electric+
                      Rock+Flying+Poison+Ice,
                    data = pokedata, subset=train.indices)

#cross-validate to find the best pruned tree size
set.seed(2019)
prune.tree.cv <- cv.tree(full.tree.fit, FUN=prune.tree)

plot(prune.tree.cv$size, prune.tree.cv$dev, type='b')
best.tree.size <- prune.tree.cv$size[order(prune.tree.cv$dev)[1]]

#build said pruned tree
prune.tree.fit <- prune.tree(full.tree.fit, best=best.tree.size)
prune.tree.pred <- predict(prune.tree.fit, newdata=pokedata[test.indices,])

#Display results
model.test.errors <- c(model.test.errors, getScore(prune.tree.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "Trees"

prune.tree.table <- getTable(test.indices, prune.tree.pred)

plot(prune.tree.fit)
text(prune.tree.fit, pretty=0)
View(prune.tree.table)


#########################################################################################

require(randomForest)

###
# RandomForests
###

set.seed(2019)
rand.forest.fit <- randomForest(formula=TierScore~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary+Mega+
                        Dragon+Ghost+Normal+Psychic+Fire+Dark+Steel+Water+Ground+Fighting+Bug+Fairy+Grass+Electric+
                        Rock+Flying+Poison+Ice,
                      data = pokedata, subset=train.indices)

rand.forest.pred <- predict(rand.forest.fit, newdata=pokedata[test.indices, ])

#Display results
model.test.errors <- c(model.test.errors, getScore(rand.forest.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "RandomForests"

rand.forest.table <- getTable(test.indices, rand.forest.pred)

plot(rand.forest.pred, pokedata$TierScore[test.indices], xlab="Random Forest Prediction (rounded)",
     ylab="Actual Score")
View(rand.forest.table)


##########################################################################################

require(gbm)
###
# Boosted Trees
###

set.seed(2019)
n.boosted.trees=5000
boosted.fit <- gbm(formula=TierScore~HP+Attack+Defense+Sp..Atk+Sp..Def+Speed+Generation+Legendary+Mega+
                                  Dragon+Ghost+Normal+Psychic+Fire+Dark+Steel+Water+Ground+Fighting+Bug+Fairy+Grass+Electric+
                                  Rock+Flying+Poison+Ice,
                                data = pokedata[train.indices,], distribution = "gaussian",
                                n.trees=n.boosted.trees, interaction.depth=4)

boosted.pred <- predict(boosted.fit, newdata=pokedata[test.indices,],n.trees=n.boosted.trees)


#Display results
model.test.errors <- c(model.test.errors, getScore(boosted.pred, pokedata$TierScore[test.indices]))
names(model.test.errors)[length(model.test.errors)] <- "Boosted Trees"

boosted.table <- getTable(test.indices, boosted.pred)

plot(boosted.pred, pokedata$TierScore[test.indices], xlab="Boosted Trees Prediction (rounded)",
     ylab="Actual Score")
summary(boosted.fit)

View(boosted.table)


##########################################################################################

###
# Compare the methods
###
print(model.test.errors)
