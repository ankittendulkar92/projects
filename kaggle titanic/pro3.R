
# set random seed
set.seed(42)

setwd("C:/project 1")
library(readr)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(ROCR)
library(caTools)

train <- read_csv("C:/project 1/train.csv")

# view structure of train and get summary
str(train)
summary(train)

# we will use pclass, sex,age,sibsp,parch to see survival 
# get general idea about the variables using plot and density

# get plot for age
plot(density(train$Age, na.rm = TRUE))
x11()

# generate barplot for sex and pclass
survival_sex<-table(train$Survived,train$Sex)
barplot(survival_sex,xlab="sex",ylab="survivors",main="survival by gender")
x11()

survival_pclass<-table(train$Survived,train$Pclass)
barplot(survival_pclass,xlab="class",ylab="survivors",main="survival by class")
x11()


survival_sibsp<-table(train$Survived,train$SibSp)
barplot(survival_sibsp,xlab="number of siblings",ylab="survived",main="survival by sibling/spouse")
x11()

survival_parch<-table(train$Survived,train$Parch)
barplot(survival_parch,xlab="number of parents",ylab="survived",main="survival by parents/children on board")
x11()

survival_embark<-table(train$Survived,train$Embarked)
barplot(survival_embark,xlab="port",ylab="survived",main="survival by port of embarkment")
x11()

train<-train[-c(1,9:11)]
#cleaning data set
# find the number of missing values in the variable columns
length(which(is.na(train$Age)))
length(which(is.na(train$Sex)))
length(which(is.na(train$Pclass)))
length(which(is.na(train$SibSp)))
length(which(is.na(train$Parch)))
length(which(is.na(train$Embarked)))
#replace na values in age column
# we see that all the data has titles like mr,mrs,miss,master and dr
# we use the mean of these titles to replace the missing values in age 

#first create vectors with indices of titles like mr,mrs,mis,mas 
mr_vec<-grep("Mr.",train$Name,fixed=TRUE)
mrs_vec<-grep("Mrs.",train$Name,fixed=TRUE)
miss_vec<-grep("Miss.",train$Name,fixed=TRUE)
master_vec<-grep("Master.",train$Name,fixed=TRUE)
dr_vec<-grep("Dr.",train$Name,fixed=TRUE)

# get mean value of vectors
master_mean=round(mean(train[master_vec,]$Age,na.rm=TRUE))
miss_mean=round(mean(train[miss_vec,]$Age,na.rm=TRUE))
mr_mean=round(mean(train[mr_vec,]$Age,na.rm=TRUE))
mrs_mean=round(mean(train[mrs_vec,]$Age,na.rm=TRUE))
dr_mean=round(mean(train[dr_vec,]$Age,na.rm=TRUE))

# rename the people only by their titles
train[master_vec,]$Name="Master"
train[miss_vec,]$Name="Miss"
train[mr_vec,]$Name="Mr"
train[mrs_vec,]$Name="Mrs"
train[dr_vec,]$Name="Dr"

# set the missing values to the corresponding mean values
for (i in 1:nrow(train)){ 
  if (is.na(train[i,5])) {
    if (train$Name[i] == "Master") {
      train$Age[i] = master_mean
    } else if (train$Name[i] == "Miss") {
      train$Age[i] = miss_mean
    } else if (train$Name[i] == "Mrs") {
      train$Age[i] = mrs_mean
    } else if (train$Name[i] == "Mr") {
      train$Age[i] = mr_mean
    } else if (train$Name[i] == "Dr") {
      train$Age[i] = dr_mean
    } else {
      print("Uncaught Title")
    }
  }
}


train<-train[-c(3)]


#first we randomise the data
# randomise row index
row<-sample(nrow(train))
#reorder data
train<-train[row,]

#split the data
#determine split row index
split_row<-round(nrow(train)*0.8)

train_data<-train[1:split_row,]

test_data<-train[(split_row+1):nrow(train),]
# build the tree
dec_tree<-rpart(Survived~.,train_data,method="class",control=rpart.control(cp=0.01),parms = list(split="information"))

# draw tree

fancyRpartPlot(dec_tree)
x11()
#save survived column as labels
train_labels<-train_data$Survived
test_labels<-test_data$Survived

#remove survived column from test data
test_data2<-test_data[-c(1)]

# predict surviavl for test data
pred_dectree<-predict(dec_tree,test_data2,type="class")

#create confusion matrix
conf_dec<-table(test_data$Survived,pred_dectree)
conf_dec

#find accuracy
accuracy_dec<-sum(diag(conf_dec))/sum(conf_dec)
accuracy_dec

## optimise knn

train_knn<-train_data[-c(1)]
test_knn<-test_data2

# scaling and normalizing data
#scaling pclass
min_class <- min(train_knn$Pclass)
max_class <- max(train_knn$Pclass)
train_knn$Pclass <- (train_knn$Pclass - min_class) / (max_class - min_class)
test_knn$Pclass <- (test_knn$Pclass - min_class) / (max_class - min_class)

# scaling Age
min_age <- min(train_knn$Age)
max_age <- max(train_knn$Age)
train_knn$Age <- (train_knn$Age - min_age) / (max_age - min_age)
test_knn$Age <- (test_knn$Age - min_age) / (max_age - min_age)

# scaling sibsp
min_sibsp <- min(train_knn$SibSp)
max_sibsp <- max(train_knn$SibSp)
train_knn$SibSp<- (train_knn$SibSp - min_sibsp) / (max_sibsp - min_sibsp)
test_knn$SibSp <- (test_knn$SibSp - min_sibsp) / (max_sibsp - min_sibsp)

# scaling parch
min_parch <- min(train_knn$Parch)
max_parch <- max(train_knn$Parch)
train_knn$Parch <- (train_knn$Parch - min_parch) / (max_parch - min_parch)
test_knn$Parch <- (test_knn$Parch - min_parch) / (max_parch - min_parch)

train_knn[is.na(train_knn)]<-0
test_knn[is.na(test_knn)]<-0

library(class)
# changing class male , female to values 0,1
train_knn$Sex<-gsub("female",0,train_knn$Sex)
train_knn$Sex<-gsub("^male",1,train_knn$Sex)

test_knn$Sex<-gsub("^male",1,test_knn$Sex)
test_knn$Sex<-gsub("female",0,test_knn$Sex)

#changing values S,Q,C to 0,1,2
train_knn$Embarked<-gsub("S",1,train_knn$Embarked)
train_knn$Embarked<-gsub("^Q",0,train_knn$Embarked)
train_knn$Embarked<-gsub("^C",2,train_knn$Embarked)

test_knn$Embarked<-gsub("S",1,test_knn$Embarked)
test_knn$Embarked<-gsub("^Q",0,test_knn$Embarked)
test_knn$Embarked<-gsub("^C",2,test_knn$Embarked)

range <- 1:round(0.2 * nrow(train_knn))
accs <- rep(0, length(range))

for (k in range) {
  #make prediction using k neighbours
  pred_knn <- knn(train_knn, test_knn,train_labels, k = k)
  
  #  construct the confusion matrix: conf
  conf_knn <- table(test_labels,pred_knn)
  # calculate the accuracy and store it in accs[k]
  accs[k] <- sum(diag(conf_knn))/sum(conf_knn)
}

# Plot the accuracies. Title of x-axis is "k".
plot(range, accs, xlab = "k")
x11()
which.max(accs)

acc_kn<-accs[43]

train_knn[,7]<-train_data[,1]

# use cross validation with knn to calculate 



ctrl <- trainControl(method="repeatedcv",repeats = 5) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Survived ~ ., data = train_knn, method = "knn", trControl = ctrl)
knnPredict <- predict(knnFit,newdata = test_knn )



colAUC(knnPredict,test_labels,plotROC=TRUE)
b<-as.data.frame(knnPredict)
for(i in 1:nrow(b)){
  if(b[i,1]>0.8458){
    b[i,1]=1}
  else{b[i,1]=0}
}

b<-b[[1]]

cong<-table(b,test_labels)
acc_n<-sum(diag(cong))/sum(cong)
acc_n
