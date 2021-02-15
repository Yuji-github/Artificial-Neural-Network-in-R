# Artificial Neuron Network
library(caTools)

# deep learning 
# install.packages('h2o') # connect computer system like gpu and a lot of options
library(h2o)

# import data
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# encoding 
dataset$Gender = as.numeric(factor(dataset$Gender, 
                        levels = c('Male', 'Female'), 
                        labels = c(0, 1)))
# must set as numerical for deep learning 

dataset$Geography = as.numeric(factor(dataset$Geography, 
                      levels = c('France', 'Spain', 'Germany'), 
                      labels = c(1, 2, 3)))

# splitting dataset
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

# feature scaling: must apply feature scaling for competitions 
train[-11] = scale(train[-11])
test[-11] = scale(test[-11])

# Build ANN

# step 1: requires/establish connections with system or server
h2o.init(nthreads = -1) # -1: all available threads currently

# define classifier 
classifier = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(train), 
                              activation = 'Rectifier', hidden = c(6, 6), 
                              epochs = 100, train_samples_per_iteration = -2)
# y is required 
# training_frame = as.h2o(train): must be h2o data frame 
# activation = 'Rectifier'
# hidden = c(6, 6) c(first hidden, second hidden layers)
# train_samples_per_iteration == batch size; -2 is auto tune

# prediction 
prob_pred = h2o.predict(classifier, newdata=as.h2o(test[-11]))
y_pred = (prob_pred > 0.5) # return as bool
y_pred = as.vector(y_pred) # for confusion matrix 
  
cm = table(test[, 13], y_pred)
print(cm)
