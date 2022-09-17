# importing necessary packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.exceptions import ConvergenceWarning
import numpy as np

# reading the training datafile
trainingData = pd.read_csv("trainingData/training_data2.dat")

# taking the features
x = trainingData.drop(['action'], axis=1).to_numpy()
# taking the target
y = trainingData["action"]

# initializing multi layered perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(18, 14, 7),
    max_iter=20,
    alpha=1e-5,
    solver="sgd",
    verbose=1,
    random_state=2,
    learning_rate_init=0.01,
    n_iter_no_change=5000,
)

##
##
##  The learning curve ploting begins here
##
##
# cross validation splits
number_of_split = 5
# number of splits in dataset
number_of_subsets = 50
sizes, training_scores, testing_scores = learning_curve(mlp, x, y, cv=number_of_split, scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, number_of_subsets))

# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted green line is for training scores and red line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="g", label="Training score")
plt.plot(sizes, mean_testing, color="r", label="Cross-validation score")
# plt.plot(sizes, Standard_Deviation_training, '--', color="r", label="Training std score")
# plt.plot(sizes, Standard_Deviation_testing, color="r", label="Cross-validation std score")

# Drawing plot
plt.title("LEARNING CURVE FOR MLP")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

##
##
##  The learning curve ploting ends here
##
##


# splitting to have testing and training data
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3,
                                                    stratify=y)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # training the neural net
    mlp.fit(X_train, y_train)

# saving the model
joblib.dump(mlp, 'modelMLP.pkl')

# printing the model evaluation
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
predicted_controls = mlp.predict(X_test)
print(classification_report(y_test, predicted_controls))
