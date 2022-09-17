# importing necessary packages
from NeuralNet.MultiLayerPerceptron import MLP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import KFold

# reading the training data
trainingData = pd.read_csv("trainingData/training_data2.dat")

# getting the features
x = trainingData.drop(['action'], axis=1).to_numpy()

# taking the target and converting it fo dummies
target = pd.get_dummies(trainingData["action"])
# taking the names order in the order
order = target.columns

# taking the target
y = target.to_numpy()

##
##
##  The learning curve ploting begins here
##
##

# cross validation splits
number_of_split = 5
train_accuracies = []
test_accuracies = []
# number of splits in dataset
number_of_subsets = 50
for i in len(x) * np.linspace(0.01, 1.0, number_of_subsets):
    x_i = x[:int(i)]
    y_i = y[:int(i)]
    if len(x_i) > number_of_split:
        kf = KFold(n_splits=number_of_split)
        kf.get_n_splits(x_i)
        train_k_accuracies = []
        test_k_accuracies = []
        for train_index, test_index in kf.split(x_i):
            X_train, X_test = x_i[train_index], x_i[test_index]
            y_train, y_test = y_i[train_index], y_i[test_index]
            # number of input neurons
            n = len(X_train[0])
            # number of the output neurons
            o = len(y_train[0])
            # number of neurons in hidden layer
            hidden = 7
            # number of epochs
            epochs = 2000
            # initializing the NN
            mpl = MLP(n=n, m=hidden, o=o, input_array=X_train,
                      output_array=y_train)
            # training
            c = mpl.train(epochs=epochs, eta=3)

            preds_train = mpl.predict(X_train)
            predicted_controls_train = [order[p] for p in preds_train]
            actual_controls_train = [order[k] for k in np.argmax(y_train, axis=1)]
            train_k_accuracies.append(accuracy_score(actual_controls_train, predicted_controls_train))
            # testing the NN
            preds = mpl.predict(X_test)
            predicted_controls = [order[p] for p in preds]
            actual_controls = [order[k] for k in np.argmax(y_test, axis=1)]
            # displaying the accuracies
            test_k_accuracies.append(accuracy_score(actual_controls, predicted_controls))
        train_accuracies.append(np.mean(train_k_accuracies))
        test_accuracies.append(np.mean(test_k_accuracies))
    else:
        print("The dataset is very small to perform the K-fold. [skipping it]")
x_axis_plot = np.linspace(1, len(train_accuracies), len(train_accuracies), dtype=int)

# dotted green line is for training scores and red line is for cross-validation score
plt.plot(x_axis_plot, train_accuracies, '--', color="g", label="Training score")
plt.plot(x_axis_plot, test_accuracies, color="r", label="Cross-validation score")

# Drawing plot
plt.title("LEARNING CURVE FOR CUSTOMNN")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

##
##
##  The learning curve ploting ends here
##
##

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# number of input neurons
n = len(X_train[0])

# number of the output neurons
o = len(y_train[0])

# number of neurons in hidden layer
hidden = 7

epochs = 20000

# initializing the NN
mpl = MLP(n=n, m=hidden, o=o, input_array=X_train,
          output_array=y_train)
# training
c = mpl.train(epochs=epochs, eta=3)

# testing the NN
preds = mpl.predict(X_test)
predicted_controls = [order[p] for p in preds]
actual_controls = [order[k] for k in np.argmax(y_test, axis=1)]

# displaying the report
print(classification_report(actual_controls, predicted_controls))

# saving the models
joblib.dump(order, 'order.pkl')
joblib.dump(mpl, 'model.pkl')

x_axis = np.linspace(1, epochs, epochs, dtype=int)
plt.plot(x_axis, c, label="Loss")
plt.legend()
plt.title("Cost depreciation Vs Epochs ")
plt.show()
