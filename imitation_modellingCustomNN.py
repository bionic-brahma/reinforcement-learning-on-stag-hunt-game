# importing necessary packages
from NeuralNet.MultiLayerPerceptron import MLP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# reading the training data
trainingData = pd.read_csv("trainingData/training_data.dat")

# getting the features
x = trainingData.drop(['action'], axis=1).to_numpy()

# taking the target and converting it fo dummies
target = pd.get_dummies(trainingData["action"])
# taking the names order in the order
order = target.columns

# taking the target
y = target.to_numpy()

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
joblib.dump(order,'order.pkl')
joblib.dump(mpl, 'model.pkl')

x_axis = np.linspace(1, epochs, epochs, dtype=int)
plt.plot(x_axis, c, label="Loss")
plt.legend()
plt.title("Cost depreciation Vs Epochs ")
plt.show()

