# importing necessary packages
import pandas as pd
from sklearn.metrics import classification_report
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# reading the training datafile
trainingData = pd.read_csv("trainingData/training_data.dat")

# taking the features
x = trainingData.drop(['action'], axis=1).to_numpy()
# taking the target
y = trainingData["action"]

# splitting to have testing and training data
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3,
                                                    stratify=y)
# initializing multi layered perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(10,7,),
    max_iter=200000,
    alpha=1e-5,
    solver="sgd",
    verbose=1,
    random_state=2,
    learning_rate_init=0.02,
    n_iter_no_change= 5000,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # training the neural net
    mlp.fit(X_train, y_train)

# saving the model
joblib.dump(mlp, 'modelMLP.pkl')

# printing the model evaluation
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
predicted_controls=mlp.predict(X_test)
print(classification_report(y_test, predicted_controls))

