import socket
import json
import random
import pandas as pd
import numpy as np
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND  # constants for the actions
from NeuralNet.MultiLayerPerceptron import MLP
from sklearn.metrics import classification_report
import joblib
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

datafile = "trainingData/serverCaught.dat"
training_threshold = 20  # training will happen is number of records in datafile
# reach to the multiple of training_threshold
train = False
PORT = 5051
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
disconnect_message = 'Disconnected'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

gsize = 5
# Columns for storing the grid value for the environment
number_of_columns = gsize * gsize
col = []

# Naming the attribute names for the training data file
for i in range(number_of_columns):
    col.append("env" + str(i))

# Column name for the reward
col.append("reward")

# Column name for random noise to increase the entropy
col.append("noise")

# Column name of the target variable. In our case it is the action taken by the agent
col.append("action")

# Creating the empty dataframe to store the training data
trainingDF = pd.DataFrame(columns=col)


def keyboard_input_to_game_instruction_code(i):
    """
    This function returns the abstract._grid_game constant by taking
    keyboard input (w,s,a,d,x).
    @param i: keyboard input (w,s,a,d,x)
    @return: abstract._grid_game constant
    """
    if i in ["w", "W"]:
        i = UP
    elif i in ["a", "A"]:
        i = LEFT
    elif i in ["s", "S"]:
        i = DOWN
    elif i in ["d", "D"]:
        i = RIGHT
    elif i in ["x", "X"]:
        i = STAND
    return i


def game_instruction_code_to_keyboard_input(code):
    """
    This function takes the abstract._grid_game constant and returns
    keyboard input (w,s,a,d,x) for agent.
    @param code: abstract._grid_game constant
    @return: character from the list [w,s,d,a,x]
    """
    i = 'x'
    if code == UP:
        i = 'w'
    elif code == LEFT:
        i = 'a'
    elif code == DOWN:
        i = 's'
    elif code == RIGHT:
        i = 'd'
    elif code == STAND:
        i = 'x'
    return i


def manual_input(message):
    """
    This function facilitates the manual game play by
    providing console as the interaction node.
    It returns the abstract._grid_game constant
    @param message: Message input for user.
    @return: abstract._grid_game constant
    """
    i = input(message)
    i = keyboard_input_to_game_instruction_code(i)
    return i


def observation_matrix_maker(obs, gsize, players_prospective='A'):
    """
    This function gives the game state in the form of the game environment matrix which contain the
    rectangular grid kind of structure with the position of entities in the matrix specified as
    1 for agent blue, 2 for agent red, 3 for the stag and others for the grasses
    @param obs: environment observation
    @param gsize: grid size of the game ground
    @param players_prospective: Agent on whose prospective you want to look environment
    @return: grid_size x grid_size matrix representing the game environment
    """
    game_matrix = np.zeros((gsize, gsize))  # creating the empty game matrix
    obs_tuples_player_A = [(obs[0][x], obs[0][x + 1]) for x in
                           range(0, len(obs[0]), 2)]  # environment with respect to agents blue perception
    obs_tuples_player_B = [(obs[1][x], obs[1][x + 1]) for x in
                           range(0, len(obs[1]), 2)]  # environment with respect to agents red perception

    # imputing the entities location of the game matrix w.r.t agents blue prospective
    if players_prospective == 'A':

        for index, t in enumerate(obs_tuples_player_A):
            if index == 0:
                game_matrix[t[0]][t[1]] = 1
            elif index == 1:
                game_matrix[t[0]][t[1]] = 2
            elif index == 2:
                game_matrix[t[0]][t[1]] = 3
            else:
                game_matrix[t[0]][t[1]] = 4

    # imputing the entities location of the game matrix w.r.t agents red prospective
    if players_prospective == 'B':

        for index, t in enumerate(obs_tuples_player_B):
            if index == 0:
                game_matrix[t[0]][t[1]] = 1
            elif index == 1:
                game_matrix[t[0]][t[1]] = 2
            elif index == 2:
                game_matrix[t[0]][t[1]] = 3
            else:
                game_matrix[t[0]][t[1]] = 4

    return game_matrix


def record_training_data(environment_obs, gsize, actionA, actionB, rewards, file_to_save="training_data.dat"):
    """
    This function writes the training data action by action. The file is saved in the specified
    @param environment_obs: Game environment object
    @param gsize: Grid size for the game ground
    @param actionA: Action taken by the agent blue after getting the game environment
    @param actionB: Action taken by the agent red after getting the game environment
    @param rewards: Previous reward
    @param file_to_save: The address of the file where the training data needs to be saved
    """
    matrixA = observation_matrix_maker(environment_obs, gsize,
                                       players_prospective='A')  # getting game mtrix w.r.t. aganet blue
    matrixB = observation_matrix_maker(environment_obs, gsize,
                                       players_prospective='B')  # getting game mtrix w.r.t. aganet blue
    rewardA = rewards[0]  # reward for agent blue
    rewardB = rewards[1]  # reward for agent red
    noise_tuple = [random.randint(1, 10), random.randint(1, 10)]  # noise tuple for agents move

    # matrixA and mtrixB are storing the observation of the game prior to actionA and actionB has taken

    # converting matrix to features
    features_A = list(matrixA.flatten())
    features_B = list(matrixB.flatten())

    # taking actions of the agents
    agent_A_keyboard_action_code = game_instruction_code_to_keyboard_input(actionA)
    agent_B_keyboard_action_code = game_instruction_code_to_keyboard_input(actionB)

    # preprocessing
    features_A.extend([rewardA])
    features_A.extend([noise_tuple[0]])
    features_A.extend(agent_A_keyboard_action_code)
    features_B.extend([rewardB])
    features_B.extend([noise_tuple[1]])
    features_B.extend(agent_B_keyboard_action_code)

    trainingDF.loc[len(trainingDF)] = features_A
    trainingDF.loc[len(trainingDF)] = features_B

    # writing the file
    print("Writting the training data file.")
    trainingDF.to_csv(file_to_save, index=False)
    return len(trainingDF)


def start():
    global conn
    df_size = 0
    print("Server is starting.")
    print("Listening to the server: ", SERVER)
    connected = True

    while connected:
        server.listen()
        conn, addr = server.accept()
        print("New connection: ", addr, " connected")

        while True:
            data = conn.recv(1024)
            if not data:
                break
            data = json.loads(data)
            print("data received:", data)
            df_size = record_training_data(environment_obs=[float_to_int(data[0]), float_to_int(data[1])],
                                           gsize=data[2], actionA=data[3],
                                           actionB=data[4],
                                           rewards=data[-1], file_to_save=datafile)
        print(df_size)
        if df_size % training_threshold == 0:
            train = True
        if not data:
            connected = False

    conn.close()


def training_by_customNN(datafile):
    # reading the training data
    trainingData = pd.read_csv(datafile)

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
    joblib.dump(order, 'order_auto.pkl')
    joblib.dump(mpl, 'model_auto.pkl')


def traing_by_MLP(datafile):
    # reading the training datafile
    trainingData = pd.read_csv(datafile)

    # taking the features
    x = trainingData.drop(['action'], axis=1).to_numpy()
    # taking the target
    y = trainingData["action"]

    # splitting to have testing and training data
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3,
                                                        stratify=y)
    # initializing multi layered perceptron
    mlp = MLPClassifier(
        hidden_layer_sizes=(10, 7,),
        max_iter=200000,
        alpha=1e-5,
        solver="sgd",
        verbose=1,
        random_state=2,
        learning_rate_init=0.02,
        n_iter_no_change=5000,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        # training the neural net
        mlp.fit(X_train, y_train)

    # saving the model
    joblib.dump(mlp, 'modelMLP_auto.pkl')

    # printing the model evaluation
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))
    predicted_controls = mlp.predict(X_test)
    print(classification_report(y_test, predicted_controls))


def float_to_int(l):
    return [int(i) for i in l]


if __name__ == '__main__':
    start()
    if train:
        training_by_customNN(datafile=datafile)
        train = False
