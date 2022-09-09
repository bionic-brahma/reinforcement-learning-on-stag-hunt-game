# importing the relevant packages
import random  # to produce random integers
from time import sleep  # to perform the delay in the game
from gym_stag_hunt.envs.gym.hunt import HuntEnv  # For rendering and executing the configuration on the environment
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND  # constants for the actions
import numpy as np  # for mathematical convenience in the operations
import pandas as pd  # to handle the dataframes and files writing and reading

######################## HANDLES THE SOCKET COMMUNICATION#####################
import socket  # for socket communication
import json

SERVER = socket.gethostbyname(socket.gethostname())  # getting the localhost
PORT = 5051  # The port used by the server
record = None
USE_SOCKET_COMMUNICATION = True
##############################################################################
# Game grid size (The window interface is (gsize x gsize))
gsize = 5

# Gives the maximum number of moves
MOVES_LIMIT = 1000

# Render delay
RENDER_DELAY = 0.4

# if true, training data will be recorded in a file
RECORD_DATA = True

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
    else:
        i = 100
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


if __name__ == "__main__":
    env = HuntEnv(grid_size=(gsize, gsize), obs_type="image", enable_multiagent=True)
    obs = env.reset()
    total_reward = 0.0
    observation_matrix = []
    first_move = False
    actions = None
    rewards = [0.0, 0.0]

    # performing the game-play
    if USE_SOCKET_COMMUNICATION:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER, PORT))
            for i in range(MOVES_LIMIT):
                if not first_move:
                    first_move = True
                    actions = [STAND, STAND]
                else:
                    obs, rewards, done, info = env.step(actions=actions)
                    total_reward += np.sum(rewards)
                    agents_config = (
                        tuple([float(i) for i in env.get_coords()]), tuple([float(i) for i in env.get_flipped_coord()]))

                    print("Total Reward in game: ", total_reward)

                    sleep(RENDER_DELAY)
                    env.render()

                    print("The only moves allowed are w,s,d,a.\nPress wrong moves for exit.")
                    action_for_A = manual_input("Move for Agent Blue: ")
                    action_for_B = manual_input("Move for Agent Red: ")
                    actions = [action_for_A, action_for_B]
                    if action_for_B + action_for_B > 99:
                        exit(1)

                    if RECORD_DATA:
                        record = [agents_config[0], agents_config[1], gsize, int(action_for_A), int(action_for_B),
                                  rewards]
                        record = json.dumps(record).encode()
                        s.sendall(record)
            env.close()
            quit()
    else:
        for i in range(MOVES_LIMIT):
            if not first_move:
                first_move = True
                actions = [STAND, STAND]
            else:
                obs, rewards, done, info = env.step(actions=actions)
                total_reward += np.sum(rewards)
                agents_config = [env.get_coords(), env.get_flipped_coord()]

                print("Total Reward in game: ", total_reward)

                sleep(RENDER_DELAY)
                env.render()

                print("The only moves allowed are w,s,d,a.\nPress wrong moves for exit.")
                action_for_A = manual_input("Move for Agent Blue: ")
                action_for_B = manual_input("Move for Agent Red: ")
                actions = [action_for_A, action_for_B]
                if action_for_B + action_for_B > 99:
                    exit(1)

                if RECORD_DATA:
                    record_training_data(environment_obs=agents_config, gsize=gsize, actionA=action_for_A,
                                         actionB=action_for_B,
                                         rewards=rewards, file_to_save="trainingData/training_data2.dat")
