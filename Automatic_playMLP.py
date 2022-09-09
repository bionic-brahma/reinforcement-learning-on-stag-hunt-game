# importing the relevant packages
from time import sleep  # to perform the delay in the game
from gym_stag_hunt.envs.gym.hunt import HuntEnv  # For rendering and executing the configuration on the environment
from gym_stag_hunt.src.games.abstract_grid_game import UP, LEFT, DOWN, RIGHT, STAND  # constants for the actions
import numpy as np  # for mathematical convenience in the operations
import joblib  # to save the model
import random  # to get the random integer

# Game grid size (The window interface is (gsize x gsize))
gsize = 5

# Gives the maximum number of moves
MOVES_LIMIT = 1000


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


def learn_behaviour_actions(environment_obs, rewards, noise, gsize):
    """
    This function produces the list containing the 2 elements as actions for agent blue
    and agent red. These actions are given by the prediction done by the model.
    @param environment_obs: game environment object
    @param rewards: previous rewards for the agents
    @param noise: noise tuple
    @param gsize: game grid size
    @return: list of size two containing actions for agent blue and red
    """
    matrixA = observation_matrix_maker(environment_obs, gsize, players_prospective='A')
    matrixB = observation_matrix_maker(environment_obs, gsize, players_prospective='B')
    mlp = joblib.load("modelMLP.pkl")
    rewardA = rewards[0]
    rewardB = rewards[1]
    noiseA = noise[0]
    noiseB = noise[1]
    features_A = list(matrixA.flatten())
    features_B = list(matrixB.flatten())
    features_A.extend([rewardA])
    features_B.extend([rewardB])
    features_A.extend([noiseA])
    features_B.extend([noiseB])
    actionA = mlp.predict([features_A])
    actionB = mlp.predict([features_B])

    agent_A_keyboard_action_code = keyboard_input_to_game_instruction_code(actionA)
    agent_B_keyboard_action_code = keyboard_input_to_game_instruction_code(actionB)

    return [agent_A_keyboard_action_code, agent_B_keyboard_action_code]


if __name__ == "__main__":
    env = HuntEnv(grid_size=(gsize, gsize), obs_type="image", enable_multiagent=True)
    obs = env.reset()
    total_reward = 0.0
    observation_matrix = []
    first_move = False
    actions = None
    rewards = [0.0, 0.0]

    for i in range(MOVES_LIMIT):
        if not first_move:
            first_move = True
            actions = [STAND, STAND]

        else:
            obs, rewards, done, info = env.step(actions=actions)
            total_reward += np.sum(rewards)
            agents_config = [env.get_coords(), env.get_flipped_coord()]

            print("Total Reward in game: ", total_reward)
            sleep(0.2)
            env.render()
            noise_tuple = [random.randint(1, 10), random.randint(1, 10)]
            actions = learn_behaviour_actions(environment_obs=agents_config, rewards=rewards, noise=noise_tuple,
                                              gsize=gsize)

    env.close()
    quit()
