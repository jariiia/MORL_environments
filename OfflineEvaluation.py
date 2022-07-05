
import numpy as np
import matplotlib.pyplot as plt


def generate_graphics_offline(environment):

    if environment == "Doors":
        from DoorsQLearning import q_learning, ethical_weight, state_to_row, num_a_paraula, Environment
    elif environment == "Sokoban":
        from SokobanQLearning import q_learning, ethical_weight, num_a_paraula, Environment
    elif environment == "Breakable":
        from Breakable_bottlesQLearning import q_learning, ethical_weight, obs_to_state, num_a_paraula, Environment
    elif environment == "Unbreakable":
        from Unbreakable_bottlesQLearning import q_learning, ethical_weight, obs_to_state, num_a_paraula, Environment


    weights = [1, ethical_weight]
    policy, V, q_table, _ = q_learning(weights)

    suma_rewards = list()
    suma_rewards.append([0, 0])

    reward_objectiu = 0
    reward_etic = 0
    number_of_actions = 0

    e = Environment()

    observation = e.env_start()

    while not e.is_terminal():

        if environment == "Doors":
            action_number = policy[state_to_row(observation)]
        elif environment == "Sokoban":
            action_number = policy[observation[0], observation[1]]
        else:
            action_number = policy[obs_to_state(observation, e)]

        action_word = num_a_paraula(action_number)
        rewards, observation = e.env_step(action_word)

        reward_objectiu += rewards['GOAL_REWARD']
        reward_etic += rewards['IMPACT_REWARD']
        suma_rewards.append([reward_objectiu, reward_etic])
        number_of_actions += 1

    print("\nIs terminal?", e.is_terminal())

    suma_rewards = np.array(suma_rewards)

    x_0 = suma_rewards[:, 0]
    x_E = suma_rewards[:, 1]

    n_of_actions_grafica = range(number_of_actions + 1)

    x_T = []

    for i in n_of_actions_grafica:
        x_T.append(x_0[i] + x_E[i])

    plt.title(label='Environment: ' + environment + ' ($w_E = $ ' + str(ethical_weight) + ')')

    individual_limit = -999999

    if environment == "Doors":
        individual_limit = 43
    elif environment == "Sokoban":
        individual_limit = 40
    elif environment == "Breakable":
        individual_limit = 36
    elif environment == "Unbreakable":
        individual_limit = 43.7

    plt.axhline(y=individual_limit, color = 'tomato', label = 'Reward individual política ètica')
    plt.axhline(y=0, color='palegreen', label='Reward ètic política ètica')
    plt.plot(n_of_actions_grafica, x_0, linewidth=2, markersize=10, marker='s', label="Reward objectiu", color='red')
    plt.plot(n_of_actions_grafica, x_E, linewidth=2, markersize=10, marker='^', label="Reward ètic", color='green')
    plt.plot(n_of_actions_grafica, x_T, markersize=10, marker='o', label="Reward total", color='black')
    plt.ylabel("Reward acumulat per acció")
    plt.xlabel("Número d'accions")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':

    env_name = "Breakable"
    generate_graphics_offline(env_name)
