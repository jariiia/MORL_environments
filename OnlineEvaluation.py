import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def generate_graphic_for_environment(environment):

    if environment == "Doors":
        from DoorsQLearning import q_learning, ethical_weight
    elif environment == "Sokoban":
        from SokobanQLearning import q_learning, ethical_weight
    elif environment == "Breakable":
        from Breakable_bottlesQLearning import q_learning, ethical_weight
    elif environment == "Unbreakable":
        from Unbreakable_bottlesQLearning import q_learning, ethical_weight

    weights = [1, ethical_weight]


    n_runs = 3

    max_episodes = 5000

    llista_runs_0 = []
    llista_runs_E = []


    for run in range(n_runs):
        _, _, _, np_graphics = q_learning(weights)
        llista_runs_0.append(np_graphics[:,0])
        llista_runs_E.append(np_graphics[:,1])

    mitjana_runs_0 = []
    mitjana_runs_E = []

    for episode in range(max_episodes):
        suma_0 = 0
        suma_E = 0
        for run in range(n_runs):
            suma_0 += llista_runs_0[run][episode]
            suma_E += llista_runs_E[run][episode]
        mitjana_runs_0.append(suma_0/n_runs)
        mitjana_runs_E.append(suma_E/n_runs)


    mitjana_total_0 = np.mean(mitjana_runs_0)
    mitjana_total_E = np.mean(mitjana_runs_E)

    print('Mitjana total rw0 = ', mitjana_total_0)
    print('Mitjana total rwE = ', mitjana_total_E)

    variance_0 = np.var(mitjana_runs_0)
    variance_E = np.var(mitjana_runs_E)

    print('Variança_0 = ', variance_0)
    print('Variança_E = ', variance_E)


    ## GENEREM LES GRÀFIQUES ##

    mean_scores0 = list()
    scores0 = deque(maxlen=250)
    for point in mitjana_runs_0:
        scores0.append(point)
        mean_score = np.mean(scores0)
        mean_scores0.append(mean_score)

    mean_scoresE = list()
    scoresE = deque(maxlen=250)
    for point in mitjana_runs_E:
        scoresE.append(point)
        mean_score = np.mean(scoresE)
        mean_scoresE.append(mean_score)

    plt.title(label = 'Environment: ' + environment + '($w_E = $ '+ str(ethical_weight) + ')')

    individual_limit = -99999

    if environment == "Doors":
        individual_limit = 43
    elif environment == "Sokoban":
        individual_limit = 40
    elif environment == "Breakable":
        individual_limit = 36
    elif environment == "Unbreakable":
        individual_limit = 43.7

    plt.axhline(y=individual_limit, color = 'tomato', label = 'Reward individual política ètica')
    plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')
    plt.plot(range(max_episodes), mean_scores0, linewidth=2, markersize=10, marker='s', markevery=500, label="Mitjana 20 runs objectiu individual $V_0$", color='red')
    plt.plot(range(max_episodes), mean_scoresE, linewidth=2, markersize=10, marker='^', markevery=500, label="Mitjana 20 runs objectiu ètic $V_N + V_E$", color='green')
    plt.ylabel("Suma de rewards al final de l'episodi")
    plt.xlabel('Episodi')
    plt.legend(loc='center right',fontsize = 'medium')
    plt.show()


if __name__ == '__main__':

    env_name = "Breakable"
    generate_graphic_for_environment(env_name)
