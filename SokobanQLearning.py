from Sokoban import Environment
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

n_episodes = 5000
ethical_weight = 6.1


def scalarisation_function(values, weights):
    f = 0
    for objective in range(len(values)):
        f += weights[objective]*values[objective]

    return f


    
def scalarised_Qs(env, Q_state, weights):

    scalarised_Q = np.zeros(len(env.actions))
    for action in range(len(Q_state)):  
        scalarised_Q[action] = scalarisation_function(Q_state[action], weights)

    return scalarised_Q



    
def choose_action_epsilon_greedy(st, eps, q_table, env, weights):
    
    random_value = random.uniform(0,1)
    
    if random_value < eps:
        return np.random.randint(len(env.actions))
    
    else:
        return np.argmax(scalarised_Qs(env, q_table[st[0], st[1]], weights))
    


def update_q_table(q_table, env, weights, alpha, gamma, action, state, new_state, reward):
    
    best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1]], weights))
    
    q_table[state[0], state[1], action] += alpha * ( reward + gamma*( q_table[new_state[0], new_state[1], best_action]) - q_table[state[0], state[1], action]) 
         
    return q_table[state[0], state[1], action]

        
def deterministic_optimal_policy_calculator(Q, env, weights): 
    policy = np.zeros([env.NUM_CELLS, env.NUM_CELLS])
    V = np.zeros([env.NUM_CELLS, env.NUM_CELLS, env.NUM_OBJECTIVES - 1])

    for cell_L in range(env.NUM_CELLS):
        for cell_R in range(env.NUM_CELLS):
            if cell_L != cell_R:

                best_action = np.argmax(scalarised_Qs(env, Q[cell_L,cell_R], weights))
                                        
                policy[cell_L, cell_R] = best_action
                V[cell_L, cell_R] = Q[cell_L, cell_R, best_action]
    return policy, V


def is_terminal_state(env,state):
    if state[0] == env.AGENT_GOAL:
        return True
    else:
        return False


def num_a_paraula(action_number):
    if action_number == 0:
        return 'up'
    if action_number == 1:
        return 'right'
    if action_number == 2:
        return 'down'
    if action_number == 3:
        return 'left'





######################################################
######################################################
######################################################


def q_learning(weights, alpha = 1.0, gamma = 1.0, epsilon = 0.9, max_episodes = 5000, original_arnau=False):
    
    env = Environment()
    epsilon_decay = 1.0/max_episodes

    original_epsilon = epsilon
    original_alpha = alpha
    
    ### Settings related to Sokovan
    n_objectives = env.NUM_OBJECTIVES - 1
    n_cells = env.NUM_CELLS
    n_actions = len(env.actions)
    
    
    ### Settings for Q-learning
    max_steps = 50
    Q = np.zeros([n_cells, n_cells, n_actions, n_objectives])
    
    
    ################################
    ## Settings per les gràfiques ##
    
    for_graphics = list()
    
    ### Algorithm starts here
    
    
    for episode in range(1,max_episodes+1):
        print(episode)
        env.env_clean_up()
        env.env_init()
        state = env.env_start()
        initial_state = state
        
        step_count = 0

        big_r = np.array([0.0, 0.0])



        if not original_arnau:
            equis = episode * epsilon_decay
            logar = max(0.0, np.log(equis) + 1)

            epsilon = original_epsilon*(1.0 - logar)
            alpha = original_alpha*(1.0 - logar)
        #print(epsilon)
        
        while not env.is_terminal() and step_count < max_steps:
            
            step_count += 1
            
            action_done_numero = choose_action_epsilon_greedy(state, epsilon, Q, env, weights)
            
            action_done_paraula = num_a_paraula(action_done_numero)
            
            rewards, new_state = env.env_step(action_done_paraula)
            
            
            vec_rewards = np.array([0,0])    
            vec_rewards[0] = rewards['GOAL_REWARD']
            vec_rewards[1] = rewards['IMPACT_REWARD']

            big_r += vec_rewards
            

            new_q_value = update_q_table(Q, env, weights, alpha, gamma, action_done_numero, state, new_state, vec_rewards)

            if env.is_terminal():
                Q[state[0],state[1],action_done_numero] = vec_rewards
            
            else:
                Q[state[0],state[1],action_done_numero] = new_q_value
            
            state = new_state
        
        #print('Episodi ', episode,' finalitzat')
        
        #####################
        ## DADES GRÀFIQUES ##
        #####################
        
        q = Q[initial_state[0],initial_state[1]].copy()

        sq = scalarised_Qs(env, q, weights)

        a = np.argmax(sq)

        # TODO: If you want to print the evolution of V(s_0), append q[a] in for_graphics instead of big_r
        print(q[a])

        appendable = q[a]

        if not original_arnau:
            appendable = big_r
        for_graphics.append(appendable)

        ###################
        
    # Output a deterministic optimal policy and its associated Value table
    policy, V = deterministic_optimal_policy_calculator(Q, env, weights)
    
    np_graphics = np.array(for_graphics)
    
    return policy, V, Q, np_graphics


if __name__ == "__main__":

    weights = [1, ethical_weight]


    policy, V, Q, np_graphics = q_learning(weights, max_episodes=n_episodes)


    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = V[1,3]
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))

    ## GENEREM LES GRÀFIQUES ##

    x_0 = np_graphics[:,0]
    x_E = np_graphics[:,1]



    reward_online_objectiu = np.mean(x_0)
    print('Reward Online Objectiu = ',reward_online_objectiu)

    reward_online_etic = np.mean(x_E)
    print('Reward Online Ètic = ',reward_online_etic)


    plt.title(label = 'Environment: Sokoban ($w_E = $ '+ str(ethical_weight) + ')')
    plt.axhline(y=40, color = 'tomato', label = 'Reward individual política ètica')
    plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')


    ### Addition by Manel to create clearer graphics

    mean_scores0 = list()
    scores0 = deque(maxlen=250)
    for point in x_0:
        scores0.append(point)
        mean_score = np.mean(scores0)
        mean_scores0.append(mean_score)

    mean_scoresE = list()
    scoresE = deque(maxlen=250)
    for point in x_E:
        scoresE.append(point)
        mean_score = np.mean(scoresE)
        mean_scoresE.append(mean_score)

    for i in range(10):
        mean_scores0[i] = 0.0
    plt.plot(range(n_episodes), mean_scores0, linewidth=2, markersize=10, marker='s', markevery=500, label="Objectiu individual $V_0$", color='red')
    plt.plot(range(n_episodes), mean_scoresE, linewidth=2, markersize=10, marker='^', markevery=500, label="Objectiu ètic $V_N + V_E$", color='green')
    plt.ylabel("Suma de rewards al final de l'episodi")
    plt.xlabel('Episodi')
    plt.legend(loc='best',fontsize = 'medium')
    plt.show()
