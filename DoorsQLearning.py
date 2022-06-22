import numpy as np
import random
import matplotlib.pyplot as plt
from Doors import Environment
from collections import deque

max_episodes = 5000
ethical_weight = 2.1
# This is an implementation of the Doors environment described in the article
# 'Potential-based multiobjective reinforcement learning approaches to low-impact agents for AI safety'
# by Vamplew et al at Engineering Applications of Artificial Intelligence Volume 100, April 2021, 104186

######
#  MULTIOBJECTIVE Q-LEARNING
#####






######
# FEM EL Q-LEARNING #

def q_learning(weights, original_arnau=False):
    # Learning phase (q-table )
    e = Environment()

    number_of_possible_states = 4*14

    q_table = np.zeros([2, number_of_possible_states, len(e.actions)])

    epsilon_decay = 1.0/max_episodes


    alpha = 0.9 #learning rate
    gamma = 1. #discount factor
    epsilon = 0.7 #epsilon greedy

    original_epsilon = epsilon
    original_alpha = alpha


    max_steps = 30

    ################################
    ## Settings per les gràfiques ##
    
    for_graphics = list()
    
    

    def is_terminal_state(state):
        if state == e.AGENT_GOAL:
            return True
        else:
            return False


    
    def create_policy(q_table):
        
        policy = np.zeros(number_of_possible_states)
        
        V = np.zeros([number_of_possible_states,2])
        
        for state in range(number_of_possible_states):
            
            q_state = q_table[:,state]
            
            scalarised_q = weights[0]*q_state[0] + weights[1]*q_state[1]
            
            index = int(np.argmax(scalarised_q))
            
            policy[state] = index
            
            V[state] = q_table[:,state,index]
        
        return policy, V
        

    for episode in range(1, max_episodes+1):


        if not original_arnau:
            equis = episode * epsilon_decay
            logar = max(0.0, np.log(equis) + 1)

            epsilon = original_epsilon*(1.0 - logar)
            alpha = original_alpha*(1.0 - logar)

        if episode % 1000 == 0:
            print(episode)
        e.env_clean_up()
        e.env_init()
        state = e.env_start()
        
        initial_state = state

        big_r = np.array([0, 0])
        
      
        
        step_count = 0
        
        while not is_terminal_state(e.agent_location) and step_count < max_steps:
            
            step_count += 1
            
            row_agent_state = state_to_row(state)
            
            #epsilon greedy to choose next action
            
            random_value = random.uniform(0,1)
            
            if random_value < epsilon:
                action = np.random.randint(5)  #explore random action
         
                
            else:
                q_state = q_table[:,row_agent_state]
                
                scalarised_q = weights[0]*q_state[0] + weights[1]*q_state[1]
                
                
                action = np.argmax(scalarised_q)
                
                

        
            #perform the action, recive the reward, update q-table, update state
            
            action_paraula = num_a_paraula(action)

            
            old_agent_state = state
            old_row_agent_state = state_to_row(old_agent_state)
            
            rewards, new_state = e.env_step(action_paraula)


            
            new_row_agent_state = state_to_row(new_state)
            
            
            
            
            q_state = q_table[:,new_row_agent_state].copy()
            
            #print(q_state)
            
            scalarised_q = weights[0]*q_state[0] + weights[1]*q_state[1]
            
            #print(scalarised_q, 'això és scalarised q')
            
            index = np.argmax(scalarised_q)
            
            
            #print(q_table[:,new_row_agent_state,index],'q_table new row agent state')
            
            next_max = q_table[:,new_row_agent_state,index]
            
            old_value = q_table[:,old_row_agent_state,action].copy()
            
            
            ########
            
            # Reward escalaritzat a partir del GOAL REWARD i l'IMPACT REWARD
            
            vec_rewards = np.array([0,0])    
            
            vec_rewards[0] = rewards['GOAL_REWARD']
            vec_rewards[1] = rewards['IMPACT_REWARD']
            big_r += vec_rewards
            
            ########
            
            #print(old_value)
            #print(next_max)
            #print(vec_rewards)
            
            
            new_q_value = (1-alpha)*old_value + alpha*(vec_rewards + gamma*next_max)
            
            if is_terminal_state(e.agent_location):
                q_table[:,old_row_agent_state,action] = vec_rewards
            
            else:
                q_table[:,old_row_agent_state,action] = new_q_value
            
            
            
            #print(new_q_value)
            
            state = new_state

    
        #print('Episodi :', episode)
        
        _, V_circumstancial = create_policy(q_table)

        if not original_arnau:
            for_graphics.append(big_r)
        else:
            for_graphics.append(V_circumstancial[state_to_row(initial_state)])
    
        
    policy, V = create_policy(q_table)
        
    np_graphics = np.array(for_graphics)
        
    return policy, V, q_table, np_graphics


def is_terminal_state(e, state):
    if state == e.AGENT_GOAL:
        return True
    else:
        return False
    
    
def state_to_row(observation):
    row_f=0
    for i in range(0,observation[0]):
        row_f +=4
        
    if observation[1] == False and observation[2] == False:
        row_f += 0
    if observation[1] == True and observation[2] == False:
        row_f += 1
    if observation[1] == True and observation[2] == True:
        row_f += 2
    if observation[1] == False and observation[2] == True:
        row_f += 3
    return row_f  


def num_a_paraula(a):
    if a == 0:
        return 'up'
    if a == 1:
        return 'right'
    if a == 2:
        return 'down'
    if a == 3:
        return 'left'
    if a == 4:
        return 'toggle_door'


if __name__ == '__main__':

    weights = [1,ethical_weight]


    policy, V, q_table, np_graphics = q_learning(weights)

    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = V[0]
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))


    x_0 = np_graphics[:,0]
    x_E = np_graphics[:,1]



    reward_online_objectiu = np.mean(x_0)
    print('Reward Online Objectiu = ',reward_online_objectiu)

    reward_online_etic = np.mean(x_E)
    print('Reward Online Ètic = ',reward_online_etic)


    plt.title(label = 'Environment: Doors ($w_E = $ '+ str(ethical_weight) + ')')
    plt.axhline(y=43, color = 'tomato', label = 'Reward individual política ètica')
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

    plt.plot(range(max_episodes), mean_scores0, linewidth=2, markersize=10, marker='s', markevery=500, label="Objectiu individual $V_0$", color='red')
    plt.plot(range(max_episodes), mean_scoresE, linewidth=2, markersize=10, marker='^', markevery=500, label="Objectiu ètic $V_N + V_E$", color='green')
    plt.ylabel("Suma de rewards al final de l'episodi")
    plt.xlabel('Episodi')
    plt.legend(loc='best',fontsize = 'medium')
    plt.show()

