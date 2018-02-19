import numpy as np
import random
from collections import defaultdict
from pingpong import Env
import csv
import pickle

class SARSA_agent:
    def __init__(self, actions) :
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda:[0.0, 0.0, 0.0])
        self.episode = 0
        
    def learn(self, state, action, reward, next_state, next_action) :
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    def get_action(self, state) :
        if np.random.rand() < self.epsilon :
            action = np.random.choice(self.actions)
        else :
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action
    
    @staticmethod
    def arg_max(state_action) :
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action) :
            if value > max_value :
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value :
                max_index_list.append(index)
        return random.choice(max_index_list)

    # def savedata(self, episode, step_cnt, catch_cnt):
    #     with open('D:\\rl_data\\sarsa_q_table_{}.pkl'.format(episode), 'wb') as f :
    #         pickle.dump(self.q_table, f)
    #
    #     with open("D:\\rl_data\\sarsa_result.csv", 'a', newline='') as f :
    #         writer = csv.writer(f, delimiter=',')
    #         writer.writerow([episode, step_cnt, catch_cnt])
    #
    #     print("save data in episode {0}.".format(episode))
    #
    # def loaddata(self, episode):
    #     with open('D:\\rl_data\\sarsa_q_table_{}.pkl'.format(episode), 'rb') as f:
    #         self.q_table = pickle.load(f)
    #
    #     print('Load Success! Start at episode {0}'.format(episode))

if __name__ == '__main__':

    env = Env()

    agent = SARSA_agent(actions=list(range(env.n_actions)))

    env.addBall()

    for episode in range(1000) :
        agent.episode = episode + 1
        print('{} episode -----------'.format(agent.episode))
        env.reset()

        while True :

            state, reward, done = env.render()

            action = agent.get_action(state)

            next_state = env.step(action)

            next_action = agent.get_action(next_state)

            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            if done :
                print('step cnt:{}  catch_cnt:{}'.format(env.step_cnt, env.catch_cnt))
                break
            
            
            
            
            
            
            