import numpy as np
import random
from pong import Env
from collections import defaultdict
import csv

class QLearning_Agent :
    def __init__(self, actions) :
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda:[0.0,0.0,0.0])
        
        self.episode = 0
        self.all_catch_cnt = 0
        self.all_step_cnt = 0
    
    def learn(self, state, action, reward, next_state) :
        q_value = self.q_table[state][action]
        q_new = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_new - q_value)

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
    
    def savedata(self, env):
        Fn = open('D:\\rl_data\\q_learning\\q_table_{}.csv'.format(self.episode), 'w', newline='')
        writer = csv.writer(Fn, delimiter=',')
        writer.writerow([self.episode])
        writer.writerow([self.all_step_cnt])
        writer.writerow([self.all_catch_cnt])
        keys = self.q_table.keys()
        
        for key in keys:
            res = list()
            res.append(key[0])
            idx = 1
            for i in range(len(env.balls)*2) :
                for j in range(len(key[idx])) :
                    res.append(key[idx][j])
                idx += 1
            res.append(key[-1])
            
            for q in self.q_table[key] :
                res.append(q)
                        
            writer.writerow(res)
        
        Fn.close()

        Fn = open("D:\\rl_data\\q_learning\\result.csv", 'a', newline='')
        writer = csv.writer(Fn, delimiter=',')
        writer.writerow([self.episode, self.all_step_cnt, self.all_catch_cnt])
        Fn.close()

        print("save data in episode {0}.".format(self.episode))
        
    def loaddata(self, episode):
        try:
            Fn = open('D:\\rl_data\\q_learning\\q_table_{}.csv'.format(episode), 'r')
            self.episode = int(Fn.readline().split(',')[0])
            self.all_step_cnt = int(Fn.readline().split(',')[0])
            self.all_catch_cnt = int(Fn.readline().split(',')[0])
            reader = csv.reader(Fn, delimiter=',')
            
            for key in reader:
                makeKey = list()
                makeKey.append(int(float(key[0])))
                
                data = list()
                for i in range(1,(len(key)-3)+1):
                    data.append(key[i])
                    if i % 2 == 0 :
                        makeKey.append(tuple(data))
                        data.clear()
                
                makeKey.append(int(float(key[-4])))
                
                value = [float(key[-3]),float(key[-2]),float(key[-1])]
                self.q_table[tuple(makeKey)] = value
               
            print('Load Success! Start at episode {0}'.format(episode))
        except Exception:
            print('Load Failed!')
            
if __name__ == '__main__':

    env = Env()
    env.addBall(1)

    agent = QLearning_Agent(actions=list(range(env.n_actions)))

    load_episode = 1
    isLearning = True

    if load_episode > 1 :
        agent.loaddata(load_episode)
    
    for episode in range(load_episode,10000) :
        agent.episode = episode
        state = env.reset()

        while True :
            
            env.render(isLearning)

            action = agent.get_action(state)

            next_state, reward, done = env.step(action)

            next_action = agent.get_action(next_state)

            agent.learn(state, action, reward, next_state)

            state = next_state

            if done :
                print('episode:{} / step:{} / catch:{}'.format(episode, env.step_cnt, env.catch_cnt))
                agent.all_catch_cnt += env.step_cnt
                agent.all_step_cnt += env.catch_cnt
                
                if episode % 500 == 0 :
                    agent.savedata(env)
                    agent.all_catch_cnt = 0
                    agent.all_step_cnt = 0
                
                break