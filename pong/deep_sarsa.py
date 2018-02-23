import copy
import pylab
import random
import numpy as np
from pong import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

class DeepSARSA_agent :
    def __init__(self) :
        self.action_space = [0,1,2]
        self.action_size = len(self.action_space)
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        
        self.all_catch_cnt = 0
        self.all_step_cnt = 0
        
    def load_model(self, episode) :
        self.model.load_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained{}.h5'.format(episode))
    
    def set_statesize(self, env) :
        self.state_size = 2 + (len(env.balls) * 4)
        self.model = self.build_model()
    
    def build_model(self) :
        model = Sequential()
        model.add(Dense(self.state_size * 2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size * 2, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def get_action(self, state) :
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        else :
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
    
    def train_model(self, state, action, reward, next_state, next_action, done) :
        if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay
        
        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        
        if done :
            target[action] = reward
        else :
            target[action] = (reward + self.discount_factor *
                  self.model.predict(next_state)[0][next_action])
        
        target = np.reshape(target, [1,3])
        self.model.fit(state, target, epochs=1, verbose=0)
        
    def state_flatten(self, env, state) :
        res = list()
        res.append(state[0])
        idx = 1
        for i in range(len(env.balls)*2) :
            for j in range(len(state[idx])) :
                res.append(state[idx][j])
            idx += 1
        res.append(state[-1])
        
        return np.reshape(res,[1,self.state_size])

if __name__ == '__main__' :
    
    EPISODES = 10000
    isLearning = True
    load_episode = 1
    
    env = Env()
    env.addBall(2)
    
    agent = DeepSARSA_agent()
    
    agent.setStateSize(env)
    
    if load_episode != 1 :
        agent.load_model(load_episode)
        
    scores, episodes = list(), list()

    for episode in range(load_episode,EPISODES) :
        agent.episode = episode
        env.reset()

        while True :
            
            state, reward, done = env.render(isLearning)
            
            state = agent.state_flatten(env,state)

            action = agent.get_action(state)

            next_state = env.step(action)
            
            next_state = agent.state_flatten(env, next_state)

            next_action = agent.get_action(next_state)

            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            state = copy.deepcopy(next_state)


            if done :
                agent.all_step_cnt += env.step_cnt
                agent.all_catch_cnt += env.catch_cnt
                
                scores.append(agent.all_catch_cnt)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('d:\\rl_data\\deep_sarsa\\deep-sarsa.png')
                print('episode:{} / score:{} / step:{} / epsilon:{}'
                      .format(episode, agent.all_catch_cnt, agent.all_step_cnt, agent.epsilon))
                agent.all_catch_cnt = 0
                agent.all_step_cnt = 0
                
                if episode % 50 == 0 :
                    agent.model.save_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained_{}.h5'.format(episode))
                
                break