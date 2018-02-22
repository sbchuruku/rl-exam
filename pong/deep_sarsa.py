import copy
import pylab
import random
import numpy as np
from pong import Env
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential

EPISODES = 10000

class DeepSARSA_agent :
    def __init__(self) :
        self.load_model = False
        self.action_space = [0,1,2]
        self.action_size = len(self.action_space)
        self.state_size = 6
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        
        self.episode = 0
        self.all_catch_cnt = 0
        self.all_step_cnt = 0
        
        if self.load_model :
            self.model.load_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained.h5')
        
    def build_model(self) :
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
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

if __name__ == '__main__' :
    env = Env()
    
    agent = DeepSARSA_agent()

    env.addBall()

    isLearning = False

    scores, episodes = list(), list()

    for episode in range(EPISODES) :
        print('{} episode -----------'.format(episode))
        env.reset()

        while True :
            
            state, reward, done = env.render(isLearning)

            state = np.reshape(state,[1,6])

            action = agent.get_action(state)

            next_state = env.step(action)
            
            next_state = np.reshape(next_state,[1,6])

            next_action = agent.get_action(next_state)

            #agent.learn(state, action, reward, next_state)

            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            state = copy.deepcopy(next_state)

            if done :
                print('step cnt:{}  catch cnt:{}'.format(env.step_cnt, env.catch_cnt))
                agent.all_catch_cnt += env.step_cnt
                agent.all_step_cnt += env.catch_cnt
                
                scores.append(agent.all_catch_cnt)
                episodes.append(agent.episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('d:\\rl_data\\deep_sarsa\\deep-sarsa.png')
                print('episode:{} / score:{} / step:{} / epsilon:{}'
                      .format(agent.episode, agent.all_catch_cnt, agent.all_step_cnt, agent.epsilon))
                
                if episode % 100 == 0 :
                    agent.all_catch_cnt = 0
                    agent.all_step_cnt = 0
                    agent.model.save_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained.h5')
                
                agent.episode = episode + 1
                
                break