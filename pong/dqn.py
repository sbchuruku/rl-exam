import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
from pong import Env

class DQN_Agent :
    def __init__(self, env) :
        # 하이퍼 파라미터        
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        
        self.memory = deque(maxlen=10000)
        
        self.state_size = 2 + (len(env.balls) * 4)
        self.action_size = env.n_actions
        self.model = self.build_model()
        self.target_model = self.build_model()
    
    def load_model(self, episode) :
        self.model.load_weights('d:\\rl_data\\dqn\\dqn_trained_{}.h5'.format(episode))
    
    def build_model(self) :
        model = Sequential()
        model.add(Dense(self.state_size * 2, input_dim=self.state_size, activation='relu', 
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.state_size * 2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model
    
    def update_target_model(self) :
        self.target_model.set_weights(self.model.get_weights())
        
    def get_action(self, state) :
        if np.random.rand() <= self.epsilon :
            return random.randrange(self.action_size)
        else :
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])
        
    def append_sample(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))
    
    def train_model(self) :
        if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay
            
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = list(), list(), list()
        
        for i in range(self.batch_size) :
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)
        
        for i in range(self.batch_size) : 
            if dones[i] :
                target[i][actions[i]] = rewards[i]
            else :
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
        
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        
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
    env.addBall(1)

    agent = DQN_Agent(env)
    
    scores, episodes = list(), list()
    
    if load_episode != 1 :
        agent.load_model(load_episode)
    
    for e in range(load_episode,EPISODES) :
        done = False
        score = 0
        state = env.reset()
        state = agent.state_flatten(env, state)
        
        while True :
            env.render(isLearning)
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = agent.state_flatten(env, next_state)
        
            agent.append_sample(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.train_start :
                agent.train_model()
                
            score += reward
            state = next_state
            
            if done :
                agent.update_target_model()
                
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('d:\\rl_data\\dqn\\dqn_result.png')
                print('episode:{} / catch:{} / step:{} / epsilon:{}'
                      .format(e, env.catch_cnt, env.step_cnt, agent.epsilon))
                
                if e % 100 == 0 :
                    agent.model.save_weights('d:\\rl_data\\dqn\\dqn_trained_{}.h5'.format(e))
                break
    
    
    
    