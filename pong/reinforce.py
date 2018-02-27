import copy
import pylab
import numpy as np
from pong import Env
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import backend as K
import pickle

class Reinforce_Agent :
    def __init__(self, env) :
        self.action_space = ['left','right','none']
        self.action_size = len(self.action_space)
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        
        self.state_size = 1 + (len(env.balls) * 4)
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        
        self.states, self.actions, self.rewards = list(), list(), list()
    
    def load_model(self, episode) :
        self.model.load_weights('d:\\rl_data\\reinforce\\reinforce_trained_{}.h5'.format(episode))
        with open('d:\\rl_data\\reinforce\\epsilon_{}.bin'.format(episode),'rb') as f :
            self.epsilon = pickle.load(f)
    
    def build_model(self) :
        model = Sequential()
        model.add(Dense(self.state_size * 2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size * 2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model
    
    def build_optimizer(self) :
        action = K.placeholder(shape=[None,3])
        discount_rewards = K.placeholder(shape=[None,])
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discount_rewards
        loss = -K.sum(cross_entropy)
        
        optimizer = RMSprop(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discount_rewards],[],updates=updates)
        return train
    
    def get_action(self, state) :
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    def discount_rewards(self, rewards) :
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0,len(rewards))) :
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
        
    def append_sample(self, state, action, reward) :
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
    
    def train_model(self) :
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = list(), list(), list()

    def state_flatten(self, env, state) :
        res = list()
        res.append(state[0])
        idx = 1
        for i in range(len(env.balls)*2) :
            for j in range(len(state[idx])) :
                res.append(state[idx][j])
            idx += 1
        
        return np.reshape(res,[1,self.state_size])

if __name__ == '__main__' :
    
    EPISODES = 10000
    isLearning = True
    load_episode = 1
    
    env = Env()
    env.addBall(1)
    
    agent = Reinforce_Agent(env)
    
    if load_episode != 1 :
        agent.load_model(load_episode)
        
    scores, episodes = list(), list()

    for episode in range(load_episode,EPISODES) :
        agent.episode = episode
        state = env.reset()
        state = agent.state_flatten(env,state)
        
        while True :
            
            env.render(isLearning)

            action = agent.get_action(state)

            next_state, reward, done = env.step(action)
            
            next_state = agent.state_flatten(env, next_state)

            next_action = agent.get_action(next_state)
            
            agent.append_sample(state, action, reward)

            state = next_state
            state = copy.deepcopy(next_state)

            if done :
                
                agent.train_model()
                
                scores.append(env.catch_cnt)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('d:\\rl_data\\reinforce\\reinforce.png')
                print('episode:{} / step:{} / catch:{}'.format(episode, env.step_cnt, env.catch_cnt))
                
                env.catch_cnt = 0
                env.step_cnt = 0
                
                if episode % 100 == 0 :
                    agent.model.save_weights('d:\\rl_data\\reinforce\\reinforce_trained_{}.h5'.format(episode))
                    with open('d:\\rl_data\\reinforce\\epsilon_{}.bin'.format(episode),'wb') as f :
                        pickle.dump(agent.epsilon,f)
                break