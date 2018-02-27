import copy
import pylab
import random
import numpy as np
from pong import Env
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Sequential
import pickle

class DeepSARSA_agent :
    def __init__(self, env) :
        # 액션 정의
        self.action_space = ['left','right','none']
        self.action_size = len(self.action_space)
        # 하이퍼 파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        
        self.state_size = 1 + (len(env.balls) * 4)
        self.model = self.build_model()
        
    def load_model(self, episode) :
        # weight, epsilon 로드
        self.model.load_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained_{}.h5'.format(episode))
        with open('d:\\rl_data\\deep_sarsa\\epsilon_{}.bin'.format(episode),'rb') as f :
            self.epsilon = pickle.load(f)
    
    def build_model(self) :
        model = Sequential()
        model.add(Dense(self.state_size * 2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size * 2, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model
    
    def get_action(self, state) :
        if np.random.rand() <= self.epsilon :
            # 무작위 행동 선택
            return random.randrange(self.action_size)
        else :
            # 모델로 부터 행동 산출
            state = np.float32(state)
            q_values = self.model.predict(state)
            # [[],[],[]] 에서 [],[],[] 로 가져오기 위해 0 을 붙인다.
            return np.argmax(q_values[0])
    
    def train_model(self, state, action, reward, next_state, next_action, done) :
        # 타임스탭 진행에 따라 epsilon 값을 줄임
        if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay
        
        state = np.float32(state)
        next_state = np.float32(next_state)
        # 예측값
        target = self.model.predict(state)[0]
        
        if done :
            # 에피소드 끝나면 즉각 보상
            target[action] = reward
        else :
            # 실제값
            target[action] = (reward + self.discount_factor *
                  self.model.predict(next_state)[0][next_action])
        
        # flatten
        target = np.reshape(target, [1,3])
        # 인공신경망을 업데이트하는데 오류함수를 감소시키는 방향으로 한번 인공신경망 업데이트
        self.model.fit(state, target, epochs=1, verbose=0)
    
    # state 를 신경망에 넣기 위해 flatten    
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
    # 사용자 파라미터
    EPISODES = 10000
    isLearning = True
    load_episode = 1
    
    # 환경 생성
    env = Env()
    env.addBall(1)
    
    # agent 생성
    agent = DeepSARSA_agent(env)
    
    # 에피소드 1부터 시작하지 않으려면 로드
    if load_episode != 1 :
        agent.load_model(load_episode)
    
    # 그래프를 그리기 위한 변수    
    scores, episodes = list(), list()

    for episode in range(load_episode,EPISODES) :
        state = env.reset()
        state = agent.state_flatten(env,state)

        while True :
            
            env.render(isLearning)
            
            action = agent.get_action(state)

            next_state,reward,done = env.step(action)
            
            next_state = agent.state_flatten(env, next_state)

            next_action = agent.get_action(next_state)

            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            state = copy.deepcopy(next_state)

            if done :
                scores.append(env.catch_cnt)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig('d:\\rl_data\\deep_sarsa\\deep-sarsa.png')
                print('episode:{} / step:{} / catch:{} / epsilon:{}'
                      .format(episode, env.step_cnt, env.catch_cnt, agent.epsilon))
                
                env.catch_cnt = 0
                env.step_cnt = 0
                
                if episode % 100 == 0 :
                    # 학습된 모델 weight 와 epsilon 저장
                    agent.model.save_weights('d:\\rl_data\\deep_sarsa\\deep_sarsa_trained_{}.h5'.format(episode))
                    with open('d:\\rl_data\\deep_sarsa\\epsilon_{}.bin'.format(episode),'wb') as f :
                        pickle.dump(agent.epsilon,f)
                break