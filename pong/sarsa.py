import numpy as np
import random
from collections import defaultdict
from pong import Env
import csv

class SARSA_agent:
    def __init__(self, actions) :
        self.actions = actions
        # hyper parameter 설정
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        # 딕셔너리를 만드는데 value값을 [0.0, 0.0, 0.0] 으로 받겠다.
        self.q_table = defaultdict(lambda:[0.0, 0.0, 0.0])
        
        # 저장시 합계를 구하기 위한 변수
        self.all_catch_cnt = 0
        self.all_step_cnt = 0
        
    # 큐함수 구현
    def learn(self, state, action, reward, next_state, next_action) :
        # q table에서 state 와 action 을 넣어서 현재 q값을 구한다.
        current_q = self.q_table[state][action]
        # 다음 state의 q값을 구한다
        next_state_q = self.q_table[next_state][next_action]
        # 살사의 큐함수 업데이트 식에 대입하여 새로운 q값 구한다.
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q))
        # 현재 상태와 현재 액션 시 q값을 q_table에 저장한다.
        self.q_table[state][action] = new_q

    # e-greedy(epsilon + greedy) 알고리즘으로 action 을 return하는 함수 
    def get_action(self, state) :
        if np.random.rand() < self.epsilon :
            action = np.random.choice(self.actions)
        else :
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action
    
    # 각 방향에 대한 가치 중 최고의 값의 인덱스를 리턴해주는 함수
    @staticmethod
    def arg_max(state_action) :
        max_index_list = []
        # max_value 값 초기화
        max_value = state_action[0]
        # 3바퀴 loop 
        for index, value in enumerate(state_action) :
            if value > max_value :
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value :
                max_index_list.append(index)
        # max 값은 같은데 max 값을 가진 index 가 여러개 일 때 그중에 하나의 index를 random으로 리턴한다.
        return random.choice(max_index_list)
    
    # 에피소드로 저장하는 함수
    def savedata(self, episode, env):
        Fn = open('D:\\rl_data\\sarsa\\q_table_{}.csv'.format(episode), 'w', newline='')
        writer = csv.writer(Fn, delimiter=',')
        # 에피소드
        writer.writerow([episode])
        # q_table 의 키(state)들
        keys = self.q_table.keys()
        
        for key in keys:
            res = list()
            # 패들의 x 좌표값
            res.append(key[0])
            # 공의 갯수만큼 위치(x,y) 값, 속도(방향) pair 로 2개씩 생기고 그 정보에 대한 idx 값으로
            # 저장하기위한 반복문
            idx = 1
            for i in range(len(env.balls)*2) :
                for j in range(len(key[idx])) :
                    res.append(key[idx][j])
                idx += 1
            
            # 위에서 적용된 키값에 대한 value 저장([0.0,0.0,0.0] 형태 이므로 반복문을 사용해 세개 다 저장)
            for q in self.q_table[key] :
                res.append(q)
            
            writer.writerow(res)
        
        Fn.close()

        Fn = open("D:\\rl_data\\sarsa\\result.csv", 'a', newline='')
        writer = csv.writer(Fn, delimiter=',')
        # 에피소드, 에피소드당 스탭수, 에피소드당 패들의 히트수 저장
        writer.writerow([episode, self.all_step_cnt, self.all_catch_cnt])
        Fn.close()

        print("save data in episode {0}.".format(episode))
    
    # 원하는 에피소드 파일을 불러오는 함수    
    def loaddata(self, episode):
        try:
            Fn = open('D:\\rl_data\\sarsa\\q_table_{}.csv'.format(episode), 'r')
            self.episode = int(Fn.readline().split(',')[0])
            reader = csv.reader(Fn, delimiter=',')
            
            for key in reader:
                # print(key)
                # ['340.0', '363.0', '5.0', '3', '3', '0.0', '-1.02', '0.0']
                # q_table 의 키를 만드는 과정
                makeKey = list()
                makeKey.append(int(float(key[0])))
                
                data = list()
                for i in range(1,(len(key)-3)+1):
                    data.append(key[i])
                    if i % 2 == 0 :
                        makeKey.append(tuple(data))
                        data.clear()
                
                # q_table 의 value 를 만드는 과정
                value = [float(key[-3]),float(key[-2]),float(key[-1])]
                # q_table 에 key 와 value 로 저장
                self.q_table[tuple(makeKey)] = value
               
            print('Load Success! Start at episode {0}'.format(episode))
        except Exception:
            print('Load Failed!')

# 실행부         
if __name__ == '__main__':

    # 환경생성
    env = Env()
    # 공 추가
    env.addBall(1)

    # agent 생성
    agent = SARSA_agent(actions=list(range(env.n_actions)))

    # 로드할 에피소드
    load_episode = 1
    # 훈련 flag
    isLearning = True

    # 로드할 에피소드가 1보다 크면 로드함수 
    if load_episode > 1 :
        agent.loaddata(load_episode)
    
    # 에피소드 수행
    for episode in range(load_episode,10000) :
        # 에피소드 시작시 reset & 현재 state 가져오기  
        state = env.reset()

        while True :
            # 현재 state 대로 그리기
            env.render(isLearning)
            
            # 현재 action = state 를 입력해서 q_table 의 value 를 argmax 한 것
            action = agent.get_action(state)
            
            # 한 타임스탭 진행 후 next_state, reward, done 가져오기
            next_state, reward, done = env.step(action)
            
            # next_state 로 next_action 구하기
            next_action = agent.get_action(next_state)

            # 공식에 의한 학습
            agent.learn(state, action, reward, next_state, next_action)
            
            # 현재 state, action 값 갱신
            state = next_state
            action = next_action

            # 에피소드 종료
            if done :
                # 결과 확인용 출력
                print('episode:{} / step:{} / catch:{}'.format(episode, env.step_cnt, env.catch_cnt))
                # 저장용 데이터 변수에 값 갱신
                agent.all_step_cnt += env.step_cnt
                agent.all_catch_cnt += env.catch_cnt                
                
                # 100 에피소드 마다 데이터 저장
                if episode % 500 == 0 :
                    agent.savedata(episode,env)
                    # 저장후 초기화
                    agent.all_step_cnt = 0
                    agent.all_catch_cnt = 0
                
                break
                        