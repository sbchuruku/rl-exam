# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:29:32 2018

@author: stu
"""


from tkinter import *
import random
import time


PADDLE_HEIGHT = 270

PADDLE_MOVE = [-100, 100]

class PongGame : 
    
    def __init__(self, canvas) :
        self.canvas = canvas
        self.balls = list()
        self.canvas_height = canvas.winfo_height()

    def addPaddle(self, paddle) :
        self.paddle = paddle

    def addBall(self, ball):
        self.balls.append(ball)

    def reset(self) :
        self.paddle.reset()
        for ball in self.balls :
            ball.reset()
    
    def state(self):
        paddle_pos = self.canvas.coords(self.paddle.id)
        
        res = []
        res.append(int(paddle_pos[0]))
        
        for ball in self.balls :
            ball_pos = self.canvas.coords(ball.id) 
            res.append(int(ball_pos[0]))
            res.append(int(ball_pos[1]))
            res.append(ball.x)
            res.append(ball.y)
       
        return res
    
    def move(self,action):
        reward=0
        done=False
        x = PADDLE_MOVE[action]
        self.paddle.setPos(x)
        self.paddle.draw()
        for ball in self.balls :
            ball.draw()
            ball_pos = self.canvas.coords(ball.id)
            if self.is_hit_paddle(ball_pos) :
                ball.setPos(random.randrange(8,15), -random.randrange(8,15))
                reward=1
            
            if self.is_no_hit(ball_pos) :
                reward=0
                done=True
                    
        return done, reward

    def is_hit_paddle(self,pos) :
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2] :
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3] :
                return True
        return False

    
    def is_no_hit(self,pos) :
        if pos[3] >= self.canvas_height:
            return True
        return False


class Ball:
    
    def __init__(self, canvas, paddle, color,idd, speed=random.randrange(8,15), x=random.randrange(8,15)) :
        self.canvas = canvas
        self.canvas_width = canvas.winfo_width()
        self.canvas_height = canvas.winfo_height()
        self.color=color
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color) 
        self.speed = speed    
        self.canvas.move(self.id, self.canvas_width/4 + idd*30, self.canvas_height/8+ idd*30)
        
        self.start_x = x
        self.start_speed = speed
        self.start_pos = self.canvas.coords(self.id)
        
        self.idd=idd
        starts = [-speed,speed]
        random.shuffle(starts)
        self.x = starts[0]
        random.shuffle(starts)
        self.y = -starts[0]
    
    def reset(self) :
        speed=random.randrange(8,15)
        starts = [-speed,speed]
        random.shuffle(starts)
        self.x = starts[0]
        speed=random.randrange(8,15)
        starts = [-speed,speed]
        random.shuffle(starts)
        self.y = starts[0]
        self.setPos(self.x,-self.y) 
  
        self.canvas.coords(self.id, self.start_pos)
      
        self.canvas.update()
    
    def draw(self):
        self.canvas.move(self.id, self.x, self.y)
        pos = self.canvas.coords(self.id)
        if pos[1] <= 0 :
            self.y = self.speed
        if pos[3] >= self.canvas_height :
            self.y = -self.speed
        if pos[0] <= 0 :
            self.x = self.speed
        if pos[2] >= self.canvas_width :
            self.x = -self.speed

    def setPos(self,x,y) :
        self.x = x
        self.y = y

    def setAgent(self, agent) :
        self.agent = agent
        
class Paddle:

    def __init__(self,canvas,color) :
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,7,fill=color)
        self.canvas.move(self.id, 100, PADDLE_HEIGHT)
        self.x = 0
        self.y = 0
        self.start_pos = self.canvas.coords(self.id)
        self.canvas_width = self.canvas.winfo_width()

    
    def reset(self) :
        self.canvas.coords(self.id, self.start_pos)

    def draw(self):
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0 :
            return
        elif pos[2] >= self.canvas_width and self.x > 0 :
            return
        self.canvas.move(self.id, self.x, self.y)
                
    def setPos(self, x):
        self.x = x
    
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading
import random
import time

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"


# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self,state_size, action_size):
        # 상태크기와 행동크기를 갖고옴
        self.state_size = state_size
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 8

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()
        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수만큼 Agent 클래스 생성
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(60 * 10)
            self.save_model("./save_model/breakout_a3c")

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        input = Input(shape=[1,self.state_size])
        fc = Dense(256,  activation='elu',
                        kernel_initializer='he_uniform')(input)
        fc = Dense(128,  activation='elu',
                        kernel_initializer='he_uniform')(fc)
        fc = Flatten()(fc)
        fc = Dense(64, activation='elu',
                        kernel_initializer='he_uniform')(fc)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 20
        self.t = 0

    def run(self):
        global episode

        state_size = 13
        action_size = 2

        step = 0
        
        tk = Tk()
        tk.title("My Ping Pong !") 
        tk.resizable(0, 0)
        tk.wm_attributes("-topmost", 1)
        
        canvas = Canvas(tk, width=300, height=300, bd=0, highlightthickness=0, bg='black')
        canvas.pack()
        
        tk.update()
        
        game = PongGame(canvas)
        
        paddle = Paddle(canvas, 'white')
        
        game.addPaddle(paddle)
        
        ball1 = Ball(canvas, paddle, 'yellow',idd=1)
        ball2 = Ball(canvas, paddle, 'blue',idd=2)
        ball3 = Ball(canvas, paddle, 'red',idd=3)
        
        game.addBall(ball1)
        game.addBall(ball2)
        game.addBall(ball3)
        
        score=0

        while episode < EPISODES:
            done = False

            game.reset()

 
            state = game.state()
            state = np.reshape(state, [1,1, state_size])

            while not done:
                step += 1
                self.t += 1
                action,policy = self.get_action(state)
                
                done,reward = game.move(action)

                next_state = game.state() 
         
                next_state = np.reshape(next_state, [1,1, state_size])


                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(state / 255.)))

                score += reward
                # 샘플을 저장
                self.append_sample(state, action, reward)

      
                state = next_state
                
                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0
                tk.update_idletasks()
                tk.update() 
                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:",
                          step)
                    score=0
                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), 1,13))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        input = Input(shape=[1,self.state_size])
        fc = Dense(256,  activation='elu',
                        kernel_initializer='he_uniform')(input)
        fc = Dense(128,  activation='elu',
                        kernel_initializer='he_uniform')(fc)
        
        fc = Flatten()(fc)
        fc = Dense(64, activation='elu',
                        kernel_initializer='he_uniform')(fc)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, state):
        state = np.float32(state / 255.)
        
        policy = self.local_actor.predict(state)[0]
        policy = np.reshape(policy, [self.action_size])
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]

        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)



if __name__ == "__main__":
    global_agent = A3CAgent(state_size=13, action_size=2)
    global_agent.train()