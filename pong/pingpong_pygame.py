import pygame, sys, random
from pygame.locals import *

# set up the window
WINDOWWIDTH = 500
WINDOWHEIGHT = 400

# set up the colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

class Env:
    def __init__(self,width=WINDOWWIDTH,height=WINDOWHEIGHT) :
        self.windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
        pygame.display.set_caption('My Ping Pong !')
        pygame.init()
        self.mainClock = pygame.time.Clock()
        
        self.action_space = ['l', 'r', 'n']
        self.n_actions = len(self.action_space)
        
        self.balls = list()
        
        self.catch_cnt = 0
        self.step_cnt = 0
        self.reward = 0
        self.done = False
    
    def render(self) :
        self.windowSurface.fill(BLACK)
        self.paddle.update()
        for ball in self.balls :
            pygame.draw.rect(ball.rect)
            ball.update()
        
        pygame.display.update()
        self.mainClock.tick(50)
        return self.getstate(self.paddle.x), self.reward, self.done
    
    def getstate(self,movement):
#        paddle_pos = self.canvas.coords(self.paddle.id)
#        res = []
#        res.append(paddle_pos[0])
#        for ball in self.balls :
#            ball_pos = self.canvas.coords(ball.id) 
#            res.append((ball_pos[0],ball_pos[1]))
#            res.append((ball.x,ball.y))
#        res.append(movement)
#        return tuple(res)
        pass
    
    def reset(self) :
        pass
    
    def addPaddle(self) :
        self.paddle = Paddle(WHITE)
        
    def addBall(self) :
        speeds = [1,2,3]
        random.shuffle(speeds)
        self.balls.append(Ball(self.paddle,speed=speeds[0]))

class Ball:
    def __init__(self, paddle, speed=3, direction=random.randrange(1,3)) :
        self.paddle = paddle
        self.speed = speed    
        
        self.start_dir = direction
        self.start_speed = speed
        
        self.x = direction
        self.y = -speed
        
        self.rect = pygame.Rect(245,200,10,10)
        
    def reset(self) :
        starts = [-self.speed,self.speed]
        random.shuffle(starts)
        self.x = starts[0]
    
    def update(self):
        pass

    def hit_paddle(self,pos) :
        pass
    
    def is_paddle_hit(self) :
        pass
    
    def is_bottom_hit(self) :
        pass

    def setPos(self,x,y) :
        self.x = x
        self.y = y
        
class Paddle:
    def __init__(self, color) :
        self.x = 0
        self.y = 0
        
        self.rect = pygame.Rect(0,0,100,10)
        
    def reset(self) :
        pass

    def update(self):
        pass
        
    def setPos(self, x):
        self.x = x