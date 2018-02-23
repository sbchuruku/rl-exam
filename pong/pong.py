from tkinter import *
import random
import time

PADDLE_MOVE = [-15, 15, 0]

class Env :
    def __init__(self) :
        self.action_space = ['left', 'right', 'none']
        self.n_actions = len(self.action_space)
        self.tk = Tk()
        self.tk.title("My Ping Pong !") 
        self.tk.resizable(0, 0)
        self.tk.wm_attributes("-topmost", 1)
        self.canvas = Canvas(self.tk, width=500, height=400, bd=0, highlightthickness=0, bg='black')
        self.canvas.pack()
        self.tk.update()

        self.catch_cnt = 0
        self.step_cnt = 0
        self.reward = 0
        self.done = False

        self.paddle = Paddle(self.canvas, 'white')
        self.balls = list()
        
    def reset(self) :
        for ball in self.balls :
            ball.reset()
        self.paddle.reset()
        self.step_cnt = 0
        self.catch_cnt = 0
        self.done = False
        rand = random.choice([0,1,2])
        return self.getstate(PADDLE_MOVE[rand])

    def addBall(self, count) :
        for i in range(count) :
            colors = ['red','green','blue','white','yellow','orange']
            random.shuffle(colors)
            speeds = [3,4,5]
            random.shuffle(speeds)
            self.balls.append(Ball(self.canvas, self.paddle, colors.pop(0),speeds[0]))

    def getstate(self,movement):
        paddle_pos = self.canvas.coords(self.paddle.id)
        res = []
        res.append(paddle_pos[0])
        for ball in self.balls :
            ball_pos = self.canvas.coords(ball.id) 
            res.append((ball_pos[0],ball_pos[1]))
            res.append((ball.x,ball.y))
        res.append(movement)
        return tuple(res)
        
    def render(self, isLearning=True) :
        self.paddle.draw()
        for ball in self.balls :
            ball.draw()
                
        self.tk.update_idletasks()
        self.tk.update() 
        
        if isLearning == False :
            time.sleep(0.01)

    def step(self, action) :
        self.step_cnt += 1
        if action == 0 or action == 1 or action == 2 :
            self.paddle.setPos(PADDLE_MOVE[action])
        else:
            rand = random.choice([0,1,2])
            self.paddle.setPos(PADDLE_MOVE[rand])
            
        next_state = self.getstate(self.paddle.x)
        
        for ball in self.balls :
            ball.draw()
            if ball.is_bottom_hit():
                self.reward -= 1
                self.done = True
            elif ball.is_paddle_hit():
                self.reward += 1
                self.done = False
                self.catch_cnt += 1
        
        return next_state, self.reward, self.done
        
class Ball:
    
    def __init__(self, canvas, paddle, color, speed=3, x=random.randrange(1,3)) :
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color) 
        self.speed = speed    
        self.canvas.move(self.id, 245, 200)
        
        self.start_x = x
        self.start_speed = speed
        self.start_pos = self.canvas.coords(self.id)
        
        self.x = x
        self.y = -speed
        
        self.canvas_width = canvas.winfo_width()
        self.canvas_height = canvas.winfo_height()
    
    def reset(self) :
        starts = [-self.speed,self.speed]
        random.shuffle(starts)
        self.x = starts[0]
        self.setPos(self.x,-self.start_speed)
        self.canvas.coords(self.id, self.start_pos)
        self.canvas.update()
    
    def draw(self):
        self.canvas.move(self.id, self.x, self.y)
        pos = self.canvas.coords(self.id)
        if pos[1] <= 0 :
            self.y = self.speed
        if pos[0] <= 0 :
            self.x = self.speed
        if pos[2] >= self.canvas_width :
            self.x = -self.speed

    def hit_paddle(self,pos) :
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2] :
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3] :
                self.setPos(self.x, -10)
                self.draw()
                return True
        return False
    
    def is_paddle_hit(self) :
        pos = self.canvas.coords(self.id)
        return self.hit_paddle(pos)
    
    def is_bottom_hit(self) :
        pos = self.canvas.coords(self.id)
        if pos[3] >= self.canvas_height :
            self.y = -self.speed
            self.canvas.move(self.id, self.x, self.y)
            return True
        return False

    def setPos(self,x,y) :
        self.x = x
        self.y = y
        
class Paddle:

    def __init__(self,canvas,color) :
        self.canvas = canvas
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)
        self.height_pos = int(self.canvas_height * 0.8)
        self.canvas.move(self.id, self.canvas_width/2, self.height_pos)
        self.start_pos = self.canvas.coords(self.id)
        self.x = 0
        self.y = 0
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)
        
    def reset(self) :
        self.canvas.coords(self.id, self.start_pos)

    def draw(self):
        pos = self.canvas.coords(self.id)
        
        if pos[0] <= 0 and self.x < 0 :
            return
        elif pos[2] >= self.canvas_width and self.x > 0 :
            return

        self.canvas.move(self.id, self.x, self.y)
        
    def turn_left(self,evt):
        self.x = -5

    def turn_right(self,evt):
        self.x = 5
        
    def setPos(self, x):
        self.x = x