from tkinter import *
import random
import time

# paddle 방향 상수
PADDLE_MOVE = [-15, 15, 0]

class Env :
    def __init__(self) :
        self.action_space = ['left', 'right', 'none']
        self.n_actions = len(self.action_space)
        #Tk 기본 설정
        self.tk = Tk()
        self.tk.title("My Ping Pong !") 
        self.tk.resizable(0, 0)
        self.tk.wm_attributes("-topmost", 1)
        self.canvas = Canvas(self.tk, width=500, height=400, bd=0, highlightthickness=0, bg='black')
        self.canvas.pack()
        self.tk.update()
        
        #agent 에서 사용되는 변수
        self.catch_cnt = 0
        self.step_cnt = 0
        self.reward = 0
        self.done = False

        self.paddle = Paddle(self.canvas, 'white')
        self.balls = list()

    # 패들, 공의 갯수만큼 전부 리셋하고 최초 state 반환    
    def reset(self) : 
        for ball in self.balls :
            ball.reset()
        self.paddle.reset()
        # 변수 초기화
        self.step_cnt = 0
        self.catch_cnt = 0
        self.done = False
        # 리셋 후 패들이 최초로 움직일 방향 랜덤 설정
        rand = random.choice([0,1,2])
        return self.getstate(PADDLE_MOVE[rand])

    # counnt 만큼 공을 생성
    def addBall(self, count) :
        for i in range(count) :
            colors = ['red','green','blue','white','yellow','orange']
            random.shuffle(colors)
            speeds = [3,4,5]
            random.shuffle(speeds)
            self.balls.append(Ball(self.canvas, self.paddle, colors.pop(0),speeds[0]))
    
    # 아래의 키를 리턴해주는 함수
    # ex : (190.0, (246.0, 291.0), (3, 3))
    # 패들 x0좌표, 공 x,y좌표, 공의 x,y값(속도) 
    def getstate(self,movement):
        paddle_pos = self.canvas.coords(self.paddle.id)
        res = []
        res.append(paddle_pos[0])
        for ball in self.balls :
            ball_pos = self.canvas.coords(ball.id) 
            res.append((ball_pos[0],ball_pos[1]))
            res.append((ball.x,ball.y))
        return tuple(res)
    
    # 공하고 패들을 지속적으로 그리는 함수
    def render(self, isLearning=True) :
        self.paddle.draw()
        for ball in self.balls :
            ball.draw()
                
        self.tk.update_idletasks()
        self.tk.update() 
        # isLearning : True면 학습을 빠르게 False면 천천히 학습
        if isLearning == False :
            time.sleep(0.03)
            
    # 한 타임스텝마다 진행되는 함수
    def step(self, action) :
        self.step_cnt += 1
        # next_state 를 구하는 코드
        # 액션을 받아서 패들의 움직임을 정해준다 (다음 state를 구하기 위해)
        if action == 0 or action == 1 or action == 2 :
            self.paddle.setPos(PADDLE_MOVE[action])
        else:
            rand = random.choice([0,1,2])
            self.paddle.setPos(PADDLE_MOVE[rand])
        
        next_state = self.getstate(self.paddle.x)
        
        # 바닥을 치면 -1 보상을 주고 done 변수를 True
        # 패들이 치면 +1 보상을 주고 done은 false 유지 받은 횟수를 1 증가 시킨다
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
        # 공 그리기 25,25 크기로 캔버스 위치 10,10에 넣겠다.
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color) 
        self.speed = speed    
        # 위에 그린 공을 캔버스 위치 245,200에 위치시킨다.
        self.canvas.move(self.id, 245, 200)
        
        self.start_x = x
        # 리셋 후 시작 속도를 일정하게 하기 위한 변수
        self.start_speed = speed
        # 리셋하면 공의 시작 위치를 같게 하기 위한 변수
        self.start_pos = self.canvas.coords(self.id)
        
        self.x = x
        self.y = -speed
        
        self.canvas_width = canvas.winfo_width()
        self.canvas_height = canvas.winfo_height()
    
    def reset(self) :
        # overload : 같은 함수 이름인데 다른 기능을 하기 위해 파라미터 개수를 다르게 받아서 사용하는 기능
        # id 를 start 좌표에 위치로 set
        self.canvas.coords(self.id, self.start_pos)
        # 왼쪽으로 갈지 오른쪽으로 갈지 방향 설정
        starts = [-self.speed,self.speed]
        random.shuffle(starts)
        self.setPos(starts[0],-self.start_speed)
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
                # 보상에 대한 카운팅 버그를 해결하기 위해 패들의 높이 만큼 공을 위로 보냄
                self.setPos(self.x, -6)
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
        # 패들을 가로 100 세로 10으로 만들기
        self.id = canvas.create_rectangle(0,0,100,6,fill=color)
        # 캔버스 전체 높이의 0.8 비율에 위치
        self.height_pos = int(self.canvas_height * 0.8)
        # 패들을 캔버스 가로 중앙에 위치 
        self.canvas.move(self.id, self.canvas_width/2, self.height_pos)
        self.start_pos = self.canvas.coords(self.id)
        self.x = 0
        self.y = 0
        #self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        #self.canvas.bind_all('<KeyPress-Right>',self.turn_right)
        
    def reset(self) :
        # overload : 같은 함수 이름인데 다른 기능을 하기 위해 파라미터 개수를 다르게 받아서 사용하는 기능
        # id 를 start 좌표에 위치로 set
        self.canvas.coords(self.id, self.start_pos)

    def draw(self):
        pos = self.canvas.coords(self.id)
        
        if pos[0] <= 0 and self.x < 0 :
            return
        elif pos[2] >= self.canvas_width and self.x > 0 :
            return

        self.canvas.move(self.id, self.x, self.y)
        
    #def turn_left(self,evt):
    #    self.x = -5

    #def turn_right(self,evt):
    #    self.x = 5
        
    def setPos(self, x):
        self.x = x