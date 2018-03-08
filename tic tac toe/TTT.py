NAMES = [' ', 'X', 'O']
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

DRAW = 3
BOARD_FORMAT = "----------------------------\n"+\
               "| {0} | {1} | {2} |\n"+\
               "|--------------------------|\n"+\
               "| {3} | {4} | {5} |\n"+\
               "|--------------------------|\n"+\
               "| {6} | {7} | {8} |\n"+\
               "----------------------------"

class Env :
    def __init__(self) :
        self.state = None
        self.reward = 0
        self.done = False
        self.switch_map = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (2, 0),
            8: (2, 1),
            9: (2, 2)
        }
        
        
    def reset(self) :
        self.state = self.emptystate()
        self.reward = 0
        return self.state

    def render(self, state):
        cells = []
    
        for i in range(3):
            for j in range(3):
                cells.append(NAMES[state[i][j]].center(6))
    
        print(BOARD_FORMAT.format(*cells))
    
    def step(self, player, action) :
        move = self.switch_map[action]
        self.state[move[0]][move[1]] = player

    def emptystate(self):
        return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    
    def getavailable(self) :
        available = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == EMPTY:
                    available.append((i, j))
        return available
                    

    def gameover(state):
        for i in range(3):
            if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
                return state[i][0]
            if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
                return state[0][i]
    
        if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
            return state[0][0]
    
        if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
            return state[0][2]
    
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    return EMPTY
    
        return DRAW
    
    def episode_over(self, result):
        if result == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(result))
