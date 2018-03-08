import random
from TTT import Env
from collections import defaultdict

class SARSA_agent :
    def __init__(self, env, player) :
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda:[i for i in range(9)])
        self.env = env
        self.player = player
    
    def get_action(self, state) :
        if random.random() < self.epsilon:
            move = random.choice(self.env.getavailable)
        else:
            move = self.greedy(state)

        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)

        return move

    def greedy(self, state):

        maxval = -50000

        maxmove = None

        for i in range(3):

            for j in range(3):

                if state[i][j] == EMPTY:

                    state[i][j] = self.player

                    val = self.lookup(state)

                    state[i][j] = EMPTY

                    if val > maxval:
                        maxval = val

                        maxmove = (i, j)

        self.backup(maxval)

        return maxmove
        
if __name__ == '__main__':
    
    env = Env()