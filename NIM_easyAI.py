# -*- coding: utf-8 -*-
"""
@authors: Marcin Grelewicz (s17692), Edyta Bartos (s17699)
Based on the example of easyAI usage on https://zulko.github.io/easyAI/examples/quick_example.html

This is a modified version of the NIM game: https://pl.wikipedia.org/wiki/Nim with rules described below:
   
    In turn, 2 players remove from one to five coins, from first, second or third heap of coins.
    The player who removes the last coin from one of the heaps looses. Players can remove coins from one heap at time.
    Example of player's move: "what do you play ? 1,3", where first number means heap and second number of coins.

To run the game you need to install:
- easyAI framework: https://zulko.github.io/easyAI/installation.html
- Python 3 environment (https://www.python.org/download/releases/3.0/)

In this game we use The Negamax algorithm: https://en.wikipedia.org/wiki/Negamax, which always look for the shortest
path to victory, or the longest path to defeat.

"""

from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

class NIM( TwoPlayersGame ):
    """This is a subclass of the class easyAI.TwoPlayersGame"""
   
    #Methods:
       
    #Initializes a game:
    def __init__(self, players, heap = None):   #The __init__ method must do the following actions:

        self.players = players  #Stores players (which must be a list of two Players) into self.players
        self.heap = [11, 16, 13]
        self.nplayer = 2    #Tells which player plays first with self.nplayer = 1 # or 2 (here AI starts)
   
    #Returns all allowed moves:
    def possible_moves(self):
        return ["%d,%d" % (i + 1, j) for i in range(len(self.heap))  
                for j in range(1, 6)]
   
    #Transforms the game according to the move:
    def make_move(self, move):
        move = list(map(int, move.split(',')))
        self.heap[move[0] - 1] -= move[1]
   
    #Returns the conditions of winning the game:
    def win(self):
        return self.heap[0]<=0 or self.heap[1]<=0 or self.heap[2]<=0
   
    #Checks whether the game has ended:
    def is_over(self):
        return self.win()   #Game stops when someone wins.
       
    #Displays the game:
    def show(self):
        print(self.heap)
       
    #Gives a score to the current game (for the AI learning)
    def scoring(self):
        return 100 if self.win() else 0

# Start a match (and store the history of moves when it ends)
ai = Negamax(6)     #The AI will think 6 moves in advance
game = NIM( [ Human_Player(), AI_Player(ai) ] )
history = game.play()

if game.nplayer == 2:
    print('AI wins!')
else:
    print('You won!')
