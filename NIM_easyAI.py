# -*- coding: utf-8 -*-
"""
@authors: Marcin Grelewicz (s17692), Edyta Bartos (s17699)
Based on the example of easyAI usage on https://zulko.github.io/easyAI/examples/quick_example.html

This is a modified version of the NIM game: https://pl.wikipedia.org/wiki/Nim 
with rules described below:
   
    In turn, 2 players remove from one to five coins, from first, second 
    or third heap of coins.
    The player who removes the last coin from one of the heaps looses. 
    Players can remove coins from one heap at time.
    Example of player's move: "what do you play ? 1,3", where first number 
    means heap and second number of coins.

To run the game you need to install easyAI framework: 
    https://zulko.github.io/easyAI/installation.html

In this game we use The Negamax algorithm: 
    https://en.wikipedia.org/wiki/Negamax, 
which looks for the shortest path to victory, or the longest path to defeat.

"""

from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

class NIM(TwoPlayersGame):
    """This is a subclass of the class easyAI.TwoPlayersGame
    
    Methods:
        - __init__() : Initializes a game
        - possible_moves() : Returns all allowed moves 
        - make_move : Transforms the game according to the move
        - win() : Returns the conditions of winning the game
        - show() : Displays the game
        - is_over() : Checks whether the game has ended
        - scoring() : Gives a score to the current game (for the AI training)
    """

    
    def __init__(self, players):   
        """ Initializes a game 
        
        Parameters:
            - players : Stores players
            - heap : Sets exact size of heaps
            - nplayer : Tells which player plays first
        """
        self.players = players  # Stores players into self.players
        self.heap = [11, 16, 13]  # We use list to set sizes of heaps
        self.nplayer = 2  # AI starts
   
    
    def possible_moves(self):
        """ Returns all allowed moves """
        # Here the player will decide how many coins to take from a heap:
        return ["%d,%d" % (i + 1, j) for i in range(len(self.heap))  
                for j in range(1, 6)]  
   
    def make_move(self, move):
        """ Transforms the game according to the move """
        move = list(map(int, move.split(',')))
        ## Second value given by player (representing coins quantity) will
        #  be substracted from first value (representing heap):
        self.heap[move[0] - 1] -= move[1]  
   
    def win(self):
        """ Returns the conditions of winning/loosing the game """
        # The condition is to get 0 in first, second or third heap
        return self.heap[0]<=0 or self.heap[1]<=0 or self.heap[2]<=0

    def is_over(self):
        """ Checks whether the game has ended """
        return self.win()  # Game stops when someone wins.
       
    def show(self):
        """ Displays the game """
        print(self.heap)  # Updates heap variable status
       
    def scoring(self):
        """ Gives a score to the current game (for the AI learning only) """
        return 100 if self.win() else 0

# Start a match (and store the history of moves when it ends)
ai = Negamax(6)     #The AI will think 6 moves in advance
game = NIM([Human_Player(), AI_Player(ai)])  # game details
history = game.play()  # starts the game and initialize history
""" variable history is a list [(g1,m1),(g2,m2)...] where gi is a copy 
    of the game after i moves and mi is the move made by the player 
    whose turn it was.
"""
# ending game message:
if game.nplayer == 2:
    print('AI wins!')
else:
    print('You won!')
