"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    players_moves = len(game.get_legal_moves(player)) 
    opponents_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if opponents_moves == 0:
        return float("inf")
    
    if players_moves == 0:
        return float("-inf")
    
    
    def space_to_edge(game, player):
        height = game.height
        width = game.width
        
        position = game.get_player_location(player)
        if position is None:
            return None
        
        (h, w) = position
        
        return min(min(height - h, h), min(width - w, w))
    
    def extra_points(num):
        if num is None:
            return 0.0
        
        return max((num-3)*-1, 0) 
    
    
    def space_between(player_position, opponent_position):
        if player_position is None or opponent_position is None:
            return 0
        
        (h1, w1) = player_position
        (h2, w2) = opponent_position
        
        return math.sqrt((h1-h2)**2 + (w1-w2)**2)

    space_between_players = space_between(game.get_player_location(player), game.get_player_location(game.get_opponent(player)))
    
    player_edge_bonus = extra_points(space_to_edge(game, player))
    opponents_edge_bonus = extra_points(space_to_edge(game, game.get_opponent(player)))
    
    return players_moves + player_edge_bonus * 0.4 - (1.5 * opponents_moves) + opponents_edge_bonus * 0.4 + max(5, space_between_players)*.6


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    players_moves = len(game.get_legal_moves(player)) 
    opponents_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if opponents_moves == 0:
        return float("inf")
    
    if players_moves == 0:
        return float("-inf")
    
    def space_to_edge(game, player):
        height = game.height
        width = game.width
        
        position = game.get_player_location(player)
        if position is None:
            return None
        
        (h, w) = position
        
        return min(min(height - h, h), min(width - w, w))
    
    def extra_points(num):
        if num is None:
            return 0.0
        
        return max((num-3)*-1, 0)

    
    player_edge_bonus = extra_points(space_to_edge(game, player))
    
        
    return players_moves - player_edge_bonus * 0.5 - (1.5 * opponents_moves)



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    players_moves = len(game.get_legal_moves(player)) 
    opponents_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if opponents_moves == 0:
        return float("inf")
    
    if players_moves == 0:
        return float("-inf")
    
    return players_moves - 1.5 * opponents_moves


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def time_test(player):
            if player.time_left() < player.TIMER_THRESHOLD:
                raise SearchTimeout()

        # TODO: finish this function!
        # return game.get_legal_moves()[0]
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            return (-1,-1)
        
        
        def terminal_test(gameState):
            """ Return True if the game is over for the active player
            and False otherwise.
            """
            return not bool(gameState.get_legal_moves())  # by Assumption 1
    
    
        # This function will only return the best score of the leaf, and not the more itself
        def max_value(player, gameState, depth):
            time_test(player) 
            
            if terminal_test(gameState):
                return float("-inf")
            
            if depth <= 0:
                return player.score(gameState, player)

            v = float("-inf")
            for m in gameState.get_legal_moves():
                v = max(v, min_value(player, gameState.forecast_move(m), depth - 1))
            return v
        
        
        # This function will only return the (worst) score of the leaf, and not the more itself
        def min_value(player, gameState, depth):
            time_test(player) 
            
            if terminal_test(gameState):
                return float("inf") 
          
            if depth <= 0:
                return player.score(gameState, player)

            v = float("inf")
            for m in gameState.get_legal_moves():
                v = min(v, max_value(player, gameState.forecast_move(m), depth - 1))
                
            return v

        
        # This function will return the best move, and not he score itself        
        def minimax_decision(player, gameState, depth):
            return max(gameState.get_legal_moves(),
               key=lambda m: min_value(player, gameState.forecast_move(m), depth - 1))
      
    
        # We want to get and return the best move, and not the best score
        return minimax_decision(self, game, depth)

        

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """        
        self.time_left = time_left
        
                
        # TODO: finish this function!
        result = (-1, -1)
        
        if len(game.get_legal_moves()) == 0:
            return result  
        
        # The search timeout will be thrown from the alphabeta algortim if the max time is reached
        try:
            # Some algorith use a while true loop here, but there is no need for searchin deeper then the max number of move left on the board
            for depth in range(1, len(game.get_blank_spaces())):
                result = self.alphabeta(game, depth)
        except SearchTimeout:
            pass
            
        return result
   
            

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        def time_test(player):
            if player.time_left() < player.TIMER_THRESHOLD:
                raise SearchTimeout()

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves) == 0:
            return (-1,-1)
        
        
        def terminal_test(gameState):
            """ Return True if the game is over for the active player
            and False otherwise.
            """
            return not bool(gameState.get_legal_moves())  # by Assumption 1
    

        # This function will only return the best score of the leaf, and not the more itself
        def max_value(player, gameState, depth, alpha, beta):
            # Start by validate the state of the algoritm, and terminate if the time is used or the max depth is reached
            time_test(player) 
            
            if terminal_test(gameState):
                return float("-inf")
            
            if depth <= 0:
                return player.score(gameState, player)

            # The max algoritm starts here
            v = float("-inf")
            for m in gameState.get_legal_moves():
                v = max(v, min_value(player, gameState.forecast_move(m), depth - 1, alpha, beta))
                
                if v >= beta:
                    return v
                
                alpha = max(alpha, v)
                
            return v
        
    
        # This function will only return the (worst) score of the leaf, and not the more itself
        def min_value(player, gameState, depth, alpha, beta):
            # Start by validate the state of the algoritm, and terminate if the time is used or the max depth is reached
            time_test(player) 
            
            if terminal_test(gameState):
                return float("inf") 
          
            if depth <= 0:
                return player.score(gameState, player)

            # The min algoritm starts here
            v = float("inf")
            for m in gameState.get_legal_moves():
                v = min(v, max_value(player, gameState.forecast_move(m), depth - 1, alpha, beta))
                
                if v <= alpha:
                    return v
                
                beta = min(beta, v)
                
            return v
        
        
        # This function will return the best move, and not he score itself
        def minimax_decision(player, gameState, depth, alpha, beta):
            v = float("-inf")
            move = (-1, -1)
            for m in gameState.get_legal_moves():
                m_v = min_value(player, gameState.forecast_move(m), depth - 1, alpha, beta)
                if m_v > v:
                    v = m_v
                    move = m
                
                # no need for checking if v is bigger then beta, beta is infinity in this step
                alpha = max(alpha, v)
                
            return move
        
        
        # We want to get and return the best move, and not the best score
        return minimax_decision(self, game, depth, alpha, beta)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
