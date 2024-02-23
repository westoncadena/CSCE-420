from __future__ import print_function

# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from util import manhattan_distance
from search import breadth_first_search
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):

    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        food_list = new_food.as_list()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghost_state.scared_timer for ghost_state in new_ghost_states]
        "*** YOUR CODE HERE ***"


        # get ghost position 
        ghost_positions = successor_game_state.get_ghost_positions()
        
        # Calculate the distance to the closest ghost
        closest_ghost = min([manhattan_distance(new_pos, ghost) for ghost in ghost_positions])
        
        # Calcualte the distance to the closest food 
        closest_food = float("inf")
        for food in food_list:
            dist = manhattan_distance(new_pos, food)
            if dist < closest_food:
                closest_food = dist

        score = successor_game_state.get_score() 
        
        # penalize if it stops, as pacman may stick on walls due to nature of manhattan distance
        stop = 0
        if action == "Stop":
            stop = - 70
    

        # factor added to closest food do to importance
        return  score + stop + float(closest_ghost) / (closest_food * 8.5)
    

def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, eval_fn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action from the current game_state using self.depth
          and self.evaluation_function.

          Here are some method calls that might be useful when implementing minimax.

          game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means Pacman, ghosts are >= 1

          game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action

          game_state.get_num_agents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        # calculate the correct action based on preforming minimax
        action = self.minimax(game_state, 0, 0)
        return action[1]
    
        util.raise_not_defined()

    def minimax(self, game_state, agent_index, depth):
        
        # terminating case, retrun score of gamestate
        if depth == self.depth or len(game_state.get_legal_actions(agent_index)) == 0: 
            return [self.evaluation_function(game_state), ""]
        
        # max if agent is pacman (agent_index = 0)
        if agent_index == 0:
            max_score = float("-inf")
            max_action = ""

            for action in game_state.get_legal_actions(agent_index):
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor  
                score = self.minimax(successor, agent_index + 1,depth)[0]

                if score > max_score:
                    max_score = score
                    max_action = action

            # return min score and action
            return [max_score, max_action]
        
        # min if agent is ghost (agent_index != 0)
        else:
            min_score = float("inf")
            min_action = ""
            new_agent_index = agent_index + 1
            new_depth = depth

            # if next agent is pacman, incremnt depth 
            if new_agent_index == game_state.get_num_agents():
                new_agent_index = 0
                new_depth += 1

            for action in game_state.get_legal_actions(agent_index): 
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor 
                score = self.minimax(successor, new_agent_index ,new_depth)[0]

                if score < min_score:
                    min_score = score
                    min_action = action

            # return min score and action
            return [min_score, min_action]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        action = self.alpha_beta(game_state, 0, 0, float("-inf"), float("inf"))
        return action[1]

        util.raise_not_defined()

    def alpha_beta(self, game_state, agent_index, depth, alpha, beta):
        
        # terminating case, retrun score of gamestate
        if depth == self.depth or len(game_state.get_legal_actions(agent_index)) == 0: 
            return [self.evaluation_function(game_state), ""]
        
        # max if agent is pacman (agent_index = 0)
        if agent_index == 0:
            max_score = float("-inf")
            max_action = ""

            for action in game_state.get_legal_actions(agent_index):
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor  
                score = self.alpha_beta(successor, agent_index + 1,depth, alpha, beta)[0]

                if score > max_score:
                    max_score = score
                    max_action = action

                # check alpha beta pruning
                if max_score > beta:
                    return [max_score, max_action]
                alpha = max(max_score, alpha)


            # return min score and action
            return [max_score, max_action]
        
        # min if agent is ghost (agent_index != 0)
        else:
            min_score = float("inf")
            min_action = ""
            new_agent_index = agent_index + 1
            new_depth = depth

            # if next agent is pacman, incremnt depth 
            if new_agent_index == game_state.get_num_agents():
                new_agent_index = 0
                new_depth += 1

            for action in game_state.get_legal_actions(agent_index): 
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor 
                score = self.alpha_beta(successor, new_agent_index ,new_depth, alpha, beta)[0]

                if score < min_score:
                    min_score = score
                    min_action = action

                 # check alpha beta pruning
                if min_score < alpha:
                    return [min_score, min_action]
                beta = min(min_score, beta)

            # return min score and action
            return [min_score, min_action]
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluation_function

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        action = self.expectimax(game_state, 0, 0)
        return action[1]


        util.raise_not_defined()

    def expectimax(self, game_state, agent_index, depth):
        

        # terminating case, retrun score of gamestate
        if depth == self.depth or len(game_state.get_legal_actions(agent_index)) == 0: 
            return [self.evaluation_function(game_state), ""]
        
        # max if agent is pacman (agent_index = 0)
        if agent_index == 0:
            max_score = float("-inf")
            max_action = ""

            for action in game_state.get_legal_actions(agent_index):
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor  
                score = self.expectimax(successor, agent_index + 1,depth)[0]

                if score > max_score:
                    max_score = score
                    max_action = action

            # return min score and action
            return [max_score, max_action]
        
        # expected value if agent is ghost (agent_index != 0)
        else:
            exp_score = 0
            exp_action = ""
            new_agent_index = agent_index + 1
            new_depth = depth

            # if next agent is pacman, incremnt depth 
            if new_agent_index == game_state.get_num_agents():
                new_agent_index = 0
                new_depth += 1
            
            probability = 1.0 / len(game_state.get_legal_actions(agent_index))
            
            for action in game_state.get_legal_actions(agent_index): 
                successor = game_state.generate_successor(agent_index, action)

                # Recusivly Calculate the score for successor 
                score = self.expectimax(successor, new_agent_index ,new_depth)[0]

                # update the expected score based on probability
                exp_score += probability * score

            # return random score and action
            return [exp_score, exp_action]
        
    

def better_evaluation_function(current_game_state):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      In order to do a better evaluation, I used important features multiplied 
      with coefficient in order to give them meaningful weight. Here is a 
      break down of what features I chose and why there coefficient
      
      score : 175
            * Score is the best representation of a "good" move by pacman,
            other features more or less aid the decision. that it why it is
            awarded the highest weight
      closest_food_recp : 100
            * If you can close the gap between you and the closest food, DO IT
      closest_ghost : 1
            * in order to keep pacman alive, did not need to give a higher value
            in order to give other features higher weights
      food_left: -100
            * important feature to minimize, a state with less food should be 
            weighted higher
      capsule_left: -50 
            * negative coefficent, as a smaller amount of capsules = more
            advantagous move. Not as advantagous as food_left

    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    pos = current_game_state.get_pacman_position()
    food_list = (current_game_state.get_food()).as_list()
    ghost_states = current_game_state.get_ghost_states()
    scared_times = [ghost_state.scared_timer for ghost_state in ghost_states]

    # amount of food and capsules left
    food_left = len(food_list)
    capsule_left = len(current_game_state.get_capsules())


    # get ghost position 
    ghost_positions = current_game_state.get_ghost_positions()
    
    # Calculate the distance to the closest ghost
    closest_ghost = min([manhattan_distance(pos, ghost) for ghost in ghost_positions])
    
    # Calcualte the distance to the closest food 
    closest_food = float("inf")
    for food in food_list:
        dist = manhattan_distance(pos, food)
        if dist < closest_food:
            closest_food = dist

    score = current_game_state.get_score() 
    closest_food_recp = 1/closest_food

    coe_dict = {"score" : 175,
                "closest_food_recp" : 100,
                "closest_ghost": 1,
                "food_left": -100,
                "capsule_left": -50}

    weighted_sum = 0
    # Iterate over the keys and linearly add them with their resepctive coefficients
    for key in coe_dict:
        weighted_sum += coe_dict[key] * locals()[key]
    
    
    return weighted_sum

    util.raise_not_defined()

# Abbreviation
better = better_evaluation_function

