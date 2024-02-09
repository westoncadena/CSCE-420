# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""

from builtins import object
import util

def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tiny_maze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tiny_maze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depth_first_search(problem):
    "*** YOUR CODE HERE ***"
    from util import Stack

    # Initalize stack with start state and actions
    stack = Stack()
    stack.push((problem.get_start_state(),[]))

    # create a visted set
    visited = set()
    

    while not stack.is_empty():
        # grab the current state and actions
        current_state, actions = stack.pop()

        
        # Check to see if current state is the goal state
        if problem.is_goal_state(current_state):
            return actions
        
        # Added curret state to visited
        visited.add(current_state)
        
        # go through the possible transitions, add them to stack if not visited
        transitions = problem.get_successors(current_state)
        for transition in transitions:
            if transition.state not in visited:
                new_actions = actions + [transition.action]
                stack.push((transition.state, new_actions))
        
    # return if goal state was not found
    return

    # What does this function need to return?
    #     list of actions that reaches the goal
    # 
    # What data is available?
    #     start_state = problem.get_start_state() # returns a string
    # 
    #     problem.is_goal_state(start_state) # returns boolean
    # 
    #     transitions = problem.get_successors(start_state)
    #     transitions[0].state
    #     transitions[0].action
    #     transitions[0].cost
    # 
    #     print(transitions) # would look like the list-of-lists on the next line
    #     [
    #         [ "B", "0:A->B", 1.0, ],
    #         [ "C", "1:A->C", 2.0, ],
    #         [ "D", "2:A->D", 4.0, ],
    #     ]
    # 
    # Example:
    #     start_state = problem.get_start_state()
    #     transitions = problem.get_successors(start_state)
    #     return [  transitions[0].action  ]
    
    util.raise_not_defined()


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    
    from util import Queue

    # Initalize queue with start state and actions
    queue= Queue()
    queue.push((problem.get_start_state(),[]))

    # create a visted set
    visited = set()
    

    while not queue.is_empty():
        # grab the current state and actions
        current_state, actions = queue.pop()
        
        # Check to see if current state is the goal state
        if problem.is_goal_state(current_state):
            return actions
        
        # Do not look at a state twice. This might happen if you add a state to the queue multiple times
        # before it is popped
        if current_state in visited:
            continue

        # Added curret state to visited
        visited.add(current_state)
        
        # go through the possible transitions, add them to queue if not visited
        transitions = problem.get_successors(current_state)
        
        for transition in transitions:
            if transition.state not in visited:
                new_actions = actions + [transition.action]
                queue.push((transition.state, new_actions))
        
    # return if goal state was not found
    return



    util.raise_not_defined()


def uniform_cost_search(problem, heuristic=None):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    # create priority queue with the problem's cost function
    priorityQueue = PriorityQueue()
    priorityQueue.update((problem.get_start_state(),[]), 0.0)

    # dictionary to store the actions
    state_actions = {problem.get_start_state(): []}

    # create a visted set
    visited = set()
    

    while not priorityQueue.is_empty():
        # grab the current state and actions
        current_state, actions = priorityQueue.pop()
        
        # Check to see if current state is the goal state
        if problem.is_goal_state(current_state):
            return actions
        
        # Do not look at a state twice. This might happen if you add a state to the queue multiple times
        # before it is popped
        if current_state in visited:
            continue
        
        # Added curret state to visited
        visited.add(current_state)
        
        # go through the possible transitions, add them to queue if not visited
        transitions = problem.get_successors(current_state)
        
        for transition in transitions:
            if transition.state not in visited:
                # add the new action to the list of actions
                new_actions = actions + [transition.action]
                # Derive the cost
                cost = problem.get_cost_of_actions(new_actions)
                
                priorityQueue.update((transition.state,new_actions), cost)
                
        
    # return if goal state was not found
    return
    


    util.raise_not_defined()


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    # create priority queue with the problem's cost function
    priorityQueue = PriorityQueue()
    priorityQueue.update((problem.get_start_state(),[]), 0.0)

    # dictionary to store the actions
    state_actions = {problem.get_start_state(): []}

    # create a visted set
    visited = set()
    

    while not priorityQueue.is_empty():
        # grab the current state and actions
        current_state, actions = priorityQueue.pop()
        
        # Check to see if current state is the goal state
        if problem.is_goal_state(current_state):
            return actions
        
        # Do not look at a state twice. This might happen if you add a state to the queue multiple times
        # before it is popped
        if current_state in visited:
            continue
        
        # Added curret state to visited
        visited.add(current_state)
        
        # go through the possible transitions, add them to queue if not visited
        transitions = problem.get_successors(current_state)
        
        for transition in transitions:
            if transition.state not in visited:
                # add the new action to the list of actions
                new_actions = actions + [transition.action]
                # Derive the cost with the heuristic
                cost = problem.get_cost_of_actions(new_actions) + heuristic(transition.state, problem)
                
                priorityQueue.update((transition.state,new_actions), cost)
                
        
    # return if goal state was not found
    return


    
    # What does this function need to return?
    #     list of actions that reaches the goal
    # 
    # What data is available?
    #     start_state = problem.get_start_state() # returns a string
    # 
    #     problem.is_goal_state(start_state) # returns boolean
    # 
    #     transitions = problem.get_successors(start_state)
    #     transitions[0].state
    #     transitions[0].action
    #     transitions[0].cost
    # 
    #     print(transitions) # would look like the list-of-lists on the next line
    #     [
    #         [ "B", "0:A->B", 1.0, ],
    #         [ "C", "1:A->C", 2.0, ],
    #         [ "D", "2:A->D", 4.0, ],
    #     ]
    # 
    # Example:
    #     start_state = problem.get_start_state()
    #     transitions = problem.get_successors(start_state)
    #     return [  transitions[0].action  ]
    
    util.raise_not_defined()


# (you can ignore this, although it might be helpful to know about)
# This is effectively an abstract class
# it should give you an idea of what methods will be available on problem-objects
class SearchProblem(object):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, step_cost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'step_cost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()

# fallback on a_star_search
for function in [breadth_first_search, depth_first_search, uniform_cost_search, ]:
    try: function(None)
    except util.NotDefined as error: exec(f"{function.__name__} = a_star_search", globals(), globals())
    except: pass

# Abbreviations
bfs   = breadth_first_search
dfs   = depth_first_search
astar = a_star_search
ucs   = uniform_cost_search