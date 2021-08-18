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
Pacman agents (in searchAgents.py).
"""

from collections import defaultdict

import numpy as np

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    visited = set()
    nodes_to_visit = util.Stack()
    initial_node = (problem.getStartState(), [])
    nodes_to_visit.push(initial_node)

    while not nodes_to_visit.isEmpty():
        node, path = nodes_to_visit.pop()
        if problem.isGoalState(node):
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in problem.getSuccessors(node):
            neighbor_name = neighbor[0]
            neighbor_action = neighbor[1]
            if neighbor_name not in visited:
                nodes_to_visit.push((neighbor_name, path + [neighbor_action]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = set()
    nodes_to_visit = util.Queue()
    initial_node = (problem.getStartState(), [])
    nodes_to_visit.push(initial_node)

    while not nodes_to_visit.isEmpty():
        node, path = nodes_to_visit.pop()
        if problem.isGoalState(node):
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in problem.getSuccessors(node):
            neighbor_node = neighbor[0]
            neighbor_action = neighbor[1]
            if neighbor_node not in visited:
                nodes_to_visit.push((neighbor_node, path + [neighbor_action]))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    distances_pq = util.PriorityQueue()
    node_name = problem.getStartState()
    node_path = []
    distances_pq.push((node_name, node_path), 0)
    explored = set()

    distances = defaultdict(lambda: np.inf)
    distances[problem.getStartState()] = 0

    while not distances_pq.isEmpty():
        current_node, current_node_path = distances_pq.pop()
        if problem.isGoalState(current_node):
            return current_node_path
        if current_node in explored:
            continue
        explored.add(current_node)
        for neighbor, neighbor_path, neighbor_weight in problem.getSuccessors(current_node):
            new_dist = distances[current_node] + neighbor_weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                distances_pq.update((neighbor, current_node_path + [neighbor_path]), new_dist)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    distances_pq = util.PriorityQueue()
    node_name = problem.getStartState()
    node_path = []
    distances_pq.push((node_name, node_path), 0)
    explored = set()

    graph_distances = defaultdict(lambda: np.inf)
    graph_distances[problem.getStartState()] = 0

    while not distances_pq.isEmpty():
        current_node, current_node_path = distances_pq.pop()
        if problem.isGoalState(current_node):
            return current_node_path
        if current_node in explored:
            continue
        explored.add(current_node)
        for neighbor, neighbor_path, neighbor_weight in problem.getSuccessors(current_node):
            new_dist = graph_distances[current_node] + neighbor_weight
            if new_dist < graph_distances[neighbor]:
                graph_distances[neighbor] = new_dist
                heuristic_distance = new_dist + heuristic(neighbor, problem)
                distances_pq.update((neighbor, current_node_path + [neighbor_path]), heuristic_distance)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
