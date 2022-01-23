from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        self.queue.put(uct_search(state))


import math
import random
from copy import deepcopy
from time import time

TIME_LIMIT_MS = 600  # Smaller than overall timeout


class MctsNode:
    def __init__(self, state, parent=None):
        self.q = 0
        self.n = 0
        self.state = state
        self.parent = parent
        self.children = []  # list of pairs (child_node, child_action)
        self.untried_actions = state.actions()

    def tree_policy(self):
        node = self
        while not node.state.terminal_test():
            if not node._fully_expanded():
                return node._expand()
            node, _ = node.best_child()
        return node

    def default_policy(self):
        player = 1 - self.state.player()  # Current player is the opponent
        game_state = deepcopy(self.state)
        while not game_state.terminal_test():
            action = random.choice(game_state.actions())
            game_state = game_state.result(action)
        utility = game_state.utility(player)
        if utility == float("inf"):
            return 1
        elif utility == float("-inf"):
            return -1
        else:
            return 0

    def backup(self, reward):
        self.q += reward
        self.n += 1
        if self.parent:
            # Reverse sign for the parent reward
            self.parent.backup(-reward)

    def best_child(self, c=1):
        return max(self.children, key=lambda child: child[0]._score(c))

    def _fully_expanded(self):
        return len(self.untried_actions) == 0

    def _expand(self):
        assert len(self.untried_actions) > 0
        action = self.untried_actions.pop()
        next_state = self.state.result(action)
        child_node = MctsNode(next_state, self)
        self.children.append((child_node, action))
        return child_node

    def _score(self, c=1):
        return (self.q / self.n) + c * math.sqrt(2 * math.log(self.parent.n) / self.n)


def uct_search(initial_state, time_limit=TIME_LIMIT_MS):
    if initial_state.terminal_test():
        return random.choice(initial_state.actions())
    root_node = MctsNode(initial_state)
    start_time = time()
    while (time() - start_time) * 1000 < time_limit:
        child_node = root_node.tree_policy()
        reward = child_node.default_policy()
        child_node.backup(reward)
    _, best_action = root_node.best_child()
    return best_action
