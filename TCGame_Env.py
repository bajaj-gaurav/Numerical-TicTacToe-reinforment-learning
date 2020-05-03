import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        rows = [[0,1,2], [3,4,5], [6,7,8]]
        columns = [[0,3,6], [1,4,7], [2,5,8]]
        diagonal = [[0,4,8], [2,4,6]]
        total_checks = rows + columns + diagonal
        for row in total_checks:
            sum = 0
            count = 0
            for pos in row:
                if np.isnan(curr_state[pos]):
                    break
                else:
                    sum = sum + curr_state[pos]
                    count = count + 1
            if sum == 15 and count == 3:
                return True
        return False
                
        
 
    def is_terminal(self, curr_state):
        """Takes state as an input and returns whether we have reached a terminal state
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False, 'Resume'"""
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state
    
    def reward(self, player, winning_state):
        """Takes player name and state("Tie", "Wins", "Resume") and returns the reward point.
        Example: player- "agent", winning_state - "Tie"
        Output = 0
        """
        if winning_state == "Tie":
            return 1
        elif winning_state == "Resume":
            return -1
        else:
            if player == "agent":
                return 10
            else:
                return -10
                
    def initial_step(self, state, action):
        """Takes state and action and calculate the next state.
        Then it moves a action on behalf of the environment and then return the combined state.
        Example: Input state- [nan, nan, nan, nan, nan, nan, nan, nan, nan], action- [7, 9]
        next_state = [nan, nan, nan, nan, nan, nan, nan, 9, nan]
        Then environment action: (2,2)
        next_state = [nan, nan, 2, nan, nan, nan, nan, 9, nan]
        Output = [nan, nan, 2, nan, nan, nan, nan, 9, nan]"""
        next_state = self.state_transition(state, action)
        env_action = random.choice(list(self.action_space(next_state)[1]))
        next_state = self.state_transition(next_state, env_action)
        return next_state

    def step(self, curr_state, curr_action):
        """Takes current state and action and calculates the next state, reward and whether the state is terminal.
        if the state is terminal then return "next state, reward and whether the state is terminal"
        else
        moves a action on behalf of the environment then check for the terminal state.
        return "next state, reward and whether the state is terminal"
        Example: Input state- [nan, nan, 2, nan, nan, nan, nan, 9, nan], action- [0, 1] or [position, value]
        Then environment action: (4,4)
        Output = ([1, nan, 2, nan, 4, nan, nan, 9, nan], -1, False)"""
        next_state = self.state_transition(curr_state, curr_action)
        terminal_state = self.is_terminal(next_state)
        if terminal_state[0]:
            r = self.reward("agent", terminal_state[1])
            return (next_state, r, terminal_state[0])
        env_action = random.choice(list(self.action_space(next_state)[1]))
        next_state = self.state_transition(next_state, env_action)
        terminal_state = self.is_terminal(next_state)
        if terminal_state[0]:
            r = self.reward("env", terminal_state[1])
            return (next_state, r, terminal_state[0])       
        return (next_state, -1, False)

    def reset(self):
        return self.state
