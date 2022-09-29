import heapq
import numpy as np
from constants import *
from environment import *
from state import State
"""
solution.py

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22

Code Reference:
    Some code is adapted from COMP3702 Tutorial 7 solution code
"""

class Solver:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.policy = {}
        self.states = {}

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.max_diff = 0
        self.states[self.environment.get_init_state()] = 0
        self.states_array = [self.environment.get_init_state()]
        self.policy = {}

        queue = [self.environment.get_init_state()]

        while len(queue) > 0:
            node = heapq.heappop(queue)

            self.states[node] = 0
            self.policy[node] = FORWARD

            for action in ROBOT_ACTIONS:
                reward, next_state = self.environment.apply_dynamics(node, action)
                if next_state not in self.states:
                    self.states_array.append(next_state)
                    heapq.heappush(queue, next_state)

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        if self.max_diff != 0 and self.max_diff < self.environment.epsilon:
            return True
        return False

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        max_diff = 0
        new_values = dict(self.states)

        for state in self.states:
            if self.environment.is_solved(state):
                self.states[state] = 1.0
                continue
            action_values = dict()
            for action in ROBOT_ACTIONS:
                total = 0
                probabilities, movements = self.environment.stoch_action(action)
                for movement in movements:
                    reward, next_state = self.get_next_state(state, movements[movement])
                    total += probabilities[movement] * (
                        reward + self.environment.gamma * self.states[next_state]
                    )
                action_values[action] = total

            self.states[state] = max(action_values.values())
            self.policy[state] = max(action_values, key=action_values.get)
            if abs(new_values[state] - self.states[state]) > max_diff:
                max_diff = abs(new_values[state] - self.states[state])

        self.max_diff = max_diff
        
    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.states[state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.t_model = []
        self.r_model = []
        self.la_policy = []
        self.states_array = []
        self.converged = False

        self.vi_initialise()
        self.states_array = list(self.states.keys())
        num_states = len(self.states)
        num_actions = len(ROBOT_ACTIONS)
        self.t_model = np.zeros([num_states, num_actions, num_states])
        self.r_model = np.zeros([num_states, num_actions])

        for state in self.states_array:
            state_index = self.states_array.index(state)
            if self.environment.is_solved(state):
                self.t_model[state_index][:][:] = 1.0
                self.r_model[state_index][:] = 1.0
                continue
            for action in ROBOT_ACTIONS:
                total = 0
                probabilities, movements = self.environment.stoch_action(action)
                for movement in movements:
                    reward, next_state = self.get_next_state(state, movements[movement])
                    self.t_model[state_index][action][
                        self.states_array.index(next_state)
                    ] += probabilities[movement]
                    total += reward * probabilities[movement]

                self.r_model[state_index][action] = total
                self.policy_array = np.zeros([num_states], dtype=np.int64) + FORWARD

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        self.policy_evaluation()
        self.policy_improvement()

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Helper Methods ===============================================================================================

    def get_next_state(self, state, movements):
        """
        Retrieve the next state and minimum reward required to get there.
        :param state: the current state
        :param movements list of movements to be performed
        :return: reward (integer), list of movemens (element of ROBOT_ACTIONS)
        """
        min_reward = 0
        next_state = state
        for movement in movements:
            reward, next_state = self.environment.apply_dynamics(next_state, movement)
            if reward < min_reward:
                min_reward = reward
        return min_reward, next_state

    def policy_evaluation(self):
        num_states = len(self.states_array)
        self.la_policy = list(self.policy.values())
        state_numbers = np.array(range(num_states))
        t_pi = self.t_model[state_numbers, self.la_policy]
        r = self.r_model[state_numbers, self.la_policy]
        values = np.linalg.solve(
            np.identity(num_states) - (self.environment.gamma * t_pi), r
        )
        self.states = {
            state: values[index] for index, state in enumerate(self.states_array)
        }

    def policy_improvement(self):
        new_values = dict(self.policy)
        for state in self.states:
            if self.environment.is_solved(state):
                self.states[state] = 1.0
                continue
            action_values = dict()
            for action in ROBOT_ACTIONS:
                total = 0
                probabilities, movements = self.environment.stoch_action(action)
                for movement in movements:
                    reward, next_state = self.get_next_state(state, movements[movement])
                    if self.environment.is_solved(next_state):
                        reward = 1.0
                    total += probabilities[movement] * (
                        reward + self.environment.gamma * self.states[next_state]
                    )
                action_values[action] = total

            self.policy[state] = max(action_values, key=action_values.get)
            
        if all(self.policy[s] == new_values[s] for s in self.states_array):
            self.converged = True

