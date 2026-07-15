from typing import Iterable, Tuple
import gymnasium as gym
import numpy as np

from interfaces.policy import RandomPolicy
from interfaces.solver import Solver, Hyperparameters

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def on_policy_n_step_td(
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    n: int,
    alpha: float,
    initV: np.array,
    gamma: float = 1.0
) -> Tuple[np.array]:
    """
    Runs the on-policy n-step TD algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 144, "n-step TD Prediction"

    Parameters:
        trajs (list): N trajectories generated using an unknown policy. Each trajectory is a 
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n (int): The number of steps (the "n" in n-step TD)
        alpha (float): The learning rate
        initV (np.ndarray): initial V values; np array shape of [nS]
        gamma (float): The discount factor

    Returns:
        V (np.ndarray): $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # ✅TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = np.array(initV)

    for traj in trajs:
        T = len(traj)
        t = 0
        while True:
            if t < T:
            # Take an action and observe the next reward and state
                _, _, r_t2, s_t2 = traj[t]
                if t + n >= T:
                    # Not enough steps to look ahead n steps
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    _, _, r_i, _ = traj[i-1]
                    G += (gamma ** (i - tau - 1)) * r_i
                if tau + n < T:
                    _, _, _, s_taun = traj[tau + n]
                    G += (gamma **n) * V[s_taun]
                s_tau = traj[tau][0]
                V[s_tau] += alpha * (G - V[s_tau])
            
            if tau == (T - 1): 
                break
            t += 1

    return V


class NStepSARSAHyperparameters(Hyperparameters):
    """ Hyperparameters for NStepSARSA algorithm """
    def __init__(self, gamma: float, alpha: float, n: int):
        """
        Parameters:
            gamma (float): The discount factor
            alpha (floa): The learning rate
            n (int): The number of steps (the "n" in n-step SARSA)
        """
        super().__init__(gamma)
        self.alpha = alpha
        """The learning rate"""
        self.n = n
        """The number of steps (the "n" in n-step SARSA)"""

class NStepSARSA(Solver):
    """
    Solver for N-Step SARSA algorithm, good for discrete state and action spaces.

    Off-policy algorithm, using weighted importance sampling.
    """
    def __init__(self, env: gym.Env, hyperparameters: NStepSARSAHyperparameters):
        super().__init__("NStepSARSA", env, hyperparameters)
        self.pi = Policy_DeterministicGreedy(np.ones((env.observation_space.n, env.action_space.n)))

    def action(self, state):
        """
        Chooses an action based on the current policy.

        Parameters:
            state (int): The current state
        
        Returns:
            int: The action to take
        """
        return self.pi.action(state)

    def train_episode(self):
        """
        Trains the agent for a single episode.

        Returns:
            float: The total (undiscounted) reward for the episode
        """

        #####################
        # ✅TODO: Implement Off Policy n-Step SARSA algorithm
        #   - Hint: Sutton Book p. 149
        #   - Hint: You'll need to build your trajectories using a behavior policy (RandomPolicy)
        #   - Hint: You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
        #   - Hint: Be sure to check both terminated and truncated variables.
        #####################        
        n = self.hyperparameters.n
        gamma = self.hyperparameters.gamma
        alpha = self.hyperparameters.alpha

        # Define your behavior policy for exploration
        bpi = RandomPolicy(self.env.action_space.n)

        # A trajectory will be a tuple of (s, a)
        traj = []           # List to store the behavior policy trajectory
        rewards = []        # List to store the rewards

        # Initialize the environment and store initiate state
        # Note: Assuming the gym environments like Taxi and FrozenLake returns a valid initial state when reset() is called.
        s, _ = self.env.reset()

        # Select and store the initial action from the behavior policy 
        a = bpi.action(s)

        traj.append((s, a))

        T = float('inf')    # initalize to infinity
        t = 0               # time step counter for the loop

        while True:
            if t < T:
                # Take action action at time step t
                s_next, r, terminated, truncated, _ = self.env.step(a)
                # Store the reward
                rewards.append(r)

                if terminated or truncated:
                    T = t + 1   # set the time of termination
                else:
                    # Select and store the next action from the behavior policy
                    # Store the next state in the trajectory
                    a = bpi.action(s_next)
                    traj.append((s_next, a))

            # tau is the time whose estimate is being updated, so it is negative until we have enough steps to update the estimate
            tau = t - n + 1 

            # Indicates that we have observed enough steps in the trajectory to update the action-value function for the state-action pair at time tau + n
            if tau >= 0:
                rho = 1.0   # Weighted importance sampling ratio
                # Compute the weighted importance sampling ratio.  This does the product of the ratios of the target policy to the behavior policy
                for i in range(tau + 1, min(tau + n, T - 1) + 1):
                    s_i, a_i = traj[i]
                    rho *= self.pi.action_prob(s_i, a_i) / bpi.action_prob(s_i, a_i)

                # Compute the return G
                episode_G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    episode_G += (gamma ** (i - tau -1)) * rewards[i-1]

                # Update the return G with the bootstrapped value from the target policy if we have enough steps to do so.
                if tau + n < T:
                    s_tau_n, a_tau_n = traj[tau + n]
                    episode_G += (gamma ** n) * self.pi.Q[s_tau_n, a_tau_n]

                # Update the target policy
                s_tau, a_tau = traj[tau]
                self.pi.Q[s_tau][a_tau] += alpha * rho * (episode_G - self.pi.Q[s_tau][a_tau])

            if tau == (T - 1):
                break
            
            t += 1

        return episode_G
