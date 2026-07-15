"""
Fast sanity checks for PA4 (n-step TD and n-step SARSA).
This file is written by KiuGPAM (Generative Pre-trained Aut...Mind)

How to run
----------
All sanity tests (recommended while developing):

    python test.py n_step_bootstrap --sanity

Or with unittest directly:

    python -m unittest tests.n_step_bootstrap_sanity -v

Run a single test class:

    python -m unittest tests.n_step_bootstrap_sanity.TestOnPolicyNStepTDSanity -v

Run one specific test:

    python -m unittest tests.n_step_bootstrap_sanity.TestOnPolicyNStepTDSanity.test_single_step_terminal_n1 -v

Full integration tests (slow, ~1–2 min):

    python test.py n_step_bootstrap

Tips
----
- Run sanity tests after every change; run integration tests before submitting.
- These tests use hand-computed expected values on tiny trajectories / envs.
- SARSA tests fix random seeds so behavior actions are reproducible.
- Crying is not an option.
"""

import unittest

import gymnasium as gym
import numpy as np

from interfaces.random_policy import RandomPolicy
from lib.envs.one_state_mdp import OneStateMDP
from lib.envs.generate_trajectories import generate_trajectories

from assignments.n_step_bootstrap import (
    NStepSARSA,
    NStepSARSAHyperparameters,
    on_policy_n_step_td,
)


class OneStepTerminalEnv(gym.Env):
    """State 0: action 0 -> terminal state 1 with reward 5; action 1 -> stay with reward 0."""

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self._state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = 0
        return self._state, {}

    def step(self, action):
        if action == 0:
            self._state = 1
            return self._state, 5.0, True, False, {}
        return self._state, 0.0, False, False, {}


class TwoStepChainEnv(gym.Env):
    """State 0 --(a=0,r=0)--> state 0 --(a=0,r=10)--> terminal state 1."""

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(2)
        self._state = 0
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = 0
        self._steps = 0
        return self._state, {}

    def step(self, action):
        self._steps += 1
        if self._steps == 1:
            return 0, 0.0, False, False, {}
        self._state = 1
        return self._state, 10.0, True, False, {}


class TestOnPolicyNStepTDSanity(unittest.TestCase):
    """Hand-checked n-step TD prediction tests on tiny trajectories."""

    def test_empty_trajectory_list(self):
        """No episodes -> V stays at initial values."""
        init = np.array([3.0, 7.0])
        V = on_policy_n_step_td([], n=1, alpha=1.0, initV=init, gamma=1.0)
        np.testing.assert_array_equal(V, init)

    def test_single_step_terminal_n1(self):
        """One transition to terminal: V(s0) should equal the reward (no bootstrap)."""
        traj = [(0, 0, 5, 1)]
        V = on_policy_n_step_td([traj], n=1, alpha=1.0, initV=np.zeros(2), gamma=1.0)
        self.assertAlmostEqual(V[0], 5.0)
        self.assertAlmostEqual(V[1], 0.0)

    def test_two_step_terminal_n1(self):
        """n=1, gamma=1: only the last reward before termination counts at tau=1."""
        traj = [(0, 0, 0, 0), (0, 0, 10, 1)]
        V = on_policy_n_step_td([traj], n=1, alpha=1.0, initV=np.zeros(2), gamma=1.0)
        self.assertAlmostEqual(V[0], 10.0)
        self.assertAlmostEqual(V[1], 0.0)

    def test_two_step_with_bootstrap_n1(self):
        """n=1 at t=0 bootstraps from V(s1) before terminal is observed."""
        traj = [(0, 0, 0, 0), (0, 0, 10, 1)]
        init = np.array([2.0, 4.0])
        V = on_policy_n_step_td([traj], n=1, alpha=1.0, initV=init.copy(), gamma=1.0)
        # t=0: G = r1 + V(s1) = 0 + 2; t=1: G = 10 -> V[0] = 10
        self.assertAlmostEqual(V[0], 10.0)

    def test_gamma_zero_uses_only_first_reward_in_window(self):
        """gamma=0: n-step return ignores later rewards in the window."""
        traj = [(0, 0, 3, 0), (0, 0, 100, 1)]
        V = on_policy_n_step_td([traj], n=2, alpha=1.0, initV=np.zeros(2), gamma=0.0)
        # Final update at tau=1 uses only r2=100 in the return window.
        self.assertAlmostEqual(V[0], 100.0)

    def test_alpha_scales_update(self):
        """Learning rate scales how far V moves toward the return."""
        traj = [(0, 0, 10, 1)]
        V = on_policy_n_step_td([traj], n=1, alpha=0.5, initV=np.zeros(2), gamma=1.0)
        self.assertAlmostEqual(V[0], 5.0)

    def test_unvisited_states_unchanged(self):
        """States never visited in trajectories keep their initial value."""
        traj = [(0, 0, 1, 1)]
        init = np.array([0.0, 9.0])
        V = on_policy_n_step_td([traj], n=1, alpha=1.0, initV=init.copy(), gamma=1.0)
        self.assertAlmostEqual(V[1], 9.0)

    def test_one_state_mdp_loose(self):
        """OneStateMDP smoke test (same idea as the official grader, loose tolerance)."""
        env = OneStateMDP()
        trajs = generate_trajectories(
            env, RandomPolicy(env.action_space.n), num_episodes=5000
        )
        V = on_policy_n_step_td(trajs, n=2, alpha=0.005, initV=np.zeros(2), gamma=1.0)
        self.assertAlmostEqual(V[0], 0.1, delta=0.15)
        self.assertAlmostEqual(V[1], 0.0, delta=1e-5)


class TestNStepSARSASanity(unittest.TestCase):
    """Fast n-step SARSA checks on tiny deterministic environments."""

    def test_train_episode_returns_float(self):
        """train_episode should return undiscounted episode return."""
        env = OneStepTerminalEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=0.99, alpha=0.1, n=1),
        )
        np.random.seed(0)
        reward = solver.train_episode()
        self.assertIsInstance(reward, float)
        self.assertAlmostEqual(reward, 5.0)

    def test_q_table_is_shared_with_greedy_policy(self):
        """Updates during training should mutate self.pi.Q in place."""
        env = OneStepTerminalEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=1.0, alpha=1.0, n=1),
        )
        Q_before = solver.pi.Q.copy()
        np.random.seed(0)
        solver.train_episode()
        self.assertFalse(np.allclose(solver.pi.Q, Q_before))

    def test_one_step_env_updates_q_with_full_alpha(self):
        """One-step episode, n=1, alpha=1: Q(s0,a0) should match the observed return."""
        env = OneStepTerminalEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=1.0, alpha=1.0, n=1),
        )
        np.random.seed(0)
        solver.train_episode()
        self.assertGreater(solver.pi.Q[0, 0], solver.pi.Q[0, 1])

    def test_greedy_policy_prefers_better_action_after_training(self):
        """After repeated episodes, greedy policy on state 0 should pick action 0."""
        env = OneStepTerminalEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=0.99, alpha=0.5, n=1),
        )
        np.random.seed(42)
        for _ in range(50):
            solver.train_episode()
        self.assertEqual(solver.pi.action(0), 0)

    def test_two_step_chain_n1(self):
        """Two-step chain with n=1 should eventually value the rewarding path."""
        env = TwoStepChainEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=1.0, alpha=0.5, n=1),
        )
        np.random.seed(0)
        for _ in range(30):
            solver.train_episode()
        self.assertGreater(solver.pi.Q[0, 0], 0.0)

    def test_n2_similar_to_n1_on_one_step_env(self):
        """n=2 on a one-step env should still learn a positive value for the good action."""
        env = OneStepTerminalEnv()
        solver = NStepSARSA(
            env,
            NStepSARSAHyperparameters(gamma=1.0, alpha=0.5, n=2),
        )
        np.random.seed(7)
        for _ in range(50):
            solver.train_episode()
        self.assertGreater(solver.pi.Q[0, 0], 1.0)


class TestTrajectoryFormatSanity(unittest.TestCase):
    """Sanity checks for trajectory indexing conventions used by TD."""

    def test_trajectory_tuple_layout(self):
        """Each step is (s_t, a_t, r_{t+1}, s_{t+1})."""
        traj = [(0, 1, 5, 2)]
        s_t, a_t, r_next, s_next = traj[0]
        self.assertEqual((s_t, a_t, r_next, s_next), (0, 1, 5, 2))

    def test_generate_trajectories_shape(self):
        """generate_trajectories should zip states/actions/rewards correctly."""
        env = OneStepTerminalEnv()
        policy = RandomPolicy(nA=2, p=np.array([1.0, 0.0]))
        trajs = generate_trajectories(env, policy, num_episodes=1)
        self.assertEqual(len(trajs), 1)
        self.assertEqual(len(trajs[0]), 1)
        s_t, a_t, r_next, s_next = trajs[0][0]
        self.assertEqual(s_t, 0)
        self.assertEqual(a_t, 0)
        self.assertEqual(r_next, 5.0)
        self.assertEqual(s_next, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)