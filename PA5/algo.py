import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        r"""
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        r"""
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #✅TODO: implement this function
    for ep in range(num_episode):
        # A trajectory will be a tuple of (s, r)
        traj = []           # List to store the behavior policy trajectory
        
        # Initialize and store S_0
        s = env.reset()
        traj.append((s, 0))

        T = float('inf')    # initialize to infinity
        t = 0

        while True:
            if t < T:
                # Take action according to the policy given S_t
                a = pi.action(s)
                # Observe and store the next reward as R_{t+1} and the next state as S_{t+1}
                s, r, done, _ = env.step(a)
                traj.append((s, r))
                 
                if done:
                    T = t + 1   # set the time of termination

            # tau is the time whose estimate is being updated, so it is negative until we have enough steps to update the estimate
            tau = t - n + 1

            # Indicates that we have observed enough steps in the trajectory to update the action-value function for the state-action pair at time tau + n
            if tau >= 0:
                # Compute the return G
                episode_G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    r_i = traj[i][1]
                    episode_G += (gamma ** (i - tau - 1)) * r_i

                # Update the return G with the bootstrapped value from the target policy if we have enough steps to do so.
                if tau + n < T:
                    s_tau_n = traj[tau + n][0]
                    episode_G += (gamma ** n) * V(s_tau_n)

                # Update the weight vector of the value function approximation
                s_tau = traj[tau][0]
                V.update(alpha, episode_G, s_tau)

            if tau == (T - 1): 
                    break
            
            t += 1

