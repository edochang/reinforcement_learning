from typing import Iterable
import numpy as np
import torch
import torch.nn as nn

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # ✅TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size
        
        # Softmax activation function is used to convert the output of the neural network into a probability distribution over actions.  The softmax function takes a vector of real-valued scores (logits) and transforms them into probabilities that sum to 1.
        # dim=-1 indicates that the softmax function is applied along the last dimension of the input tensor, which corresponds to the action dimension.
        self.nn = NN(state_dims, output_dim=num_actions)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr = alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        # ✅TODO: this should return an action sampled from the policy according to probability distribution pi(.|s)
        # Note: some students implemented this method as a function that returns the probability of each action. This is incorrect,
        # and although it might pass locally, it will fail the autograder. 
        
        # Set the network model to evaluation mode
        #self.nn.eval()

        # Convert the state to a tensor and add a batch dimension
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        # Get raw logits from NN, apply softmax to get probability distribution
        with torch.no_grad():
            logits = self.nn(s_tensor)
            action_prob = nn.functional.softmax(logits, dim=-1).flatten() # shape(num_actions, )

            # Sample an action from the probability distribution.  Using torch.multinomial to sample an action.  The input to torch.multinomial is a tensor of probabilities, and it returns the index of the sampled action.
            return torch.multinomial(action_prob, num_samples=1).item()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # ✅TODO: implement this method
        # We will do this at the caller level
        # Set the network model to training mode
        #self.nn.train()

        # Convert the state to a tensor and add a batch dimension
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)        

        # Get raw logits from NN
        logits = self.nn(s_tensor)  # shape(1, num_actions)

        # Two ways to compute the loss: 
        # 1) Calculating directly and get the log probability of the action taken.
        #log_prob = nn.functional.log_softmax(logits, dim=-1)
        #neg_log_prob = -log_prob[0, a]
        
        # 2) Using pytorch cross entropy loss which is used for multi-class classification problems like this one with multiple actions.  It combines log_softmax and negative log likelihood loss in one function.  This is supposedly more numerically stable.
        a_tensor = torch.tensor([a], dtype=torch.long) # shape (1, )
        neg_log_prob = nn.functional.cross_entropy(logits, a_tensor, reduction='none')

        # Compute REINFORCE loss: gamma^t * delta * log(pi(a_t|s_t))
        loss = (neg_log_prob * gamma_t * delta).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # ✅TODO: implement here
        self.nn = NN(state_dims)
        # Potential to give a slower learning rate for the value function than the policy.  If the value function learns too quickly, it can lead to instability in the policy updates.
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr = alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        # ✅TODO: implement this method
        # Convert the state to a tensor and add a batch dimension
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        # Set the network model to evaluation mode
        #self.nn.eval()

        # Pass the state through NN model
        with torch.no_grad():
            value = self.nn(s_tensor).item()

            # Return the value estimate as a float
            return float(value)

    def update(self,s,G):
        # ✅TODO: implement this method
        # We will do this at the caller level
        # Set the network model to training mode
        #self.nn.train()

        # convert G (scalar) to a tensor and add a batch dimension to address shape warning
        G_tensor = torch.tensor([[G]], dtype=torch.float32)

        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        logits = self.nn(s_tensor)

        loss = torch.nn.functional.mse_loss(logits, G_tensor, reduction='sum')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
# Universal Neural Network class for both PiApproximationWithNN and VApproximationWithNN
class NN (nn.Module):
    def __init__(self, state_dims, output_dim=1):
        super().__init__()
        # 2 hidden layers
        # 32 neurons for each hidden layer
        self.hidden = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # head as the output layer
        self.head = nn.Linear(32, output_dim)  

    def forward(self, x):
        # Pass the input through the hidden layers and then through the output layer.
        return self.head(self.hidden(x))

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # ✅TODO: implement this method
    # The output to return
    G_0s = []

    # pi is the differentiable policy approximation input pi(a|s, theta)
    for ep in range(num_episodes):         
        # Generate an episode following pi(·|·, theta): S_0, A_0, R_1,...,S_T-1, A_T-1, R_T
        traj = []
        done = False
        s = env.reset()
        while not done:
            a = pi(s)    
            s_next, r, done, _ = env.step(a)
            traj.append((s, a, r))
            s = s_next
        
        # Pre-calculate G for each time step t to improve performance to O(T) using the reversed() function used in previous PAs.
        T = len(traj)
        Gs = [0.00] * T
        G = 0
        for t in reversed(range(T)):
            _, _, r = traj[t]
            G = gamma * G + r
            Gs[t] = G

        G_0s.append(Gs[0])

        for t, step in enumerate(traj):
            # This is the book algorithm but it not efficient (O(T^2) time complexity).
            '''
            G = 0
            T = len(traj) 
            # Note: R_1 is at index 0 and R_T is at index T-1 in trajs, so k = t and gamma exponent is k-t.
            for k in range(t, T):
                _, _, r_k = step[k]
                G += (gamma ** (k - t)) * r_k
            
            if t == 0:
                G_0s.append(G)
            '''

            G = Gs[t]

            s, a, _ = step
            # Note: no baseline is controlled by setting V to a constant like Baseline(0.) as shown in test_reinforce.py.
            baseline = V(s)
            delta = G - baseline
            # Update the baseline (value function) using the observed return G at each time step (t).
            V.update(s, G)
            pi.update(s, a, gamma ** t, delta)

    return G_0s

