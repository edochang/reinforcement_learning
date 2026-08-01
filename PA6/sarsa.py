import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # ✅TODO: implement here
        # For mountain car, the dimensions are position on the x-axis and velocity, so state_low gives you the minimum position and velocity, and state_high gives you the maximum position and velocity.
        self.state_low = state_low
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        # The tile_width is the width of each tile in each dimension - position and velocity.
        self.tile_width = tile_width

        # Gives the number of tiles for each dimension; shape(2, ) for mountain car, where the first element is for position dimension and the second element is for velocity dimension.
        n_tiles_array = np.ceil((state_high - state_low) / tile_width) + 1
        self.n_tiles_array_int = n_tiles_array.astype(int)

        # n_weights is the product of the number of tiles in each dimension.
        self.n_weights = np.prod(self.n_tiles_array_int)

        # Initialize weights to zero.  Shape of the weight is (n_weights, num_tilings)
        self.w = np.zeros((self.n_weights, num_tilings))

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # ✅TODO: implement this method
        # Note: self.n_weights is num_tiles, which is the product of the number of tiles in each dimension.
        return self.num_actions * self.num_tilings * self.n_weights

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # ✅TODO: implement this method
        # Note: S+ = set of all states, including the terminal state
        # Note: A = set of all actions
        # Remember the d formulation in feature_vector_len() = num_actions * num_tilings * num_tiles.  So [0,1]^d is the feature vector.  It will have a length of d, and will have a single 1 at the index that corresponds to the tile that the state-action pair (s, a) belongs to across all tilings. 0 for the rest of the tiles in the tiling.
        # In other words can expect a 1 in each tiling slice in the feature vector and 0 for the rest of the tiles in the tiling.
        x = np.zeros(self.feature_vector_len())
        if not done:
            for tiling in range(self.num_tilings):
                # This is the index of the tile for the given state and tiling.
                flat_index = self.get_flat_index(s, tiling)
                # Expand the index to account for the action.  Each action has its own set of tiles, so we need to offset the index by the number of tiles fo each action.
                index = flat_index + tiling * self.n_weights + a * self.num_tilings * self.n_weights
                x[index] = 1
        return x

    def get_flat_index(self, s, t) -> int:
        """
        Get the index of the tile for a given state and tiling

        input:
            self: the instance of the class
            s: the state
            t: the tiling index
        output:
            int: the index
        """
        offset = self.state_low - ((t * self.tile_width) / self.num_tilings)
        index = np.floor((s - offset) / self.tile_width).astype(int)

        # Clip indices to valid range [0, n_tiles-1]
        # np.floor can give negative indices, so we need to clip to be at least 0.  Can set upper bound
        index = np.clip(index, 0, self.n_tiles_array_int - 1)

        # Note 1: np.ravel_multi_index expects the dimensional coordinates in a list and not a numpy array, so we need to convert it.  Example: [dim1, dim2] instead of np.array([dim1, dim2])  
        # Note 2: tuple converts the numpy array to a tuple to know how many tiles there are in each dimension, so if there are 2 tiles per dimension this will be (2, 2).
        return np.ravel_multi_index(list(index), tuple(self.n_tiles_array_int))


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    # Initialize w
    w = np.zeros((X.feature_vector_len()))

    #✅TODO: implement this function
    # This is section 12.7 Sarsa Lambda page 307 that goes over the True oneline Sarsa Lambda algorithm
    # Loop for each episode
    for ep in range(num_episode):
        done = False

        # Initialize S
        s = env.reset()

        # Choose A using epsilon-greedy policy
        a = epsilon_greedy_policy(s, done, w)

        x = X(s, done, a) # x(S, A)

        # Initialize the elgibility trace: z = 0
        z = np.zeros((X.feature_vector_len()))

        Q_old = 0

        # Loop for each step of episode
        while not done:
            # Take action A, observe R, S'
            # We capture done here because we need to know if the episode has terminated to satisfy the condition "until S' is terminal".
            s, r, done, _ = env.step(a)

            # Choose A' using epsilon-greedy policy
            a = epsilon_greedy_policy(s, done, w)

            x_new = X(s, done, a) # x(S', A')
            Q = w @ x
            Q_new = w @ x_new
            delta = r + gamma * Q_new -Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * (z @ x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_new
            x = x_new

            # a = a_new is already done in the loop above, so we don't need to do it again here.

    # Return the learned `w` after all episodes have been completed.  `w` is used to compute the action-value function: Q(s, a) = w^T * x(s, a).
    return w
