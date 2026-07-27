import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # ✅TODO: implement this method
        # For mountain car, the dimensions are position on the x-axis and velocity, so state_low gives you the minimum position and velocity, and state_high gives you the maximum position and velocity.
        self.state_low = state_low
        self.num_tilings = num_tilings
        # The tile_width is the width of each tile in each dimension - position and velocity.
        self.tile_width = tile_width

        # Gives the number of tiles for each dimension
        n_tiles = np.ceil((state_high - state_low) / tile_width) + 1
        self.n_tiles_int = n_tiles.astype(int)

        # n_weights is the product of the number of tiles in each dimension.
        self.n_weights = np.prod(self.n_tiles_int)

        # Initialize weights to zero.  Shape of the weight is (n_weights, num_tilings)
        self.w = np.zeros((self.n_weights, num_tilings))

    def __call__(self,s):
        # ✅TODO: implement this method
        V = 0.0
        for t in range(self.num_tilings):
            flat_index = self.get_flat_index(s, t)

            V += self.w[flat_index, t]
        return V

    def update(self,alpha,G,s_tau):
        # ✅TODO: implement this method
        delta = alpha * (G - self.__call__(s_tau))
        for t in range(self.num_tilings):
            flat_index = self.get_flat_index(s_tau, t)
            self.w[flat_index, t] += delta
        return None

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
        index = np.clip(index, 0, self.n_tiles_int - 1)

        return np.ravel_multi_index(list(index), tuple(self.n_tiles_int))