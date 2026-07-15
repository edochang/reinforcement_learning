# Instruction


In this programming assignment, we will implement the n-step semi-gradient TD(0) algorithm described on page 209 (2nd edition of the textbook) with two function approximation methods: (1) linear approximation with tile-coding and (2) neural networks.

## Setup

Please familiarize yourself with the Gym interface since it is widely used currently in the RL community. In this assignment, you will also be using it to collect trajectories on which to execute the function approximation methods you will implement.

Please install the required dependencies by running `pip install -r requirements.txt`. Note that we are using Gym instead of Gymnasium (which was used in the previous assignments).

You need to implement one function and two classes. Similar to the previous assignment, you have to keep the function and class interface intact sincpython polie it will be auto-graded later.

We provide a policy in policy.py to be evaluated for testing. The policy is for an OpenAI gym environment called "MountainCar-v0". "MountainCar-v0" is a simple but traditional RL environment that has a continuous two-dimensional state space and a discrete action space. You can check your installation by running `python policy.py`. If you see a popup window with a car escaping from the valley, then you are all set.

## N-Step TD Algorithm (0 pt)
In this part, we implement n-step semi-gradient TD(0) algorithms as described on page 209 of the textbook. Open `algo.py` and implement the  `semi_gradient_n_step_td` method. You should use the two methods of `ValueFunctionWithApproximation` class.

Note that while you do not need to submit your code for this part, you will need it to test the subsequent tile coding and NN approximation codes.

## Tile Coding (1 pt)
Now, we will implement a linear function approximation with tile coding. Open tc.py and fill in the parts indicated as 'TODO'.

Here are a few tips & specific details you need to be careful with when implementing tile coding:

- Each tiling should cover the entire space completely so that every point is included by the same number of tilings.
- In order to achieve that, # tiles in each dimension should be: ceil[(high-low)/tile_width] + 1.
- Each tiling should start from `(low - tiling_index / # tilings * tile width)` where the tiling index starts from 0. With the starting offset, you can easily find out the corresponding tile for each tiling given the state.
- You might find the Numpy function `ravel_multi_index` useful when assigning tiles to a given state.

Note: It's possible to have the wrong number of tilings in `test.py` due to floating point errors that cause your implementation to use one tile more or less. If that happens, try slightly increasing the tile width specified in `test.py` by changing line 20 to

`tile_width=np.array([.451,.0351]))`

## Neural Networks (1 pt)
Now, we will approximate a value function using a neural network. Open nn.py and fill in the parts indicated as 'TODO'. Please use Pytorch (and not Tensorflow) for this assignment. If you are not familiar with Pytorch, please see some tutorials before working on this part. For the autograder, we use pytorch 1.13.1 but pytorch 2.x.x should work.

To guarantee convergence, here are a few specifics to follow:

Use AdamOptimizer with beta1=0.9, beta2=0.999. For the learning rate \alpha, you can assume the \alpha is always 0.001.
Use four layers in total. After the hidden layers, use a ReLU for activation.  The final layer does not have any activation function.
For clarification, the input layer should go to 32 nodes. These 32 nodes should go to another 32 nodes. These nodes should then go to the output layer.  There are four layers and three connections between layers. (The connections are really what you are implementing with your code when you use torch.nn.Linear(previous_layer_dims, next_layer_dims)).
Set your model to eval mode with `model.eval()` for implementing `__call__()` for the value function approximator
Set it back to train mode with `model.train()` and call `optimizer.zero_grad()` before calling `loss.backward()` when implementing the gradient update.

## Testing
You can test your implementation with the provided `test.py` script. Please run the script with `python3 test.py`, and check whether it passes all the assertions we've placed there. However, note that your code will be tested with a different policy and on different states. Hence, you might want to test your implementation more thoroughly with different policies, such as a random policy.

## Tips
You should try to reuse as much of the code you implemented for n-step algorithms in the previous assignment as you can.
As `test.py` indicates, the tests might fail due to stochasticity. This may happen with tile coding and/or Neural Networks (although more so with Neural Networks).
For the Neural Network, you might want to start by allowing more training iterations (say 2000) before trying to get it working with the default 1000.
Even so, the tests might fail, and that's not necessarily bad. For either part, getting values in the same ballpark (deviations of about 30) as the true values is a good sign that things are working fine. The grader is set to a high tolerance, so do not be afraid to submit if this is consistently the case.
