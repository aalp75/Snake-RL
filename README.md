# Snake-RL

Implementation of Reinforcement learning algorithm with Q-learning and Deep Q-learning for the snake game.

**Summary**

1. Article
2. Code and instructions
3. Data
4. References
5. Author

**1. Execution**

To execute the RL algorithm type in console: *python agent.py*

You can change few parameters, either you train the model by modifying the variable *MODE="train"* or either you play with already trained model *MODE="play" *.

2 models are available: Q-learning and Deep Q-learning, you can between the both by changing the variable * *MODEL* *.

**2. Presentation**

We use a vector of size 11 to representating the state, each value can take 0 or 1.

[danger straight, danger right, danger left, direction left, direction right, direction up, direction down, food left, food right, food up, food down].

And the actions available are [straight, right, left] representing by for example [1,0,0] = straight.

Q-learning is representaing in a matrix of * *state x action* *matrix.

Deep Q-learning is representating by approching the Q function by a neural network of *11x128x128x3*  with *relu* function on each layer and * *softmax* * for output.
We use * *Tensorflow* * to implement the neural network. We use replay experience with last 10000 states in memory and a batch size of 32.


**3.Files**

- agent.py: The main files, representing the agent working with the environment and the model.

- Qlearn.py: the Q-learn model.

- NN: the Deep Q-learning model implemented with * *Tensorflow* *.

- game.py: the snake game using * *pygame* *.

- save_parameters: folder containing models already trained.

**3. Result**

We need around 500 games for achieve good results.

Q-learning: we achieved a record of 98 and average score of 50.

Deep Q-learning: we achieved a record of 75 and an average score of 32.

Q-learning perform better on this case due to the low number of states. If we pass in argument the whole picture in argument of our NN it will achieve better results.

**3. References**

I take the game on Geeksforgeeks and inspired myself about this article: https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/.

*Playing Atari with Deep Reinforcement Learning* Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller, 2013

**4. Author**

aalp75

Last update 25/01/2022
