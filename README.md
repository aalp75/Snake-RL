# Snake-RL

Implementation of Reinforcement learning algorithm with Q-learning and Deep Q-learning for the snake game.

**Summary**

1. Article
2. Code and instructions
3. Data
4. References
5. Author

**1. Execution**

To execute the snake
-- python agent.py --

You can change few parameters, either you train the model by modifying the variable * *MODE="train"* * or either you play with already trained model * *MODE="play" * *.

2 models are available: Q-learning and Deep Q-learning, you can between the both by changing the variable * *MODEL* *.

**2. Presentation**

We use a vector of size 11 to representating the state, each value can take 0 or 1.

[danger straight, danger right, danger left, direction left, direction right, direction up, direction down]

**3.Files **

- agent.py: The main files, representing the agent

- Qlearn.py: the Q-learn model

- Financial data.ipynb: Application of the theorical result on real financial datas. First on S&P daily returns then Euro Stoxx 50.

- Improving density.ipynb: First extension of the article

- Multi-dimensional.ipynb: Second extension of the article

**3. Result**
**3. References**

**4. Author**

aalp75

Last update 25/01/2022
