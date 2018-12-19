## deepPongRL
A mini-project to make a deep RL agent to play pong in the openAI gym. This was made in the summer school CCNSS.

Training of the RL agent using [policy gradients](http://karpathy.github.io/2016/05/31/rl/). 

Trainig was done in [OpenAI Gym](https://gym.openai.com/) environment with Python 2.7

Used packages:
- OpenAI Gym
- Keras (TF backend)


[![Example trained](https://github.com/Immiora/deepPongRL/blob/master/openaigym.video.0.8268.video000001.mp4_snapshot.jpg?raw=true)](https://github.com/Immiora/deepPongRL/blob/master/openaigym.video.0.8268.video000001.mp4?raw=true)


![problem_statement](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide2.PNG?raw=true)
We would like to train an RL agent to win at Pong game 

![policy gradients](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide3.PNG?raw=true)
A. Karpathy introduced a policy gradient algorith for training an RL agent. The agent learns by processing pixel information from each frame of the game. The optimal policy (prob(action|image)) is calculated by adding a reward function to the neural network gradient. 

