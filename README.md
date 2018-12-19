## DeepPongRL
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

![deep q-learning](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide4.PNG?raw=true)
As an alternative to policy gradients we looked at the deep Q-learning algorithm. 

![best model performance](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide6.PNG?raw=true)
One of our best models (Policy gradient, MLP 1 hidden layer, 200 hidden units, relu activation) learnt to beat the built-in AI agent in OpenAI Pong environment.

![best model training](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide7.PNG?raw=true)
The policy gradient model was trained on ~5000 episodes and almost reached 0-reward.

![model comparison](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide11.PNG?raw=true)
We did some preliminary model comparison and observed that shallower models (1-layer) converged faster (final performance subject to number of training episodes). We also saw that policy gradient training resulted in faster and more graduate training compared to deep Q-learning.

![training details](https://github.com/Immiora/deepPongRL/blob/master/final_report/Slide13.PNG?raw=true)
Our final cote concerned GPU vs CPU computational time differences. We report that smaller networks are trained faster on a CPU unit.