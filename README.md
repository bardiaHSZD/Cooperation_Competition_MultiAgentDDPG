
# Collaboration-and-Competition
Deep Reinforcement Learning Nanodegree Project 3 (Multiagent RL)

<div>
<img src="tennis.gif" width="600" align="left"/>
</div>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Getting Started
1.Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click her](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

2.Place the file in the DRLND GitHub repository, in the p3_collab-compet/ folder, and unzip (or decompress) the file.

### Instructions
Follow the instructions in Tennis.ipynb to get started with training your own agent!

The MADDPG algorithm for mixed cooperative-competitive environments is summarized below:
<div>
<img src="algorithm.PNG" width="600" align="left"/>
</div>

## Results
Once all of the above components were in place, the agents were able to solve the Tennis environment. Again, the performance goal is an average reward of at least +0.5 over 100 episodes, taking the best score from either agent for a given episode.

`Episodes 0000-0100	Maximum Score: 0.200	 Average Over 100 Episodes: 0.022`

`Episodes 0100-0200	Maximum Score: 0.400	 Average Over 100 Episodes: 0.049`

`Episodes 0200-0300	Maximum Score: 0.500	 Average Over 100 Episodes: 0.082`

`Episodes 0300-0400	Maximum Score: 0.800	 Average Over 100 Episodes: 0.114`

`Episodes 0400-0500	Maximum Score: 2.900	 Average Over 100 Episodes: 0.385`

`Environment solved in 534 episodes with an average of 0.501 over the last 100 episodes`



The graph below shows the final training results. The best-performing agents were able to solve the environment in 534 episodes. The complete set of results and steps can be found in this notebook.

<div>
<img src="results.png" width="400" align="left"/>
</div>


