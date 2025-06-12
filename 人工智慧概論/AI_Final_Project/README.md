# Intro To AI Final Project Report
## RLQunat
### Introduction
Quantitative trading, also known as algorithmic trading, has gained significant popularity in the financial industry due to its potential for generating profitable trades with speed and efficiency. In recent years, there has been a growing interest in utilizing reinforcement learning (RL) techniques to develop trading strategies that adapt and optimize their decision-making process in dynamic market environments.
In this project, we have used DQN, DDQN, Policy Gradient and Trajectory Transformer. There are many approach applying seqence to sequence ML algorithm to Quantitative market, but there is no one combine RL, seq2seq and Quatitative trading together(at least we can not find on the web). Thus we decide to apply these algorithm and evaluate how they performed. This is the report of our result and analysis why it perform good/bad.

### Related Work
[Personae - RL & SL Methods and Envs For Quantitative Trading](https://github.com/Ceruleanacg/Personae)
[Reinforcement Learning for Quantitative Trading](
https://dl.acm.org/doi/pdf/10.1145/3582560)
[RLQuant](https://github.com/yuriak/RLQuant)

### Platform / Environment
[gym-anytrading](https://github.com/AminHP/gym-anytrading)
AnyTrading is a collection of OpenAI Gym environments. Our works are based on this envionment, but modified lots of core part to make the environment more suitable for our algorithm, such as obsevation, reward, and the way to calculate profit.


- Observation space
The obsevation is a 2-D array with shape `window size * signal features`. 
    - Window size 
    The Window size represents the number of previous days' data to be included in observation. 
    - Signal feature
    The signal feature of each day contains 4 elements, `opening price, closing price, high position, low position`.
- Action space 
Unlike other trading environment, we only have two actions, **0=Sell** and **1=Buy**. To achieve hold action, simply do **buy** action in **long** position or do **sell** action in **short** position.
- Reward
In various algorithms, we examined different approaches for calculating rewards. After testing, we identified two distinct reward calculation methods. One of these methods was selected for integration within the Reinforce algorithm, while the other was employed in DQN、DDQN、TT.
    - First way (for DQN、DDQN、TT)
$p_t$ is the stock price at $t$ time step. $p_{t+1}$ means stock price of next time step.

$$
R1(t,a) = 
\begin{cases}
  p_t - p_{t+1}, & \text{if sell} \\
  p_{t+1} - p_t, & \text{if buy}
\end{cases}
$$

This reward simply calculates the price difference, and doesn't care about whether the you are in long or short position. It's derived from the idea that whenever the price rises you should buy and vice versa.
    - Second way (for Reinforce)
$$
R2(t,a) = 
\begin{cases}
  p_t - p_{buy}, & \text{if sell and long position} \\
  0, & \text{otherwise}
\end{cases}
$$

This reward is calculated bases on the actual profit you will get. When you sell and you have stock at that time, your reward is the price difference of current price and the price you bought before.
- Profit
The environment starts with initial fund 1, and everytime you trade, the fund becomes:

$$
fund(t)=fund(t-1)*(p_t/p_{buy})
$$

The final fund is our profit.
- Trade fee
Due to high trading frequency, we disable the trade fee for simplicity in our environment.

We use TSMC(2330) stock to train & test out agent.
Dataset is web-crawled from [證券交易所](https://www.twse.com.tw/zh/index.html)
Due to high trading frequency, we disable the 2 trading fees for simplicity.
- Training Data:
    - Maximum Possible Profit: 251.61
    - ![](https://imgur.com/6xf9YvK.png)

- Testing Data:
    - Maximum Possible Profit: 31.38
    - ![](https://imgur.com/6xf9YvK.png)

### Baseline
#### Moving average crossover method
Since most trading strategies require more info. such as financial statements of the company or intraday trading information, we picked the moving average crossover method, which is well-known and simple.
For each trading point, all the information we use is the last 200 day's Open, Close, High, Low price, which is approximately same but more than what our agent use for making decisions.
##### concept : 
For each day, we will calculate
1. the average price of last 50 days
2. the average price of last 200 days.

If 1 is greater than 2 -> indicates the stock is doing good recently -> buy
If 2 is greater than 1 -> indicates the stock is doing bad recently -> sell 

### Main Approach
#### Normalization
- Reason for normalizing
    - Avoiding Scale Bias
    - Generalization
- method
    - calculate relative change for previous datas

$$
change_{relative} = (data_{now} - data_{previous}) / data_{now}
$$

```python=
for i in range(12):
    for j in range(4):
        tempstate[i*4+j] = (state[44+j] - state[4*i+j])/state[44+j]
        // tempstate is fed into the NN
```
#### Reinforce with GAE
- Introduction
The Reinforce algorithm includes collecting trajectories, computing the policy gradient using Monte Carlo returns, and updating the policy network through gradient ascent. The Reinforce algorithm with GAE extends the basic Reinforce algorithm by incorporating Generalized Advantage Estimation (GAE). GAE addresses the issue of high variance in the policy gradient estimates by providing more accurate estimates of the advantages. This is achieved by combining temporal difference (TD) errors over multiple time steps. By considering future rewards and values, GAE reduces variance and enhances the stability of the training process.

#### DQN
- Inroduction
DQNs work by using a neural network to approximate the action-value function, which maps states of the environment to the expected return for each possible action. The goal of the DQN is to learn the optimal policy, which is the action that will maximize the expected return for each state.

$$
\\
Q_{evaluate}(s_t,a) = Q(s,a) + a(r + \gamma maxQ(s_{t+1},a) - Q(s,a))
\\
Target Q = r + \gamma maxQ(s_{t+1},a)
$$

#### DDQN
- introduction
DDQN is an extension of the DQN algorithm. In DQN, the Q-values are often overestimated, and DDQN is used to address this issue. By using a separate target network and decoupling the action selection and value estimation steps, DDQN reduces the overestimation bias observed in DQN and leads to more accurate Q-value estimates. This, in turn, can result in improved performance and faster convergence in reinforcement learning tasks.

$$
\\
Q_{evaluate}(s_t,a) = Reward_{t+1} + \gamma Q_{target}(s_{t+1}, \max_a(Q_{evaluate}(s_{t+1}, a)))
\\
$$

#### Trajectory Transformer (TT)
- Offline algorithm
- Introduction
Trajectory transformer is a sequence-to-sequence model that can predict sequences of states, actions, and rewards. This makes it well-suited for tasks such as planning and imitation learning. Trajectory transformers are trained on a dataset of trajectories, which are sequences of states, actions, and rewards that an agent has taken in an environment. The model learns to predict the next state, action, and reward given the current state and action.
![](https://hackmd.io/_uploads/B1M2os_82.png)
TT is a Transformer decoder mirroring the GPT architecture. TT use a smaller architecture than those typically used in large-scale language modeling, consisting of four layers and four self-attention heads. 
- Training
The traning process is like the picture above. We use precollected trajectory datas and separate $(s_1, a_1, r_1, s_2, a_2, ..., s_t, a_t, r_t, ...)$ tuple to $(s_1, s_2, ..., s_t, a_1, a_2, ..., a_t, r_t...)$ $t=1,...,T$
Training is performed with the standard teacher-forcing procedure used to train sequence models. Denoting the parameters of the Trajectory Transformer as $\theta$ and induced conditional probabilities as $P_\theta$, the objective maximized during training is:

$$
{\cal{L}}(\bar{\tau})=\sum_{t=0}^{T-1}(
    \sum_{i=0}^{N-1}logP_{\theta}(\bar{s}_{t}^{i}|\bar{s}_{t}^{<i},\bar{\tau}_{<t}) + 
    \sum_{i=0}^{M-1}logP_{\theta}(\bar{a}_{t}^{i}|\bar{a}_{t}^{<i},\bar{s}_t,\bar{\tau}_{<t}) +
    logP_{\theta}(\bar{r}_{t}|\bar{a}_{t}^{<i},\bar{s}_t,\bar{\tau}_{<t})
)
$$

- Planning
Beam search is a search algorithm that can be used to find the most likely sequence of tokens given a probability distribution over sequences. In the context of trajectory transformer, beam search is used to find the most likely trajectory given a model that predicts the probability of a sequence of states and actions.

$$
trajectory = (s_1, a_1, r_1, R_1, s_2, a_2, r_2, R_2, ,...,s_t, a_t, r_t, R_t, s_{t+1}) \\
a_{t+1} = TT_{\theta}(trajectory)
$$

### Evaluation Metric


#### Total Profit
#### Testing Environment

### Result & Analysis
#### Baseline
#### Policy Gradient
- Training Loss
- Rewards
- Total Profit
- Discussion
#### DQN
- Training Loss
- Rewards
- Total Profit
- Discussion
#### DDQN
- Training Loss
- Defferent training epoch get differet testing result.Training 1000 epoch and test every 200 epoch.
    - Training bound
    
    | Training epoch | Total Reward | Total Profit |
    | -------- | -------- | -------- |
    | 200      | 1931.5     | 109.31     |
    | 400      | 2063.5     | 150.94     |
    | 600      | 2162.5     | 193.63     |
    | 800      | 2220.5     | 217.85     |
    | 1000     | 2254.5     | 232.90     |
    - Testing bound
    
    | Training epoch | Total Reward | Total Profit |
    | -------- | -------- | -------- |
    | 200      | 102.0     | 1.63     |
    | 400      | -14.0     | 1.54     |
    | 600      | -26.0     | 1.39     |
    | 800      | -95.0     | 1.27     |
    | 1000     | -250.0     | 1.05     |

- Discussion
    - Overfitting Issues

#### Trajectory Transformer
- Training Loss
Use DDQN-1000 to collect data trajectory.
![](https://hackmd.io/_uploads/r17KqejI3.png)
- Defferent training epoch get differet testing result.
    - Training bound
    
    | Training epoch | Total Reward | Total Profit |
    | -------- | -------- | -------- |
    | 100      | 2202.5     | 212.68     |
    | 200      | 2202.5     | 212.68     |
    | 250      | 2202.5     | 212.68     |
    | 300      | 2202.5     | 212.68     |
    | 400      | 2202.5     | 212.68     |
    | 450      | 2202.5     | 212.68     |
    | 500      | 2202.5     | 212.68     |
    
    - Testing bound
    
    | Training epoch | Total Reward | Total Profit |
    | -------- | -------- | -------- |
    | 100      | -175.0     | 1.15     |
    | 200      | -181.0     | 1.104     |
    | 250      | -230.0     | 1.014     |
    | 300      | -126.0     | 1.16     |
    | 400      | 150.0     | 1.587     |
    | 450      | 58.0     | 1.564     |
    | 500      | 236.0     | 1.615     |
    
- Discussion
In TT, we can get higher profit from testing environment compared to data resource DDQN.

### Conclusion
- In online algorithm, DDQN performs best in three RL algorithms.
- Trajectory Transformer perform best in four algorithm
- Add more algorithm in the future (like PPO, DDPG, Decision Transformer)
- Test on more stock data in the fure
