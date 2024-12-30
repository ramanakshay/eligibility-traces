# Continual-Policy-Gradient

## About

Policy gradient methods are a family of algorithms in reinforcement learning that optimize and model the policy directly. In this project, I implement incremental learning versions of policy gradient methods which allows an agent to simultaneously learn while interacting with the environment. This is achieved by using eligibility traces with deep neural networks in PyTorch.

## Run

```

1. Install dependencies from requirements file. Make sure to create a virtual environment before running this command.
```
pip install -r requirements.txt
```

2. cd into source folder and then run the code.
```
cd continual-learning
python main.py
```


## Batch vs. Continual Policy Gradient

Batch policy gradient methods essentially use stochastic gradient descent to estimate the gradient. Therefore, larger batch sizes and larger batch rollouts would give better and more accurate results. This however is not scalable since large rollouts imply huge data buffers to store encountered states, actions, and rewards.

<p align = "center">
<img align="center" src="assets/images/agent_env.jpg" alt="Hash Tree" width = "750" />
</p>

Continual policy gradient method do not update the agent using stored batches. Instead, the algorithm must update the agent only using information available at the current (and last) timestep.

## GAE and Eligibility Traces

Generalized Advantage Estimation (GAE) is a popular advantage estimate, often preferred, due to its ability to control the bias and variance of the gradient estimate. Another benefit of GAE is that it can be reformulated to work with continual policy gradients. This is done by employing a trick known as eligibility traces which stores a temporary record marking the occurrence of an event. For deep neural networks, we associate a trace, along with the gradient, for each weight in the network. At each step, we update the trace by decaying the existing trace and adding the gradient associated with the action.

Eligibility traces can be implemented in Pytorch by designing a custom optimizer for the actor and critic networks. A trace is associated for each weight in the network. Furthermore, special functions must be implemented to set/reset the trace and also broadcasting the TD error across the network.

## Results

The first figure compares the performance of batch and continual GAE. We can see that both approaches converge to the same solution. The second figure tests the algorithm on the same environment with different values of lambda.

<p>
<img  src="assets/images/traces.png" alt="Hash Tree" width="350"/>

<img src="assets/images/lambda.png" alt="Hash Tree" width="350"/>
</p>

## Note

A more detailed explanation (with math) can be found in report.pdf


