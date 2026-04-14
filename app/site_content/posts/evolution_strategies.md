---
author: Nima Manafzadeh Dizbin
title: Evolution Strategies for Solving Reinforcement Learning Problems
date: 2021-11-20
description: A brief guide to using evolution strategy for solving reinforcement learning problems
math: true
draft: true
---

# Evolution Strategies (ES)
Evolution Strategy is a population based stoachstic continuous optmization methodology. It is related to the familiy of evolutionary population based algorithms such as the Genetic AAlgorithm. 
ES has been traditionally used for optimizing lower dimensional poroblems. However, recent works demonstrate that evolutionary algorithms can be scaled to optimize neural networks 
with millions of parametrs. 

In this tutorial, we will first intorduce the basics of different variants of the ES. Then, we show how to use ES for optimizing supervised learning and reinforcement learning tasks. 

## Simple Evolution Strategies
The most simple variant of the ES consists of modelling the optmization parametr space with  

is the most basic and canonical version of evolution strategies. It models $p_\theta(x)$ as a $n$-dimensional isotropic Gaussian distribution, in which $\theta$ only tracks the mean $\mu$ and standard deviation $\sigma$.
    \begin{equation}
        \theta = (\mu, \sigma),\;p_\theta(x) \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 I) = \mu + \sigma \mathcal{N}(0, I)
    \end{equation}

The process of Simple-Gaussian-ES, given $x \in \mathcal{R}^n$:
\begin{enumerate}
    \item Initialize $\theta = \theta^{(0)}$ and the generation counter $t=0$
    \item  Generate the offspring population of size $\Lambda$ by sampling from the Gaussian distribution:
    \begin{equation}
        D^{(t+1)}=\{ x^{(t+1)}_i \mid x^{(t+1)}_i = \mu^{(t)} + \sigma^{(t)} y^{(t+1)}_i, 
    \end{equation}
    where 
    \begin{equation}
        y^{(t+1)}_i \sim \mathcal{N}(x \vert 0, \mathbf{I}),;i = 1, \dots, \Lambda\}.
    \end{equation}
## Natural Evolution Strategies 
The main idea behind the natural evolution strategy is to use all of the population in estimating a gradient signal that can generate a better population in the iteration of the algorithm. 

## Covariance MAtrix Adaptation Evolution Strategies



## Optmization with Evolution Strategy
In this tutorial, we use the ask/tell framework for optmizing functions summerized as follows. At each iteration of the algorithm, we ask the optimizer for a new set of candidate solutions. Then, we evaluate the performance of each candidate solution using the ```get_rewards_method``` and tell the evaluated fitnesses to the optimizer. The optmizer uses the new sets of rewards to update its parameters. In the next step of the optmization a new set of solutions are generated based on the updated parameters. The main body of the optmization procedure can be written as: 
```python 
    es = ES()
    for epoch in range(1, number_of_training_epochs + 1):
        solutions = es.ask()
        rewards =  get_rewards_method(solutions) 
        es.tell(rewards)
```

### Example

```python
def sum_of_squares(x):
    """
    Returns the sum of squared of the values in the inout vector x 
    """
    sum_value = 0.0
    n = len(x) 
    for i in range(0,n):
        sum_value += x[i]*x[i]
    
    return sum_value


def get_fitness_method(solutions):
    population_fitness = []
    for sol in solutions: 
        individual_fitness = sum_of_squares(sol)
        population_fitness.append(individual_fitness)
    
    return population_fitness

problem_dimension = 100
number_of_optimization_episodes = 1000

initial_solutions = np.array([1]*problem_dimension)
es_population_size = 50 
sigma = 0.1
esopt = ESOpt(x0=initial_solutions, sigma=sigma, population_size=es_population_size) 
for _ in range(number_of_optimization_episodes):
    solutions = esopt.ask()
    population_fitness = get_fitness_method(solutions)
    esopt.tell(population_fitness)
```



# ES for Deep Reinforcement Learning
The application of evolutionary algorithms for solving RL problem is not new. [this](https://arxiv.org/abs/1106.0221). 

However, it was the work of [Salimans](https://arxiv.org/abs/1703.03864) that showed ES as a scalable alternative to the RL based algorithms. 

## Evolution Strategy for Optimizing Neural Networks using Pytorch
In order to 
We use a module called 
```python

import torch

class ESModule(torch.nn.Module):

    def get_model_shapes(self):
        """
        returns the shapes of the model parameters
        """
        model_shapes = []
        for param in self.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            param.requires_grad = False
        return model_shapes

    @property
    def model_shapes(self, ):
        return self.get_model_shapes()

    def set_model_params(self, flat_params):
        """
        Sets the current parametrs of the  neural network to the flat_params based on the model shapes
        """
        assert 
        model_shapes = self.model_shapes
        idx = 0

        for i, param in enumerate(self.parameters()):
            delta = np.product(model_shapes[i])
            block = flat_params[idx: idx + delta]
            block = np.reshape(block, model_shapes[i])
            idx += delta
            block_data = torch.from_numpy(block).float()
            param.data = block_data

        return self



