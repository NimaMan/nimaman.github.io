---
author: Nima Manafzadeh Dizbin
title: Optiver Quantitative Researcher Question - Mean Absorbing Time of a Two dimensional Random Walk
date: 2022-09-03
description: Mean Absorbing Time of a Two dimensional Random Walk
math: true
draft: false
---

# Question

An ant leaves its anthill in order to forage for food. It moves with the
speed of 1 unit per time step, but it doesn’t know where to go, therefore
every time step it moves randomly 1 unit directly north, south, east or west
with equal probability.

1.  If the food is located on east-west lines 2 units to the north and 2 units to
the south, as well as on north-south lines 2 units to the east and 2 units to
the west from the anthill, how long will it take the ant to reach it on
average?

2.  What is the average time the ant will reach food if it is located
    only on a diagonal line passing through (1, 0) and (0, 1) points?

3.  Can you write a program that comes up with an estimate of average
    time to find food for any closed boundary around the anthill? 
    
# Solution

I am not sure whether the food is located on the set of points in \\({(0, 2), (0, -2), (2, 0), (-2, 0)}\\) or horizontal and vertical lines
passing through these points. If the question is asking for the mean time
to reach the lines passing through these points the answer is pretty straightforward.
It can be found using the code for solving question number three which
solves a set of linear equations for the mean time to get absorbed into
states outside a given boundary. Otherwise, I solve this question in the following way.

Let \\(s_t = (i_t, j_t)\\) denote the distance of the ant on the
east-west and south-north directions at time t where
\\(t \in {0, 1, 2, 3, \dots\} \\) denotes the number of time steps passed
since the ant has started moving. Notice that the movement of the ant
has a Markovian property as the position of the ant only depends on its
latest position. Mathematically speaking:

$$
P \left(s_{t+1}=(i_{t+1}, j_{t+1}) | s_{t}, s_{t-1}, s_{0}\right) = P \left(s_{t+1}=(i_{t+1}, j_{t+1}) | s_{t}\right).
$$
Markovian property states that regardless of the history of the
movements of the ant the mean number of steps to enter a given state is
only dependent on the current state or position of the ant. I will use
this property in calculating the time it takes for the ant to reach the
food on average starting from a given position.

## Q1

Let \\( Z^{A} = \{(0, 2), (0, -2), (2, 0), (-2, 0) \} \\) be the set of
coordinates where the food is located and
\\( V(s) = \mathop{\mathrm{\mathbb{E}}}\left[\min\{n: s_n \in Z^{A}|s_0=s\} \right], \\)
be the firs time that the ant enters a state in \\(\(Z^{A}\) \\) (the time that
the ant reaches the food) starting from the state \\( s = (i, j)\\). In
order to find the mean number of steps to reach the food, I condition on
the current state of the ant. Assume that the ant is currently in state
\\((i, j) \notin Z^{A}\\). The mean number of steps for the ant to enter a
state in \\(Z^{A} \\) is equal to taking one step in a random direction
plus the mean number of steps to enter \\(Z^{A}\\) from the new
state. Since we move in each direction randomly with probability
\\(\frac{1}{4}\\) we can write:
$$
V((i, j)) = 1 +  \frac{1}{4} V((i+1, j)) + \frac{1}{4} V((i-1, j)) + \frac{1}{4} + V((i, j+1)) + \frac{1}{4}V((i, j-1)).
$$ 
We are interested in finding \\(V((0, 0))\\) to calculate the mean number
of steps for the ant to reach a food. We need to solve the two-dimensional recursive equations given in the above Equation 
for \\(i, j \in \{ -\infty, \dots, +\infty\}\\) to calculate \\(V((0, 0))\\).
Solving these equations maybe cumbersome. Hence, I will partition the state
space into a set of points that have the same distance to the closest
point in \\(Z^A\\) to simplify the problem. Define \\((i^N, j^N)\\) as the
nearest point in \\(Z^A\ \\) to position \\((i, j)\\)
$$
(i^N, j^N) = \mathop{\mathrm{arg\,min}}_{(u, v) \in Z^{A}} |u - i| + |v - j|,
$$
where \\(|i|, |j| \geq 1\\). I cluster the states into the following sets:


$$
\begin{aligned}
 AR(p): Y_i &= c + \epsilon_i + \phi_i Y_{i-1} \dots \\
 Y_{i} &= c + \phi_i Y_{i-1} \dots
\end{aligned}
$$


\begin{align*}
    Z^{0} &=  \{(0, 0)\}, \\ 
    Z^{-1} &=  \{(1, 0), (-1, 0), (0, 1), (0, -1)\}, \\
    Z^{-2} &=  \{(1, 1), (1, -1), (-1, 1), (-1, -1)\}, \\
    Z^{d} &=  \{(i, j): |i - i^N| + |j - j^N| = d \},  d\geq 1.
\end{align*}


$$
\begin{aligned}
  2x+3 &= 7 & 2x+3-3 &= 7-3 \\\\\\
  2x &= 4 & \frac{2x}2 &= \frac{4}{2}\\\\\\
  x &= 2
\end{aligned}
$$

One-step transition from each of these sets results in moving to the
neighboring sets. Following the Markovian property and symmetric nature
of the problem \\(V^\gamma = V((i, j)), \ \forall (i, j) \in Z^\gamma\\)
where \\(\gamma \in \{-2, -1, 0, 1, 2, \dots \}\\). The mean number of steps
to enter \\(Z^A\\) starting from a point in \\(Z^\gamma\\) can be written as

$$
\begin{align}\label{r1}
    V^{-2} &=  1 + \frac{1}{2}V^{-1} + \frac{1}{2}V^{1}, \nonumber \\
    V^{-1} &=  1 + \frac{1}{2}V^{-2} +  \frac{1}{4}V^{0}, \nonumber \\
    V^{0} &=  1 + V^{-1}, \nonumber \\ 
    V^{1} &=  1 + \frac{1}{2}V^{-2} +  \frac{1}{2}V^{2},\nonumber \\
    V^{d} &=  1 + \frac{1}{2}V^{d-1} + \frac{1}{2}V^{d+1}, \ \forall \in \{2, 3, \dots\}. 
\end{align}
$$
After some algebra these equations can be re-written in the following
form:
$$
\begin{align}
 \label{r12}
    V^{-1} &=  \frac{2}{3}V^{-2} + \frac{5}{3}, \nonumber\\
    V^{-2} &=  \frac{3}{4}V^{1} +  \frac{5+6}{4}, \nonumber\\
    V^{1} &=  \frac{4}{5}V^{2} +  \frac{5+6+8}{5}, \nonumber\\
    V^{n} &=  \frac{n+3}{n+4}V^{n+1} + \frac{5+\sum\limits_{j=0}^{n} (6+2(n-j))} {n+4}.
\end{align}
$$
The summation term in the above equation can be further simplified as
```
\[\alpha (n) = 5+\sum\limits_{j=0}^{n} (6+2(n-j)) = n^2 +7n + 11.\]
```
Re-writing ```\(V^1\)``` as a function of ```\(V^n\)``` we get 
```
\[\label{oneton}
    V^1 = \frac{4}{n+3} V^n + 4\sum\limits_{j=1}^{n-1} \frac{\alpha(j)}{(j+3)(j+4)} =  \frac{4}{n+3} V^n + 4\sum\limits_{j=1}^{n-1} \frac{\alpha(j)}{\alpha(j)+1}.\]
```
The summation ```\(\sum\limits_{j=1}^{n-1} \frac{\alpha(j)}{\alpha(j)+1}\)```
in Equation ([\[oneton\]](#oneton)) diverges to ```\(\infty\)``` as we let
```\(n\to\infty\)```. That is to say that the average number of steps to enter
an absorbing state in ```\(Z^A\)``` starting from a point in ```\(Z^1\)```, is the
summation of infinity and the average number of steps to enter \(Z^A\)
starting from a point in ```\(Z^\infty\)``` divided by ```\((n+3)\)```. The result
of this summation is equal to infinity. Using Equation ([\[r1\]](#r1))
```\(V^0\)``` will be equal to infinity as well. Hence, starting from the
anthill the mean number of steps to find a food is infinity.

## Question 2

I take a similar approach to the question one to find the mean number of
steps to reach the food located on the line \(i+j = 1\) starting from
\((0, 0)\). I cluster the states below the line into the parallel line
that are distance \(d\) away from \(i+j = 1\). 

```
\[\begin{aligned}
    Z^{d} &=  \{(i, j): i + j = 1 - d \}, \ for \ d \in \{1, 2, 3, \dots\} \end{aligned}\]
```
Starting from a state in \(Z^d\) with probability \(\frac{1}{2}\) we
move to a state in \(Z^{d+1}\) and with probability \(\frac{1}{2}\) to a
state in \(Z^{d-1}\). Similar to the previous question, letting \(V^d\)
be the mean number of steps to reach a food starting from a point in
```
\(Z^d\), we can write \[\begin{aligned}
\label{r21}
    V^{1} &=  1 + \frac{1}{2}V^{2}, \nonumber\\
    V^{d} &=  1 + \frac{1}{2}V^{d-1} + \frac{1}{2}V^{d+1}, \ \forall \in \{2, 3, \dots\}.\end{aligned}\]
```
Re-writing \(V^{1}\) as a function of \(V^n\) we get the
\[\label{one2n2}
    V^1 = n-1 + \frac{1}{n} V^n.\] Equation ([\[one2n2\]](#one2n2))
means that the average number of steps to enter any point on line
\(i+j = 1\) starting from a point on set \(Z^1\) is the sum of infinity
and mean number of time to hit the line starting from a point in
\(Z^\infty\) divided by \(n\). Hence, the mean number of points to reach
the food starting from \((0, 0)\) is infinity.

## Question 3

The mean number of steps to reach the food outside the boundary given in
question number three is 14. The code below can be used to find the
number of steps to reach the food using simulation and analytic methods.
The number of steps to reach the food can be generated either serially
or in parallel using the simulation.

It is also possible to find the exact solution by extracting the states
\((i, j)\) inside the boundary, generating the probability transition
matrix of the transient states, and solving the set of equations in the
form of Equation ([\[twodmean\]](#twodmean)).

``` python
import numpy as np


class RandomWalK2D():
    def __init__(self, ):
        self.p = 1/4
        self.max_number_steps = int(1e8)
        self.init_state = (0, 0)  # (x, y)
        self.states_trajectory = [self.init_state]
        self.absorbing_states = {(2, 2), (2, -2), (-2, 2), (-2, -2)}
    
    @staticmethod
    def puzzle_1_boundry_func(state):
        
        return ((abs(state[0]) < 2) and (abs(state[1]) < 2))
    
    @staticmethod
    def puzzle_3_boundry_func(state):
        
        return ((state[0]-0.25)/3)**2 + ((state[1]-0.25)/4)**2 < 1
    
    def one_step_transition(self, current_state):
        rnd = np.random.random()
        new_state = [None, None]
        if rnd < self.p:
            new_state[0] = current_state[0] + 1
            new_state[1] = current_state[1]
        elif (rnd >= self.p) & (rnd < 2*self.p):
            new_state[0] = current_state[0] - 1
            new_state[1] = current_state[1]
        elif (rnd >= 2*self.p) & (rnd < 3 * self.p):
            new_state[0] = current_state[0]
            new_state[1] = current_state[1] + 1
        else:
            new_state[0] = current_state[0]
            new_state[1] = current_state[1] - 1

        return tuple(new_state)

    def get_num_steps_to_absorbing_states_in_set(self, track_states=False,
                                                 init_state=(0, 0)):
        num_steps = 0
        current_state = init_state
        while current_state not in self.absorbing_states:
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
            
        return num_steps

    
    def get_num_steps_to_absorbing_states_on_linear_line(self, p1=(1, 0),
                                                         p2=(0, 1), 
                                                         track_states=False,
                                                         init_state=(0, 0)):
        line_slope = (p1[1]- p2[1])/(p1[0]- p2[0])
        line_intercept = p1[1] - p1[0] * line_slope
        
        num_steps = 0
        current_state = init_state
        while (current_state[1] - current_state[0]*line_slope) != line_intercept:
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
        return num_steps     

    def get_num_steps_to_absorbing_states_outside_boiundry(self, 
                                                            track_states=False,
                                                            init_state=(0, 0)):
        
        num_steps = 0
        current_state = init_state
        while self.puzzle_3_boundry_func(current_state):
            current_state = self.one_step_transition(current_state)
            if track_states:
                self.states_trajectory.append(current_state)
            if num_steps > self.max_number_steps:
                return num_steps
            num_steps += 1
            
        return num_steps
    
     def get_neighboring_states(self, current_state):
        i, j = current_state
        
        return ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
        
    def get_states_inside_baoundry(self, boundry_func):
        current_state = (0, 0)
        is_explored = []
        to_explore = [current_state]
        states_in_boundry = {(0, 0)}
        while True:
            new_states = self.get_neighboring_states(current_state)
            for state in new_states:
                if boundry_func(state):
                    states_in_boundry.add(state)
                    if state not in is_explored:
                        to_explore.append(state)
                        
            to_explore.remove(current_state)
            is_explored.append(current_state)
            if len(to_explore)>=1:
                current_state = to_explore[-1]
            else:
                break 
            
        return states_in_boundry
    
    def get_states_in_boundry_transition_matrix(self, boundry_func):
        states_in_boundry = list(self.get_states_inside_baoundry(boundry_func))
        
        probability_transition = np.zeros((len(states_in_boundry), 
                                           len(states_in_boundry)))
        
        for state_idx, state in enumerate(states_in_boundry):
            neighbor_states = self.get_neighboring_states(state)
            p = 0
            for ns in neighbor_states:
                if ns in states_in_boundry:
                    ns_idx = states_in_boundry.index(ns)
                    probability_transition[state_idx, ns_idx] = 0.25
                    
        return probability_transition, states_in_boundry
    
    def get_mean_time_to_boundry_func_analytic(self, boundry_func):
        p, s = self.get_states_in_boundry_transition_matrix(boundry_func)
        n = p.shape[0]
        origin_idx = s.index((0, 0))
        return np.linalg.inv(np.eye(n) - p)[origin_idx].sum()
    
    def run_experiments_parallel(self, experiment_func, num_samples, 
                                 mp_num_processors=4):
        import multiprocessing as mp 
        pool = mp.Pool(processes=mp_num_processors)
        num_steps_samples = [
            pool.apply_async(experiment_func)
            for idx in range(num_samples)
            ]
        
        num_steps_samples = [result.get() for result in num_steps_samples]
            
        return num_steps_samples
    
    def run_experiments_serial(self, experiment_func, num_samples):
        num_steps_samples = []
        for _ in range(num_samples):
            num_steps = experiment_func()
            num_steps_samples.append(num_steps)
        
        return num_steps_samples
    

if __name__ == "__main__":
    
    rw = RandomWalK2D()
    boundry_func_1 = rw.puzzle_1_boundry_func
    puzzle_1_mean_time = rw.get_mean_time_to_boundry_func_analytic(boundry_func)
    
    """
    experiment_func = rw.get_num_steps_to_absorbing_states_outside_boundry
    num_samples = int(1e6)
    
    samples = rw.run_experiments_parallel(experiment_func, num_samples, mp_num_processors=4)
    mean_number_of_steps = np.mean(samples)
    boundry_func_3 = rw.puzzle_3_boundry_func
    puzzle_3_mean_time = rw.get_mean_time_to_boundry_func_analytic(boundry_func_3)
    assert  np.isclose(mean_number_of_steps, puzzle_3_mean_time, rtol=1e-3)
    """
```