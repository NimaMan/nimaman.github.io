
---
author: Nima Manafzadeh Dizbin
title: Summary and Future Directions of My Research  
date: 2021-12-13
description: A Sumamry and Future Directions of My Research 
---

My primary and future research interests focus on the ***methodological areas of reinforcement learning and stochastic optimization with applications in inventory management, production planning, resource allocation, robotics, and healthcare operations***.

With the advances in data collection methods, decision-makers gather a massive amount of data describing the environment of the operations. The objective of data collection is to transform it into useful information for improved decision-making. However, translation of the swell of data into actionable information is challenging in operational problems. The recent advances in Machine Learning (ML) introduce the possibility for improving the decision-making in these problems. However, current ML algorithms are not directly applicable to these problems. My research bridges this gap by focusing on two different threads. First, developing intelligent data-driven decision-making algorithms. Second, applying the ML and optimization methods in solving challenging real-world problems.   

During my Ph.D. research, I focused on improving the performance of production/inventory systems using a large dataset, with millions of rows corresponding to several thousands of products processed on hundreds of machines, collected in the semiconductor manufacturing plant of the Robert Bosch Company in Reutlingen, Germany. I conducted my research as part of the Productive4.0 project, Europe's biggest research project in the field of digital industry. In the first stage of my research, I performed an Exploratory Data Analysis (EDA) to understand the sources of uncertainties that arise in the life-cycle of the products inside the system, and the resulting operational challenges that they introduce summarized as follows:
- I showed that production data has several statistical properties that has not been studied in the production/inventory literature.  
- I demonstrated how to model the observed statistical properties in the production line using Markovian models introduced in the telecommunications literature. 
- I modeled the optimal control problem as a Markov Decision Process and identified the optimal control policy of the problem.  
- I developed machine learning and statistical methods to predict the production times of the products in the production network.

My production system-level EDA revealed that the uncertainties in the total production times of the products arise due to inefficiencies in controlling the system. I joined the Reinforcement Learning and Optimization team of Bosch Center for Artificial Intelligence (BCAI) to improve these inefficiencies using Machine Learning. During my sabbatical time in BCAI: 
- I analyzed the data collected from the Ion Implantation of the semiconductor wafer fabrication and extracted the existing patterns in the product waiting times. 
- I contributed to a project applying the AlphaGo algorithm to the parallel machine scheduling problems that arise in semiconductor wafer fabrication using Q-Learning and Monte-Carlo Tree Search methodologies. 
- I showed that a single forward-pass of the GNN can generate near-optimal solutions to the parallel machine scheduling problem by integrating the problem structure into Graph Neural Networks (GNN).
- I defined a research project called Reinforced Genetic Algorithm (RGA) on learning the mutation operator of the genetic algorithm for the parallel machine scheduling problem using GNN and reinforcement learning. 

During my postdoctoral research at the Eindhoven University of Technology (TUE), I further extended RGA to solve the Travelling Salesman and Maximum Cut problems. In addition, I proposed a new project called Block Evolutionary Training (BET) together with Stefan Falkner from BCAI on integrating the neural network structure into Evolutionary Training. BET achieves higher rewards than RL methods using smaller neural network architectures in different PyBullet physics engine environments. BET is currently under the patent application process. I can provide further information about the performance of BET during the interview process.

In addition to RGA and BET, I also researched using reinforcement learning for solving inventory management problems. I demonstrate that Covariance Matrix Adaptation Evolution Strategies (CMA-ES) can learn significantly smaller neural networks to manage inventory in comparison to the learned networks using RL. In addition, CMA-ES does not require as much hyper-parameter optimization as RL methods. For further detail about the algorithm, please refer to my blog post on the topic. My current results on using Evolutionary methods for solving decision-making problems are quite promising. In my future research program, I am planning to:
- further enhance the performance of Reinforcement Learning and Evolutionary algorithms in solving decision-making problems. My ultimate goal is to develop algorithms for automated decision-making. 
- apply the developed optimization method to real-world problems from different domains, in particular, robotics, production planning, and healthcare operations. 

In summary, my ultimate goal is to develop data-driven optimization methodologies with a focus on their applications in areas such as healthcare operations, production planning, and robotics.
