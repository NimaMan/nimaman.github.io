

---
author: Nima Manafzadeh Dizbin
title: A Sumamry and Future Directions of My Research  
date: 2021-12-12
description: A Sumamry and Future Directions of My Research 
---

With the advancement in data collections methods, decision-makers are now swamped with a massive amount of data describing the environment of the operations. The exponentially increasing computational power and abundance of data, along with the advances in Machine Learning (ML) introduce the possibility of transforming the gathered data into useful information for improved decision making. Different areas including robotic systems, autonomous automobiles, and drug discovery are being transferred to a new era thanks to these advances. However, translation of the swell of data into actionable information for improved decision making has been rather slow in operational problems. Such a transition requires fundamental changes in the way we use data in decision making by bringing statistics, machine learning, and operations research techniques together. To this end, my research focuses on two different threads. First, developing intelligent data-driven decision making algorithms. Second, applying the optimization methods in solving challenging real-world problems.

During my Ph.D. research, I focused on improving the performance of production/inventory systems using a huge dataset, with millions of rows corresponding to several thousands of products processed on hundreds of machines, collected in the semiconductor manufacturing plant of the Robert Bosch Company in Reutlingen, Germany. I conducted my research as part of the Productive4.0 project, Europe's biggest research project in the field of digital industry. In the first stage of my research, I performed an Exploratory Data Analysis (EDA) to understand the sources of uncertainties that arise in the life-cycle of the products inside the system, and the resulting operational challenges that they introduce summarized as follows:
- I showed that production data has several statistical properties that has not been studies in the production/inventory literature.  
- I demonstrated how to model the observed statistical properties in the production line using Markovian models that had been introduced in the telecommunications literature. 
- I modelled the optimal control problem as a Markov Decision Process, and identified the optimal control policy of the problem.  
- I developed machine learning and statistical methods to predict the production times of the products in the production network.

My production system level EDA revealed that the uncertainties in the total production times of the products arise due to inefficiencies in controlling the system. I joined the Reinforcement Learning and Optimization team of Bosch Center for Artificial Intelligence (BCAI) to improve these inefficiencies using Machine Learning. During my sabbatical time in BCAI: 
- I conducted data analysis on the data collected from the Ion Implantation of the semiconductor wafer fabrication, and extracted the existing patterns in the product waiting times. 
- I contributed to a project applying the AlphaGo algorithm to the parallel machine scheduling problems that arise in the semiconductor wafer fabrication using Q-Learning and Monte-Carlo Tree Search methodologies. 
- I showed that one can generate near optimal solutions to the parallel machine scheduling problem by integrating the problem structure into Graph Neural Networks (GNN) with a single forward-pass of the GNN.
- I defined a research project called Reinforced Genetic Algorithm (RGA) on learning the mutation operator of the genetic algorithm for the parallel machine scheduling problem using GNN and reinforcement learning (RGA is currently under patent application process).

During my postdoctoral research in Eindhoven University of Technology (TUE), I further extended RGA to solve the Travelling Salesman and Maximum Cut problems. In addition, I proposed a new project called Block Evolutionary Training (BET) together with Stefan Falkner from BCAI on integrating the neural network structure into Evolutionary Training. Applications of BET into PyBullet physics engine (a harder version of the MuJoCo physics engine) shows that it can achieve higher rewards than RL methods using smaller neural network architectures. BET is currently under patent application process. Please visit my YouTube channel for watching videos of the performance of BET on different PyBullet environments. Further details can be provided upon request. 

My main research in TUE consists of using reinforcement learning for solving inventory management problems. I demonstrate that Covariance Matrix Adaptation Evolution Strategies (CMA-ES) can be used to train a significantly smaller neural networks in comparison to the existing literature on using RL for solving inventory management problems. In addition, CMA-ES does not require much need for hyper parameter optimization in comparison to RL methods. For further detail about the algorithm please refer to my blog post on the topic.    

My current results on using Evolutionary methods for solving decision making problems are quite promising. In my future research program, I am planning to:
- Further enhance the performance of Reinforcement Learning and Evolutionary algorithms in solving decision making problems. My ultimate goal is to develope algorithms for automated decision making. 
- Apply the developed optimization method into real-world problems from different domains, in particular robotics, production planning, and healthcare operations.
