# Reviving DeepMind's DQN: A Replication Study
This repo is about **rebuilding DeepMind’s Deep Q-Network (DQN)** for reinforcement learning, starting from scratch with a minimal setup. 

---

## Overview
We re-implemented the original DQN step by step, focusing on **CartPole-v1** using **PyTorch** (for the neural net) and **Gymnasium** (for the environment). Along the way, we added the core tricks that make DQN work:  

- **Experience replay**  
- **Target networks**  
- **ε-greedy exploration**  
- **Reward clipping**  

The goal wasn’t just to “make it run,” but to see how each piece matters and what happens when parameters shift.  

---

## Results
- The agent learns to balance the pole after enough training (often within ~800 episodes).  
- Training curves aren’t smooth — exploration vs. exploitation creates ups and downs.  
- Across multiple runs, performance varies: some reach perfect scores, others get stuck.  
- Overall, The repolication if the DQN works, but it’s sensitive to hyperparameters and setup.  

---

## Challenges
- Building a working DQN without relying on existing RL libraries.  
- Tuning learning rates, discount factors, and exploration strategies.  
- Debugging instability and checking results across repeated runs.  

---

## References
- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (2013)  
- Mnih et al., *Human-level control through deep reinforcement learning* (2015)  
- Sentdex, *Reinforcement Learning Tutorials*  
- PyTorch Reinforcement Learning Guide  
