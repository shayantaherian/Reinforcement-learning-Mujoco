# Reinforcement Learning for MuJoCo

<p align="center">
<img src="https://user-images.githubusercontent.com/51369142/113700464-c8ba1600-96ce-11eb-9b84-f735d156648b.png" width="400" height="300"/>                                     <img src="https://user-images.githubusercontent.com/51369142/113700507-d40d4180-96ce-11eb-9911-14de573f344d.png" width="400" height="300"/>
</p>


This repository contains different reinforcement learning implementations for continous control tasks in [Mujoco](http://www.mujoco.org/) environment. These are meant to serve as learning tool to complement the learning materials from:

* DDPG [[1]](https://arxiv.org/pdf/1509.02971.pdf)
* Actor-Critic [[2]](http://incompleteideas.net/book/first/ebook/node66.html#:~:text=The%20policy%20structure%20is%20known,being%20followed%20by%20the%20actor.)
* PPO [[3]](https://arxiv.org/pdf/1707.06347.pdf)
* SAC [[4]](https://arxiv.org/abs/1801.01290)

## Dependencies
The main package dependencies are `Mujoco`, `python=3.8`, `gym` and `PyTorch`.

## Installation

1) Required license key.
2) Install mujoco-py according to [this](https://www.youtube.com/watch?v=xG8oujhD9lA).
3) Clone `
git clone https://github.com/shayantaherian/Reinforcement-learning-Mujoco/.git`

## Testing 
To test the trained agent, simply run `Test.py` file to see agent behaviour. This code snippet is for DDPG algorithm, however, it can be extended for the other methods simply by loading the main file as well as loading the weights file.
