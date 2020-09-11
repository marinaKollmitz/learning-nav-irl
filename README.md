# Learning Human-Aware Robot Navigation from Physical Interaction via Inverse Reinforcement Learning 

We learn the parameters of our navigation reward function via inverse reinforcement learning to eliminate the need for manual tuning of the parameters. Our navigation reward function balances social constraints and the desire to reach a goal state. We conducted a user study where 13 test subjects interacted with our mobile robot in our office hallway. The test subjects demonstrated how they wanted the robot to pass them by pushing on its force-sensitive shell. 

This repository includes our experiment data and IRL code and provides a ros node to retrain on the experiment demonstrations.

If you are using our code or experiment data in your research, please consider citing our paper:

```
@INPROCEEDINGS{kollmitz20iros,
  author = {Marina Kollmitz and Torsten Koller and Joschka Boedecker and Wolfram Burgard},
  title = {Learning Human-Aware Robot Navigation from Physical Interaction via Inverse Reinforcement Learning},
  booktitle = {Proc.~of the IEEE/RSJ Int.~Conf.~on Intelligent Robots and Systems (IROS)},
  year = {2020},
  url = {http://ais.informatik.uni-freiburg.de/publications/papers/kollmitz20iros.pdf}
}
```

## Get the code and experiment data

1. Clone repository, which includes the code and experiment data:

```
cd ~/catkin_ws/src
git clone https://github.com/marinaKollmitz/learning-nav-irl/
```

2. Unpack experiment data:

```
cd learning-nav-irl/experiments/
tar -xzf runs.tar.gz
```

3. Install pytorch

Our code uses pytorch for gradient-based optimization. Follow the installation instructions from https://pytorch.org/ to install pytorch for your system and architecture. The standard will be:
```
pip install torch torchvision
```

## Run IRL on experiment demos

Use the ```launch/experiment_demo_irl.launch``` file to run inverse reinforcement learning on the experiment demonstrations. The launch file takes the path of the experiment run folder as argument:

```
roslaunch learning-nav-irl experiment_demo_irl.launch experiment_folder:=<path_to_folder>
```

An rviz window should pop up where you can see the demonstration trajectories, the state visitations and the learned reward. 

![visualization](https://github.com/marinaKollmitz/learning-nav-irl/blob/master/img/viz.png?raw=true)

The reward parameters and the gradient are logged in the terminal window and you should find a plot of the optimization objective (maximizing the log-likelihood of the demonstrations) as a .png file in the ```learning-nav-irl``` folder.

## License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
