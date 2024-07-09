# Advanced-DRL-Renegades-Game-Bots

A collection of my implemented advanced & complex RL agents for complex games like Soccer, Rubik's Cube, VizDoom, Montezuma, Kungfu-Master, super-mario-bros and more by implementing various DRL algorithms 
using gym, unity-ml, pygame, sb3, rl-zoo, rubiks_cube_gym and sample factory libraries. I have added some links in **Acknowledgement** section below. 

## DRL-Renegades-Game-Bots
To see my simple & intermediate agents for relatively simple environments; visit this repository: https://github.com/hishamcse/DRL-Renegades-Game-Bots

## Covered DRL Topics
  * Unity Ml Agents
  * Curiosity Driven RL - Random Network Distillation(RND)
  * PPO + RND
  * MultiAgent Reinforcement Learning(MARL)
  * Self Play
  * Sample Factory
  * Asynchronous Advantage Actor-Critic (A3C)
  * Intrinsic Curiosity Module (ICM)
  * Decision Transformer (both online & offline)

## Table of Implemented Agents

| **Environments**                       | **Libraries Used(includes HF)**                                       | **Algos**                    | **Kaggle Notebooks** |
|----------------------------------------|-----------------------------------------------------------------------|------------------------------|----------------------|
| MontezumaRevengeNoFrameskip-v4         | pytorch, gym, atari                                                   | PPO + RND(CNN) - Scratch     | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-5-rnd-montezuma-mario-bros)                     |
| Super-Mario-Bros                       | pytorch, gym, atari                                                   | PPO + RND(CNN) - Scratch     | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-5-rnd-super-mario-bros)                     |
| SoccerTwos                             | unity-mlagents                                                        | MA-POCA                      | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-7-marl-soccer-2vs2)                     |
| VizDoom (doom_health_gathering_supreme)| sample_factory                                                        | PPO                          | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-8-ii-smpfc-vizdoom-deathmatch)                     |
| Doom Deathmatch (doom_deathmatch_bots) | sample_factory                                                        | APPO                         | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-8-ii-smpfc-vizdoom-deathmatch)                     |
| KungFuMaster-v5                        | pytorch, gym                                                          | A3C-ICM-scratch              | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-adv-drl-a3c-icm-kungfu-master)
| RubiksCube-v0                          | pytorch, gym_rubiks_cube, decision transformer                        | Decision Transformer         | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-advanced-decisiontransformer-rubikscube)

## HuggingFace Models
Find all my traned agents at [hishamcse agents](https://huggingface.co/hishamcse)

## Game Previews
  ![Montezuma](https://www.gymlibrary.dev/_images/montezuma_revenge.gif) <img src="https://media.tenor.com/pKgBbArPChQAAAAC/mario-super.gif" height="200"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1LFs0griZFmQBd1Pw_odjoUN1vdxBY1iz3de2HQFxHMHAlqJA9un1zJRQk8F6LuNjFiM&usqp=CAU" height="200" width="200"/> 
  <img src="https://www.gocoder.one/static/unity-ml-agents-soccertwos-b6e8a229df44d79b8d8a559338680026.gif" width="390" height="200"/> <img src="https://www.gymlibrary.dev/_images/kung_fu_master.gif" height="200"/> 
  <img src="https://i.sstatic.net/wgKuG.gif" height="200"/> 


## Acknowledgements & Resources
   * [Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
   * [Thomas Simonini](https://x.com/ThomasSimonini)
   * [Open RL Leaderboard - HuggingFace](https://huggingface.co/spaces/open-rl-leaderboard/leaderboard)
   * [PPO-RND for Montezuma & Mario](https://github.com/alirezakazemipour/PPO-RND/tree/main)
   * [RND - Random Network Distillation](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)
   * [CleanRL Single File Implementations](https://docs.cleanrl.dev/)
   * [Unity-mlagents](https://github.com/Unity-Technologies/ml-agents)
   * [gym](https://www.gymlibrary.dev/index.html)
   * [Sample_Factory](https://www.samplefactory.dev/)
   * [RL-Zoo3](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
   * [QRDQN - Quantile Regression DQN](https://advancedoracademy.medium.com/quantile-regression-dqn-pushing-the-boundaries-of-value-distribution-approximation-in-620af75ec5f3)
   * [Deep RL Paradise](https://github.com/alirezakazemipour/DeepRL-Paradise)
   * [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
   * [Awesome Deep RL](https://github.com/kengz/awesome-deep-rl)
   * [Dennybritz RL](https://github.com/dennybritz/reinforcement-learning)
   * [HuggingFace Hub Python Library](https://huggingface.co/docs/huggingface_hub/index)
   * [Uvipen](https://github.com/uvipen)
   * [Ted Deng](https://github.com/tedtedtedtedtedted)
   * [Street-fighter-A3C-ICM-pytorch](https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch)
   * [Solve-Rubiks-Cube-using-Transformer](https://github.com/tedtedtedtedtedted/Solve-Rubiks-Cube-Via-Transformer/tree/main)
   * [CubeGPT](https://github.com/tedtedtedtedtedted/Solve-Rubiks-Cube-Via-Transformer/tree/main/CubeGPT)
   * [GymRubiksCube](https://github.com/mgroling/GymRubiksCube)
