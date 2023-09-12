from abc import ABC, abstractmethod
import torch
import random

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
#from torch_ac.agents.py_djinn_agent import DJINNAgent
from torch_ac.opt_helpers.replay_buffer import discount_reward
import numpy as np


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, local_guide):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        local_guide : agent
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.local_guide = local_guide

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        #Develope the local agent. (Heuristic rules, Decision Tree, DJINN, PPO)
        #AGENT_TYPE = 'djinn'
        #if AGENT_TYPE == 'djinn':
            #guide_agent = DJINNAgent(bot_name='djinn_lava',input_dim=6,output_dim=2)
            #guide_agent = DJINNAgent(bot_name='djinn_lava',input_dim=len(self.obs[0]['image']),output_dim=2)


        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            rules = True 
            rules = False 
            #The rules for keydoor
            if(rules and self.env.envs[0].spec.id == 'MiniGrid-DoorKey-6x6-v0'):
             for k in (0,len(self.obs)-1):
                # If is in front a door
                carrying = self.env.envs[k].carrying
                front = self.obs[k]['image'][3][-2]
                if(front[0] == 5):#key
                    action[k] = 3 #pickup
                if(front[0] == 4 and front[2] ==2 and carrying != None): #Door
                    action[k] = 5 #toggle
               
            #The rules for 4rooms
            elif(rules and 'MiniGrid-FourRooms-v0' == self.env.envs[0].spec.id):
              for k in (0,len(self.obs)-1):
                #If the hole is in front of the agent(not work well)
                for j in (0,len(self.obs[k]['image'][3])-1):
                        hole = self.obs[k]['image'][3][j][0]
                        hole_l = self.obs[k]['image'][2][j][0]
                        hole_r = self.obs[k]['image'][4][j][0]
                        if(hole == 1 and hole_l == 2 and hole_r == 2):
                            action[k] = 2
                #For 4rooms
                #If the target 8 is in front of the agent
                for a in (0,len(self.obs[k]['image'])-1):
                    for b in (0,len(self.obs[k]['image'][a])-2):
                        target = self.obs[k]['image'][a][b][0]
                        if(target == 8):
                            action[k] = 2
                #If it is in the local area
                if(self.obs[k]['image'][3][-2][0] == 2):
                      has_hole = 0
                      for l in (0,len(self.obs[k]['image'])-1):
                               hole = self.obs[k]['image'][l][-2][0]
                               if(hole == 1):
                                   has_hole = 1
                                   if l < 3:
                                       action[k] = 0
                                   if l > 3:
                                       action[k] = 1
                                   break
                      if(has_hole == 0):
                           right  = self.obs[k]['image'][-1][-1][0]
                           left  = self.obs[k]['image'][0][-1][0]
                           if (right == 2):
                               action[k] = 0
                           if (left == 2):
                               action[k] = 1
            #For lavacrossing 
            #DJINN based guide
            elif(rules):
             '''
             for k in (0,len(self.obs)-1):
                #If it is in the local area
                if(self.obs[k]['image'][3][-2][0] == 9 and action[k] == 2):
                      lava_obs = [self.obs[k]['image'][0][-2][0],self.obs[k]['image'][1][-2][0],self.obs[k]['image'][2][-2][0],self.obs[k]['image'][4][-2][0],self.obs[k]['image'][5][-2][0],self.obs[k]['image'][6][-2][0]]
                      action[k] = self.local_guide.get_action(lava_obs)
                      #print("actionkkkkk:",action[k]," lava_obs:",lava_obs)

             '''
             for k in (0,len(self.obs)-1):
                #If it is in the local area
                if(self.obs[k]['image'][3][-2][0] == 9 and action[k] == 2):
                      has_hole = 0
                      for l in (0,len(self.obs[k]['image'])-1):
                               hole = self.obs[k]['image'][l][-2][0]
                               if(hole == 1):
                                   has_hole = 1
                                   if l < 3:
                                       action[k] = 0
                                   if l > 3:
                                       action[k] = 1
                                   break
                      if(has_hole == 0):
                           right_front = self.obs[k]['image'][-1][-2][0]
                           if(right_front == 0 or right_front == 2):
                               action[k] = 0
                           else:
                               action[k] = 1
#            for k in (0,len(self.obs)-1):
#                 if(self.obs[k]['image'][3][-2][0] == 9 and action[k] == 2):
#                            action[k] = 1
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            #self.local_guide.save_reward(sum(reward))
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            '''       
            reward_sum = np.sum(self.local_guide.replay_buffer.rewards_list)
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(self.local_guide.replay_buffer.rewards_list,self.local_guide.replay_buffer.value_list, self.local_guide.replay_buffer.deeper_value_list)
            self.local_guide.replay_buffer.rewards_list = rewards_list
            self.local_guide.replay_buffer.advantage_list = advantage_list
            self.local_guide.replay_buffer.deeper_advantage_list = deeper_advantage_list
            '''
        # Add advantage and return to experiences
       
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
