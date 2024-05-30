# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size, extras_size):
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space, rec_state_size)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)
        
class TSOGGlobalRolloutStorage(object):

    def __init__(self, num_steps, num_processes, detection_results_shape, state_shape, rec_state_size, gat_mem_len):
        self.n_actions = 4

        self.states = torch.zeros(num_steps + 1, num_processes, *state_shape)
        self.detection_results = torch.zeros(num_steps + 1, num_processes, *detection_results_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes, 2, 2, rec_state_size)   # (hx, cx), layers: 2
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions), dtype=torch.float32)
        self.goal_cat_ids = torch.zeros((num_steps + 1, num_processes), dtype=torch.long)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.gat_embedding_memorys = torch.zeros(num_steps + 1, num_processes, gat_mem_len, 15+1, 256)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.states = self.states.to(device)
        self.detection_results = self.detection_results.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.goal_cat_ids = self.goal_cat_ids.to(device)
        self.masks = self.masks.to(device)
        self.gat_embedding_memorys = self.gat_embedding_memorys.to(device)
        return self

    def insert(self, states, detection_results, rec_states, actions, action_log_probs, value_preds, rewards, goal_cat_ids, masks, gat_embedding_memory):
        self.states[self.step + 1].copy_(states)
        self.detection_results[self.step + 1].copy_(detection_results)
        # self.rec_states[self.step + 1].copy_(rec_states)        # Recurrent states
        self.rec_states[self.step + 1][:, 0].copy_(rec_states[0].transpose(0, 1))    # hx
        self.rec_states[self.step + 1][:, 1].copy_(rec_states[1].transpose(0, 1))    # cx
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs.squeeze(1))
        self.value_preds[self.step].copy_(value_preds.squeeze(1))
        self.rewards[self.step].copy_(rewards)
        self.goal_cat_ids[self.step + 1].copy_(goal_cat_ids)            # Masks, 任务是否完成
        self.masks[self.step + 1].copy_(masks)
        self.gat_embedding_memorys[self.step + 1].copy_(gat_embedding_memory)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.detection_results[0].copy_(self.detection_results[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.goal_cat_ids[0].copy_(self.goal_cat_ids[-1])
        self.masks[0].copy_(self.masks[-1])
        self.gat_embedding_memorys[0].copy_(self.gat_embedding_memorys[-1])

    def reset_gat_mem(self, e):
        # new episode
        self.gat_embedding_memorys[:, e, :, :, :] = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value.squeeze(1)
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value.squeeze(1)
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'state': self.states[:-1].view(-1, *self.states.size()[2:])[indices],
                'detection_results': self.detection_results[:-1].view(-1, *self.detection_results.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(-1, *self.rec_states.size()[2:])[indices],     # (-1, 2, 2, rec_state_size)
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'goal_cat_ids': self.goal_cat_ids.view(-1)[indices],
                'gat_embedding_memory': self.gat_embedding_memorys[:-1].view(-1, *self.gat_embedding_memorys.size()[2:])[indices],
            }

class TSOGSemExpGlobalRolloutStorage(RolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size, extras_size, detction_results_shape, gat_mem_len):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.detection_results = torch.zeros(num_steps + 1, num_processes, *detction_results_shape)
        # self.gat_embedding_memorys = torch.zeros(num_steps + 1, num_processes, gat_mem_len, 15+1, 256)        # for sog w.o. gat_memory

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = True
        self.extras_size = extras_size

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.detection_results = self.detection_results.to(device)
        # self.gat_embedding_memorys = self.gat_embedding_memorys.to(device)        # for sog w.o. gat_memory
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras, detection_results, gat_embedding_memory):
        self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.extras[self.step + 1].copy_(extras)
        self.detection_results[self.step + 1].copy_(detection_results)
        # self.gat_embedding_memorys[self.step + 1].copy_(gat_embedding_memory)     # for sog w.o. gat_memory

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.detection_results[0].copy_(self.detection_results[-1])
        # self.gat_embedding_memorys[0].copy_(self.gat_embedding_memorys[-1])       # for sog w.o. gat_memory
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def reset_gat_mem(self, e):
        # new episode
        # self.gat_embedding_memorys[:, e, :, :, :] = 0         # for sog w.o. gat_memory
        pass

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
                'detection_results': self.detection_results[:-1].view(-1, *self.detection_results.size()[2:])[indices],
                # 'gat_embedding_memory': self.gat_embedding_memorys[:-1].view(-1, *self.gat_embedding_memorys.size()[2:])[indices],        # for sog w.o. gat_memory
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            detection_results = []
            gat_embedding_memory = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                detection_results.append(self.detection_results[:-1, ind])
                gat_embedding_memory.append(self.gat_embedding_memorys[:-1, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            detection_results = torch.stack(detection_results, 1)
            gat_embedding_memory = torch.stack(gat_embedding_memory, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
                'detection_results': _flatten_helper(T, N, detection_results),
                # 'gat_embedding_memory': _flatten_helper(T, N, gat_embedding_memory),      # for sog w.o. gat_memory
            }