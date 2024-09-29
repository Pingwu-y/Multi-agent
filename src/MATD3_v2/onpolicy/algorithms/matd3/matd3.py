import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check


class MATD3():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

        self.actor_pointer = 0
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.gamma = 0.95
        self.policy_update_freq = 2
        self.tau = 0.01

    def td3_update(self, sample, update_actor=True):
        self.actor_pointer += 1

        share_obs_batch, obs_batch, share_obs_next_batch, obs_next_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, rewards_batch, dones_batch, masks_batch = sample

        share_obs_batch = torch.tensor(share_obs_batch).cuda()
        obs_batch = torch.tensor(obs_batch).cuda()
        share_obs_next_batch = torch.tensor(share_obs_next_batch).cuda()
        obs_next_batch = torch.tensor(obs_next_batch).cuda()
        rnn_states_batch = torch.tensor(rnn_states_batch).cuda()
        rnn_states_critic_batch = torch.tensor(rnn_states_critic_batch).cuda()
        actions_batch = torch.tensor(actions_batch).cuda()
        rewards_batch = torch.tensor(rewards_batch).cuda()
        dones_batch = torch.tensor(dones_batch).cuda()
        masks_batch = torch.tensor(masks_batch).cuda()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            actions_next_batch, rnn_states = self.policy.actor_target(obs_next_batch, rnn_states_batch, masks_batch)
            noise = torch.normal(0, 0.1, actions_next_batch.shape).clamp(-self.noise_clip, self.noise_clip).cuda()
            actions_next_batch = (actions_next_batch + noise).clamp(-1, 1)

            # Trick 2:clipped double Q-learning
            Q1_next, rnn_states_critic = self.policy.critic1_target(share_obs_next_batch, actions_next_batch,
                                                                    rnn_states_critic_batch, masks_batch)
            Q2_next, rnn_states_critic = self.policy.critic2_target(share_obs_next_batch, actions_next_batch,
                                                                    rnn_states_critic_batch, masks_batch)

            target_Q = rewards_batch + self.gamma * (1 - dones_batch) * torch.min(Q1_next, Q2_next)

        # Compute current_Q
        current_Q1, rnn_states_critic = self.policy.critic1(share_obs_batch, actions_batch, rnn_states_critic_batch,
                                                            masks_batch)
        current_Q2, rnn_states_critic = self.policy.critic2(share_obs_batch, actions_batch, rnn_states_critic_batch,
                                                            masks_batch)
        td_error = nn.functional.mse_loss(current_Q1, target_Q, reduction='none') + nn.functional.mse_loss(current_Q2,
                                                                                                           target_Q,
                                                                                                           reduction='none')
        critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + nn.functional.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.policy.critic1_optimizer.zero_grad()
        self.policy.critic2_optimizer.zero_grad()
        critic_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic1.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.critic2.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic1.parameters())

        self.policy.critic1_optimizer.step()
        self.policy.critic2_optimizer.step()

        actor_grad_norm = 0
        actor_loss = torch.zeros(1)
        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            actions, rnn_states = self.policy.actor(obs_batch, rnn_states_batch, masks_batch)
            Q1, rnn_states_critic = self.policy.critic1(share_obs_batch, actions, rnn_states_critic_batch, masks_batch)
            actor_loss = -Q1.mean()

            # Optimize the actor
            self.policy.actor_optimizer.zero_grad()
            actor_loss.backward()

            '''for para in self.policy.actor.parameters():
                print(para.grad)
            print(a)'''

            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

            self.policy.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.policy.critic1.parameters(), self.policy.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.policy.critic2.parameters(), self.policy.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, actor_loss, critic_grad_norm, actor_grad_norm

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['critic_loss'] = 0
        train_info['actor_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0

        for _ in range(self.ppo_epoch * self.num_mini_batch):  # 15*16
            sample = buffer.sample_buffer()  # (2000,:)
            critic_loss, actor_loss, critic_grad_norm, actor_grad_norm = self.td3_update(sample, update_actor)

            train_info['critic_loss'] += critic_loss.item()
            train_info['actor_loss'] += actor_loss.item()
            train_info['actor_grad_norm'] += actor_grad_norm
            train_info['critic_grad_norm'] += critic_grad_norm

        '''for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                # here
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                critic_loss, critic_grad_norm, actor_grad_norm = self.td3_update(sample, update_actor)

                train_info['critic_loss'] += critic_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm'''

        num_updates = self.ppo_epoch * self.num_mini_batch

        '''for k in train_info.keys():
            train_info[k] /= num_updates'''
        train_info['critic_loss'] /= num_updates
        train_info['critic_grad_norm'] /= num_updates
        train_info['actor_grad_norm'] /= (num_updates / 2)
        train_info['actor_loss'] /= (num_updates / 2)

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic1.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic1.eval()
