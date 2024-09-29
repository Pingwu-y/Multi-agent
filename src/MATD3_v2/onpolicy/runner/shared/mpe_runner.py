import time
import csv
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
from onpolicy import global_var as glv

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.use_train_render = False
        self.no_imageshow = True
        self.buffer_size = 500000

    def run(self):
        if self.all_args.save_data:
            # csv
            file = open('Rewards.csv', 'w', encoding='utf-8', newline="")
            writer = csv.writer(file)
            writer.writerow(['step', 'average', 'min', 'max', 'std'])
            file.close()
    
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads  # 5e6 / 200 / 256 [每个线程的episode总数]

        current_size = 0
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            if self.use_train_render == True and episode>episodes/5:
                image = self.envs.render('rgb_array')[0][0]

            glv.set_value('CL_ratio', episode/episodes)  #curriculum learning
            self.envs.set_CL(glv.get_value('CL_ratio'))  # env_wrappers
            # print('the global value is {}'.format(glv.get_value('CL_ratio')))
            
            total_rewards = []
            for step in range(self.episode_length):
                calc_start = time.time()

                # Sample actions
                obs, share_obs, actions_env, rnn_states, rnn_states_critic = self.collect(current_size)
                current_size = (current_size + 1) % self.buffer_size

                # obs.shape:(32,5,28), share_obs.shape:(32,5,140)
                #print(actions_env)

                # Obser reward and next obs
                obs_next, rewards, dones, infos = self.envs.step(actions_env)
                total_rewards.append(rewards)

                data = obs, share_obs, obs_next, actions_env, rewards, dones, infos, rnn_states, rnn_states_critic
                
                # render while training
                if self.use_train_render == True and step < 140 and episode>episodes/5:
                    image = self.envs.render('rgb_array')[0][0]
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            #self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information （for each thread）
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, CL {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                                glv.get_value('CL_ratio')))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews

                total_rewards = np.array(total_rewards)
                r = total_rewards.mean(2).sum(axis=(0, 2))
                
                Average = np.mean(r)
                Min = np.min(r)
                Max = np.max(r)
                Std = np.std(r)

                if self.all_args.save_data:
                    file = open('Rewards.csv', 'a', encoding='utf-8', newline="")
                    writer = csv.writer(file)
                    writer.writerow([total_num_steps, Average, Min, Max, Std])
                    file.close()
                
                train_infos["average_episode_rewards"] = Average
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # print(obs)
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        # print(share_obs)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    # get action
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        obs = self.buffer.obs[step]
        share_obs = self.buffer.share_obs[step]
        
        # 通过trainer.policy.get_actions()函数来采样动作
        value, action, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # 多个进程的, 分开.由[   ]变为[[][][]]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        #action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):  # action_space[0]:其实action_space[i]都是一样的，一个multi_discrete
                # np.eye: one hot form of action
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions  # 需要写成[[ar,at],[ar,at]...]
            # print(actions)
        else:
            raise NotImplementedError

        return obs, share_obs, actions_env, rnn_states, rnn_states_critic

    def insert(self, data):  # memory
        obs, share_obs, obs_next, actions_env, rewards, dones, infos, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs_next = obs_next.reshape(self.n_rollout_threads, -1)
            share_obs_next = np.expand_dims(share_obs_next, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs_next = obs_next
        
        dones = dones.reshape((self.n_rollout_threads,self.num_agents,1))
        
        self.buffer.insert(obs, share_obs, obs_next, share_obs_next, actions_env, rewards, dones, rnn_states, rnn_states_critic, masks)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                if self.no_imageshow:
                    envs.render('rgb_array')
                else:
                    image = envs.render('rgb_array')[0][0]  # imshow
                    all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                #rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                #masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                #masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    if self.no_imageshow:
                        envs.render('rgb_array')
                    else:
                        image = envs.render('rgb_array')[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs and self.no_imageshow == False:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)


