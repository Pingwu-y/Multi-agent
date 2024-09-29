import csv
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from onpolicy import global_var as glv

# update bounds to center around agent
cam_range = 8
INFO=[]  # render时可视化数据用

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,  # 以上callback是通过MPE_env跑通的
                 done_callback=None, post_step_callback=None,  # MPE游戏没用到的参数
                 shared_viewer=True, discrete_action=False):
        # discrete_action为false,即指定动作为Box类型

        self.no_imageshow = True  # 管理是否需要用teamviewer
        self.INFO_flag = 0

        # set CL
        self.use_policy = 0
        self.use_CL = 0  # training:1, render/tune:0
        self.CL_ratio = 0
        self.Cp= 0.6 # 1.0 # 0.3
        self.JS_thre = 0
        self.start_ratio = 0.80  # for JS thre

        # terminate
        self.is_ternimate = False

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents
        self.landmarks = self.world.landmarks
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback  # dont know why

        self.post_step_callback = post_step_callback

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:  # WHAT WE NEED
                u_action_space = spaces.Box(
                    low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)  # [ar, at], 2维
            
            if agent.movable:
                total_action_space.append(u_action_space)
            
            # total action space(u and c)
            if len(total_action_space) > 1:  # u_action & c_action
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])  # 只有u_action
        
            # observation space
            obs_dim = len(observation_callback(agent, self.world))  # callback from senario, changeable
            share_obs_dim += obs_dim  # simple concatenate
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
            agent.action.c = np.zeros(self.world.dim_c)
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):  # action_n: action for all policy agents, concatenated, from MPErunner
        self.current_step += 1
        obs_n = []
        reward_n = []  # 所有智能体的横向拼接
        done_n = []
        info_n = []
        self.agents = self.world.policy_agents  # adversaries only

        self.JS_thre = int(self.world_length*self.start_ratio*set_JS_curriculum(self.CL_ratio/self.Cp))
        
        terminate = []
        for i, agent in enumerate(self.agents):
            terminate.append(agent.done)
        
        if all(terminate)==True:
            self.is_ternimate = True
            # pass
            if self.CL_ratio > self.Cp:
                # print('terminate triggered')
                pass
            elif self.use_policy:
                # print('terminate triggered')
                pass
            else:
                pass
        else:
            self.is_ternimate = False

        # set action for each agent
        policy_u = self.policy_u(self.landmarks, self.agents, self.world.scripted_agents[0])
        for i, agent in enumerate(self.agents):  # adversaries only
            self._set_action(action_n[i], policy_u[i], agent, self.action_space[i])
        
        # advance world state
        self.world.step()  # core.step(), after done, all stop. 不能传参

        # record observation for each agent
        # is_good_action = []
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n  # [[reward] [reward] [reward] ...]

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, policy_u, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:  # Box
            action = [action]

        if agent.movable:
            # physical action, obtain agent.action.u for each agent
            # WE NEED BOX
            if self.discrete_action_input:  # multi_discrete
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                d = self.world.dim_p
            else:
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    d = 5
                else:  # 连续动作空间
                    if self.force_discrete_action:  # false
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0
                    # # 以下是给agent设置动作，与小车端的物理接口是一样的。
                    # 都是-1~1之间的[u0, u1]数组
                    network_output = action[0][0:self.world.dim_p]  # [ar, at] 1*2

                    if self.is_ternimate:
                        if (self.use_CL and self.CL_ratio > self.Cp) or self.use_CL==False or self.use_policy:
                            # agent 减速到 0
                            target_v = np.linalg.norm(agent.state.p_vel)
                            if target_v < 1e-3:
                                acc = np.array([0,0])
                            else:
                                acc = -agent.state.p_vel/target_v*agent.max_accel*1.1
                            network_output[0], network_output[1] = acc[0], acc[1]

                    # elif self.is_ternimate and self.use_policy:
                    #     # agent 减速到 0
                    #     target_v = np.linalg.norm(agent.state.p_vel)
                    #     if target_v < 1e-3:
                    #         acc = np.array([0,0])
                    #     else:
                    #         acc = -agent.state.p_vel/target_v*agent.max_accel
                    #     network_output[0], network_output[1] = acc[0], acc[1]

                    # rescale to 0~1
                    # rescale = 0.5*(network_output+1)  # 0~1
                    # 认为输出是[r, theta]
                    # r_, theta_ = rescale[0], rescale[1]
                    # act = 2*np.array([np.cos(2*np.pi*theta_), np.sin(2*np.pi*theta_)])
                    # act = limit_action_inf_norm(act, 1)
                    # act_norm = np.linalg.norm(act)
                    # network_output = r_*act_norm*np.array([np.cos(2*np.pi*theta_), np.sin(2*np.pi*theta_)])

                    policy_output = (policy_u.T)[0]
                    if self.use_CL == True:
                        if self.CL_ratio < self.Cp:
                            # fa2 = np.dot(network_output, policy_output)/np.dot(policy_output, policy_output)*policy_output if np.linalg.norm(policy_output - network_output)<1.0 else policy_output
                            # agent.action.u = (1-self.CL_ratio/self.Cp)*policy_output+self.CL_ratio/self.Cp*fa2
                            # agent.policy_action = policy_output
                            # agent.network_action = network_output

                            # act = (1-self.CL_ratio/self.Cp)*policy_output+self.CL_ratio/self.Cp*network_output
                            # agent.action.u = limit_action_inf_norm(act, 1)
                            # agent.policy_action = policy_output
                            # agent.network_action = network_output

                            # act = policy_output + self.CL_ratio/self.Cp*network_output
                            # agent.action.u = limit_action_inf_norm(act, 1)

                            # agent.action.u = 0.6*policy_output + 0.4*network_output
                            if self.current_step < self.JS_thre:
                                agent.action.u = policy_output
                            else:
                                agent.action.u = network_output
                        else:
                            # act = policy_output + network_output
                            act = network_output
                            agent.action.u = limit_action_inf_norm(act, 1)

                            # agent.action.u = 0.6*policy_output + 0.4*network_output
                    elif self.use_policy:
                        agent.action.u = policy_output
                    else:
                        act = network_output
                        agent.action.u = limit_action_inf_norm(act, 1)
                    # network_output = action[0][0:self.world.dim_p]
                    # agent.action.u = network_output
                    d = self.world.dim_p
                    # print("action in env is {}".format(action))
            # print("1 action in env is {}".format(action))

            if (not agent.silent) and (not isinstance(action_space, MultiDiscrete)):
                action = action[0][d:]
            else:
                action = action[1:]
        # print("2 action in env is {}".format(action))
        # print(len(action))

        # make sure we used all elements of action
        assert len(action) == 0, 'some action not used'

    def policy_u(self, landmarks, agents, target):
        num_agents = len(agents)
        U = np.zeros((num_agents, 2, 1))

        d_cap = 1.0
        L = 2*d_cap*np.sin(np.pi/num_agents)
        k_ic = 2.0
        k_icv = 1.5
        k_ij = 4.5
        k_b = 1.5  # 速度阻尼
        k_obs = 4.0
        k1, k2 = 1.2, 1.2
        k3, k4 = 0.25, 2.0
        for i, agent in enumerate(agents):
            # 与目标之间的吸引力
            r_ic = target.state.p_pos - agent.state.p_pos
            norm_r_ic = np.linalg.norm(r_ic)
            vel_vec = target.state.p_vel - agent.state.p_vel
            if norm_r_ic - d_cap > 0:
                if norm_r_ic - d_cap > 1.5:
                    f_c = 1.5/norm_r_ic*r_ic + k_icv*vel_vec
                else:
                    f_c = k_ic*(norm_r_ic - d_cap)/norm_r_ic*r_ic + k_icv*vel_vec
            else:  # 不能穿过目标
                f_c = 20 * k_ic * (norm_r_ic - d_cap) / norm_r_ic * r_ic + k_icv * vel_vec

            # if norm_r_ic - d_cap > 0.15:
            #     if norm_r_ic - d_cap > 1.5:
            #         f_c = 1.5/norm_r_ic*r_ic + k_icv*vel_vec
            #     else:
            #         f_c = k_ic*(norm_r_ic - d_cap)/norm_r_ic*r_ic + k_icv*vel_vec
            # elif norm_r_ic - d_cap < -0.15:
            #     f_c = 20 * k_ic * (norm_r_ic - d_cap) / norm_r_ic * r_ic + k_icv * vel_vec
            # else:
            #     f_c = -k_b*agent.state.p_vel

            # 势阱
            # if abs(norm_r_ic - d_cap) < 0.1:
            #     x_ = norm_r_ic
            #     R_ = d_cap + 0.1
            #     r_ = d_cap - 0.1
            #     q_c = d_cap
            #     # x_ = norm_r_ic - d_cap
            #     if x_ > q_c:  # 外侧。引力。
            #         f_p = 0.1 * (x_-q_c)**2/(R_-x_)**4/(R_-q_c)**2 * r_ic/norm_r_ic
            #     else:  # 内侧
            #         f_p = - 0.1 * (x_-q_c)**2/(x_-r_)**4/(q_c-r_)**2 * r_ic/norm_r_ic
            #     f_c = f_c + f_p

            # 与其他agt之间的斥力
            f_r = np.array([0, 0])
            for adv in agents:
                if adv is agent: continue
                r_ij = agent.state.p_pos - adv.state.p_pos
                norm_r_ij = np.linalg.norm(r_ij)
                if norm_r_ij < L:
                    f_ = k_ij*(L - norm_r_ij)/norm_r_ij*r_ij
                    if np.dot(f_, r_ic) < 0 and norm_r_ij > 2*L/3:  # 把与目标方向相反的部分力给抵消了
                        f_ = f_ - np.dot(f_, r_ic) / np.dot(r_ic, r_ic) * r_ic
                    f_r = f_r + f_

            # 与obs的斥力
            f_obs = np.array([0, 0])
            for landmark in landmarks:
                d_ij = agent.state.p_pos - landmark.state.p_pos
                norm_d_ij = np.linalg.norm(d_ij)
                L_min = agents[0].R + agents[0].delta + landmark.R + landmark.delta
                Ls = L_min+0.3
                if norm_d_ij < Ls:
                    f_obs = f_obs + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

            u_i = f_c + f_r + f_obs - k_b*agent.state.p_vel

            u_i = limit_action_inf_norm(u_i, 1)

            U[i] = u_i.reshape(2, 1)
        return U

    def _set_CL(self, CL_ratio):
        # 通过多进程set value，与env_wrapper直接关联，不能改。
        # 此处glv是这个进程中的！与mperunner中的并不共用。
        glv.set_value('CL_ratio', CL_ratio)
        self.CL_ratio = glv.get_value('CL_ratio')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False):
        if self.no_imageshow:
            # 只保留数据，不显示图像
            for i in range(len(self.viewers)):
                # steps： 1,2,3...199,0   1,2,3...
                if self.current_step == 1:
                    self.INFO_flag = 0
                    self.is_ternimate=False
                #csv
                if self.is_ternimate==True and self.INFO_flag == 0:  # 在常规时间内完成围捕
                    data_ = ()
                    data_ = data_ + (self.current_step, int(self.is_ternimate),)
                    INFO.append(data_)  # 增加行
                    self.INFO_flag = 1
                #csv
                elif self.is_ternimate==False and self.current_step == 0 and self.INFO_flag == 0:  # 终端也没有抓住
                    data_ = ()
                    data_ = data_ + (self.current_step, int(self.is_ternimate),)
                    INFO.append(data_)  # 增加行
                '''#csv
                data_ = ()
                for j in range(len(self.world.agents)):
                    data_ = data_ + (j, self.world.agents[j].state.p_pos[0], self.world.agents[j].state.p_pos[1], \
                                    self.world.agents[j].state.p_vel[0], self.world.agents[j].state.p_vel[1], \
                                        self.world.agents[j].state.phi)
                data_ = data_ + (int(self.is_ternimate),)
                INFO.append(data_)
                #csv'''
        else:
            if close:
                # close any existic renderers
                for i, viewer in enumerate(self.viewers):
                    if viewer is not None:
                        viewer.close()
                    self.viewers[i] = None
                return []

            if mode == 'human':
                alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                message = ''
                for agent in self.world.agents:
                    comm = []
                    for other in self.world.agents:
                        if other is agent:
                            continue
                        if np.all(other.state.c == 0):
                            word = '_'
                        else:
                            word = alphabet[np.argmax(other.state.c)]
                        message += (other.name + ' to ' +
                                    agent.name + ': ' + word + '   ')
                print(message)

            for i in range(len(self.viewers)):
                # create viewers (if necessary)
                if self.viewers[i] is None:
                    # import rendering only if we need it (and don't import for headless machines)
                    #from gym.envs.classic_control import rendering
                    from . import rendering
                    self.viewers[i] = rendering.Viewer(700, 700)

            # create rendering geometry
            if self.render_geoms is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                self.render_geoms = []
                self.render_geoms_xform = []
                self.line = {}

                self.comm_geoms = []
                for entity in self.world.entities:
                    # if 'agent' in entity.name:
                    radius = entity.R
                    geom = rendering.make_circle(radius)

                    xform = rendering.Transform()

                    entity_comm_geoms = []
                    if 'agent' in entity.name:
                        geom.set_color(*entity.color, alpha=0.5)

                        if not entity.silent:
                            dim_c = self.world.dim_c
                            # make circles to represent communication
                            for ci in range(dim_c):
                                comm = rendering.make_circle(entity.size / dim_c)
                                comm.set_color(1, 1, 1)
                                comm.add_attr(xform)
                                offset = rendering.Transform()
                                comm_size = (entity.size / dim_c)
                                offset.set_translation(ci * comm_size * 2 -
                                                    entity.size + comm_size, 0)
                                comm.add_attr(offset)
                                entity_comm_geoms.append(comm)

                    else:
                        geom.set_color(*entity.color)
                        if entity.channel is not None:
                            dim_c = self.world.dim_c
                            # make circles to represent communication
                            for ci in range(dim_c):
                                comm = rendering.make_circle(entity.size / dim_c)
                                comm.set_color(1, 1, 1)
                                comm.add_attr(xform)
                                offset = rendering.Transform()
                                comm_size = (entity.size / dim_c)
                                offset.set_translation(ci * comm_size * 2 -
                                                    entity.size + comm_size, 0)
                                comm.add_attr(offset)
                                entity_comm_geoms.append(comm)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                    self.comm_geoms.append(entity_comm_geoms)
                
                for wall in self.world.walls:
                    corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                            (wall.axis_pos - 0.5 *
                                wall.width, wall.endpoints[1]),
                            (wall.axis_pos + 0.5 *
                                wall.width, wall.endpoints[1]),
                            (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                    if wall.orient == 'H':
                        corners = tuple(c[::-1] for c in corners)
                    geom = rendering.make_polygon(corners)
                    if wall.hard:
                        geom.set_color(*wall.color)
                    else:
                        geom.set_color(*wall.color, alpha=0.5)
                    self.render_geoms.append(geom)

                # add geoms to viewer
                # for viewer in self.viewers:
                #     viewer.geoms = []
                #     for geom in self.render_geoms:
                #         viewer.add_geom(geom)
                for viewer in self.viewers:
                    viewer.geoms = []
                    for geom in self.render_geoms:
                        viewer.add_geom(geom)
                    for entity_comm_geoms in self.comm_geoms:
                        for geom in entity_comm_geoms:
                            viewer.add_geom(geom)

            results = []
            
            for i in range(len(self.viewers)):
                
                # 1st place for not showing fig when render
                from . import rendering

                if self.shared_viewer:
                    pos = np.zeros(self.world.dim_p)
                else:
                    pos = self.agents[i].state.p_pos
                self.viewers[i].set_bounds(
                    pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range+6.5, pos[1]+cam_range+6.5)
                
                
                #csv
                data_ = ()
                for j in range(len(self.world.agents)):
                    data_ = data_ + (j, self.world.agents[j].state.p_pos[0], self.world.agents[j].state.p_pos[1], \
                                    self.world.agents[j].state.p_vel[0], self.world.agents[j].state.p_vel[1], \
                                        self.world.agents[j].state.phi)
                data_ = data_ + (int(self.is_ternimate),)
                INFO.append(data_)
                #csv
                
                # update geometry positions
                for e, entity in enumerate(self.world.entities):
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                    self.line[e] = self.viewers[i].draw_line(entity.state.p_pos, entity.state.p_pos+entity.state.p_vel*1.0)
                    # 绘制agent速度

                    if 'agent' in entity.name:
                        self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                        self.line[e].set_color(*entity.color, alpha=0.5)

                        if not entity.silent:
                            for ci in range(self.world.dim_c):
                                color = 1 - entity.state.c[ci]
                                self.comm_geoms[e][ci].set_color(
                                    color, color, color)
                    else:
                        self.render_geoms[e].set_color(*entity.color)
                        if entity.channel is not None:
                            for ci in range(self.world.dim_c):
                                color = 1 - entity.channel[ci]
                                self.comm_geoms[e][ci].set_color(
                                    color, color, color)

                # render to display or array
                results.append(self.viewers[i].render(
                    return_rgb_array=mode == 'rgb_array'))

            return results
            

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx

def limit_action_inf_norm(action, max_limit):
    action = np.float32(action)
    action_ = action
    if abs(action[0]) > abs(action[1]):
        if abs(action[0])>max_limit:
            action_[1] = max_limit*action[1]/abs(action[0])
            action_[0] = max_limit if action[0] > 0 else -max_limit
        else:
            pass
    else:
        if abs(action[1])>max_limit:
            action_[0] = max_limit*action[0]/abs(action[1])
            action_[1] = max_limit if action[1] > 0 else -max_limit
        else:
            pass
    return action_

def set_JS_curriculum(CL_ratio):
    # func_ = 1-CL_ratio
    k = 2.0
    delta = 1-(np.exp(-k*(-1))-np.exp(k*(-1)))/(np.exp(-k*(-1))+np.exp(k*(-1)))
    x = 2*CL_ratio-1
    y_mid = (np.exp(-k*x)-np.exp(k*x))/(np.exp(-k*x)+np.exp(k*x))-delta*x**3
    func_ = (y_mid+1)/2
    return func_
