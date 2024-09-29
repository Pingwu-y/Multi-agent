import numpy as np
from onpolicy.envs.mpe.core import World, Agent
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy import global_var as glv

class Scenario(BaseScenario):
    
    def __init__(self) -> None:
        super().__init__()
        self.cd = 1.0  # 取消Cd
        self.cp = 0.4
        self.cr = 1.0  # 取消Cr
        self.d_cap = 1.5 # 期望围捕半径,动态变化,在set_CL里面
        self.init_target_pos = 4.5
        self.use_CL = False # 是否使用课程式训练(render时改为false)

    # 设置agent,landmark的数量，运动属性。
    def make_world(self,args):
        world = World()
        world.collaborative = True
        # set any world properties first
        num_good_agents = 1 # args.num_good_agents
        num_adversaries = 3 # args.num_adversaries
        num_agents = num_adversaries + num_good_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):  # i 从0到5
            agent.i = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False  # agent 0 1 2 3 4:adversary.  5: good
            agent.size = 0.03 if agent.adversary else 0.045
            # agent.accel = 3.0 if agent.adversary else 3.0  # max acc 一阶模型不用设置这个
            agent.max_speed = 2.0 if agent.adversary else 0.3
            agent.max_angular = 2.0 if agent.adversary else 2.0

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # properties and initial states for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            # agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            if i == 0:
                agent.state.p_pos = np.array([-1.0, 0.0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.phi = np.pi/2
            elif i == 1:
                agent.state.p_pos = np.array([0.0, 0.0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.phi = np.pi/2
            elif i == 2:
                agent.state.p_pos = np.array([1.0, 0.0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.phi = np.pi/2
            elif i == 3:
                rand_pos = np.random.uniform(0, 1, 2)  # 1*2的随机数组，范围0-1
                r_, theta_ = 0.5*rand_pos[0], np.pi*2*rand_pos[1]  # 半径为0.5，角度360，随机采样。圆域。
                if self.use_CL:
                    init_dist = self.init_target_pos*(self.cr + (1-self.cr)*glv.get_value('CL_ratio')/self.cp)
                else:
                    init_dist = self.init_target_pos
                agent.state.p_pos = np.array([r_*np.cos(theta_), init_dist+r_*np.sin(theta_)])  # (0,5)为圆心
                # agent.state.p_pos = np.array([0.0, init_dist+(r_-0.5)*2])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.action_callback = escape_policy
                agent.done = False
                # callback只调用函数名。escape_policy的出入参数应该与agent.action_callback()保持一致
                # print('111111', agent.state.p_pos)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def Get_antiClockAngle(self, v1, v2):  # 向量v1逆时针转到v2所需角度。范围：0-2pi
        # 2个向量模的乘积
        TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
        assert TheNorm!=0.0, "0 in denominator"
        # 叉乘
        rho = np.arcsin(np.cross(v1, v2)/TheNorm)
        # 点乘
        cos_ = np.dot(v1, v2)/TheNorm
        if 1.0 < cos_:
            cos_ = 1.0
            rho = 0
        elif cos_ < -1.0:
            cos_ = -1.0
        theta = np.arccos(cos_)
        if rho < 0:
            return np.pi*2 - theta
        else:
            return theta

    def GetAcuteAngle(self, v1, v2):  # 计算较小夹角(0-pi)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-4 or norm2 < 1e-4:
            # print('0 in denominator ')
            cos_ = 1  # 初始化速度为0，会出现分母为零
        else:
            cos_ = np.dot(v1, v2)/(norm1*norm2)
            if 1.0 < cos_:
                cos_ = 1.0
            elif cos_ < -1.0:
                cos_ = -1.0
        return np.arccos(cos_)

    '''
    返回左右邻居下标(论文中邻居的定义方式)和夹角
        agent: 当前adversary agent
        adversary: 所有adversary agents数组
        target: good agent
    '''
    def find_neighbors(self, agent, adversary, target):
        angle_list = []
        for adv in adversary:
            if adv == agent:
                angle_list.append(-1.0)
                continue
            agent_vec = agent.state.p_pos-target.state.p_pos
            neighbor_vec = adv.state.p_pos-target.state.p_pos
            angle_ = self.Get_antiClockAngle(agent_vec, neighbor_vec)
            if np.isnan(angle_):
                # print("angle_list_error. agent_vec:{}, nb_vec:{}".format(agent_vec, neighbor_vec))
                if adv.i==0:
                    print("tp{:.3f} tv:{:.3f}".format(target.state.p_pos, target.state.p_vel))
                    print("0p{:.1f} 0v:{:.1f}".format(adversary[0].state.p_pos, adversary[0].state.p_vel))
                    print("1p{:.3f} 1v:{:.3f}".format(adversary[1].state.p_pos, adversary[1].state.p_vel))
                    print("2p{:.3f} 2v:{:.3f}".format(adversary[2].state.p_pos, adversary[2].state.p_vel))
                    print("3p{:.3f} 3v:{:.3f}".format(adversary[3].state.p_pos, adversary[3].state.p_vel))
                    print("4p{:.3f} 4v:{:.3f}".format(adversary[4].state.p_pos, adversary[4].state.p_vel))
                angle_list.append(0)
            else:
                angle_list.append(angle_)

        min_angle = np.sort(angle_list)[1]  # 第二小角，把自己除外
        max_angle = max(angle_list)
        min_index = angle_list.index(min_angle)
        max_index = angle_list.index(max_angle)
        max_angle = np.pi*2 - max_angle

        return [max_index, min_index], max_angle, min_angle

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def set_CL(self, CL_ratio):
        d_cap = 1.5
        if CL_ratio < self.cp:
            # print('in here Cd')
            self.d_cap = d_cap*(self.cd + (1-self.cd)*CL_ratio/self.cp)
        else:
            self.d_cap = d_cap
    
    # agent 和 adversary 分别的reward
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary == True:
            main_reward = self.adversary_reward(agent, world) # else self.agent_reward(agent, world
        else:
            print('error')
        return main_reward

    # individual adversary award
    def adversary_reward(self, agent, world):  # agent here is adversary
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'))
        
        # print("dcap is {}".format(self.d_cap))
        # Agents are rewarded based on individual position advantage
        r_step = 0
        target = self.good_agents(world)[0]  # moving target
        adversaries = self.adversaries(world)
        N_adv = len(adversaries)
        dist_i_vec = target.state.p_pos - agent.state.p_pos
        dist_i = np.linalg.norm(dist_i_vec)  #与目标的距离
        d_i = dist_i - self.d_cap  # 剩余围捕距离
        d_list = [np.linalg.norm(adv.state.p_pos - target.state.p_pos) - self.d_cap for adv in adversaries]   # left d for all adv
        d_mean = np.mean(d_list)
        sigma_d = np.std(d_list)
        exp_alpha = np.pi*2/N_adv
        # find neighbors (方位角之间不存在别的agent)
        nb_idx, left_nb_angle, right_nb_angle = self.find_neighbors(agent, adversaries, target)  # nb:neighbor
        delta_alpha = abs(left_nb_angle - right_nb_angle)
        # find min d between allies
        d_min = 20
        for adv in adversaries:
            if adv == agent:
                continue
            d_ = np.linalg.norm(agent.state.p_pos - adv.state.p_pos)
            if d_ < d_min:
                d_min = d_
        # if dist_i < d_min: d_min = dist_i  # 与目标的碰撞也考虑进去，要围捕不能撞上

        ####### calculate dones ########
        dones = []
        for adv in adversaries:
            di_adv = np.linalg.norm(target.state.p_pos - adv.state.p_pos) - self.d_cap
            _, left_nb_angle_, right_nb_angle_ = self.find_neighbors(adv, adversaries, target)
            # print('i:{}, leftE:{}, rightE:{}'.format(adv.i, abs(left_nb_angle_ - exp_alpha), abs(right_nb_angle_ - exp_alpha)))
            # 0.2, 0.3
            if abs(di_adv)<0.1 and abs(left_nb_angle_ - exp_alpha)<0.3 and abs(right_nb_angle_ - exp_alpha)<0.3: # 30°
                dones.append(True)
            else: dones.append(False)
            '''if abs(left_nb_angle_ - exp_alpha)<0.3 and abs(right_nb_angle_ - exp_alpha)<0.3: # 30°
                dones.append(True)
            else: dones.append(False)'''
        if all(dones)==True:
            agent.done = True
            target.done = True
            return 10
        else:  agent.done = False
        #################################

        '''if self.use_CL == True:
            # Cp之前不考虑避障，Cp之后考虑避障
            if glv.get_value('CL_ratio') < self.cp:
                k1, k2 = 0.2, 0.1
                w1, w2 = 0.4, 0.6
                # formaion reward r_f
                form_vec = np.array([0.0, 0.0])
                for adv in adversaries:
                    form_vec = form_vec + (adv.state.p_pos - target.state.p_pos)
                r_f = np.exp(-k1*np.linalg.norm(form_vec)) - 1
                # distance coordination reward r_d
                r_d = np.exp(-k2*np.sum(np.square(d_list))) - 1
                
                r_step = w1*r_f + w2*r_d
            else:
                k1, k2, k3 = 0.2, 0.1, 2.0
                w1, w2, w3 = 0.35, 0.4, 0.25
                # formaion reward r_f
                form_vec = np.array([0.0, 0.0])
                for adv in adversaries:
                    form_vec = form_vec + (adv.state.p_pos - target.state.p_pos)
                r_f = np.exp(-k1*np.linalg.norm(form_vec)) - 1
                # distance coordination reward r_d
                r_d = np.exp(-k2*np.sum(np.square(d_list))) - 1
                # neighbor coordination reward r_l
                r_l = 2/(1+np.exp(-k3*d_min))-2

                r_step = w1*r_f + w2*r_d + w3*r_l
        else:
            # render，考虑避障
            k1, k2, k3 = 0.2, 0.1, 2.0
            w1, w2, w3 = 0.35, 0.4, 0.25
            # formaion reward r_f
            form_vec = np.array([0.0, 0.0])
            for adv in adversaries:
                form_vec = form_vec + (adv.state.p_pos - target.state.p_pos)
            r_f = np.exp(-k1*np.linalg.norm(form_vec)) - 1
            # distance coordination reward r_d
            r_d = np.exp(-k2*np.sum(np.square(d_list))) - 1
            # neighbor coordination reward r_l
            r_l = 2/(1+np.exp(-k3*d_min))-2

            r_step = w1*r_f + w2*r_d + w3*r_l'''
            
        
        #k1, k2 = 0.2, 0.1
        w1, w2 = 0.4, 0.6
        k1, k2 = 0.05, 0.1
        # formaion reward r_f
        form_vec = np.array([0.0, 0.0])
        for adv in adversaries:
            form_vec = form_vec + (adv.state.p_pos - target.state.p_pos)
        r_f = np.exp(-k1*np.linalg.norm(form_vec)) - 1
        # distance coordination reward r_d
        r_d = np.exp(-k2*np.sum(np.square(d_list))) - 1
        
        r_step = w1*r_f + w2*r_d
        
        # 0.2, 0.3
        if abs(d_i)<0.2 and abs(left_nb_angle - exp_alpha)<0.6 and abs(right_nb_angle - exp_alpha)<0.6: # 30°
            return 5 # 5    # terminate reward
        else:
            return r_step
        '''if abs(left_nb_angle - exp_alpha)<0.3 and abs(right_nb_angle - exp_alpha)<0.3:
            return 5 # 5    # terminate reward
        else:
            return r_step'''
         
    # observation for adversary agents
    def observation(self, agent, world):
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'))

        target = self.good_agents(world)[0]  # moving target
        adversaries = self.adversaries(world)
        dist_vec = agent.state.p_pos - target.state.p_pos
        vel_vec = agent.state.p_vel - target.state.p_vel
        # calculate o_loc：
        Q_iM = self.GetAcuteAngle(-dist_vec, agent.state.p_vel)
        Q_iM_dot = np.dot(vel_vec, -dist_vec)/np.sum(np.square(dist_vec))
        d_i = np.linalg.norm(dist_vec) - self.d_cap
        d_i_dot = (dist_vec[0]*vel_vec[0]+dist_vec[1]*vel_vec[1])/np.linalg.norm(dist_vec)
        o_loc = [Q_iM, Q_iM_dot, d_i, d_i_dot, np.linalg.norm(agent.state.p_vel), agent.state.last_a]  # 1*6
        # print(o_loc)
        # calculate o_ext：
        _, left_nb_angle, right_nb_angle = self.find_neighbors(agent, adversaries, target)  # nb:neighbor
        delta_alpha = left_nb_angle - right_nb_angle
        d_list = [np.linalg.norm(adv.state.p_pos - target.state.p_pos) - self.d_cap for adv in adversaries]   # left d for all adv
        d_mean = np.mean(d_list)
        o_ext = [delta_alpha, d_mean]  # 1*2
        # communication of all other agents
        o_ij = np.array([])  # 1*N*5
        for adv_j in adversaries:
            if adv_j is agent: continue
            # [id1, id2], _, _ = self.find_neighbors(agent, adversaries, target)
            # if adv_j.i == id1 or adv_j.i == id2: # 只考虑邻居的
            d_ij_vec = agent.state.p_pos - adv_j.state.p_pos
            d_ij = np.linalg.norm(d_ij_vec)
            Q_ij = self.GetAcuteAngle(-d_ij_vec, agent.state.p_vel)
            Q_ji = self.GetAcuteAngle(adv_j.state.p_vel, d_ij_vec)
            dist_j_vec = adv_j.state.p_pos - target.state.p_pos
            d_j = np.linalg.norm(dist_j_vec) - self.d_cap
            delta_d_ij = d_i - d_j
            alpha_ij = self.GetAcuteAngle(dist_vec, dist_j_vec)
            o_ij = np.concatenate([o_ij]+[np.array([d_ij, Q_ij, Q_ji, delta_d_ij, alpha_ij])])
        
        # assert len(o_ij) == 20, ('o_ij length not right')
        # 只取邻居的特征
        # print(o_ij)
        # if len(o_ij) == 10:
        #     o_ij = 0.5*(o_ij[0:5]+o_ij[5:10])
        # else: print('o_ij length not right')

        obs_concatenate = np.concatenate([o_loc] + [o_ext] + [o_ij]) # concatenate要用两层括号
        return obs_concatenate

    def done(self, agent, world):
        target = self.good_agents(world)[0]
        adversaries = self.adversaries(world)
        exp_alpha = np.pi*2/len(adversaries)
        dones = []
        for adv in adversaries:
            d_i = np.linalg.norm(adv.state.p_pos - target.state.p_pos) - self.d_cap
            _, left_nb_angle, right_nb_angle = self.find_neighbors(adv, adversaries, target)
            if d_i<0 and abs(left_nb_angle - exp_alpha)<0.5 and abs(right_nb_angle - exp_alpha)<0.5: # 30°
                dones.append(True)
            else: dones.append(False)
            '''if abs(left_nb_angle - exp_alpha)<0.5 and abs(right_nb_angle - exp_alpha)<0.5: # 30°
                dones.append(True)
            else: dones.append(False)'''
        
        # print("dones is: ",dones)
        if all(dones)==True:
            agent.done = True  # 在env中改变dones
            return True
        else:
            agent.done = False
            return False
            
# # 逃逸目标的策略
def escape_policy(agent, adversaries):
    set_CL = False
    Cp = 0.4
    Cv = 0.4
    action = agent.action
    if agent.done==True:  # terminate
        escape_v = np.array([0.0, 0.0])
    else:
        if set_CL:
            max_v = agent.max_speed
            CL_ratio = glv.get_value('CL_ratio')
            if CL_ratio < Cp:  # Cp
                max_speed = max_v*(Cv + (1-Cv)*CL_ratio/Cp)  # Cv = 0.2
                # print('in here Cv')
            else:
                max_speed = max_v
        else:
            max_speed = agent.max_speed
        # print("simple, CL is {}, maxV is {}".format(CL_ratio, max_speed))
        
        '''
        # potential based
        escape_v = np.array([0, 0])
        for adv in adversaries:
            d_vec_ij = agent.state.p_pos - adv.state.p_pos
            d_vec_ij = d_vec_ij / np.square(np.linalg.norm(d_vec_ij))
            escape_v = escape_v+d_vec_ij
        
        escape_v = escape_v/np.square(np.linalg.norm(escape_v))  # 原文这么设的，有点不合理

        # 超过最大速度,归一化
        if np.linalg.norm(escape_v) > max_speed:
            escape_v = escape_v/np.linalg.norm(escape_v) * max_speed
        '''

        escape_v = np.array([0, 0])
        for adv in adversaries:
            d_vec_ij = agent.state.p_pos - adv.state.p_pos
            d_vec_ij = d_vec_ij / np.square(np.linalg.norm(d_vec_ij))
            escape_v = escape_v+d_vec_ij
        
        # 计算此刻与上一时刻速度方向
        v_vector = escape_v/np.linalg.norm(escape_v)  # 此刻期望速度方向
        last_v_norm = np.linalg.norm(agent.state.p_vel)
        if last_v_norm > 0:
            last_v_vec = agent.state.p_vel/last_v_norm  # 上一速度方向
        else:
            last_v_vec = np.array([0.0, 1.0]) # 初始速度方向
        
        # 新旧方向夹角delta_theta(-pi ~ pi)
        rho = np.arcsin(np.cross(last_v_vec, v_vector))
        cos_ = np.dot(last_v_vec, v_vector)
        if 1.0 < cos_:
            cos_ = 1.0
            rho = 0
        elif cos_ < -1.0:
            cos_ = -1.0
        delta_theta = np.arccos(cos_)
        if rho < 0:
            delta_theta =  np.pi*2 - delta_theta
        if delta_theta > np.pi: delta_theta = delta_theta - np.pi*2

        # constrain
        max_theta = 0.1*agent.max_angular  # self.dt
        if abs(delta_theta) > max_theta:
            max_theta = max_theta if delta_theta > 0 else -max_theta
            # rotate
            v_vector = np.array([last_v_vec[0] * np.cos(max_theta) - last_v_vec[1] * np.sin(max_theta),
                                last_v_vec[0] * np.sin(max_theta) + last_v_vec[1] * np.cos(max_theta)])

        escape_v = max_speed * v_vector

    action.u = escape_v  # 1*2
    return action
