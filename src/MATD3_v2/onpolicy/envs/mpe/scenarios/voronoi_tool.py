from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
import numpy as np

def get_init_bnd(evader, pursuer, n_pursuer):
    max_x_adv = pursuer[0, :]
    max_y_adv = pursuer[0, :]
    del_x, del_y = 0, 0
    for i in range(n_pursuer):
        adv_i = pursuer[i, :]
        _del_x, _del_y = abs(adv_i[0] - evader[0]), abs(adv_i[1] - evader[1])
        if _del_x > del_x:
            del_x = _del_x
            max_x_adv = adv_i
        if _del_y > del_y:
            del_y = _del_y
            max_y_adv = adv_i
    # print(max_x_adv, del_x, max_y_adv, del_y)
    bounds = np.array([[evader[0] - del_x, evader[1] - del_y],
                       [evader[0] + del_x, evader[1] - del_y],
                       [evader[0] + del_x, evader[1] + del_y],
                       [evader[0] - del_x, evader[1] + del_y]])
    return bounds

def update_bound(bound, control_inputs, all_agents):
    evader = all_agents[0]
    pursuer = all_agents[1:]
    xmin = np.min(bound[:, 0])
    xmax = np.max(bound[:, 0])
    ymin = np.min(bound[:, 1])
    ymax = np.max(bound[:, 1])
    deltaX, deltaY = (xmax-xmin)/2, (ymax-ymin)/2
    gamma = deltaX/(deltaX+deltaY)
    # 选出离evader最远的X向和Y向的pursuer
    max_x_adv = -1  # 编号
    max_y_adv = -1
    del_x, del_y = 0, 0
    for i in range(len(all_agents)-1):
        if i == 0:
            pass
        adv_i = all_agents[i, :]
        _del_x, _del_y = abs(adv_i[0] - evader[0]), abs(adv_i[1] - evader[1])
        if _del_x > del_x:
            del_x = _del_x
            max_x_adv = i
        if _del_y > del_y:
            del_y = _del_y
            max_y_adv = i
    # print(control_inputs[max_y_adv])
    X_deltaT = control_inputs[max_x_adv, 0]*0.1
    Y_deltaT = control_inputs[max_y_adv, 1]*0.1

    D_deltaT = abs(X_deltaT) if abs(X_deltaT) < abs(Y_deltaT) else abs(Y_deltaT)
    new_x = deltaX - D_deltaT*gamma
    new_y = deltaX - D_deltaT * (1-gamma)
    bounds = np.array([[evader[0] - new_x, evader[1] - new_y],
                       [evader[0] + new_x, evader[1] - new_y],
                       [evader[0] + new_x, evader[1] + new_y],
                       [evader[0] - new_x, evader[1] + new_y]])
    return bounds

def bounded_voronoi(bnd, pnts):
    """
    有界区域的维诺图计算
    """

    # 为了使所有的母点的Voronoi区域保持有界，需要添加三个虚拟母点。
    gn_pnts = np.concatenate([pnts, np.array([[100, 100], [100, -100], [-100, 0]])])

    # 维诺图计算
    vor = Voronoi(gn_pnts)

    # 分割多边形
    bnd_poly = Polygon(bnd)

    # 存储各个维诺图区域的列表
    vor_polys = []

    # 对于除虚拟点以外的母点进行循环
    for i in range(len(gn_pnts) - 3):

        # 不考虑闭空间的维诺图区域
        vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
        # 计算要分割的区域与维诺图区域的交集
        i_cell = bnd_poly.intersection(Polygon(vor_poly))

        # 存储考虑闭空间的维诺图区域的顶点坐标
        vor_polys.append(list(i_cell.exterior.coords[:-1]))

    return vor_polys

def add_obs_hyperplane(all_agents, vor, obs):
    global i_cell
    new_vor = []

    agt = all_agents[0]  # agent顺序和维诺图顺序保持一致
    points = np.array(vor[0])  # 维诺cell i的顶点
    hull0 = Polygon(points)
    bound_points = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]])
    for j in range(len(obs)):
        x_, y_, R_, delta_ = obs[j, 0], obs[j, 1], obs[j, 2], obs[j, 3]
        center = np.array([x_, y_])
        w = agt - center  # 超平面wx=b
        P0 = center + w/np.linalg.norm(w)*(R_+delta_)
        A_, B_, C_ = -w[0], -w[1], np.dot(w, P0)
        big_poly_bnd = []

        # 选择三个初始方向,基于P0沿每个方向上走100作为点
        e1 = w/np.linalg.norm(w)
        e2 = np.array([e1[1], -e1[0]])
        e3 = -e2
        P1 = P0 + e1*100
        P2 = P0 + e2*100
        P3 = P0 + e3*100
        big_poly_bnd = [P1, P2, P3]
        '''
        for k in range(len(bound_points)):
            if np.dot(w, bound_points[k]) > C_ and len(big_poly_bnd) < 2:  # 有可能3个点都在超平面一侧
                big_poly_bnd.append(bound_points[k])
                # print(bound_points[k])
        if abs(B_) < 1e-4:  # B == 0
            big_poly_bnd.append([-C_/A_, 100])
            big_poly_bnd.append([-C_/A_, -100])
        else:
            big_poly_bnd.append([100, (-C_-A_*100)/B_])
            big_poly_bnd.append([-100, (-C_-A_ * (-100))/B_])

        # 这一段对取交集的区域进行polish
        dist_thre = 50
        del_flag = False
        for l in range(2):
            dist_ = np.linalg.norm(np.array(big_poly_bnd[0]) - np.array(big_poly_bnd[l+2]))
            if dist_ < dist_thre:
                big_poly_bnd.pop(0)
                del_flag = True
                break
        for l in range(2):
            if del_flag == True:
                break
            else:
                dist_ = np.linalg.norm(np.array(big_poly_bnd[1]) - np.array(big_poly_bnd[l+2]))
                if dist_ < dist_thre:
                    # print('deleted1', big_poly_bnd)
                    a = big_poly_bnd.pop(1)
                    # print('after deleted1', big_poly_bnd)
                    # print('deleted1', a)
                    del_flag = True
                    break
        '''

        # print(del_flag)
        # print(big_poly_bnd)
        big_poly = Polygon(big_poly_bnd)
        i_cell = hull0.intersection(big_poly)  # 取交集后的区域
        # print(hull0)
        hull0 = Polygon(i_cell.exterior.coords[:-1])

    # 第一个vor区域遍历完所有障碍物后
    new_vor = list(i_cell.exterior.coords[:-1])

    return new_vor

def compute_target_pts(vor_CA, all_agents):

    pts = np.array(vor_CA)
    tri_cent = []
    for j in range(len(pts) - 2):
        pt1, pt2, pt3 = pts[0], pts[j + 1], pts[j + 2]
        area = 1 / 2 * np.cross(pt2 - pt1, pt3 - pt1)
        tri_cent.append([1 / 3 * (pt1[0] + pt2[0] + pt3[0]), 1 / 3 * (pt1[1] + pt2[1] + pt3[1]), area])
    Area, sumx, sumy = 0, 0, 0
    for j in range(len(tri_cent)):
        sumx = sumx + tri_cent[j][0] * tri_cent[j][2]
        sumy = sumy + tri_cent[j][1] * tri_cent[j][2]
        Area = Area + tri_cent[j][2]
    Cx = sumx / Area
    Cy = sumy / Area

    target_pt = [Cx, Cy]

    return target_pt

