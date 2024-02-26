import casadi as ca
import numpy as np
from moveit_commander import MoveGroupCommander
import rospy
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest

class ArmModel(object):
    def __init__(self, moveit_commander_model):
        rospy.init_node("ik_test")#正运动学解算节点
        # 系统状态：四个关节的角度
        theta1 = ca.SX.sym('theta1')
        theta2 = ca.SX.sym('theta2')
        theta3 = ca.SX.sym('theta3')
        theta4 = ca.SX.sym('theta4')
        states = ca.vertcat(theta1, theta2, theta3, theta4)

        # 控制输入：四个关节的角速度
        omega1 = ca.SX.sym('omega1')
        omega2 = ca.SX.sym('omega2')
        omega3 = ca.SX.sym('omega3')
        omega4 = ca.SX.sym('omega4')
        controls = ca.vertcat(omega1, omega2, omega3, omega4)

        # 我们假设状态的导数与控制输入直接相关 (dx/dt = u）
        rhs = controls

        # 定义状态和控制变量与系统方程之间的关系
        self.f = ca.Function('f', [states, controls], [rhs])

        # 存储状态和控制变量的数量
        self.nx = states.size(1)
        self.nu = controls.size(1)
        self.kinematics_model = moveit_commander_model
    def forward_kinematics(self, joint_state):
        ik_service_name = "compute_ik"
        rospy.wait_for_service(ik_service_name)
        ik_service = rospy.ServiceProxy(ik_service_name, GetPositionIK)
        ik_request = GetPositionIKRequest()
        # Populate ik_request here

        try:
            response = ik_service(ik_request)
            print(response)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))


class ArmMPC(object):
    def __init__(self, model, N, dt):
        self.model = model
        self.N = N  # 预测范围
        self.dt = dt  # 控制间隔

        # 初始化决策变量和约束列表
        self.X = ca.SX.sym('X', model.nx, N+1) # 状态向量
        self.U = ca.SX.sym('U', model.nu, N)   # 控制向量

        # 代价函数
        self.J = 0

        # 构建MPC问题
        self.__build_mpc_problem()

        # 设置和求解NLP
        self.__setup_solver()

    def __build_mpc_problem(self):
        # 设置权重矩阵
        Q = ca.diag(ca.DM([1, 1, 1, 1]))
        R = ca.diag(ca.DM([0.1, 0.1, 0.1, 0.1]))

        # 初始状态为参数
        self.X0 = ca.SX.sym('X0', self.model.nx)
        self.U_ref = ca.SX.sym('U_ref', self.model.nu)
        self.y_ref = ca.MX.sym('X_ref', 7, self.N)

        # 构建代价函数和动态约束
        for k in range(self.N):
            Y_k = self.model.forward_kinematics(self.X[:,k])
            self.J += (Y_k - Y_des).T @ Q @ (X_k - X_des) + (self.U[:,k]).T @ R @ (self.U[:,k])
            # 添加动力学约束（简化）
            X_next = self.X[:,k] + self.model.rhs * self.dt
            self.X[:,k+1] = X_next  # 更新状态

        # 添加状态更新的约束
        self.g = []
        for k in range(self.N):
            x_next = self.X[:,k] + self.model.f(self.X[:,k], self.U[:,k]) * self.dt
            self.g.append(self.X[:,k+1] - x_next)
    
    def __setup_solver(self):
        # 使用CasADi的非线性规划求解器
        opts = {'ipopt': {'print_level': 0, 'acceptable_tol': 1e-8, 'max_iter': 100},
                'print_time': 0}
        self.nlp = {'x': ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1)), 'f': self.J, 'g': ca.vertcat(*self.g), 'p': ca.vertcat(self.X0, self.U_ref)}
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, opts)

    def solve(self, x0, u_ref):
        # 设置初始条件和参考值
        p = ca.vertcat(x0, u_ref)
        # 设置初始猜测和约束界
        x0 = ca.DM.zeros(self.nlp['x'].size(1))
        lbx = -ca.DM.inf * ca.DM.ones(self.nlp['x'].size(1))
        ubx = ca.DM.inf * ca.DM.ones(self.nlp['x'].size(1))
        lbg = ca.DM.zeros(len(self.g))
        ubg = ca.DM.zeros(len(self.g))
        
        sol = self.solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        u_opt = ca.reshape(sol['x'][self.model.nx*(self.N+1):], self.model.nu, self.N)
        
        return u_opt[:,0]

# 示例使用
arm_model = ArmModel()
arm_mpc = ArmMPC(arm_model, N=20, dt=0.1)

x0 = ca.DM([0, 0, 0, 0])  # 初始状态
u_ref = ca.DM([0.1, 0, 0, 0])  # 目标状态
u_optimal = arm_mpc.solve(x0, u_ref)
print(u_optimal)