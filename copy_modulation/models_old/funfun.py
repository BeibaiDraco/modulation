import numpy as np
import matplotlib.pyplot as plt

# 时间设置
dt = 0.1
T = 500
time = np.arange(T)*dt

# 初始状态 (Hz)
E1 = np.zeros(T)
I1 = np.zeros(T)
E2 = np.zeros(T)
I2 = np.zeros(T)

E1[0] = 0.0
I1[0] = 200.0
E2[0] = 0.0
I2[0] = 50.0

# 参数设置
# 外部输入
E1_input = 15.0  # 正确方向E外部输入较强，一打进来可能瞬间升高
E2_input = 9.0  # 反方向E外部输入较弱，初期缓慢上升

I1_input = 5.0   # 正确方向的I外部输入稍大
I2_input = 1.0   # 反方向的I外部输入较小

# 抑制和兴奋参数
# 简化为：E下个状态依赖于外界输入和抑制，I依赖于E的输入和外部输入
tau_E = 10.0
tau_I = 5.0


# 抑制对E的效果
I_to_E_weight = -0.5
# E对I的效果(兴奋)，但有非线性
# 基本思路：I的输入 = I外部输入 + (E对I的兴奋度)
# 当E < 50Hz 用较小增益，当E > 50Hz 用较大增益

for t in range(1, T):
    E1_prev = E1[t-1]
    I1_prev = I1[t-1]
    E2_prev = E2[t-1]
    I2_prev = I2[t-1]

    # 对正确方向I的激发增益选择

    E1_to_I1 = E1_prev 
    E2_to_I2 = E2_prev 



    # 更新I的状态
    # 简单模型：dI/dt = -(I/tau_I) + I外部输入 + E->I的兴奋
    dI1 = (-I1_prev + I1_input + E1_to_I1) / tau_I
    dI2 = (-I2_prev + I2_input + E2_to_I2) / tau_I

    I1_new = I1_prev + dt * dI1
    I2_new = I2_prev + dt * dI2

    # 更新E的状态
    # E受到：外部输入 + I的抑制(I_to_E_weight * I) 
    # dE/dt = -(E/tau_E) + E外部输入 + I抑制
    dE1 = (-E1_prev + E1_input + I_to_E_weight * I1_new) / tau_E
    dE2 = (-E2_prev + E2_input + I_to_E_weight * I2_new) / tau_E

    E1_new = E1_prev + dt * dE1
    E2_new = E2_prev + dt * dE2

    # 保证E和I不为负数（ReLU-like）
    E1_new = max(0, E1_new)
    E2_new = max(0, E2_new)
    I1_new = max(0, I1_new)
    I2_new = max(0, I2_new)

    E1[t] = E1_new
    I1[t] = I1_new
    E2[t] = E2_new
    I2[t] = I2_new

# 绘图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(time, E1, label='E Correct')
plt.plot(time, E2, label='E Opposite')

plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.title('E populations activity')
plt.legend()

plt.subplot(1,2,2)
plt.plot(time, I1, label='I Correct')
plt.plot(time, I2, label='I Opposite')
plt.xlabel('Time (ms)')
plt.ylabel('Firing rate (Hz)')
plt.title('I populations activity')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 时间参数
dt = 0.1
T = 500
time = np.arange(T)*dt

# 初始状态
E1 = np.zeros(T)
I1 = np.zeros(T)
E2 = np.zeros(T)
I2 = np.zeros(T)

E1[0] = 0.0
I1[0] = 12.0
E2[0] = 0.0
I2[0] = 0.0

# 参数设定（两方向除输入外都相同）
tau = 10.0      # E和I相同的时间常数
G_E = 1.0       # E对外部输入的增益
G_I = 2.0       # I对外部输入的增益，比E更大，I更敏感
w_EI = -0.5     # I对E的抑制权重
g_IE = 0.5      # E对I的兴奋权重

# 外部输入
# 正确方向输入更大
E1_input = 15
I1_input = 5.0

# 反方向输入更小
E2_input = 12
I2_input = 4.5

for t in range(1, T):
    E1_prev, I1_prev = E1[t-1], I1[t-1]
    E2_prev, I2_prev = E2[t-1], I2[t-1]

    # 正确方向I更新
    dI1 = (-I1_prev + G_I*I1_input + g_IE*E1_prev) / tau
    I1_new = I1_prev + dt * dI1

    # 正确方向E更新
    dE1 = (-E1_prev + G_E*E1_input + w_EI*I1_new) / tau
    E1_new = E1_prev + dt * dE1

    # 反方向I更新
    dI2 = (-I2_prev + G_I*I2_input + g_IE*E2_prev) / tau
    I2_new = I2_prev + dt * dI2

    # 反方向E更新
    dE2 = (-E2_prev + G_E*E2_input + w_EI*I2_new) / tau
    E2_new = E2_prev + dt * dE2

    # 非负约束（可选）
    E1_new = max(0, E1_new)
    I1_new = max(0, I1_new)
    E2_new = max(0, E2_new)
    I2_new = max(0, I2_new)

    E1[t], I1[t] = E1_new, I1_new
    E2[t], I2[t] = E2_new, I2_new

# 绘图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(time, E1, label='E Correct')
plt.plot(time, E2, label='E Opposite')
plt.xlabel('Time')
plt.ylabel('Firing rate (Hz)')
plt.title('E populations activity')
plt.legend()

plt.subplot(1,2,2)
plt.plot(time, I1, label='I Correct')
plt.plot(time, I2, label='I Opposite')
plt.xlabel('Time')
plt.ylabel('Firing rate (Hz)')
plt.title('I populations activity')
plt.legend()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 时间参数
dt = 0.1
T = 500
time = np.arange(T)*dt

# 初始条件
E1 = np.zeros(T)
I1 = np.zeros(T)
E2 = np.zeros(T)
I2 = np.zeros(T)

# 参数设定（两方向除输入外一致）
tau = 10.0         # E和I相同的时间常数
G_E = 1.0          # E对外部输入的增益
G_I = 3.0          # I对外部输入的增益，比E大，使I对输入更敏感
w_EI = -0.5        # I对E的抑制权重(负值)
g_IE = 0.5         # E对I的正反馈权重

# 饱和参数
E_max = 20.0
E_half = 10.0
I_max = 20.0
I_half = 10.0

def saturate(x, X_max, X_half):
    # 平滑饱和函数，x/(x+X_half)*X_max，当x>>X_half时接近X_max
    # 确保非负
    x = np.maximum(x, 0)
    return (x / (x + X_half)) * X_max

# 外部输入 (正确方向与反方向不同)
# 正确方向有更大的E与I输入, 使I初期快速跃升
E1_input = 1.0
I1_input = 10.0   # 给I更大输入，让其早期快速上升并饱和

# 反方向外部输入较小
E2_input = 0.3
I2_input = 0.1

for t in range(1, T):
    E1_prev, I1_prev = E1[t-1], I1[t-1]
    E2_prev, I2_prev = E2[t-1], I2[t-1]

    # 正确方向的I更新(线性更新后再饱和)
    I1_lin = I1_prev + dt/tau * (-I1_prev + G_I*I1_input + g_IE*E1_prev)
    I1_new = saturate(I1_lin, I_max, I_half)

    # 正确方向的E更新(线性更新后再饱和)
    E1_lin = E1_prev + dt/tau * (-E1_prev + G_E*E1_input + w_EI*I1_new)
    E1_new = saturate(E1_lin, E_max, E_half)

    # 反方向的I更新(线性更新后再饱和)
    I2_lin = I2_prev + dt/tau * (-I2_prev + G_I*I2_input + g_IE*E2_prev)
    I2_new = saturate(I2_lin, I_max, I_half)

    # 反方向的E更新(线性更新后再饱和)
    E2_lin = E2_prev + dt/tau * (-E2_prev + G_E*E2_input + w_EI*I2_new)
    E2_new = saturate(E2_lin, E_max, E_half)

    E1[t], I1[t] = E1_new, I1_new
    E2[t], I2[t] = E2_new, I2_new

# 绘图
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(time, E1, label='E Correct')
plt.plot(time, E2, label='E Opposite')
plt.xlabel('Time')
plt.ylabel('Firing rate (Hz)')
plt.title('E populations activity')
plt.legend()

plt.subplot(1,2,2)
plt.plot(time, I1, label='I Correct')
plt.plot(time, I2, label='I Opposite')
plt.xlabel('Time')
plt.ylabel('Firing rate (Hz)')
plt.title('I populations activity')
plt.legend()

plt.tight_layout()
plt.show()