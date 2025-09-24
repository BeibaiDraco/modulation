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
E_max = 10.0
E_half = 5.0
I_max = 15.0
I_half = 5.0

def saturate(x, X_max, X_half):
    # 平滑饱和函数，x/(x+X_half)*X_max，当x>>X_half时接近X_max
    # 确保非负
    x = np.maximum(x, 0)
    return (x / (x + X_half)) * X_max

# 外部输入 (正确方向与反方向不同)
# 正确方向有更大的E与I输入, 使I初期快速跃升
E1_input = 15.0
I1_input = 10.0   # 给I更大输入，让其早期快速上升并饱和

# 反方向外部输入较小
E2_input = 9.0
I2_input = 2.0

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
