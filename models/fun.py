import time
import os
import random

# 需要滚动显示的文本
text = "徐云龙是大帅哥"

# 终端宽度（可以根据需要自行调整）
width = 80

# 定义几种颜色的ANSI码（可自行增加或修改）
colors = [
    "\033[31m", # 红
    "\033[32m", # 绿
    "\033[33m", # 黄
    "\033[34m", # 蓝
    "\033[35m", # 紫
    "\033[36m", # 青
    "\033[91m", # 亮红
    "\033[95m", # 亮紫
]

# 重置颜色
reset = "\033[0m"

# 制造一个随机的星空背景
def generate_background(width, height=1):
    line = ""
    for _ in range(width):
        # 随机决定是否放星号，不放则用空格
        if random.random() < 0.02:
            line += "*"
        else:
            line += " "
    return line

# 清屏函数
def clear_screen():
    # 尝试清屏
    os.system('cls' if os.name == 'nt' else 'clear')

positions = list(range(width, -len(text)-1, -1))

try:
    for pos in positions:
        clear_screen()
        
        # 生成背景
        bg_line = generate_background(width)
        
        # 构造打印行
        line = list(bg_line)
        for i, ch in enumerate(text):
            color = colors[(i + pos) % len(colors)]
            idx = pos + i
            if 0 <= idx < width:
                # 给对应位置上色赋值
                line[idx] = color + ch + reset
        
        print("".join(line))
        time.sleep(0.05)
except KeyboardInterrupt:
    # 用户中断的话就结束动画
    pass
finally:
    # 最后清一下屏幕结束
    clear_screen()
    print("动画结束！")
