import asyncio
import platform
import pygame
import random
import math
import os

# 初始化 Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# 创建用于保存帧的文件夹
FRAME_DIR = "frames"
if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)

# 初始化字体
font = pygame.font.SysFont("arial", 24)  # 使用 Arial 字体，大小 24

# 粒子类
class Particle:
    def __init__(self, x, y, radius, speed):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.angle = random.uniform(0, 2 * math.pi)
        self.color = (255, 255, 255)  # 白色粒子

    def move(self):
        # 随机改变方向
        self.angle += random.uniform(-0.2, 0.2)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # 边界反弹
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.angle = math.pi - self.angle
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        if self.y - self.radius < 0 or self.y + self.radius > HEIGHT:
            self.angle = -self.angle
            self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

# 全局变量
particle = Particle(WIDTH // 2, HEIGHT // 2, 10, 10)  # 初始位置中心，半径20，速度5
running = True
frame_count = 0
start_time = 0  # 记录动画开始时间
max_frames = 1800  # 限制总帧数（例如 30秒，30*60=1800帧）

def setup():
    global running, start_time
    pygame.display.set_caption("Random Moving Particle with Time Display")
    running = True
    start_time = pygame.time.get_ticks() / 1000.0  # 获取初始时间（秒）

def update_loop():
    global running, frame_count
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 计算当前时间（秒）
    current_time = pygame.time.get_ticks() / 1000.0 - start_time
    cycle_time = current_time % 16

    # 确定当前状态
    show_particle = False
    if 0 <= cycle_time < 10:  # 前10秒黑屏
        show_particle = False
    elif 1 <= cycle_time < 16:  # 5秒粒子运动
        show_particle = True
        particle.move()


    # 绘制
    screen.fill((0, 0, 0))  # 黑色背景
    if show_particle:
        particle.draw(screen)

    # 绘制时间文本（左下角）
    time_text = font.render(f"Time: {current_time:.1f}s", True, (255, 255, 255))  # 白色文本
    screen.blit(time_text, (10, HEIGHT - 30))  # 左下角，距离底部30像素

    pygame.display.flip()

    # 保存当前帧
    if frame_count < max_frames:
        filename = os.path.join(FRAME_DIR, f"frame_{frame_count:04d}.png")
        pygame.image.save(screen, filename)
        frame_count += 1
    else:
        running = False  # 达到最大帧数后停止

async def main():
    setup()
    while running:
        update_loop()
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())