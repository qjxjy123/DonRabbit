# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import math
import sys


class Brush:
    def __init__(self, screen):
        # pygame.Surface 对象
        self.screen = screen
        self.color = (0, 0, 0, 255)
        # 初始时候默认设置画笔大小为 1
        self.size = 1
        self.drawing = False
        self.last_pos = None
        # 如果 style 是 True ，则采用 png 笔刷
        # 若是 style 为 False ，则采用一般的铅笔画笔
        self.style = False
        # 加载刷子的样式
        self.brush = pygame.image.load("images/brush.png").convert_alpha()
        self.brush = pygame.transform.scale(self.brush,(64,64))
        self.brush_now = self.brush.subsurface((0, 0), (1, 1))
        
        # 创建 surface绘图对象，绘制图像到该区域上
        self.draw_area = pygame.Surface((500,500),SRCALPHA)
        self.draw_area.fill((255,255,255,0))
        # 若是 paint 为 True ，进入填充状态
        self.paint = False
        # 若是 circle 为 True ，进入画圆状态
        self.circle = False
        # 若是 eraser 为 True ，进入橡皮状态
        self.eraser = False
        
    def start_draw(self, pos):
        self.drawing = True
        self.last_pos = pos
        
    def end_draw(self):
        self.drawing = False

    def set_brush_style(self, style):
        print("* set brush style to", style)
        self.style = style

    def get_brush_style(self):
        return self.style

    def get_current_brush(self):
        return self.brush_now

    def set_size(self, size):
        if size < 1:
            size = 1
        elif size > 32:
            size = 32
        print("* set brush size to", size)
        self.size = size
        self.brush_now = self.brush.subsurface((0, 0), (size * 2, size * 2))

    def get_size(self):
        return self.size

    # 设定笔刷颜色
    def set_color(self, color):
        self.color = color
        for i in range(self.brush.get_width()):
            for j in range(self.brush.get_height()):
                self.brush.set_at((i, j),
                                  color)

    def get_color(self):
        return self.color

    # 绘制
    def draw(self, pos):
        if self.drawing:
            for p in self._get_points(pos):
                if self.style:
                    self.screen.blit(self.brush_now, p)
                else:
                    pygame.draw.circle(self.draw_area, self.color, (p[0]-100,p[1]-10), self.size)
                    self.screen.blit(self.draw_area,(100,10))
            self.last_pos = pos

    # 获取前一个点与当前点之间的所有需要绘制的点
    def _get_points(self, pos):
        points = [(self.last_pos[0], self.last_pos[1])]
        len_x = pos[0] - self.last_pos[0]
        len_y = pos[1] - self.last_pos[1]
        length = math.sqrt(len_x**2 + len_y**2)
        step_x = len_x / length
        step_y = len_y / length
        for i in range(int(length)):
            points.append((points[-1][0] + step_x, points[-1][1] + step_y))
        # 对 points 中的点坐标进行四舍五入取整
        points = map(lambda x: (int(0.5 + x[0]), int(0.5 + x[1])), points)
        # 去除坐标相同的点
        return list(set(points))
    
    def draw_circle(self, pos, draw_color):
        pos = (pos[0]-100,pos[1]-10)
        pygame.draw.circle(self.draw_area,draw_color,pos,7)
        self.screen.blit(self.draw_area,(100,10))
    
    def erase(self, pos, draw_color):
        pos = (pos[0]-100,pos[1]-10)
        # pygame.draw.rect(self.draw_area,draw_color,pos,4)
        sub = pygame.Surface((self.size,self.size),SRCALPHA)
        sub.fill((0,0,0,0))
        self.draw_area.blit(sub,pos)
        # pygame.draw.rect(self.screen, (0, 0, 0, 0), (pos[0], pos[1], self.size, self.size), 1)
        
        self.screen.blit(self.draw_area,(100,10))
        
    def inSurface(self, pos):
        if 0<=pos[0]<500 and 0<=pos[1]<500:
            return True
        else:
            return False
        

    def floodfill(self, position, fill_color):
        position = (position[0]-100,position[1]-10)
        surface = self.draw_area
        fill_color = surface.map_rgb(fill_color)  # Convert the color to mapped integer value.
        
        surf_array = pygame.surfarray.pixels2d(surface)  # Create an array from the surface.
        current_color = surf_array[position]  # Get the mapped integer color value.

        # 'frontier' is a list where we put the pixels that's we haven't checked. Imagine that we first check one pixel and 
        # then expand like rings on the water. 'frontier' are the pixels on the edge of the pool of pixels we have checked.
        #
        # During each loop we get the position of a pixel. If that pixel contains the same color as the ones we've checked
        # we paint it with our 'fill_color' and put all its neighbours into the 'frontier' list. If not, we check the next
        # one in our list, until it's empty.

        frontier = [position]
        while len(frontier) > 0:
            x, y = frontier.pop()
            try:  # Add a try-except block in case the position is outside the surface.
                if surf_array[x, y] != current_color:
                    continue
            except IndexError:
                continue
            surf_array[x, y] = fill_color
            # Then we append the neighbours of the pixel in the current position to our 'frontier' list.
            if self.inSurface(pos=(x+1,y)):
                frontier.append((x + 1, y))  # Right.
            if self.inSurface(pos=(x-1,y)):
                frontier.append((x - 1, y))  # Left.
            if self.inSurface(pos=(x,y+1)):
                frontier.append((x, y + 1))  # Down.
            if self.inSurface(pos=(x,y-1)):
                frontier.append((x, y - 1))  # Up.

        pygame.surfarray.blit_array(self.draw_area, surf_array)
        del surf_array
        self.screen.blit(self.draw_area,(100,10))
        self.paint = False
    
class Menu:
    def __init__(self, screen):
        self.screen = screen
        self.brush = None
        # 画板预定义的颜色值
        self.colors = [
            (0xff, 0xff, 0xff, 0x00), (0x80, 0x00, 0x80),
            (0x00, 0x00, 0xff), (0x00, 0x00, 0x80),
            (0x00, 0xff, 0xff), (0x00, 0x80, 0x80),
            (0x00, 0xff, 0x00), (0x00, 0x80, 0x00),
            (0xff, 0xff, 0x00), (0x80, 0x80, 0x00),
            (0xff, 0x00, 0x00), (0x80, 0x00, 0x00),
            (0xc0, 0xc0, 0xc0), (0xff, 0xff, 0xff),
            (0x00, 0x00, 0x00, 0xff), (0x80, 0x80, 0x80),
        ]
        # 计算每个色块在画板中的坐标值，便于绘制
        self.colors_rect = []
        for (i, rgb) in enumerate(self.colors):
            rect = pygame.Rect(10 + i % 2 * 32, 254 + i // 2 * 32, 32, 32)
            self.colors_rect.append(rect)
        # 两种笔刷的按钮图标
        self.pens = [
            pygame.transform.scale(pygame.image.load("images/pen1.png").convert_alpha(),(64,64)),
            pygame.transform.scale(pygame.image.load("images/pen2.png").convert_alpha(),(64,64))
        ]
        
        # 计算坐标，便于绘制
        self.pens_rect = []
        for (i, img) in enumerate(self.pens):
            rect = pygame.Rect(10, 10 + i * 64, 64, 64)
            self.pens_rect.append(rect)

        # 调整笔刷大小的按钮图标
        self.sizes = [
            pygame.transform.scale(pygame.image.load("images/big.png").convert_alpha(),(32,32)),
            pygame.transform.scale(pygame.image.load("images/small.png").convert_alpha(),(32,32))
        ]
        # 计算坐标，便于绘制
        self.sizes_rect = []
        for (i, img) in enumerate(self.sizes):
            rect = pygame.Rect(10 + i * 32, 138, 32, 32)
            self.sizes_rect.append(rect)
            
        
        # save按钮和调整位置
        self.save_rect = []
        self.save_img = [pygame.transform.scale(pygame.image.load("images/save.png").convert_alpha(),(64,64))]
        for (i, img) in enumerate(self.save_img):
            rect = pygame.Rect(10, 530, 64, 64)
            self.save_rect.append(rect)
            
        # 方向键按钮和调整位置
        self.directions_rect = []
        self.direction_imgs = [
            pygame.transform.scale(pygame.image.load("images/leftAngle.png").convert_alpha(),(32,32)),
            pygame.transform.scale(pygame.image.load("images/rightAngle.png").convert_alpha(),(32,32))
        ]
        for (i, img) in enumerate(self.direction_imgs):
            rect = pygame.Rect(10 + i % 2 * 32, 594 + i // 2 * 32, 32, 32)
            self.directions_rect.append(rect)
        
        # 填充按钮和调整位置
        self.paint_rect = []
        self.paint_img = [pygame.transform.scale(pygame.image.load("images/brush.png").convert_alpha(),(64,64))]
        for (i, img) in enumerate(self.paint_img):
            rect = pygame.Rect(10, 630, 64, 64)
            self.paint_rect.append(rect)
        # 画圆按钮和调整位置
        self.circle_rect = []
        self.circle_img = [pygame.transform.scale(pygame.image.load("images/circle.png").convert_alpha(),(64,64))]
        for (i, img) in enumerate(self.circle_img):
            rect = pygame.Rect(10, 700, 64, 64)
            self.circle_rect.append(rect)
        # 画圆按钮和调整位置
        self.eraser_rect = [] 
        self.eraser_img = [pygame.transform.scale(pygame.image.load("images/eraser.png").convert_alpha(),(64,64))]
        for (i, img) in enumerate(self.eraser_img):
            rect = pygame.Rect(10, 764, 64, 64)
            self.eraser_rect.append(rect)
        
        # 预显示的图片
        self.img_num = 1
        self.imgs = []
        self.imgs_rect = []
        for i in range(4):
            rect = pygame.Rect(100 + i % 2 * 510, 10 + i // 2 * 510, 500, 500)
            self.imgs_rect.append(rect)
        
        self.showClassificationFlag = False
        self.showDetectionFlag = False
        self.showDrawAreaFlag = False
        
    def showClassification(self):
        if self.showClassificationFlag == True:
            self.showClassificationFlag = False
        else:
            self.showClassificationFlag = True
            
    def showDetection(self):
        if self.showDetectionFlag == True:

            self.showDetectionFlag = False
        else:
            self.showDetectionFlag = True
    
    def showDrawArea(self):
        if self.showDrawAreaFlag == True:
            self.showDrawAreaFlag = False
        else:
            self.showDrawAreaFlag = True
            
    def set_brush(self, brush):
        self.brush = brush

        
    # 绘制菜单栏
    def draw(self):
        # 绘制画笔样式按钮
        for (i, img) in enumerate(self.pens):
            self.screen.blit(img, self.pens_rect[i].topleft)
        # 绘制 + - 按钮
        for (i, img) in enumerate(self.sizes):
            self.screen.blit(img, self.sizes_rect[i].topleft)
        # 绘制用于实时展示笔刷的小窗口
        self.screen.fill((255, 255, 255), (10, 180, 64, 64))
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 180, 64, 64), 1)
        size = self.brush.get_size()
        x = 10 + 32
        y = 180 + 32
        
        # 如果当前画笔为 png 笔刷，则在窗口中展示笔刷
        # 如果为铅笔，则在窗口中绘制原点
        if self.brush.get_brush_style():
            x = x - size
            y = y - size
            self.screen.blit(self.brush.get_current_brush(), (x, y))
        else:
            # BUG
            pygame.draw.circle(self.screen, 
                               self.brush.get_color(), (x, y), size)
        # 绘制色块
        for (i, rgb) in enumerate(self.colors):
            pygame.draw.rect(self.screen, rgb, self.colors_rect[i])
        # 绘制 save按钮
        for (i,img) in enumerate(self.save_img):
            self.screen.blit(img, self.save_rect[i].topleft)
        # 绘制 <- -> 按钮
        for (i, img) in enumerate(self.direction_imgs):
            self.screen.blit(img, self.directions_rect[i].topleft)
        # 绘制 paint按钮
        for (i, img) in enumerate(self.paint_img):
            self.screen.blit(img, self.paint_rect[i].topleft)
        # 绘制 circle按钮
        for (i, img) in enumerate(self.circle_img):
            self.screen.blit(img, self.circle_rect[i].topleft)
        # 绘制 eraser按钮
        for (i, img) in enumerate(self.eraser_img):
            self.screen.blit(img, self.eraser_rect[i].topleft)
    # 加载区域图片
    def loadImages(self):
        self.imgs = [
                    pygame.transform.scale(pygame.image.load("data/orig_img/img{}.png".format(self.img_num)),(500,500)),
                    pygame.transform.scale(pygame.image.load("data/orig_img/img{}.png".format(self.img_num)),(350,350)),
                    pygame.transform.scale(pygame.image.load("data/point_mask_detection/img{}_plus.png".format(self.img_num)),(350,350)),
                    pygame.transform.scale(pygame.image.load("data/point_mask_classification/{}_plus.png".format(self.img_num)),(350,350)),
                    pygame.transform.scale(pygame.image.load("data/point_mask_detection/RGBA_3/img{}.png".format(self.img_num)),(500,500)),
                    pygame.transform.scale(pygame.image.load("data/point_mask_classification/RGBA_3/{}.png".format(self.img_num)),(500,500)),
                    pygame.transform.scale(pygame.image.load("data/region_mask/RGBA_3/img{}.png".format(self.img_num)),(500,500)),
                ]
    # 重新绘制画图屏幕
    def showAll(self):
        self.screen.blit(self.imgs[0],(100,10))
        if self.showDetectionFlag == True:
            self.screen.blit(self.imgs[4],(100,10))
        if self.showClassificationFlag == True:
            self.screen.blit(self.imgs[5],(100,10))
        if self.showDrawAreaFlag == True:
            self.brush.draw_area.blit(self.imgs[6],(0,0))
            self.screen.blit(self.brush.draw_area,(100,10))
            
        
            
    # 定义菜单按钮的点击响应
    def click_button(self, pos):
        # 笔刷
        for (i, rect) in enumerate(self.pens_rect):
            if rect.collidepoint(pos):
                self.brush.set_brush_style(bool(i))
                return True
        # 笔刷大小
        for (i, rect) in enumerate(self.sizes_rect):
            if rect.collidepoint(pos):
                # 画笔大小的每次改变量为 1
                if i:
                    self.brush.set_size(self.brush.get_size() - 1)
                else:
                    self.brush.set_size(self.brush.get_size() + 1)
                return True
        # 颜色
        for (i, rect) in enumerate(self.colors_rect):
            if rect.collidepoint(pos):
                self.brush.set_color(self.colors[i])
                return True
        # 1.区域绘制
        
        # 保存绘制图片
        for (i, rect) in enumerate(self.save_rect):
            if rect.collidepoint(pos):
                # rect = pygame.Rect(0, 0, 500, 500)
                # sub = self.brush.draw_area.subsurface(rect)
                # sub = pygame.transform.scale(sub,(500,500))
                # sub = self.brush.draw_area.convert()
                
                sub = pygame.Surface((500,500),SRCALPHA)
                sub.blit(self.brush.draw_area,(0,0),(0,0,500,500))
                
#                 sub = pygame.transform.scale(sub,(500,500))
#                 sub = pygame.Surface((500,500))
#                 sub.blit(self.screen,(0,0),(100,10,500,500))
#                 sub = pygame.transform.scale(sub,(500,500))
            
                # pygame.image.save(sub, "data/region_mask/mask{}_region.png".format(self.img_num))
                pygame.image.save(sub, "data/region_mask/img{}.png".format(self.img_num))
                pygame.image.save(sub, "data/region_mask/RGBA_3/img{}.png".format(self.img_num))
                sub = pygame.Surface((500,500))
                sub.blit(self.screen,(0,0),(100,10,500,500))
                pygame.image.save(sub, "data/region_mask/visualize/img{}.png".format(self.img_num))
                return True
        # 进入涂料状态
        for (i, rect) in enumerate(self.paint_rect):
            if rect.collidepoint(pos):
                if self.brush.paint == True:
                    self.brush.paint = False
                else:
                    self.brush.paint = True
                
                return True
        # 进入画圆状态
        for (i, rect) in enumerate(self.circle_rect):
            if rect.collidepoint(pos):
                if self.brush.circle == True:
                    self.brush.circle = False
                else:
                    self.brush.circle = True
                return True
            
        # 进入橡皮状态
        for (i, rect) in enumerate(self.eraser_rect):
            if rect.collidepoint(pos):
                if self.brush.eraser == True:
                    self.brush.eraser = False
                else:
                    self.brush.eraser = True
                return True
        # 载入前一组或者后一组图片
        for (i, rect) in enumerate(self.directions_rect):
            if rect.collidepoint(pos):
                if self.imgs == []:
                    self.loadImages()
                    for (i, img) in enumerate(self.imgs):
                        if i<1:
                            self.screen.blit(img, self.imgs_rect[i].topleft)
                    self.brush.draw_area.fill((255,255,255,0))
                    self.screen.blit(self.brush.draw_area, (100,10))
                    
                if i == 0:
                    if self.img_num > 1:
                        self.img_num -= 1
                    else:
                        self.img_num = 1
                    self.loadImages()
                    for (i, img) in enumerate(self.imgs):
                        if i<1:
                            self.screen.blit(img, self.imgs_rect[i].topleft)
                    self.brush.draw_area.fill((255,255,255,0))
                    self.screen.blit(self.brush.draw_area, (100,10))
                elif i == 1:
                    if self.img_num < 100:
                        self.img_num += 1
                    else:
                        self.img_num = 100
                    self.loadImages()
                    for (i, img) in enumerate(self.imgs):
                        if i<1:
                            self.screen.blit(img, self.imgs_rect[i].topleft)
                    self.brush.draw_area.fill((255,255,255,0))
                    self.screen.blit(self.brush.draw_area, (100,10))
                self.showAll()
                pygame.display.set_caption(str(self.img_num)+'.png')
                return True
        
        return False

def judgeBorder(pos):
        if 600 >= pos[0] >= 100 and 510 >= pos[1] >= 10:
            return True
        else:
            return False

class Painter:
    def __init__(self):
        # 设置了画板窗口的大小与标题
        self.screen = pygame.display.set_mode((1100, 900))
        
        pygame.display.set_caption("Painter")
        # 创建 Clock 对象
        self.clock = pygame.time.Clock()
        # 创建 Brush 对象
        self.brush = Brush(self.screen)
        # 创建 Menu 对象，并设置了默认笔刷
        self.menu = Menu(self.screen)
        self.menu.set_brush(self.brush)

        
    def run(self):
        self.screen.fill((255, 255, 255))
        # 程序的主体是一个循环，不断对界面进行重绘，直到监听到结束事件才结束循环
        while True:
            # 设置帧率
            self.clock.tick(60)
            # 监听事件
            for event in pygame.event.get():
                # 结束事件
                if event.type == QUIT:
                    return
                # 键盘按键事件
                elif event.type == KEYDOWN:
                    # 按下 ESC 键，清屏
                    if event.key == K_LEFT:
                        self.menu.showClassification()
                        self.menu.showAll()
                    if event.key == K_UP:
                        self.menu.showDrawArea()
                        self.menu.showAll()
                    if event.key == K_RIGHT:
                        self.menu.showDetection()
                        self.menu.showAll()
                    if event.key == K_ESCAPE:
                        self.screen.fill((255, 255, 255))
                # 鼠标按下事件|
                elif event.type == MOUSEBUTTONDOWN:
                    # 若是当前鼠标位于菜单中，则忽略掉该事件
                    # 否则调用 start_draw 设置画笔的 drawing 标志为 True
                    if event.pos[0] <= 74:
                        self.menu.click_button(event.pos)
                    elif judgeBorder(event.pos):
                        if self.brush.paint == True:
                            pass
                        elif self.brush.circle == True:
                            pass
                        elif self.brush.eraser == True:
                            pass
                        else:
                            self.brush.start_draw(event.pos)        
                # 鼠标移动事件
                elif event.type == MOUSEMOTION:
                    if judgeBorder(event.pos):
                        if self.brush.paint == True:
                            pass
                        elif self.brush.circle == True:
                            pass
                        elif self.brush.eraser == True:
                            pass
                        else:
                            self.brush.draw(event.pos)   
                    else:
                        self.brush.end_draw()
                # 松开鼠标按键事件
                elif event.type == MOUSEBUTTONUP:
                    # 调用 end_draw 设置画笔的 drawing 标志为 False
                    if judgeBorder(event.pos):
                        if self.brush.paint == True:
                            self.brush.floodfill(event.pos, (0,0,0,255))
                        elif self.brush.circle == True:
                            self.brush.draw_circle(event.pos, (0,0,0,255))
                        elif self.brush.eraser == True:
                            self.brush.erase(event.pos,(0,0,0,0))
                        else:
                            self.brush.end_draw()
            # 绘制菜单按钮
            self.menu.draw()
            # 刷新窗口
            pygame.display.update()

def main():
    app = Painter()
    app.run()
    pygame.quit()
if __name__ == '__main__':
    main()