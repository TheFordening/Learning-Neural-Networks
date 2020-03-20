#Best Version so far - Works
import pygame
from random import *
import random
import math
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import numpy as np
import tensorflow
pygame.init()
screen = pygame.display.set_mode((800,600)) # generic syntax
running = True

percentages = [0.5,0.6,0.7,0.8,0.9]

def round(num):
    return -1 if num < 0 else 1
###################################################################
class net:
    def __init__(self):
        pass
    def create_net(self):
        input = Input(shape = (1,2))
        x = Dense(3,activation = 'relu')(input)
        output = Dense(2,activation = 'tanh')(x)
        return Model(input,output)
###################################################################
class target:
    def __init__(self,radius):
        self.t_x_pos = randint(200,600)
        self.t_y_pos = randint(0,300)
        self.radius = radius
    def draw(self):
        pygame.draw.circle(screen,[0,255,0],[self.t_x_pos,self.t_y_pos],self.radius,0)
    def reset(self):
        self.t_x_pos = randint(200,600)
        self.t_y_pos = randint(0,300)
        pygame.draw.circle(screen,[0,255,0],[self.t_x_pos,self.t_y_pos],self.radius,0)
####################################################################
class ball:
    def __init__(self,target_x,target_y, target_radius):
        self.target_radius = target_radius
        self.rank = 0
        self.goal_y = target_y
        self.distances = []
        self.goal_radius = target_radius
        self.movement = tensorflow.keras.models.load_model('best_ball.h5')
        self.hit = False

        self.goal_x = target_x
        self.x_pos = 395
        self.y_pos = 590
        self.color = [255,0,0]
        self.stop = False
        self.initial_position = [395,595]

    def draw(self):
        pygame.draw.circle(screen,self.color,[self.x_pos,self.y_pos],5,0)

    def move(self):
        self.distances.append(self.distance())
        out = self.movement.predict(np.reshape([self.x_pos - self.goal_x,self.goal_y - self.y_pos],(1,1,2)))
        if self.y_pos < 5:
            self.stop = True
            return
        elif self.y_pos > 595:
            self.stop = True
            return
        elif self.x_pos < 5:
            self.stop = True
            return
        elif self.x_pos > 795:
            self.stop = True
            return
        self.x_pos +=  round(out[0][0][0])
        self.y_pos -=  round(out[0][0][1])

    def distance(self):
        return math.sqrt((self.goal_x - self.x_pos)**2 + (self.goal_y - self.y_pos)**2)

    def true_distance(self):
        self.distance_val = min(self.distances)

    def reset(self):
        self.x_pos = self.initial_position[0]
        self.y_pos = self.initial_position[1]
        self.color = [255,0,0]
        self.stop = False
        self.hit = False

    def target_hit(self):
        if self.y_pos >= self.goal_y - 5 and self.y_pos <= self.goal_y + 5:
            if self.x_pos >= self.goal_x - 5 and self.x_pos <= self.goal_x + 5:
                self.color = [0,0,255]
                self.stop = True
                self.hit = True
#####################################################################
test_target = target(5)
test_target.draw()
x = test_target.t_x_pos
y = test_target.t_y_pos
rad = test_target.radius
ball = ball(x,y,rad)
ball.draw()

press = False
generation = 1
#####################################################################
while running:
        t_x_pos_change = 0
        t_y_pos_change = 0
        for event in pygame.event.get(): # runs through all the events in pygame
            if event.type == pygame.QUIT: # capital quit
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    press = True
                if event.key == pygame.K_a:
                    t_x_pos_change -= 30
                if event.key == pygame.K_d:
                    t_x_pos_change += 30
                if event.key == pygame.K_w:
                    t_y_pos_change -= 30
                if event.key == pygame.K_s:
                    t_y_pos_change += 30
                if event.key == pygame.K_UP:
                    test_target.reset()
                    ball.goal_x = test_target.t_x_pos
                    ball.goal_y = test_target.t_y_pos
                    ball.reset()
                    press = False

        test_target.t_x_pos += t_x_pos_change
        test_target.t_y_pos += t_y_pos_change
        if test_target.t_x_pos < 5:
            test_target.t_x_pos = 0
        elif test_target.t_x_pos > 795:
            test_target.t_x_pos = 795
        if test_target.t_y_pos < 5:
            test_target.t_y_pos = 5
        elif test_target.t_y_pos > 595:
            test_target.t_y_pos = 595


        ball.goal_y = test_target.t_y_pos
        ball.goal_x = test_target.t_x_pos

        if press:
            screen.fill([0,0,0])
            ball.move()
            ball.target_hit()

        screen.fill([0,0,0])
        ball.draw()
        test_target.draw()
        pygame.display.update()
