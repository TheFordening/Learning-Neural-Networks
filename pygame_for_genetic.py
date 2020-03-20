import pygame
from random import *
import random
import math
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import numpy as np
pygame.init()
screen = pygame.display.set_mode((800,600)) # generic syntax
running = True

percentages = [0.5,0.6,0.7,0.8,0.9]

def round(num):
    return -1 if num < 0 else 1
###################################################################
class genetic_algo:
    def __init__(self,population_size):
        self.population_size = population_size
        self.population = [ball(x, y, rad) for i in range(population_size)]
        self.initial_pop = True
        self.next_pop = []

    def all_stopped(self):
        stopped = []
        for i in self.population:
            if i.stop == True:
                stopped.append(True)
            else:
                stopped.append(False)
        for i in stopped:
            if i == False:
                return False
        return True

    def fitness_sort(self):
        self.population.sort(key=lambda x: x.distance_val, reverse=False)
        #self.population.sort(key=lambda y: y.y_fitness,reverse=True)

    def ranking(self):
        ranker = self.population_size
        for i in self.population:
            i.rank = ranker
            ranker -= 1


    def selection(self,generation):
        self.next_pop.append(ball(self.population[0].goal_x,self.population[0].goal_y,self.population[0].goal_radius))
        self.next_pop[-1].movement.set_weights(self.population[0].movement.get_weights())
        self.ranking()
        num  = len(self.next_pop)
        """
        S = 0
        for i in self.population:
            S += i.rank
        """
        pie_wheel = []
        for i in self.population:
            for z in range(i.rank):
                pie_wheel.append(i)

        while(num < self.population_size):
            """
            P = random.uniform(0,S)
            m = 0
            i = -1
            while m < P:
                i += 1
                m += self.population[i].rank
            m = 0
            P = random.uniform(0,S)
            parent_a = self.population[i]
            i = -1
            while m < P:
                i += 1
                m += self.population[i].rank
            parent_b = self.population[i]
            """
            parent_a = pie_wheel[int(random.uniform(0,len(pie_wheel)))]
            parent_b = pie_wheel[int(random.uniform(0,len(pie_wheel)))]
            self.mate(parent_a,parent_b,generation)
            num += 1
        self.population = [i for i in self.next_pop]
        del self.next_pop[:]

    def mate(self,a,b,generation):
        half = len(a.movement.get_weights()) // 2
        self.next_pop.append(ball(self.population[0].goal_x,self.population[0].goal_y,self.population[0].goal_radius))
        self.next_pop[-1].movement.set_weights(a.movement.get_weights()[0:half] + b.movement.get_weights()[half:])
        self.mutate(generation)

    def mutate(self,generation):
        """
        if generation//10 < 5:
            percentage = percentages[generation//10]
        else:
            percentage = percentages[-1]

        p = random.random()
        if p < percentage:
            return
        """
        p = random.random()
        if p < 0.8:
            return
        shape = np.shape(self.next_pop[-1].movement.get_weights())
        noise =np.random.random(shape)
        self.next_pop[-1].movement.set_weights(self.next_pop[-1].movement.get_weights() + noise)

    def most_hit(self):
        numberhit = 0
        for i in self.population:
            if i.color == [0,0,255]:
                numberhit += 1
        return numberhit

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
        self.movement = net().create_net()
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

mem = genetic_algo(20)
for i in mem.population:
    i.draw()

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
                if event.key == pygame.K_RIGHT:
                    for i in mem.population:
                        i.stop = True
                if event.key == pygame.K_a:
                    t_x_pos_change -= 10
                if event.key == pygame.K_d:
                    t_x_pos_change += 10
                if event.key == pygame.K_w:
                    t_y_pos_change -= 10
                if event.key == pygame.K_s:
                    t_y_pos_change += 10
                if event.key == pygame.K_e:
                    mem.population[0].movement.save("best_ball.h5")
                if event.key == pygame.K_UP:
                    test_target.reset()
                    for i in mem.population:
                        i.goal_x = test_target.t_x_pos
                        i.goal_y = test_target.t_y_pos

        test_target.t_x_pos += t_x_pos_change
        test_target.t_y_pos += t_y_pos_change

        for i in mem.population:
            i.goal_y = test_target.t_y_pos
            i.goal_x = test_target.t_x_pos

        if press:
            screen.fill([0,0,0])
            for i in mem.population:
                i.move()
                i.draw()
                i.target_hit()


        if mem.all_stopped():
            print("Generation: %d"%generation)
            print("The number of populants that hit the target are: %d" %(mem.most_hit()))
            print(len(mem.population))
            for i in mem.population:
                if i.hit == True:
                    print(min(i.distances))
            for i in mem.population:
                i.true_distance()
            generation += 1
            mem.fitness_sort()
            mem.selection(generation)

        test_target.draw()
        pygame.display.update()
