import random
import math
import matplotlib.pyplot as plt
import copy
import numpy as np

ENERGY_POINTS = []
CIRCLE_RADIUS = 1
TEMPERATURE = 1

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def make_energy_system(number_of_points):

    for i in range(number_of_points):
        random_distance = random.uniform(0, 1)
        random_angle = random.uniform(0, 2*math.pi)
        (x,y) = pol2cart(random_distance, random_angle)
        ENERGY_POINTS.append(EnergyBall(x,y))


def distance_between(E1, E2):
    diff_x = abs(E1.x - E2.x)
    diff_y = abs(E1.y - E2.y)
    return math.sqrt(diff_x**2+diff_y**2)


def total_energy(points = ENERGY_POINTS):
    energy = 0
    for i in range(len(points)):
        for j in range(i):
            dis = distance_between(points[i], points[j])
            if dis == 0:
                energy = 1000
            else:
                energy += 1/abs(dis)
    return energy

def find_force_vector(ball_number):
    without_ball_number = ENERGY_POINTS
    right_ball = ENERGY_POINTS[ball_number]
    ball_x_force = []
    ball_y_force = []

    for ball in without_ball_number:
        if ball != right_ball:
            ball_x_force.append((1/((ball.x - right_ball.x)**3)))
            ball_y_force.append((1/((ball.x - right_ball.x)**3)))

    x = sum(ball_x_force)
    y = sum(ball_y_force)
    return x, y

def expo_variet(low, high):
    high_pos_neg = high/abs(high)
    low_pos_neg = low/abs(low)

    scale = random.expovariate(1)
    total = abs(high+low)
    number_scale = total*scale

    if number_scale>abs(low):
        return number_scale*high_pos_neg
    elif number_scale<abs(high):
        return number_scale*low_pos_neg
    else:
        return 0

def random_move():
    global ENERGY_POINTS, TEMPERATURE
    old_config = copy.deepcopy(ENERGY_POINTS)
    ball_nr = random.randrange(0,len(ENERGY_POINTS))

    ball = ENERGY_POINTS[ball_nr]

    x, y = find_force_vector(ball_nr)
    delta_x = expo_variet(-x*0.4, x)
    delta_y = expo_variet(-y*0.4, y)

    new_ball = ball.move(delta_x, delta_y)


    ENERGY_POINTS[ball_nr] = new_ball
    new_energy = total_energy(points=ENERGY_POINTS)
    old_energy = total_energy(points=old_config)
    alpha = min(np.exp(-(new_energy-old_energy)/TEMPERATURE), 1)
    if random.uniform(0,1) > alpha:
        ENERGY_POINTS = old_config


def plotsystem():
    angles = []
    lengths = []
    for ball in ENERGY_POINTS:
        length, angle = cart2pol(ball.x, ball.y)
        angles.append(angle)
        lengths.append(length)
    figure = plt.figure()
    ax = figure.add_subplot(polar=True)
    ax.set_rmax(1)
    ax.set_rmin(1)
    ax.plot(angles, lengths, 'o')
    plt.show()


class EnergyBall:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.forceV_x = 0
        self.forceV_y = 0

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        if self.x**2 + self.y**2 >1:
            r, phi = cart2pol(self.x, self.y)
            self.x, self.y = pol2cart(1, phi)
        return self

    def __str__(self):
        return "Length: " + str(self.length) + ", angle = " + str(self.angle)
    # def add_force_on(self, amount_balls, x, y):
    #     self.forceV_x = self.forceV_x + (x/amount_balls)
    #     self.forceV_y = self.forceV_y + (y/amount_balls)


a=1
b=2
store_iterations = []

for i in range(50):
    ENERGY_POINTS = []
    make_energy_system(12)
    print(len(ENERGY_POINTS))
    for j in range(100):
        random_move()
        TEMPERATURE = a/math.log(10000 + j+b)
    store_iterations.append(total_energy(points=ENERGY_POINTS))
        # print(TEMPERATURE)
    #plotsystem()
#plotsystem()
f = open('controled vector.txt', 'w')
print(store_iterations, file=f)
f.close()
