import random
import math
import matplotlib.pyplot as plt
import copy
import numpy as np

ENERGY_POINTS = []
CIRCLE_RADIUS = 1
TEMPERATURE = 1

# Function to convert from cartesian to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

# Function to convert from polar to cartesian coordinates
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

# Randomly initialize system with given number of energy points
def make_energy_system(points):
    number_of_points = points
    global ENERGY_POINTS
    ENERGY_POINTS = []
    for i in range(number_of_points):
        random_distance = random.uniform(0, 1)
        random_angle = random.uniform(0, 2*math.pi)
        (x,y) = pol2cart(random_distance, random_angle)
        ENERGY_POINTS.append(EnergyBall(x,y))

# Calculate distance between two vectors in cartesian coordinates
def distance_between(E1, E2):
    diff_x = abs(E1.x - E2.x)
    diff_y = abs(E1.y - E2.y)
    return math.sqrt(diff_x**2+diff_y**2)

# Calculate the total energy of the system
def total_energy(points = ENERGY_POINTS):
    energy = 0
    for i in range(len(points)):
        for j in range(i):
            energy += 1/abs(distance_between(points[i], points[j]))
    return energy

# Make a random move and accept/reject the move based on the calculated alpha
def random_move():
    global ENERGY_POINTS, TEMPERATURE
    old_config = copy.deepcopy(ENERGY_POINTS)
    ball_nr = random.randrange(0, len(ENERGY_POINTS))
    ball = ENERGY_POINTS[ball_nr]
    step = 0.3
    delta_x = random.uniform(-step, step)
    delta_y = random.uniform(-step, step)
    new_ball = ball.move(delta_x, delta_y)
    ENERGY_POINTS[ball_nr] = new_ball
    new_energy = total_energy(points=ENERGY_POINTS)
    old_energy = total_energy(points=old_config)
    if new_energy < old_energy:
        alpha = 1
    else:
        alpha = min(math.exp(-(new_energy-old_energy)/TEMPERATURE), 1)
    if random.uniform(0,1) > alpha:
        ENERGY_POINTS = old_config
        return old_energy
    return new_energy

# Plot the system as a polar plot, which shows the particles located in the circle
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
    return figure


class EnergyBall:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.forceV_x = 0
        self.forceV_y = 0

    # Move over a given x and y distance
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

# Execute the simulated annealing with an inverse logarithmic cooling schedule
def perform_optimalization(nr_points, nr_iterations):
    global TEMPERATURE
    TEMPERATURE = 1
    a = 1
    b = 2
    e = []
    t = []
    make_energy_system(nr_points)
    for j in range(nr_iterations):
        e.append(random_move())
        TEMPERATURE = a / math.log(j + b)
        t.append(TEMPERATURE)
        # print(TEMPERATURE)
    # plt.plot(t)
    # plt.show()
    return e

# Execute the simulated annealing with a linear cooling schedule
def linear_cooling(nr_points, nr_iterations, dt):
    global TEMPERATURE
    TEMPERATURE = 1
    e = []
    t = []
    make_energy_system(nr_points)
    for j in range(nr_iterations):
        e.append(random_move())
        TEMPERATURE = max(TEMPERATURE - dt, 0.0001)
        t.append(TEMPERATURE)
        # print(TEMPERATURE)
    # plt.plot(t)
    # plt.show()
    return e


# Execute the simulated annealing with an exponential cooling schedule
def exponential_cooling(nr_points, nr_iterations, a):
    global TEMPERATURE
    TEMPERATURE = 1
    e = []
    t = []
    make_energy_system(nr_points)
    for j in range(nr_iterations):
        e.append(random_move())
        TEMPERATURE = max(a * TEMPERATURE, 0.0001)
        t.append(TEMPERATURE)
        # print(TEMPERATURE)
    # plt.plot(t)
    # plt.show()
    return e


log = []
lin = []
exp = []
for i in range(50):
    log.append(perform_optimalization(15,1000))
    # plotsystem()

    lin.append(linear_cooling(15,1000, 0.001))
    #plotsystem()
    exp.append(exponential_cooling(15,1000, 0.98))
    #plotsystem()
# plt.boxplot([log, lin, exp])
lin = list(zip(*lin))
log = list(zip(*log))
exp = list(zip(*exp))
lin_mean = [np.mean(i) for i in lin]
exp_mean = [np.mean(i) for i in exp]
log_mean = [np.mean(i) for i in log]

lin_std = [np.sqrt(np.var(i)) for i in lin]
exp_std = [np.sqrt(np.var(i)) for i in exp]
log_std = [np.sqrt(np.var(i)) for i in log]
x = range(1000)
plt.semilogx(x,lin_mean, label="Linear cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(lin_mean, lin_std)], [sum(x) for x in zip(lin_mean,[ -i for i in lin_std])], alpha=0.3)
plt.semilogx(x,exp_mean, label="Exponential cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(exp_mean, exp_std)], [sum(x) for x in zip(exp_mean,[ -i for i in exp_std])], alpha=0.3)
plt.semilogx(x,log_mean, label="Inverse logarithmic cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(log_mean, log_std)], [sum(x) for x in zip(log_mean,[ -i for i in log_std])], alpha=0.3)

plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Energy of configuration")
plt.show()
plt.plot(x,lin_mean, label="Linear cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(lin_mean, lin_std)], [sum(x) for x in zip(lin_mean,[ -i for i in lin_std])], alpha=0.3)
plt.plot(x,exp_mean, label="Exponential cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(exp_mean, exp_std)], [sum(x) for x in zip(exp_mean,[ -i for i in exp_std])], alpha=0.3)
plt.plot(x,log_mean, label="Inverse logarithmic cooling schedule")
plt.fill_between(x, [sum(x) for x in zip(log_mean, log_std)], [sum(x) for x in zip(log_mean,[ -i for i in log_std])], alpha=0.3)

plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Energy of configuration")
plt.show()

#
# for i in range(2,21):
#     perform_optimalization(i,100000)
#     fig = plotsystem()
#     fig.savefig("Configuration_for_"+ str(i)+"_balls.png")
