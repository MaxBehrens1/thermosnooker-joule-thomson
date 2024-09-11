'''Simulations script
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from thermosnooker.sim import Simulation
from thermosnooker.balls import Container, Ball
from thermosnooker.squares import Capsule, Valve, Piston
from thermosnooker.ball_generation import rtrings

def generate(height, width, multi):
    """Generated balls in a square shape

    Args:
        height (float): height of square of balls
        width (float): width of square of balls
        multi (int): how many balls per row

    Yields:
        float, float: coordinates of balls
    """
    for i in range(multi+1):
        y = i * height / multi 
        for j in range(multi+1):
            x = j * width / multi
            yield x,y

class SingleJouleThompson(Simulation):
    """Single ball simulation of Joule Thomspon system
    """
    def __init__(self, capsule = Capsule(), top_valve = Valve(init_anchor=[-0.5, 0.5]),
                 bottom_valve = Valve(init_anchor=[-0.5, -5]),
                 left_piston = Piston(init_anchor=[-10.0,-5.0]),
                 right_piston = Piston(init_anchor=[9.0,-5.0]),
                 ball = Ball(pos = [-5.0, -0.2], vel = [1.0, 1.0], radius= 0.25)):
        self.__capsule = capsule
        self.__ball = ball
        self.__right_piston = right_piston
        self.__left_piston = left_piston
        self.__bottom_valve = bottom_valve
        self.__top_valve = top_valve

    def capsule(self):
        """To call capsule

        Returns:
            Box: all information of capsule
        """
        return self.__capsule

    def top_valve(self):
        """To call top vakve

        Returns:
            Box: all information of top valve
        """
        return self.__top_valve

    def bottom_valve(self):
        """To call bottom valve

        Returns:
            Box: all information of bottom valve
        """
        return self.__bottom_valve

    def right_piston(self):
        """To call right piston

        Returns:
            Box: all information of right piston
        """
        return self.__right_piston

    def left_piston(self):
        """To call left piston

        Returns:
            Box: all information of left piston
        """
        return self.__left_piston

    def ball(self):
        """To call ball

        Returns:
            Ball: all information of ball
        """
        return self.__ball

    def setup_figure(self):
        """To setup visual figure
        """
        x_min = self.capsule().anchor()[0]
        x_max = self.capsule().anchor()[0] + self.capsule().width()
        y_min = self.capsule().anchor()[1] 
        y_max = self.capsule().anchor()[1] + self.capsule().height()
        plt.figure()
        ax = plt.axes(xlim=(x_min - 1, x_max + 1), ylim=(y_min - 1, y_max + 1))
        ax.set_aspect('equal')
        ax.add_artist(self.capsule().patch())
        ax.add_artist(self.top_valve().patch())
        ax.add_artist(self.bottom_valve().patch())
        ax.add_artist(self.right_piston().patch())
        ax.add_artist(self.left_piston().patch())
        ax.add_patch(self.ball().patch())

    def next_collision(self):
        """Finds next collision and moves all objects to that point
        """
        collision_times = {
            "capsule": self.capsule().time_to_collision(self.ball()),
            "left_piston": self.left_piston().time_to_collision(self.ball()),
            "right_piston": self.right_piston().time_to_collision(self.ball()),
            "top_valve": self.top_valve().time_to_collision(self.ball()),
            "bottom_valve": self.bottom_valve().time_to_collision(self.ball())
        }

        filtered_collision_times = {k: v for k, v in collision_times.items() if v is not None}
        print(filtered_collision_times)
        key_of_next_collision = str(min(filtered_collision_times,
                                        key=filtered_collision_times.get)) #type: ignore
        dt = float(min(filtered_collision_times.values()))
        self.ball().move(dt)

        if key_of_next_collision == "capsule":
            self.capsule().collide(self.ball())
            print('collision with capsule')
        elif key_of_next_collision == "left_piston":
            self.left_piston().collide(self.ball())
            print('collision with left piston')
        elif key_of_next_collision == "right_piston":
            self.right_piston().collide(self.ball())
            print('collision with right piston')
        elif key_of_next_collision == "top_valve":
            self.top_valve().collide(self.ball())
            print('collision with top valve')
        else:
            self.bottom_valve().collide(self.ball())
            print('collision with bottom valve')

        print('pos:', self.ball().pos())
        print('vel:', self.ball().vel())

class MultiJouleThomsponValve(Simulation):
    """Simulation of Joule Thompson effect with multiple balls after equilibrium si reached
    """
    def __init__(self, capsule = Capsule(), stopper = Valve(init_anchor=[0, -10],
                init_height=20.0, init_width=0.0),left_piston = Piston(init_anchor=[-22.5,-10.0]),
                 right_piston = Piston(init_anchor=[0,-10.0]),
                 balls = [], dp = 0.0, time = 0.0):
        self.__balls = balls
        self.__capsule = capsule
        self.__right_piston = right_piston
        self.__left_piston = left_piston
        self.__stopper = stopper
        self._time_current = time
        self._final_vol = 2 * self.left_vol()
        self.__right_piston.set_dp(dp/2)
        self.__left_piston.set_dp(dp)
        self.con_left_pres = self.left_pressure()
        self._con_right_pres = self.left_pressure() / 2 #Pressure is half of lhs

    def capsule(self):
        """To call capsule

        Returns:
            Box: all information of capsule
        """
        return self.__capsule

    def right_piston(self):
        """To call right piston

        Returns:
            Box: all information of right piston
        """
        return self.__right_piston

    def left_piston(self):
        """To call left piston

        Returns:
            Box: all information of left piston
        """
        return self.__left_piston

    def stopper(self):
        """To call stopper valve

        Returns:
            Box: all information of stopper valve
        """
        return self.__stopper

    def kinetic_energy_left(self):
        """Calculates kinetic energy of left part of system 

        Returns:
            float: the left kinetic energy
        """
        total_ke = (0.5 * self.left_piston().mass() *
                    self.left_piston().vel().dot(self.left_piston().vel()))
        for i in self.__balls:
            if i.pos()[0] < 0:
                total_ke += 0.5 * i.mass() * i.vel().dot(i.vel())
        return total_ke

    def kinetic_energy_right(self):
        """Calculates kinetic energy of right part of system 

        Returns:
            float: the right kinetic energy
        """
        total_ke = (0.5 * self.right_piston().mass() *
                    self.right_piston().vel().dot(self.right_piston().vel()) +
                    0.5 * self.left_piston().mass() *
                    self.left_piston().vel().dot(self.left_piston().vel()))
        for i in self.__balls:
            if i.pos()[0] > 0:
                total_ke += 0.5 * i.mass() * i.vel().dot(i.vel())
        return total_ke

    def left_temp(self):
        """Calculates temperature of left part of system 

        Returns:
            float: the left temperature
        """
        temp = self.kinetic_energy_left() / (len(self.balls()) * Boltzmann)
        return temp

    def right_temp(self):
        """Calculates temperature of right part of system 

        Returns:
            float: the right temperature
        """
        temp = self.kinetic_energy_right() / (len(self.balls()) * Boltzmann)
        return temp

    def left_vol(self):
        """Calculates volume of left part of system 

        Returns:
            float: the left volume
        """
        width = self.stopper().anchor()[0] - (self.left_piston().anchor()[0] + self.left_piston().width())
        return width * self.capsule().height()

    def right_vol(self):
        """Calculates volume of right part of system 

        Returns:
            float: the right volume
        """
        width = self.right_piston().anchor()[0] - self.stopper().anchor()[0]
        return width * self.capsule().height()

    def balls(self):
        """To call list of balls

        Returns:
            list: lsit of all balls 
        """
        return self.__balls

    def time(self):
        """To call current time

        Returns:
            float: time the system has been running
        """
        return self._time_current

    def left_pressure(self):
        """Calculates pressure of left part of system 

        Returns:
            float: the left pressure
        """
        pressure = self.left_piston().dp_tot() / (self.time() * self.left_piston().height())
        return pressure

    def right_pressure(self):
        """Calculates pressure of right part of system 

        Returns:
            float: the right pressure
        """
        pressure = self.right_piston().dp_tot() / (self.time() * self.right_piston().height())
        return pressure

    def enth_left(self):
        """Calculates enthalpy of left part of system 

        Returns:
            float: the left enthalpy
        """
        enthalpy = self.kinetic_energy_left() + self.left_pressure() * self.left_vol()
        return enthalpy

    def enth_right(self):
        """Calculates enthalpy of right part of system 

        Returns:
            float: the right enthalpy
        """
        enthalpy = self.kinetic_energy_right() + self.right_pressure() * self.right_vol()
        return enthalpy

    def setup_figure(self):
        """Sets up visual figure for simulation
        """
        x_min = self.capsule().anchor()[0]
        x_max = self.capsule().anchor()[0] + self.capsule().width()
        y_min = self.capsule().anchor()[1] 
        y_max = self.capsule().anchor()[1] + self.capsule().height()
        plt.figure()
        ax = plt.axes(xlim=(x_min - 1, x_max + 1), ylim=(y_min - 1, y_max + 1))
        ax.set_aspect('equal')
        ax.add_artist(self.capsule().patch())
        ax.add_artist(self.stopper().patch())
        ax.add_artist(self.right_piston().patch())
        ax.add_artist(self.left_piston().patch())
        for i in self.balls():
            ax.add_patch(i.patch())   

    def next_collision(self):
        """Calculates the next collision, and movex objects to that point
        """
        len_balls = len(self.balls())
        time_ball =  np.array([[100.0 for col in range(len_balls)] for row in range(len_balls)])
        time_container = {}

        for i in range(len_balls):
            ball_i = self.balls()[i]
            collision_times = {
            "capsule": self.capsule().time_to_collision(self.balls()[i]),
            "leftpiston": self.left_piston().time_to_collision(self.balls()[i]),
            "rightpiston": self.right_piston().time_to_collision(self.balls()[i]),
            "stopper": self.stopper().time_to_collision(self.balls()[i])
            }
            filtered_collision_times={k:v for k, v in collision_times.items() if v is not None}
            key_of_smallest_time = str(min(filtered_collision_times,
                                            key=filtered_collision_times.get)) #type: ignore
            smallest_time = float(min(filtered_collision_times.values())) 
            time_container[f'{i}_{key_of_smallest_time}'] = smallest_time
            for j in range(i+1, len_balls):
                ball_j = self.balls()[j]
                rel_r = ball_j.pos() - ball_i.pos()
                if np.sqrt(rel_r.dot(rel_r)) < 20:  #Change to check all balls 
                    collide_time = ball_i.time_to_collision(ball_j)
                    if collide_time is None:
                        pass
                    else:
                        time_ball[i][j] = collide_time

        ball_min_time = float(time_ball.min())
        container_min_time = float(min(time_container.values()))
        object_min_time = str(min(time_container, key=time_container.get)) #type: ignore

        if container_min_time < ball_min_time and (container_min_time > 0):
            dt = container_min_time
            self._time_current += float(dt)
            for i in self.balls():
                i.move(dt)
            ball_no = int(''.join([char for char in object_min_time if char.isdigit()]))
            object_type = str(''.join([char for char in object_min_time if char.isalpha()]))
            if object_type == "capsule":
                self.capsule().collide(self.balls()[ball_no])
            elif object_type == "leftpiston":
                self.left_piston().collide(self.balls()[ball_no])
                vel_right = self.right_piston().vel()+self.right_piston().acc()*dt
                if vel_right[0]>0:
                    self.right_piston().set_vel(vel_right)
                acc_left = (self.capsule().height() *
                            (self.con_left_pres -self.left_pressure()) / self.left_piston().mass())
                acc_right = (self.capsule().height() *  
                    (self.right_pressure() - self._con_right_pres) / self.right_piston().mass())
                self.left_piston().set_acc([acc_left, 0])
                if acc_right>0:
                    self.right_piston().set_acc([acc_right, 0])
            elif object_type == "rightpiston":
                self.right_piston().collide(self.balls()[ball_no])
                self.left_piston().set_vel(self.left_piston().vel()+self.left_piston().acc()*dt)
                acc_left = (self.capsule().height() *
                            (self.con_left_pres - self.left_pressure()) /self.left_piston().mass())
                acc_right = (self.capsule().height() * (self.right_pressure()
                            - self._con_right_pres) / self.right_piston().mass())
                self.left_piston().set_acc([acc_left, 0])
                if acc_right>0:
                    self.right_piston().set_acc([acc_right, 0])
            else:
                self.stopper().collide(self.balls()[ball_no]) 

            if object_type != "leftpiston" and object_type != "rightpiston":
                self.left_piston().set_vel(self.left_piston().vel() + self.left_piston().acc() * dt)
                vel_right = self.right_piston().vel() + self.right_piston().acc() * dt
                if vel_right[0] >0:
                    self.right_piston().set_vel(self.right_piston().vel() + self.right_piston().acc() * dt)
                acc_left = self.capsule().height() * (self.con_left_pres - self.left_pressure()) / self.left_piston().mass()
                acc_right = self.capsule().height() * (self.right_pressure() - self._con_right_pres) / self.right_piston().mass()
                self.left_piston().set_acc([acc_left, 0])
                if acc_right>0:
                    self.right_piston().set_acc([acc_right, 0])
        else:
            dt = ball_min_time
            self._time_current += float(dt)
            argmin = np.unravel_index(time_ball.argmin(), time_ball.shape)
            ball1 = self.balls()[int(argmin[0])]
            ball2 = self.balls()[int(argmin[1])]
            self.left_piston().set_vel(self.left_piston().vel() + self.left_piston().acc() * dt)
            vel_right = self.right_piston().vel() + self.right_piston().acc() * dt
            if vel_right[0] >0:
                self.right_piston().set_vel(self.right_piston().vel() + self.right_piston().acc() * dt)
            acc_left = self.capsule().height() * (self.con_left_pres - self.left_pressure()) / self.left_piston().mass()
            acc_right = self.capsule().height() * (self.right_pressure() - self._con_right_pres) / self.right_piston().mass()
            self.left_piston().set_acc([acc_left, 0])
            if acc_right>0:
                self.right_piston().set_acc([acc_right, 0])
            for i in self.balls():
                i.move(dt)
            ball1.collide(ball2) 

class MultiJouleThompsonEqui(Simulation):
    """Simulation of Joule Thompson effect with multiple balls until they reach equilibrium
    """
    def __init__(self, b_radius = 0.1, b_speed = 40.0, b_mass = 1.0,
                 height = 17, width = 18, multi = 8, capsule = Capsule(),
                 right_piston = Piston(init_anchor=[0.0, -10.0]),
                 left_piston = Piston(init_anchor=[-22.5,-10.0])):
        self.__balls = []
        self.__capsule = capsule
        self.__left_piston = left_piston
        self.__valve = right_piston
        self._time_current = 0.0

        for i in generate(height, width, multi):
            random_vel = np.array([np.random.uniform(-100,100), np.random.uniform(-100,100)] )
            norm_random_vel = (b_speed * random_vel / (np.sqrt(random_vel.dot(random_vel))))
            ball = Ball([i[0] - 20, i[1] - 8.9], norm_random_vel, b_radius, b_mass) #generates a square for lhs
            self.__balls.append(ball)

    def dp(self):
        """Returns total momentum of pistons

        Returns:
            list: left piston momentum
        """
        return self.left_piston().dp_tot()

    def capsule(self):
        """To call capsule

        Returns:
            Box: all information on capsule
        """
        return self.__capsule

    def valve(self):
        """To call valve

        Returns:
            Box: all information on valve
        """
        return self.__valve

    def left_piston(self):
        """To call left piston

        Returns:
            Box: all information on leftpiston
        """
        return self.__left_piston

    def balls(self):
        """To call list of balls

        Returns:
            list: lsit of all information on balls
        """
        return self.__balls

    def time(self):
        """To call current simulation time

        Returns:
            float: how long simulation is running
        """
        return self._time_current

    def left_pressure(self):
        """To calculate current pressure on left

        Returns:
            float: current left pressure
        """
        pressure = self.left_piston().dp_tot() / (self.time() * self.left_piston().height())
        return pressure

    def setup_figure(self):
        """Sets up figure for visual animation
        """
        x_min = self.capsule().anchor()[0]
        x_max = self.capsule().anchor()[0] + self.capsule().width()
        y_min = self.capsule().anchor()[1] 
        y_max = self.capsule().anchor()[1] + self.capsule().height()
        plt.figure()
        ax = plt.axes(xlim=(x_min - 1, x_max + 1), ylim=(y_min - 1, y_max + 1))
        ax.set_aspect('equal')
        ax.add_artist(self.capsule().patch())
        ax.add_artist(self.valve().patch())
        ax.add_artist(self.left_piston().patch())
        for i in self.balls():
            ax.add_patch(i.patch()) 

    def next_collision(self):
        """Finds the next collision in simulation and moves objects to that position
        """
        len_balls = len(self.balls())
        time_ball =  np.array([[100.0 for col in range(len_balls)] for row in range(len_balls)])
        time_container = {}

        for i in range(len_balls):
            ball_i = self.balls()[i]  
            collision_times = {
                "capsule": self.capsule().time_to_collision(self.balls()[i]),
                "leftpiston": self.left_piston().time_to_collision(self.balls()[i]),
                "valve": self.valve().time_to_collision(self.balls()[i])
            }
            filtered_collision_times = {k: v for k, v in collision_times.items() if v is not None}
            key_of_smallest_time = str(min(filtered_collision_times, key=filtered_collision_times.get)) #type: ignore
            smallest_time = float(min(filtered_collision_times.values())) 
            time_container[f'{i}_{key_of_smallest_time}'] = smallest_time
            for j in range(i+1, len_balls):
                ball_j = self.balls()[j]
                rel_r = ball_j.pos() - ball_i.pos()
                if np.sqrt(rel_r.dot(rel_r)) < 10.0:  #Cheks balls within radius 10
                    collide_time = ball_i.time_to_collision(ball_j)
                    if collide_time is None:
                        pass
                    else:
                        time_ball[i][j] = collide_time

        ball_min_time = float(time_ball.min())
        container_min_time = float(min(time_container.values()))
        object_min_time = str(min(time_container, key=time_container.get)) #type: ignore

        if container_min_time < ball_min_time and (container_min_time > 0):
            dt = container_min_time
            self._time_current += float(dt)
            for i in self.balls():
                i.move(dt)
            ball_no = int(''.join([char for char in object_min_time if char.isdigit()]))
            object_type = str(''.join([char for char in object_min_time if char.isalpha()]))
            if object_type == "capsule":
                self.capsule().collide(self.balls()[ball_no])
            elif object_type == "leftpiston":
                self.left_piston().collide(self.balls()[ball_no])
            else:
                self.valve().collide(self.balls()[ball_no])

        else:
            dt = ball_min_time
            self._time_current += float(dt)
            argmin = np.unravel_index(time_ball.argmin(), time_ball.shape)
            ball1 = self.balls()[int(argmin[0])]
            ball2 = self.balls()[int(argmin[1])]
            for i in self.balls():
                i.move(dt)
            ball1.collide(ball2)       
                