'''Squares module
'''
import numpy as np
from matplotlib.patches import Rectangle
from thermosnooker.balls import Ball

class Box:
    '''Rectangle class
    '''
    def __init__(self, init_anchor = [0.0, 0.0], vel = [0.0,0.0], acc = [0.0, 0.0], init_width = 1.0, init_height = 1.0, mass = 1.0):
        self.__anchor = np.array(init_anchor, dtype = float)  #bottom left corner of box
        self.__vel = np.array(vel, dtype = float)
        self.__acc = np.array(acc, dtype = float)
        self.__width = float(init_width)
        self.__mass = mass
        self.__height = float(init_height)
        self.__box_patch = Rectangle(init_anchor, init_width, init_height, ec = 'Black', fc = 'None')
        self._tot_p = 0.0  

    def width(self):
        """To call width of box

        Returns:
            float: width
        """
        return self.__width

    def height(self):
        """To call height of box

        Returns:
            float: height
        """
        return self.__height

    def vel(self):
        """To call velocity of box

        Returns:
            list: velocity of box
        """
        return self.__vel
    
    def acc(self):
        """To call acceleration of box

        Returns:
            list: acceleration of box
        """
        return self.__acc
    
    def anchor(self):
        """To call anchor of box

        Returns:
            list: coordinates of anchor
        """
        return self.__anchor

    def mass(self):
        """To call mass of box

        Returns:
            float: mass
        """
        return self.__mass
    
    def dp_tot(self):
        """To call total momentum of box

        Returns:
            float: momentum
        """
        return self._tot_p

    def patch(self):
        """To call patch of box

        Returns:
            Rectangle: patch of box
        """
        return self.__box_patch

    def set_vel(self, new_vel = [0.0, 0.0]):
        """Set new volcity of box

        Args:
            new_vel (list, optional): new velocity. Defaults to [0.0, 0.0].

        Raises:
            ValueError: If velocity is too long or short

        Returns:
            Box: self
        """
        if len(new_vel) != 2:
            raise ValueError('New velocity argument is invalid')
        self.__vel = np.array(new_vel, dtype = float)
        return self
    
    def set_acc(self, new_acc = [0.0, 0.0]):
        """Set new acceleration of box

        Args:
            new_acc (list, optional): new acceleration. Defaults to [0.0, 0.0].

        Raises:
            ValueError: If acceleration is too long or short

        Returns:
            Box: self
        """
        if len(new_acc) != 2:
            raise ValueError('New acceleration argument is invalid')
        self.__acc = np.array(new_acc, dtype = float)
        return self

    def move(self, dt):
        """calculates new velocity, and acceleration

        Args:
            dt (float): time to move

        Returns:
            Box: self
        """
        self.__anchor += (self.vel() * dt + 0.5 * self.acc() * dt * dt)
        self.__box_patch.set_x(self.__anchor[0])
        self.__box_patch.set_y(self.__anchor[1])
        return self

class Capsule(Box):
    """The outer capsule
    """
    def __init__(self, init_anchor=[-22.5, -10], vel=[0.0, 0.0], acc = [0.0, 0.0], init_width=67.5, init_height=20, mass=1000000000):
        super().__init__(init_anchor, vel, acc, init_width, init_height, mass)

    def collide(self, ball = Ball()):
        """Collision between box and ball

        Args:
            ball (Ball, opttional): ball that is in collision. Defaults to Ball().
        """
        vel_ball = ball.vel()
        if isinstance(ball, Ball):
            ball.set_vel([vel_ball[0], - vel_ball[1]]) 

    def time_to_collision(self, ball = Ball()):
        """Time until ball collides next with Capsule

        Args:
            ball (Ball, optional): ball that is in collision. Defaults to Ball().

        Returns:
            float: time to collision
        """
        yvel_ball = ball.vel()[1]
        ypos = ball.pos()[1]
        radius = ball.radius()
        container_top = self.anchor()[1] + self.height()
        container_bottom = self.anchor()[1] 
        if yvel_ball == 0:
            return None
        elif yvel_ball > 0:
            time_collide = (container_top - ypos - radius) / yvel_ball
        else:
            time_collide = (container_bottom - ypos + radius) / yvel_ball
        return time_collide

class Valve(Box):
    def __init__(self, init_anchor=[0.0, 0.0], vel=[0.0, 0.0], acc = [0.0, 0.0], init_width=1.0, init_height=0.0, mass=1000000000):
        super().__init__(init_anchor, vel, acc, init_width, init_height, mass)

    def collide(self, ball = Ball()):
        """Collision between box and ball

        Args:
            ball (Ball, opttional): ball that is in collision. Defaults to Ball().
        """
        pos = ball.pos()
        vel = ball.vel()
        left_side = self.anchor()[0]
        right_side = self.anchor()[0] + self.width()
        if (pos[0] < left_side or pos[0] > right_side): #collision with sides
            ball.set_vel([- vel[0], vel[1]])

    def time_to_collision(self, ball = Ball()):
        """Time until ball collides next with Valve

        Args:
            ball (Ball, optional): ball that is in collision. Defaults to Ball().

        Returns:
            float: time to collision
        """
        pos = ball.pos()
        vel = ball.vel()
        right_side = self.anchor()[0] + self.width()
        radius = ball.radius()

        if pos[0] > right_side and vel[0] <0:
            return (right_side + radius - pos[0]) / vel[0]
        else:
            return None

class Piston(Box):

    def __init__(self, init_anchor=[0.0, 0.0], vel=[0.0, 0.0], acc = [0.0, 0.0], init_width=1.0, init_height=20.0, mass = 10.0):
        super().__init__(init_anchor, vel, acc, init_width, init_height, mass)

    def set_dp(self, dp):
        """Sets new momentum for piston

        Args:
            dp (float): new momentum

        Returns:
            Box: self
        """
        self._tot_p = dp
        return self 

    def collide(self, ball = Ball()):
        """Collision between box and ball

        Args:
            ball (Ball, opttional): ball that is in collision. Defaults to Ball().
        """
        vel_ball_init = ball.vel()[0]
        vel_pist_init = self.vel()[0]
        mass_ball = ball.mass()
        mass_pist = self.mass()
        vel_ball_fin = ((mass_ball - mass_pist) * vel_ball_init  + 2 * mass_pist * vel_pist_init) / (mass_pist + mass_ball)
        ball.set_vel([vel_ball_fin, ball.vel()[1]])
        ball_dv = (ball.vel() - vel_ball_init)
        dp = ball.mass() * np.sqrt(ball_dv.dot(ball_dv))
        self._tot_p += dp

    def time_to_collision(self, ball = Ball()):
        """Time until ball collides next with piston

        Args:
            ball (Ball, optional): ball that is in collision. Defaults to Ball().

        Returns:
            float: time to collision
        """
        right_side = self.anchor()[0] + self.width() + ball.radius()
        left_side = self.anchor()[0] - ball.radius()
        if ball.vel()[0] < self.vel()[0] and ball.pos()[0] > right_side: #ball moving to left
            a = 0.5 * self.acc()[0] 
            if a == 0:              
                    return (ball.pos()[0] - right_side)/(self.vel()[0] - ball.vel()[0])
            b = self.vel()[0] - ball.vel()[0]
            c = right_side - ball.pos()[0]
            dis = b * b - 4 * a * c
            if dis < 0:
                return None
            roots = [(-b+np.sqrt(dis))/(2*a), (-b-np.sqrt(dis))/(2*a)]
            if min(roots) > 0:
                return min(roots)
            elif max(roots) > 0:
                return max(roots)    
        elif ball.vel()[0] > self.vel()[0] and ball.pos()[0] < left_side: #moving to right
            a = 0.5 * self.acc()[0] 
            if a == 0:
                return (ball.pos()[0] - left_side)/(self.vel()[0] - ball.vel()[0])
            b = self.vel()[0] - ball.vel()[0]
            c = left_side - ball.pos()[0]
            dis = b * b - 4 * a * c
            if dis < 0:
                return None
            roots = [(-b+np.sqrt(dis))/(2*a), (-b-np.sqrt(dis))/(2*a)]
            if min(roots) > 0:
                return min(roots)
            elif max(roots) > 0:
                return max(roots) 
        else:
            return None
