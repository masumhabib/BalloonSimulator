
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
import copy as cp

from typing import NewType

from recordclass import recordclass, RecordClass

#import contextlib
#with contextlib.redirect_stdout(None):
#    import pixiedust

System = namedtuple('System', [
    'radius',   # inital, relaxed radius
    'num_balls',     # number of balls to model the balloon
    'center',   # center of the balloon
    'm',        # mass of the balls
    'ball_radius',  # radius of the balls for visualization
    'k',        # spring constant for each spring
    'x0',       # sprint length at equilibrium
    'delP',     # pressure difference between outside and inside
    'dt',       # time step
    'max_iters',  # max number of sol iterations
])

Point = NewType('Point', np.array)


class Ball(object):
    X = 0
    Y = 1

    def __init__(self, x: float = 0, y: float = 0, mass: float = 1, color=None, radius:float = 1):
        self._pos = np.array([x, y])
        self._color = color
        self._mass = mass
        self._radius = radius

    def __str__(self):
        return 'pos=(' + str(self._pos[self.X]) + ', ' + str(self._pos[self.Y]) + '), mass=' + \
               str(self._mass)

    def set_pos(self, new_pos: np.array):
        self._pos = new_pos

    def get_pos(self):
        return self._pos

    def move(self, vector: np.array):
        self._pos = self._pos + vector

    def draw(self, ax):
        pass


class Spring(object):
    #def __init__(self, point1: np.array, point2: np.array, k: float = 1.0):
    def __init__(self, point1: np.array, point2: np.array, k: float = 1.0):
        self._p = point1
        self._q = point2
        self._k = k
        self._length0 = 0

    def get_force(self, x):
        return -self._k*x


class BallsAndSpringsSystem(object):
    Neighbor = namedtuple('Neighbor', ['ball', 'spring', ])

    def __init__(self, ):
        self._balls = []
        self._springs = []
        self._neighbors = OrderedDict()

    def draw(self, ax=None):
        pass


class StraightChain(BallsAndSpringsSystem):
    BOUNDARY_RIGID = 0
    BOUNDARY_OPEN = 1

    def __init__(self, num_balls: int, ball_mass: float = 1.0, spring_const: float = 1.0, spring_length: float = 1.0,
                 left_boundary: int = BOUNDARY_OPEN, right_boundary: int = BOUNDARY_RIGID):
        super().__init__()
        self._num_balls = num_balls
        self._ball_mass = ball_mass
        self._spring_const = spring_const
        self._spring_length = spring_length
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary
        self._create()

    def _create(self):
        y = 0
        x = 0
        prev_ball = None
        prev_spring = None
        if self._left_boundary == self.BOUNDARY_RIGID:
            spring = Spring(np.array([x - self._spring_length, y]), np.array(x, y), k=self._spring_const)
            self._springs.append(spring)
            prev_spring = spring

        for idx in range(self._num_balls):
            new_x = x + idx*self._spring_length
            ball = Ball(x=new_x, y=y, mass=self._ball_mass)
            self._balls.append(ball)

            last_element = (idx < self._num_balls)
            spring = None
            if not last_element or self._right_boundary == self.BOUNDARY_RIGID:
                spring = Spring(np.array([new_x, y]), np.array([new_x + self._spring_length, y]), k=self._spring_const)
                self._springs.append(spring)

            valid_neigh = not (prev_spring is None and prev_ball is None)
            if valid_neigh:
                neigh = self.Neighbor(ball=prev_ball, spring=prev_spring)
                self._neighbors[ball] = neigh
            prev_ball, prev_spring = ball, spring

        if self._right_boundary == self.BOUNDARY_RIGID:
            neigh = self.Neighbor(ball=None, spring=prev_spring)
            self._neighbors[ball] = neigh


class Simulator(object):
    def __init__(self, system: BallsAndSpringsSystem):
        self._sys = system

    def run(self):
        pass

class Ball2(object):
    X = 0
    Y = 1

    def __init__(self, x=0, y=0, neigh1=None, neigh2=None):
        self.xy = np.array([x,y])
        self.neigh1 = neigh1
        self.neigh2 = neigh2
        self.color = None
        self.mass = 1

    def __str__(self):
        return '(' + str(self.xy[self.X]) + ', ' + str(self.xy[self.Y]) + ', ' + \
               str(type(self.neigh1)) + ', ' + str(type(self.neigh2)) + ')'
        
    def set_xy(self, xy):
        self.xy = xy
    
    def get_xy(self):
        return self.xy
 
    def set_neigh1(self, b1):
        self.neigh1 = b1   
        
    def set_neigh2(self, b2):
        self.neigh2 = b2  
        
    def calc_norm_and_dir(self, v):
        #print("DBG: v = " + str(v))
        x = v[self.X]
        y = v[self.Y]
        r = np.sqrt(x*x + y*y)
        #print("DBG: r = " + str(r))
        u = v/r
        #print("DBG: u = " + str(u))
        return r, u
    
    def get_neigh_vector(self, neigh):
        cv = self.xy - neigh._pos
        #print("DBG: cv = " + str(cv))
        c, cu = self.calc_norm_and_dir(cv)
        return c, cu
 
    def calc_del_area(self, c1, c2):
        return (c1+c2)/2
 
    def calc_tension(self, c, cu, k, c0):
        del_c = c - c0
        del_cv = del_c * cu
        #spring force
        F = -del_cv*k
        return F
    
    def calc_pressure_force(self, del_area, pressure, center):
        rv = self.xy - center
        r, ru = self.calc_norm_and_dir(rv)
        #del_area = 1
        F = del_area * pressure * ru
        return F
       
    def calc_next_position(self, sys):
        if self.neigh1 is not None:
            c1, cu1 = self.get_neigh_vector(self.neigh1)
            #print("DBG: cu1 = " + str(cu1))
            T1 = self.calc_tension(c1, cu1, sys.k, sys.x0)
        else:
            T1 = np.array([0, 0])
        if self.neigh2 is not None:
            c2, cu2 = self.get_neigh_vector(self.neigh2)
            T2 = self.calc_tension(c2, cu2, sys.k, sys.x0)
        else:
            T2 = np.array([0, 0])
            
        
        #del_area = self.calc_del_area(c1, c2)
        del_area = 1
        Fp = self.calc_pressure_force(del_area, sys.delP, sys.center)
       
        #print("DBG T1 = " + str(T1) + " T2 = " + str(T2) + " FP = " + str(Fp))
        
        Fnet = T1 + T2 + Fp
        a = Fnet/sys.m
        #print(a)
        delxy = 1/2 * a * sys.dt * sys.dt
        xy = self.xy + delxy
        
        return xy




class Balloon(object):
    def __init__(self, radius=1, num_balls=25, center=np.array([0, 0])):
        self._balls = []
        self._sticks = []
        self._radius = radius
        self._num_balls = num_balls
        self._center = center
        self._create()

    def _create(self):
        xs, ys = create_polygon(radius=self._radius, N=self._num_balls, x0=self._center[0], y0=self._center[1])
        self._balls = []
        self._sticks = []
        for x, y in zip(xs, ys):
            b = Ball2(x, y)
            self._balls.append(b)
            self._num_balls = len(self._balls)
        for idx, b in enumerate(self._balls):
            b.set_neigh1(self._balls[idx - 1])
            b.set_neigh2(self._balls[(idx + 1) % self._num_balls])

    def relax(self, sys, verbosity=1):
        iter_results = []
        for it in range(sys.max_iters):
            new_xys = []
            for idx, b in enumerate(self._balls):
                new_xy = b.calc_next_position(sys)
                # print("new XY: " + str(new_xy))
                new_xys.append(new_xy)

            del_xys = 0
            for idx, ball in enumerate(self._balls):
                # print("OLD: " + str(ball))
                del_xy = ball.get_xy() - new_xys[idx]
                del_xy = np.sqrt(del_xy[0] * del_xy[0] + del_xy[1] * del_xy[1])
                del_xys += del_xy / np.sqrt(sys.num_balls)

                ball.set_xy(new_xys[idx])
                # print("NEW: " + str(ball))
                if verbosity > 1:
                    print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
            new_balloon = cp.deepcopy(self)
            iter_results.append(new_balloon)
            if del_xys < 0.1:
                if verbosity > 0:
                    print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
                    print('Converged')
                break

        return iter_results

    def inflate(self, sys, verbosity=1):
        return self.relax(sys, verbosity)

    def puncture(self, sys, verbosity=1):
        ### balloon[len(balloon)-1].set_neigh2(None)
        ### balloon[0].set_neigh1(None)
        self._balls[0].set_neigh2(None)
        self._balls[1].set_neigh1(None)
        return self.relax(sys, verbosity)

    def draw(self):
        pass

class BalloonViewer(object):
    def __init__(self, balloon):
        pass













# Todo
# --------
# - Reduce pressure with time
# - Going back to initial shape after busting
# - Volume to pressure conversion
# 
# - Plotting improvements
# - Animation


def plot_balls_and_sticks(xs, ys, radius,
                          stick_color=None, ball_color=None, ball_alpha=0.9,
                          xlim=[-50,50], ylim=[-50,50],
                          draw_axes='off', ax=None):
    #print(xs)
    #print(ys)
    radii = radius*np.ones(xs.size)

    if ax is None:
        fig, ax = plt.subplots()
        #plt.ion()
        #plt.show()

    patches = []
    dr = 0
    for x1, y1, r in zip(xs, ys, radii):
        circle = Circle((x1, y1), r+dr)
        patches.append(circle)
        dr += r/len(xs)


    #colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=ball_alpha)
    #p.set_array(colors)
    ax.plot(xs, ys, '-')
    ax.add_collection(p)
    ax.set_aspect('equal')
    #fig.colorbar(p, ax=ax)
    plt.axis(draw_axes)
    #plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.grid()

    #plt.draw()
    #plt.show()

    return ax


def plot_balls(balls, radius, stick_color=None, ball_color=None, ball_alpha=0.9, ax=None, draw_axes='off'):
    xs = []
    ys = []
    for ball in balls:
        xs.append(ball._pos[ball.X])
        ys.append(ball._pos[ball.Y])
    #xs.append(balls[0].xy[ball.X])
    #ys.append(balls[0].xy[ball.Y])
    #print(xs)
    #print(ys)
    return plot_balls_and_sticks(np.array(xs), np.array(ys), radius, stick_color, ball_color, ball_alpha, ax=ax, draw_axes=draw_axes)


def create_polygon(radius, N=25, x0=0, y0=0):
    two_pi = 2*np.pi
    th = np.linspace(0, two_pi*(N-1)/N, N)
    x = x0 + radius*np.sin(th)
    y = y0 + radius*np.cos(th)
    return x, y




##small_radius=0.5
##x, y = get_circle_balloon(radius=5)
##plot_balls_and_sticks(x, y, small_radius)
##x, y = get_circle_balloon(radius=10)
##plot_balls_and_sticks(x, y, small_radius)
##
##balloon = create_balloon(sys)
##ax = plot_balls(balloon, 0.5)
### plt.show()
### plt.tight_layout()
##
##balloon = relax_balloon(sys, balloon)
##plot_balls(balloon, 0.5, draw_axes='on')
##
### balloon[len(balloon)-1].set_neigh2(None)
### balloon[0].set_neigh1(None)
##balloon[0].set_neigh2(None)
##balloon[1].set_neigh1(None)
##
##balloon = relax_balloon(sys, balloon, plot_iterations=False)
##plot_balls(balloon, 0.5, draw_axes='on')
##
##for idx, b in enumerate(balloon):
##    print("id = " + str(idx) + ": " + str(b))


 



