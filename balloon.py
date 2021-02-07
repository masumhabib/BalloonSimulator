
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
import copy as cp
from typing import Union, TypeVar

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
    'max_iter',  # max number of sol iterations
])

#Point = NewType('Point', np.array)

X = 0
Y = 1


def _vect_norm(v):
    """
    Calculate L2 norm of a vector
    >>> _vect_norm(np.array([3, 4]))
    5.0
    """
    v_mag = np.dot(v, v)
    v_norm = np.sqrt(v_mag)
    return v_norm


def _vect_unit_vector(v):
    norm = _vect_norm(v)
    u = v/norm
    return u

def _get_bbox(pt, bbox=None, big_box_side=6.4E23):
    if bbox is None:
        xmin, xmax = big_box_side, -big_box_side
        ymin, ymax = big_box_side, -big_box_side
        min_pt = np.array([xmin, ymin])
        max_pt = np.array([xmax, ymax])
        bbox =(min_pt, max_pt)

    min_pt, max_pt = bbox
    xmin, ymin = min_pt
    xmax, ymax = max_pt
    x, y = pt

    if x < xmin:
        xmin = x
    if x > xmax:
        xmax = x
    if y < ymin:
        ymin = y
    if y > ymax:
        ymax = y
    min_pt = np.array([xmin, ymin])
    max_pt = np.array([xmax, ymax])

    new_bbox =(min_pt, max_pt)
    return new_bbox


class Ball(object):
    """
    Simple ball object that has mass, position and velocity.
    >>> b = Ball()
    >>> print(b)
    Ball(x=0, y=0, mass=1)
    """

    def __init__(self, x: float = 0, y: float = 0, mass: float = 1, color='g', radius: float = 0.2):
        self._pos = np.array([x, y])
        self._vel = 0.0
        self._mass = mass
        self._color = color
        self._radius = radius

    def __str__(self):
        return 'Ball(x={:.2G}, y={:.2G}, mass={:.2G})'.format(self._pos[X], self._pos[Y], self._mass)

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, new_pos: np.array):
        self._pos = new_pos

    def move(self, vector: np.array):
        self._pos = self._pos + vector

    def get_next_pos(self, force, dt):
        a = force/self._mass
        pos = self._pos + self._vel*dt + 1/2*a*dt*dt
        return pos

    def draw(self, ax):
        circle = plt.Circle((self._pos[X], self._pos[Y]), self._radius, color=self._color)
        ax.add_artist(circle)
        return ax


class Wall(object):
    def __init__(self, pos: np.array):
        self._pos = pos

    def get_pos(self):
        return self._pos


class Spring(object):
    """
    Simple spring object that has models behavior of a spring. Springs two ends should always be connected to
    something: either a Ball or a Wall.
    >>> s = Spring(Ball(0, 0), Ball(0,1))
    >>> print(s)
    Spring(b1=Ball(x=0, y=0, mass=1), b2=Ball(x=0, y=1, mass=1), k=1)
    """

    Boundary = Union[Ball, Wall]

    def __init__(self, bound1: Boundary, bound2: Boundary, spring_const: float = 1.0, relaxed_length: float = 0):
        self._b1 = bound1
        self._b2 = bound2
        self._k = spring_const
        self._l0 = relaxed_length
        self._color = 'k'

    def get_length(self):
        p, q = self._get_end_pos()
        v = p - q
        length = _vect_norm(v)
        dl = length - self._l0
        return dl

    def get_force(self):
        return self._k * self.get_length()

    def draw(self, ax):
        p, q = self._get_end_pos()
        xs = np.array([p[X], q[X]])
        ys = np.array([p[Y], q[Y]])
        ax.plot(xs, ys, '-', color=self._color, linewidth=2)

    def _get_end_pos(self):
        return self._b1.get_pos(), self._b2.get_pos()

    def __str__(self):
        return 'Spring(b1=' + str(self._b1) + ', b2=' + str(self._b2) + ', k={:.2G})'.format(self._k)


class BallsAndSpringsGeometry(object):
    #Neighbor = namedtuple('Neighbor', ['ball', ])

    def __init__(self):
        self._balls = []
        self._springs = []
        self._neighbors = OrderedDict()
        self._connections = OrderedDict()

    #def get_neighbors(self, ball: Ball):
    #    return self._neighbors[ball]

    def get_balls(self):
        return self._balls

    #def get_springs(self):
    #    return self._springs

    def copy(self):
        new = cp.deepcopy(self)
        return new

    def draw(self, ax=None):
        for ball in self._balls:
            ball.draw(ax)
        for spring in self._springs:
            spring.draw(ax)

    def calc_net_force(self, ball: Ball):
        return self._calc_total_spring_force(ball)

    def get_bbox(self):
        bbox = None
        for ball in self._balls:
            bbox = _get_bbox(ball.get_pos(), bbox)
        return bbox

    #@classmethod
    #def _calc_vect_norm_and_dir(cls, v):
    #    x = v[X]
    #    y = v[Y]
    #    r = np.sqrt(x*x + y*y)
    #    u = v/r
    #    return r, u

    @classmethod
    def _get_unit_vector(cls, frm: np.array, to: np.array):
        v = to - frm
        u = _vect_unit_vector(v)
        return u

    def _calc_spring_force_between(self, b1: Spring.Boundary, b2: Spring.Boundary):
        ball_pair = (b1, b2)
        if ball_pair not in self._connections:
            raise RuntimeError('No spring found between (' + str(b1) + ', ' + str(b2) + ')')
        spring = self._connections[ball_pair]
        tension = spring.get_force()
        dir_from_b1_to_b2 = self._get_unit_vector(frm=b1.get_pos(), to=b2.get_pos())
        force = tension*dir_from_b1_to_b2
        return force

    def _calc_total_spring_force(self, ball):
        total_tension = 0
        for neigh in self._neighbors[ball]:
            tension = self._calc_spring_force_between(ball, neigh)
            total_tension += tension
        return total_tension

    def __str__(self):
        out_str = 'SpringsAndBalls(\nballs=[\n'
        for ball in self._balls:
            out_str += '  ' + str(ball) + '\n'
        out_str += ']\nsprings=[\n'
        for spring in self._springs:
            out_str += '  ' + str(spring) + '\n'
        out_str += ']\nneighbors={\n'
        for ball, neighs in self._neighbors.items():
            out_str += '  ' + str(ball) + ' --> ['
            for neigh in neighs:
                out_str += '' + str(neigh) + ','
            out_str += ']\n'
        out_str += '})'
        return out_str


class BalloonGeometry(BallsAndSpringsGeometry):
    """
    Represents the balloon geometry using balls and springs. Knows how to calculate force on each ball.
    Invariants:
    1. _neighbors[ball] always point to all neighbors of ball
    2. _connections[(ball1, ball2)] points to the spring that connects ball1 and ball2

    >>> balloon = BalloonGeometry(num_balls=4)
    >>> print(balloon)
    SpringsAndBalls(
    balls=[
      Ball(x=1, y=0, mass=1)
      Ball(x=6.1E-17, y=1, mass=1)
      Ball(x=-1, y=1.2E-16, mass=1)
      Ball(x=-1.8E-16, y=-1, mass=1)
    ]
    springs=[
      Spring(b1=Ball(x=1, y=0, mass=1), b2=Ball(x=-1.8E-16, y=-1, mass=1), k=1)
      Spring(b1=Ball(x=1, y=0, mass=1), b2=Ball(x=6.1E-17, y=1, mass=1), k=1)
      Spring(b1=Ball(x=6.1E-17, y=1, mass=1), b2=Ball(x=1, y=0, mass=1), k=1)
      Spring(b1=Ball(x=6.1E-17, y=1, mass=1), b2=Ball(x=-1, y=1.2E-16, mass=1), k=1)
      Spring(b1=Ball(x=-1, y=1.2E-16, mass=1), b2=Ball(x=6.1E-17, y=1, mass=1), k=1)
      Spring(b1=Ball(x=-1, y=1.2E-16, mass=1), b2=Ball(x=-1.8E-16, y=-1, mass=1), k=1)
      Spring(b1=Ball(x=-1.8E-16, y=-1, mass=1), b2=Ball(x=-1, y=1.2E-16, mass=1), k=1)
      Spring(b1=Ball(x=-1.8E-16, y=-1, mass=1), b2=Ball(x=1, y=0, mass=1), k=1)
    ]
    neighbors={
      Ball(x=1, y=0, mass=1) --> [Ball(x=-1.8E-16, y=-1, mass=1),Ball(x=6.1E-17, y=1, mass=1),]
      Ball(x=6.1E-17, y=1, mass=1) --> [Ball(x=1, y=0, mass=1),Ball(x=-1, y=1.2E-16, mass=1),]
      Ball(x=-1, y=1.2E-16, mass=1) --> [Ball(x=6.1E-17, y=1, mass=1),Ball(x=-1.8E-16, y=-1, mass=1),]
      Ball(x=-1.8E-16, y=-1, mass=1) --> [Ball(x=-1, y=1.2E-16, mass=1),Ball(x=1, y=0, mass=1),]
    })

    >>> balloon.calc_net_force(balloon._balls[0])
    array([-2.,  0.])

    >>> balloon.set_pressure(1)
    >>> balloon.calc_net_force(balloon._balls[0])
    array([-3.,  0.])
    """
    def __init__(self, radius: float = 1, num_balls: int = 25, center=np.array([0, 0]), spring_const: float = 1,
                 inside_pressure: float = 0,):
        super().__init__()
        self._radius = radius
        self._num_balls = num_balls
        self._center = center
        self._spring_const = spring_const
        self._inside_pressure = inside_pressure
        self._outside_pressure = 0.0
        self._pressure_loss_rate = 0.1
        self._punctured = False
        self._create()

    def set_pressure(self, new_pressure):
        self._inside_pressure = new_pressure

    def puncture(self, at, span):
        self._springs[at]._k = 0
        self._punctured = True

    def _create(self):
        xs, ys = create_polygon(radius=self._radius, N=self._num_balls, x0=self._center[0], y0=self._center[1])
        self._balls = []
        self._springs = []
        for x, y in zip(xs, ys):
            b = Ball(x, y)
            self._balls.append(b)
        self._num_balls = len(self._balls)
        for idx, b in enumerate(self._balls):
            b1 = self._balls[idx - 1]
            b2 = self._balls[(idx + 1) % self._num_balls]
            for neigh in (b1, b2):
                spring = Spring(bound1=b, bound2=neigh, spring_const=self._spring_const)
                self._springs.append(spring)
                self._connections[(b, neigh)] = spring
                if b not in self._neighbors:
                    self._neighbors[b] = []
                self._neighbors[b].append(neigh)

    def calc_net_force(self, ball: Ball, dt: float):
        total_tension = self._calc_total_spring_force(ball)
        pressure_force = self._calc_total_pressure_force(ball, dt)
        total_force = total_tension + pressure_force
        return total_force

    def _calc_total_pressure_force(self, ball, dt, del_area=1.0):
        dir_along_radius = self._get_unit_vector(frm=self._center, to=ball.get_pos())
        if self._punctured:
            self._lose_pressure(dt, del_area)
        force = del_area * (self._inside_pressure - self._outside_pressure) * dir_along_radius
        return force

    def _lose_pressure(self, dt, del_area):
        pressure_loss = dt*del_area*self._pressure_loss_rate
        self._inside_pressure -= pressure_loss
        if self._inside_pressure < self._outside_pressure:
            self._inside_pressure = self._outside_pressure

class BallsAndSpringsSimulator(object):
    def __init__(self, time_step, max_iter, update_pos_tol, residue_force_tol):
        self._dt = time_step
        self._max_iter = max_iter
        self._update_pos_tol = update_pos_tol
        self._residue_force_tol = residue_force_tol
        self._geom_iters = []

    def relax(self, geom: BallsAndSpringsGeometry, verbosity=1):
        prev_geom = geom.copy()
        iter_results = []
        for it in range(self._max_iter):
            balls = prev_geom.get_balls()
            new_geom = prev_geom.copy()
            new_balls = new_geom.get_balls()
            residue_force = 0
            for idx, ball in enumerate(balls):
                force = prev_geom.calc_net_force(ball, self._dt)
                new_pos = ball.get_next_pos(force, self._dt)
                if verbosity > 5:
                    print("D>> New pos: " + str(new_pos) + " force: " + str(force))
                new_balls[idx].set_pos(new_pos)
                residue_force += _vect_norm(force) / np.sqrt(len(balls))

            del_posns = 0
            for idx, ball in enumerate(balls):
                # print("OLD: " + str(ball))
                del_xy = ball.get_pos() - new_balls[idx].get_pos()
                del_xy = _vect_norm(del_xy)
                del_posns += del_xy / np.sqrt(len(balls))

                # print("NEW: " + str(ball))
            if verbosity > 1:
                print('I>> Itr # {:.0f} dx = {:.4G}, force = {:.4G}'.format(it, del_posns, residue_force))
            if del_posns < self._update_pos_tol and residue_force < self._residue_force_tol:
                if verbosity > 0:
                    print('I>> Itr # {:.0f} dx = {:.4G}, force = {:.4G}'.format(it, del_posns, residue_force))
                    print('I>> Converged')
                break
            iter_results.append(new_geom)
            prev_geom = new_geom

        return iter_results
    #def relax(self):
    #    pass

    def get_bbox_for_all_iters(self):
        bbox = None
        for geom in self._geom_iters:
            box = geom.get_bbox()
            bbox = _get_bbox(box[0], bbox)
            bbox = _get_bbox(box[1], bbox)
        return bbox


class Balloon(object):
    def __init__(self, radius=1, num_balls=25, center=np.array([0, 0]), time_step=0.5, max_iter=4000,
                 update_pos_tol=2E-3, residue_force_tol=1E-3):
        #self._balls = []
        #self._sticks = []
        #self._radius = radius
        #self._num_balls = num_balls
        #self._center = center
        #self._create()
        self._geom = BalloonGeometry(radius=radius, num_balls=num_balls, center=center)
        self._sim = BallsAndSpringsSimulator(time_step=time_step, max_iter=max_iter, update_pos_tol=update_pos_tol,
                                             residue_force_tol=residue_force_tol)

    #def _create(self):
    #    xs, ys = create_polygon(radius=self._radius, N=self._num_balls, x0=self._center[0], y0=self._center[1])
    #    self._balls = []
    #    self._sticks = []
    #    for x, y in zip(xs, ys):
    #        b = Ball2(x, y)
    #        self._balls.append(b)
    #        self._num_balls = len(self._balls)
    #    for idx, b in enumerate(self._balls):
    #        b.set_neigh1(self._balls[idx - 1])
    #        b.set_neigh2(self._balls[(idx + 1) % self._num_balls])

    #def relax(self, sys, verbosity=1):
    #    iter_results = []
    #    for it in range(sys.max_iters):
    #        new_xys = []
    #        for idx, b in enumerate(self._balls):
    #            new_xy = b.calc_next_position(sys)
    #            # print("new XY: " + str(new_xy))
    #            new_xys.append(new_xy)

    #        del_xys = 0
    #        for idx, ball in enumerate(self._balls):
    #            # print("OLD: " + str(ball))
    #            del_xy = ball.get_xy() - new_xys[idx]
    #            del_xy = np.sqrt(del_xy[0] * del_xy[0] + del_xy[1] * del_xy[1])
    #            del_xys += del_xy / np.sqrt(sys.num_balls)

    #            ball.set_xy(new_xys[idx])
    #            # print("NEW: " + str(ball))
    #            if verbosity > 1:
    #                print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
    #        new_balloon = cp.deepcopy(self)
    #        iter_results.append(new_balloon)
    #        if del_xys < 0.1:
    #            if verbosity > 0:
    #                print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
    #                print('Converged')
    #            break

    #    return iter_results

    def inflate(self, new_pressure: float, verbosity=1):
        self._geom.set_pressure(new_pressure)
        geom_iters = self._sim.relax(self._geom, verbosity)
        self._geom = geom_iters[len(geom_iters)-1]
        self._geom_iters = geom_iters

    def puncture(self, at=0, span=1, verbosity=1):
        self._geom.puncture(at, span)
        geom_iters = self._sim.relax(self._geom, verbosity)
        self._geom = geom_iters[len(geom_iters)-1]
        self._geom_iters = geom_iters

    def draw(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        bbox = self._geom.get_bbox()
        ax.set_aspect('equal')
        ax.set_xlim([bbox[0][X]*1.1, bbox[1][X]*1.1])
        ax.set_ylim([bbox[0][Y]*1.1, bbox[1][Y]*1.1])
        self._geom.draw(ax)
        return ax

    def animate(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.grid()
        plt.ion()
        lim = 2
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        #plt.ion()
        #plt.show()
        for idx, geom in enumerate(self._geom_iters):
            plt.cla()
            plt.xlim([-lim, lim])
            plt.ylim([-lim, lim])
            geom.draw(ax)
            plt.pause(1)
        #plt.ion()
        plt.show()











#class Ball2(object):
#    X = 0
#    Y = 1
#
#    def __init__(self, x=0, y=0, neigh1=None, neigh2=None):
#        self.xy = np.array([x,y])
#        self.neigh1 = neigh1
#        self.neigh2 = neigh2
#        self.color = None
#        self.mass = 1
#
#    def __str__(self):
#        return '(' + str(self.xy[self.X]) + ', ' + str(self.xy[self.Y]) + ', ' + \
#               str(type(self.neigh1)) + ', ' + str(type(self.neigh2)) + ')'
#
#    def set_xy(self, xy):
#        self.xy = xy
#
#    def get_xy(self):
#        return self.xy
#
#    def set_neigh1(self, b1):
#        self.neigh1 = b1
#
#    def set_neigh2(self, b2):
#        self.neigh2 = b2
#
#    def calc_norm_and_dir(self, v):
#        #print("DBG: v = " + str(v))
#        x = v[self.X]
#        y = v[self.Y]
#        r = np.sqrt(x*x + y*y)
#        #print("DBG: r = " + str(r))
#        u = v/r
#        #print("DBG: u = " + str(u))
#        return r, u
#
#    def get_neigh_vector(self, neigh):
#        cv = self.xy - neigh.xy
#        #print("DBG: cv = " + str(cv))
#        c, cu = self.calc_norm_and_dir(cv)
#        return c, cu
#
#    def calc_del_area(self, c1, c2):
#        return (c1+c2)/2
#
#    def calc_tension(self, c, cu, k, c0):
#        del_c = c - c0
#        del_cv = del_c * cu
#        #spring force
#        F = -del_cv*k
#        return F
#
#    def calc_pressure_force(self, del_area, pressure, center):
#        rv = self.xy - center
#        r, ru = self.calc_norm_and_dir(rv)
#        #del_area = 1
#        F = del_area * pressure * ru
#        return F
#
#    def calc_next_position(self, sys):
#        if self.neigh1 is not None:
#            c1, cu1 = self.get_neigh_vector(self.neigh1)
#            #print("DBG: cu1 = " + str(cu1))
#            T1 = self.calc_tension(c1, cu1, sys.k, sys.x0)
#        else:
#            T1 = np.array([0, 0])
#        if self.neigh2 is not None:
#            c2, cu2 = self.get_neigh_vector(self.neigh2)
#            T2 = self.calc_tension(c2, cu2, sys.k, sys.x0)
#        else:
#            T2 = np.array([0, 0])
#
#
#        #del_area = self.calc_del_area(c1, c2)
#        del_area = 1
#        Fp = self.calc_pressure_force(del_area, sys.delP, sys.center)
#
#        #print("DBG T1 = " + str(T1) + " T2 = " + str(T2) + " FP = " + str(Fp))
#
#        Fnet = T1 + T2 + Fp
#        a = Fnet/sys.m
#        #print(a)
#        delxy = 1/2 * a * sys.dt * sys.dt
#        xy = self.xy + delxy
#
#        return xy



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
        xs.append(ball.xy[ball.X])
        ys.append(ball.xy[ball.Y])
    #xs.append(balls[0].xy[ball.X])
    #ys.append(balls[0].xy[ball.Y])
    #print(xs)
    #print(ys)
    return plot_balls_and_sticks(np.array(xs), np.array(ys), radius, stick_color, ball_color, ball_alpha, ax=ax, draw_axes=draw_axes)


def create_polygon(radius, N=25, x0=0, y0=0):
    two_pi = 2*np.pi
    th = np.linspace(0, two_pi*(N-1)/N, N)
    x = x0 + radius*np.cos(th)
    y = y0 + radius*np.sin(th)
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

#class StraightChain(BallsAndSpringsGeometry):
#    BOUNDARY_RIGID = 0
#    BOUNDARY_OPEN = 1
#
#    def __init__(self, num_balls: int, ball_mass: float = 1.0, spring_const: float = 1.0, spring_length: float = 1.0,
#                 left_boundary: int = BOUNDARY_OPEN, right_boundary: int = BOUNDARY_RIGID):
#        super().__init__()
#        self._num_balls = num_balls
#        self._ball_mass = ball_mass
#        self._spring_const = spring_const
#        self._spring_length = spring_length
#        self._left_boundary = left_boundary
#        self._right_boundary = right_boundary
#        self._create()
#
#    def draw(self, ax=None):
#        super().draw(ax)
#
#    def _create(self):
#        x, y = 0, 0
#        for idx in range(self._num_balls):
#            new_x = x + idx*self._spring_length
#            ball = Ball(x=new_x, y=y, mass=self._ball_mass)
#            self._balls.append(ball)
#
#        balls = self._balls.copy()
#        if self._left_boundary == self.BOUNDARY_RIGID:
#            balls = [None] + balls
#        if self._right_boundary == self.BOUNDARY_RIGID:
#            balls = balls + [None]
#
#        for ball in self._balls:
#            print("D>> " + str(ball))
#
#        for idx in range(len(balls) - 1):
#            b1 = balls[idx]
#            b2 = balls[idx+1]
#            if b1 is not None:
#                p = b1.get_pos()
#            else:
#                p = b2.get_pos()
#                p[X] = p[X] - self._spring_length
#            if b2 is not None:
#                q = b2.get_pos()
#            else:
#                q = b1.get_pos()
#                q[X] = q[X] + self._spring_length
#            spring = Spring(p, q, k=self._spring_const)
#            self._springs.append(spring)
#
#            if b1 is not None:
#                neigh = self.Neighbor(ball=b2, spring=spring)
#                if b1 not in self._neighbors:
#                    self._neighbors[b1] = []
#                self._neighbors[b1].append(neigh)
#
#
#        #prev_ball = None
#        #prev_spring = None
#        #if self._left_boundary == self.BOUNDARY_RIGID:
#        #    spring = Spring(np.array([x - self._spring_length, y]), np.array(x, y), k=self._spring_const)
#        #    self._springs.append(spring)
#        #    prev_spring = spring
#
#        #for idx in range(self._num_balls):
#        #    new_x = x + idx*self._spring_length
#        #    ball = Ball(x=new_x, y=y, mass=self._ball_mass)
#        #    self._balls.append(ball)
#
#        #    last_element = (idx < self._num_balls)
#        #    spring = None
#        #    if not last_element or self._right_boundary == self.BOUNDARY_RIGID:
#        #        spring = Spring(np.array([new_x, y]), np.array([new_x + self._spring_length, y]), k=self._spring_const)
#        #        self._springs.append(spring)
#
#        #    valid_neigh = not (prev_spring is None and prev_ball is None)
#        #    if valid_neigh:
#        #        neigh = self.Neighbor(ball=prev_ball, spring=prev_spring)
#        #        self._neighbors[ball] = neigh
#        #        neigh = self.Neighbor(ball=prev_ball, spring=prev_spring)
#        #        self._neighbors[ball] = neigh
#        #    prev_ball, prev_spring = ball, spring
#
#        #if self._right_boundary == self.BOUNDARY_RIGID:
#        #    neigh = self.Neighbor(ball=None, spring=prev_spring)
#        #    self._neighbors[ball] = neigh
#

#class BallsAndSpringsSystem(object):
#
#    def __init__(self, time_step: float, max_iters: int):
#        super().__init__()
#        self._dt = time_step
#        self._max_iters = max_iters
#        self._geom = None
#
#    def relax(self, sys: BallsAndSpringsGeometry, verbosity=1):
#        iter_results = []
#        for it in range(self._max_iters):
#            new_xys = OrderedDict()
#            for idx, ball in enumerate(sys.get_balls()):
#                #print("DBG: " + str(ball))
#                neighs = sys.get_neighbors(ball)
#                #print(ball)
#                #print(neighs)
#                new_xy = self.calc_next_position(ball, neighs)
#                # print("new XY: " + str(new_xy))
#                new_xys[ball] = new_xy
#
#        #    del_xys = 0
#        #    for idx, ball in enumerate(self._balls):
#        #        # print("OLD: " + str(ball))
#        #        del_xy = ball.get_xy() - new_xys[idx]
#        #        del_xy = np.sqrt(del_xy[0] * del_xy[0] + del_xy[1] * del_xy[1])
#        #        del_xys += del_xy / np.sqrt(sys.num_balls)
#
#        #        ball.set_xy(new_xys[idx])
#        #        # print("NEW: " + str(ball))
#        #        if verbosity > 1:
#        #            print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
#        #    new_balloon = cp.deepcopy(self)
#        #    iter_results.append(new_balloon)
#        #    if del_xys < 0.1:
#        #        if verbosity > 0:
#        #            print('Itr # {:.0f} dx = {:.4G}'.format(it, del_xys))
#        #            print('Converged')
#        #        break
#
#        return iter_results
#
#    @classmethod
#    def calc_norm_and_dir(cls, v):
#        # print("DBG: v = " + str(v))
#        x = v[X]
#        y = v[Y]
#        r = np.sqrt(x * x + y * y)
#        # print("DBG: r = " + str(r))
#        u = v / r
#        # print("DBG: u = " + str(u))
#        return r, u
#
#    def get_neigh_vector(self, neigh):
#        cv = self.xy - neigh._pos
#        # print("DBG: cv = " + str(cv))
#        c, cu = self.calc_norm_and_dir(cv)
#        return c, cu
#
#    def calc_del_area(self, c1, c2):
#        return (c1 + c2) / 2
#
#    def calc_tension(self, c, cu, k, c0):
#        del_c = c - c0
#        del_cv = del_c * cu
#        # spring force
#        F = -del_cv * k
#        return F
#
#    def calc_pressure_force(self, del_area, pressure, center):
#        rv = self.xy - center
#        r, ru = self.calc_norm_and_dir(rv)
#        # del_area = 1
#        F = del_area * pressure * ru
#        return F
#
#    @classmethod
#    def calc_net_force(cls, ball, neighs):
#        Fnet = 0
#        for neigh in neighs:
#            if neigh.spring is None:
#                continue
#            T = neigh.spring.calc_tension()
#            print(neigh)
#
#    @classmethod
#    def calc_next_position(cls, ball, neighs):
#        Fnet = cls.calc_net_force(ball, neighs)
#
#        #if self.neigh1 is not None:
#        #    c1, cu1 = self.get_neigh_vector(self.neigh1)
#        #    # print("DBG: cu1 = " + str(cu1))
#        #    T1 = self.calc_tension(c1, cu1, sys.k, sys.x0)
#        #else:
#        #    T1 = np.array([0, 0])
#        #if self.neigh2 is not None:
#        #    c2, cu2 = self.get_neigh_vector(self.neigh2)
#        #    T2 = self.calc_tension(c2, cu2, sys.k, sys.x0)
#        #else:
#        #    T2 = np.array([0, 0])
#
#        ## del_area = self.calc_del_area(c1, c2)
#        #del_area = 1
#        #Fp = self.calc_pressure_force(del_area, sys.delP, sys.center)
#
#        ## print("DBG T1 = " + str(T1) + " T2 = " + str(T2) + " FP = " + str(Fp))
#
#        #Fnet = T1 + T2 + Fp
#        #a = Fnet / sys.m
#        ## print(a)
#        #delxy = 1 / 2 * a * sys.dt * sys.dt
#        #xy = self.xy + delxy
#
#        return ball.get_pos()

if __name__ == "__main__":
    import doctest
    doctest.testmod()


