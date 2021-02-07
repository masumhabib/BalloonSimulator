from balloon import *

def main():
    #small_radius = 0.5
    #x, y = get_circle_balloon(radius=5)
    #plot_balls_and_sticks(x, y, small_radius)
    #x, y = get_circle_balloon(radius=10)
    #plot_balls_and_sticks(x, y, small_radius)

    sys = System(
        radius=0.1,
        num_balls=8,
        center=np.array([0, 0]),
        m=1,
        ball_radius=0.1,
        k=1,
        x0=0,
        delP=3,
        dt=1.2,
        max_iter=200,
    )

    plt.rcParams.update({'figure.max_open_warning': 0})
    #fig, ax = plt.subplots()
    #ax.set_aspect('equal')
    #lim = 2
    #ax.set_xlim([-lim, lim])
    #ax.set_ylim([-lim, lim])
    #ax.grid()

    #balloon_geom = BalloonGeometry(num_balls=4)
    ##balloon_geom.draw(ax)
    #print(balloon_geom)
    ##plt.show()
    #balloon_geom2 = balloon_geom.copy()
    #balloon_geom2._balls[0]._pos[X] = 10
    #print(balloon_geom2)

    balloon = Balloon(radius=sys.radius, num_balls=sys.num_balls, center=sys.center, time_step=sys.dt, max_iter=sys.max_iter)
    ax = balloon.draw()
    balloon.inflate(new_pressure=0.5, verbosity=2)
    balloon.draw(ax=ax)
    balloon.inflate(new_pressure=1, verbosity=2)
    balloon.draw(ax=ax)
    plt.show()
    #balloon.inflate(new_pressure=0.0, verbosity=2)
    #balloon.draw(ax=ax)
    balloon.animate()
    #plt.show()

    balloon.puncture(verbosity=2)


    balloon.animate()

    #balloon = Balloon(sys.radius, sys.num_balls, sys.center)
    ##plot_balls(balloon._balls, 0.5, draw_axes='on')
    #iter_results = balloon.inflate(sys=sys, verbosity=2)
    ##plot_balls(balloon._balls, 0.5, draw_axes='on')
    ##plt.show()


    #iter_results = balloon.puncture(sys=sys, verbosity=2)
    ## #plot_balls(balloon._balls, 0.5, draw_axes='on')

    ## plt.rcParams.update({'figure.max_open_warning': 0})
    #fig, ax = plt.subplots()
    #balloon.animate()

    #fig, ax2 = plt.subplots()
    #ax2.set_aspect('equal')
    #ax2.grid()
    #plt.ion()
    #lim = 2
    #ax2.set_xlim([-lim, lim])
    #ax2.set_ylim([-lim, lim])
    #plt.ion()
    #plt.show()
    #for idx, geom in enumerate(reversed(balloon._geom_iters)):
    #    plt.cla()
    #    plt.xlim([-lim, lim])
    #    plt.ylim([-lim, lim])
    #    #ax2 = plot_balls(balloon._balls, 0.5, draw_axes='on', ax=ax2)
    #    geom.draw(ax2)
    #    plt.pause(1)
    #    #plt.draw()
    #    #print("DBG: " + str(idx))
    #plt.show()

    #fig, ax = plt.subplots()
    #plt.xlim([-1, 4])
    #plt.ylim([-1, 4])
    #ax.set_aspect('equal')
    #plt.grid()

    #simple_chain = StraightChain(num_balls=4)
    #simple_chain.draw(ax)

    #print(simple_chain)
    #plt.show()

    #sim = BallsAndSpringsSystem(time_step=0.5, max_iters=5)
    #sim.relax(simple_chain)

    #balloon = Balloon(num_balls=4, radius=1, spring_const=1)
    #
    #balloon.inflate(2, animate=True, ax=ax)
    #balloon.draw(ax=ax)
    ##sim.relax(balloon)
    #balloon.puncture(5, animate=True, ax=ax)
    #balloon.draw(ax=ax)
    ##sim.relax(balloon)



if __name__ == '__main__':
    main()

