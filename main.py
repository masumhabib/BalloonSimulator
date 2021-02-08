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
        ball_radius=0.02,
        k=1,
        x0=0,
        delP=3,
        dt=1.2,
        max_iter=200,
    )

    plt.rcParams.update({'figure.max_open_warning': 0})

    balloon = Balloon(radius=sys.radius, num_balls=sys.num_balls, center=sys.center, time_step=sys.dt, max_iter=sys.max_iter,
                      ball_radius=sys.ball_radius)
    #ax = balloon.draw()
    #plt.show()

    balloon.set_color(balls=range(0, 3), color='red')
    ax = balloon.draw()
    #plt.show()

    balloon.inflate(new_pressure=0.5, verbosity=2)
    #ax = balloon.draw()
    balloon.draw(ax=ax)
    plt.show()
    #balloon.animate()

    #balloon.inflate(new_pressure=1, verbosity=2)
    #balloon.draw(ax=ax)
    #plt.show()
    #balloon.animate()

    #balloon.puncture(verbosity=2)
    #balloon.animate()



if __name__ == '__main__':
    main()

