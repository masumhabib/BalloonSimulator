from balloon import *

def main():
    #small_radius = 0.5
    #x, y = get_circle_balloon(radius=5)
    #plot_balls_and_sticks(x, y, small_radius)
    #x, y = get_circle_balloon(radius=10)
    #plot_balls_and_sticks(x, y, small_radius)

    sys = System(
        radius=2,
        num_balls=25,
        center=np.array([0, 0]),
        m=1,
        ball_radius=0.2,
        k=1,
        x0=0,
        delP=3,
        dt=0.5,
        max_iters=4000,
    )

    # ##balloon = Balloon(sys._radius, sys.num_balls, sys.center)
    # #plot_balls(balloon._balls, 0.5, draw_axes='on')
    # iter_results = balloon.inflate(sys=sys, verbosity=2)
    # #plot_balls(balloon._balls, 0.5, draw_axes='on')


    # iter_results = balloon.puncture(sys=sys, verbosity=2)
    # #plot_balls(balloon._balls, 0.5, draw_axes='on')

    # plt.rcParams.update({'figure.max_open_warning': 0})
    # fig, ax = plt.subplots()

    # plt.ion()
    # plt.show()
    # for idx, balloon in enumerate(iter_results):
    #     plt.cla()
    #     plt.xlim([-90, 90])
    #     plt.ylim([-90, 90])
    #     ax = plot_balls(balloon._balls, 0.5, draw_axes='on', ax=ax)
    #     plt.pause(0.01)
    #     #plt.draw()
    #     print("DBG: " + str(idx))

    single_ball = StraightChain(num_balls=1)
    single_ball.draw()




if __name__ == '__main__':
    main()

