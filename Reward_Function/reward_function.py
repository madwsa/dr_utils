import math
import numpy as np


class Reward:
    def __init__(self, verbose=False, optimum_line=None):
        self.first_racingpoint_index = 0
        self.verbose = verbose
        self.optimum_line = optimum_line

    ################## HELPER FUNCTIONS ###################

    def angle_mod_360(self, angle):
        """
        Maps an angle to the interval -180, +180.
        Examples:
        angle_mod_360(362) == 2
        angle_mod_360(270) == -90
        :param angle: angle in degree
        :return: angle in degree. Between -180 and +180
        """

        n = math.floor(angle / 360.0)

        angle_between_0_and_360 = angle - n * 360.0

        if angle_between_0_and_360 <= 180.0:
            return angle_between_0_and_360
        else:
            return angle_between_0_and_360 - 360

    # thanks to https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    def polar(self, x, y):
        """
        returns r, theta(degrees)
        """

        r = (x**2 + y**2) ** 0.5
        theta = math.degrees(math.atan2(y, x))
        return r, theta

    def sech(self, x):
        return (2 * math.exp(x)) / (math.exp(2 * x) + 1)

    def opt2wps(self, line):
        wps = [(pt[0], pt[1]) for pt in line]
        return wps

    def dist(self, point1, point2):
        return (
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        ) ** 0.5

    def up_sample(self, waypoints, factor):
        """
        Adds extra waypoints in between provided waypoints
        :param waypoints:
        :param factor: integer. E.g. 3 means that the resulting list has 3 times as many points.
        :return:
        """
        p = waypoints
        n = len(p)

        return [
            [
                i / factor * p[(j + 1) % n][0] + (1 - i / factor) * p[j][0],
                i / factor * p[(j + 1) % n][1] + (1 - i / factor) * p[j][1],
            ]
            for j in range(n)
            for i in range(factor)
        ]

    def get_target_point(self, params):
        waypoints = self.up_sample(self.opt2wps(self.optimum_line), 20)

        car = [params["x"], params["y"]]

        distances = [self.dist(p, car) for p in waypoints]
        min_dist = min(distances)
        i_closest = distances.index(min_dist)

        n = len(waypoints)

        waypoints_starting_with_closest = [
            waypoints[(i + i_closest) % n] for i in range(n)
        ]

        r = params["track_width"] * 0.9

        is_inside = [
            self.dist(p, car) < r for p in waypoints_starting_with_closest
        ]
        i_first_outside = is_inside.index(False)

        if (
            i_first_outside < 0
        ):  # this can only happen if we choose r as big as the entire track
            return waypoints[i_closest]

        return waypoints_starting_with_closest[i_first_outside]

    def get_target_steering_degree(self, params):
        tx, ty = self.get_target_point(params)
        car_x = params["x"]
        car_y = params["y"]
        dx = tx - car_x
        dy = ty - car_y
        heading = params["heading"]

        _, target_angle = self.polar(dx, dy)

        steering_angle = target_angle - heading

        return self.angle_mod_360(steering_angle)

    def score_steer_to_point_ahead(self, params):
        best_stearing_angle = self.get_target_steering_degree(params)
        steering_angle = params["steering_angle"]

        error = (
            abs(steering_angle - best_stearing_angle) / 60.0
        )  # 60 degree is already really bad

        score = 1.0 - (error ** (1.0 / 3.0))

        return max(
            score, 1e-4
        )  # optimizer is rumored to struggle with negative numbers and numbers too close to zero

    def dist_2_points(self, x1, x2, y1, y2):
        return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5

    def closest_2_racing_points_index(self, racing_coords, car_coords):
        # Calculate all distances to racing points
        distances = []
        for i in range(len(racing_coords)):
            distance = self.dist_2_points(
                x1=racing_coords[i][0],
                x2=car_coords[0],
                y1=racing_coords[i][1],
                y2=car_coords[1],
            )
            distances.append(distance)

        # Get index of the closest racing point
        closest_index = distances.index(min(distances))

        # Get index of the second closest racing point
        distances_no_closest = distances.copy()
        distances_no_closest[closest_index] = 999
        second_closest_index = distances_no_closest.index(
            min(distances_no_closest)
        )

        return [closest_index, second_closest_index]

    def get_optimum_V(self, params):
        racing_track = self.optimum_line
        # Get closest indexes for racing line (and distances to all points on racing line)
        (
            closest_index,
            second_closest_index,
        ) = self.closest_2_racing_points_index(
            racing_track, [params["x"], params["y"]]
        )

        # Get optimal speed for closest index
        return racing_track[closest_index][2]

    def reward_function(self, params):
        racing_track = self.optimum_line

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params["all_wheels_on_track"]
        distance_from_center = params["distance_from_center"]
        is_offtrack = params["is_offtrack"]
        progress = params["progress"]
        speed = params["speed"]
        track_width = params["track_width"]
        theta = params["heading"]
        steering_angle = params["steering_angle"]

        # zero out reward if not two wheels on the track
        if distance_from_center > (track_width / 2):
            reward = 1e-4
            return float(reward)

        # point in track direction
        r_point_car = self.score_steer_to_point_ahead(params)

        # match optimum speed
        optimum_V = self.get_optimum_V(params)
        r_V = self.sech(2 * math.e * (speed - optimum_V))

        # make progress, and actually finish
        r_prog = (progress / 100) ** 3

        r_goatifi = 1e-6
        if r_prog > 0.95:
            r_goatifi = 1e4

        # try to do as little steering as possible
        r_input = max(
            float(1e-6),
            float(
                ((-7.1 * (abs(steering_angle) ** (1.0 / 3.0))) + 22.0) / 22.0
            ),
        )

        # actually stay on the track
        r_verstappen = max(
            float(1e-6),
            float(-((distance_from_center / (track_width / 2.0)) ** 32) + 1),
        )

        # add together
        reward = (
            (60 * r_point_car)
            + (80 * r_V)
            + (10 * r_verstappen)
            + (10 * r_input)
            + (10 * r_prog)
            + r_goatifi
        )

        ####################### VERBOSE #######################

        if self.verbose == True:
            print(
                f"r_point_car: {r_point_car} "
                + f"r_V: {r_V} "
                + f"r_verstappen: {r_verstappen} "
                + f"r_input: {r_input} "
                + f"r_prog: {r_prog} "
                + f"r_goatifi: {r_goatifi}"
            )

        #################### RETURN REWARD ####################

        # Always return a float value
        return float(reward)


reward_object = Reward(
    verbose=True,
    optimum_line=[
        [-7.4033, 0.7633, 4.0, 0.06938],
        [-7.18846, 0.58381, 4.0, 0.06999],
        [-6.96566, 0.41042, 4.0, 0.07058],
        [-6.7357, 0.24273, 4.0, 0.07115],
        [-6.49931, 0.08035, 4.0, 0.0717],
        [-6.25719, -0.07714, 4.0, 0.07221],
        [-6.01, -0.23015, 4.0, 0.07268],
        [-5.75835, -0.37908, 4.0, 0.0731],
        [-5.50279, -0.52429, 4.0, 0.07348],
        [-5.24382, -0.66615, 4.0, 0.07382],
        [-4.9818, -0.80486, 4.0, 0.07412],
        [-4.71722, -0.94081, 4.0, 0.07436],
        [-4.45043, -1.07422, 4.0, 0.07457],
        [-4.18171, -1.20528, 4.0, 0.07474],
        [-3.9113, -1.33413, 4.0, 0.07489],
        [-3.6394, -1.46089, 4.0, 0.075],
        [-3.36618, -1.58562, 4.0, 0.07509],
        [-3.09177, -1.70838, 4.0, 0.07516],
        [-2.81627, -1.8292, 4.0, 0.07521],
        [-2.53976, -1.94807, 4.0, 0.07524],
        [-2.26231, -2.06499, 4.0, 0.07527],
        [-1.98397, -2.17996, 4.0, 0.07529],
        [-1.70478, -2.29297, 4.0, 0.0753],
        [-1.42478, -2.40402, 4.0, 0.0753],
        [-1.14398, -2.51307, 4.0, 0.07531],
        [-0.86239, -2.62009, 4.0, 0.07531],
        [-0.58003, -2.72507, 4.0, 0.07531],
        [-0.2969, -2.82795, 4.0, 0.07531],
        [-0.013, -2.92874, 4.0, 0.07531],
        [0.27166, -3.0274, 4.0, 0.07532],
        [0.55707, -3.12396, 4.0, 0.07532],
        [0.84321, -3.21843, 4.0, 0.07533],
        [1.13006, -3.31086, 4.0, 0.07534],
        [1.4176, -3.40131, 4.0, 0.07536],
        [1.70578, -3.48988, 4.0, 0.07537],
        [1.99456, -3.57666, 4.0, 0.07539],
        [2.28391, -3.66178, 4.0, 0.0754],
        [2.57378, -3.74536, 4.0, 0.07542],
        [2.86411, -3.82755, 4.0, 0.07544],
        [3.15485, -3.90852, 4.0, 0.07545],
        [3.44596, -3.98833, 4.0, 0.07546],
        [3.73743, -4.06706, 4.0, 0.07548],
        [4.02926, -4.14471, 4.0, 0.07549],
        [4.32145, -4.2212, 4.0, 0.07551],
        [4.61405, -4.29638, 3.60848, 0.08372],
        [4.90711, -4.37011, 3.13917, 0.09626],
        [5.20072, -4.44198, 2.78462, 0.10855],
        [5.49498, -4.51157, 2.43896, 0.12398],
        [5.78997, -4.57822, 2.17347, 0.13914],
        [6.08389, -4.63928, 1.93913, 0.15481],
        [6.37405, -4.68922, 1.71902, 0.17128],
        [6.65773, -4.72339, 1.71902, 0.16621],
        [6.9319, -4.73726, 1.71902, 0.1597],
        [7.19378, -4.72749, 1.71902, 0.15245],
        [7.44078, -4.69175, 1.71902, 0.14518],
        [7.66948, -4.62701, 1.71902, 0.13827],
        [7.87631, -4.53091, 1.71902, 0.13267],
        [8.05617, -4.40023, 1.71902, 0.12933],
        [8.19968, -4.22947, 2.0862, 0.10692],
        [8.3154, -4.03266, 2.36165, 0.09668],
        [8.40645, -3.81573, 2.69324, 0.08735],
        [8.47598, -3.58384, 3.11328, 0.07776],
        [8.52755, -3.3417, 3.80996, 0.06498],
        [8.56615, -3.09378, 4.0, 0.06273],
        [8.59764, -2.843, 4.0, 0.06319],
        [8.63456, -2.58085, 4.0, 0.06618],
        [8.66945, -2.31057, 4.0, 0.06813],
        [8.70117, -2.03887, 4.0, 0.06839],
        [8.72935, -1.76606, 4.0, 0.06856],
        [8.75361, -1.49239, 4.0, 0.06869],
        [8.77354, -1.21807, 3.93508, 0.0699],
        [8.7887, -0.9433, 3.49067, 0.07884],
        [8.79844, -0.66829, 3.11501, 0.08834],
        [8.80193, -0.39331, 2.74659, 0.10012],
        [8.7982, -0.11873, 2.7003, 0.1017],
        [8.78602, 0.155, 2.7003, 0.10147],
        [8.76393, 0.42725, 2.7003, 0.10116],
        [8.72959, 0.6971, 2.7003, 0.10074],
        [8.68032, 0.96325, 2.7003, 0.10024],
        [8.61256, 1.22377, 2.7003, 0.09969],
        [8.52228, 1.47599, 2.7003, 0.09921],
        [8.40421, 1.71544, 2.7003, 0.09887],
        [8.25902, 1.93892, 3.01527, 0.08838],
        [8.0935, 2.14758, 3.27713, 0.08127],
        [7.91182, 2.34236, 3.50499, 0.07599],
        [7.71692, 2.52403, 3.72706, 0.07149],
        [7.51107, 2.69337, 3.90219, 0.06831],
        [7.29591, 2.85089, 3.66442, 0.07277],
        [7.07256, 2.99677, 3.37917, 0.07894],
        [6.84209, 3.13127, 3.04033, 0.08777],
        [6.60525, 3.25428, 2.71085, 0.09845],
        [6.36281, 3.36573, 2.41816, 0.11035],
        [6.11542, 3.46528, 2.16664, 0.12308],
        [5.86368, 3.5522, 2.16664, 0.12292],
        [5.60815, 3.62536, 2.16664, 0.12267],
        [5.34937, 3.68191, 2.16664, 0.12226],
        [5.08847, 3.71881, 2.16664, 0.12162],
        [4.82731, 3.7316, 2.16664, 0.12068],
        [4.56904, 3.71472, 2.16664, 0.11946],
        [4.31832, 3.66197, 2.16664, 0.11825],
        [4.08161, 3.56655, 2.82353, 0.09039],
        [3.85497, 3.44661, 3.28022, 0.07817],
        [3.63646, 3.30843, 2.95682, 0.08744],
        [3.42397, 3.15736, 2.56579, 0.10162],
        [3.21572, 2.99738, 2.56579, 0.10235],
        [3.00901, 2.83176, 2.56579, 0.10324],
        [2.79847, 2.67362, 2.56579, 0.10263],
        [2.58282, 2.52589, 2.56579, 0.10188],
        [2.36097, 2.39153, 2.56579, 0.10109],
        [2.13178, 2.27413, 2.56579, 0.10036],
        [1.89366, 2.17942, 2.56579, 0.09987],
        [1.64516, 2.11519, 2.80053, 0.09165],
        [1.38896, 2.07729, 3.05557, 0.08476],
        [1.12687, 2.06238, 3.37664, 0.07774],
        [0.86031, 2.06719, 3.64679, 0.07311],
        [0.59018, 2.08972, 3.94413, 0.06873],
        [0.31722, 2.12815, 4.0, 0.06891],
        [0.04203, 2.1807, 4.0, 0.07004],
        [-0.23485, 2.24574, 4.0, 0.07111],
        [-0.51301, 2.32164, 4.0, 0.07208],
        [-0.79208, 2.40662, 4.0, 0.07293],
        [-1.07178, 2.49904, 4.0, 0.07364],
        [-1.35193, 2.59723, 4.0, 0.07421],
        [-1.63239, 2.69973, 4.0, 0.07465],
        [-1.91313, 2.80535, 4.0, 0.07499],
        [-2.19409, 2.91297, 4.0, 0.07522],
        [-2.47608, 3.02212, 4.0, 0.07559],
        [-2.75823, 3.13085, 4.0, 0.07559],
        [-3.04053, 3.23917, 4.0, 0.07559],
        [-3.32299, 3.34711, 4.0, 0.07559],
        [-3.60558, 3.45467, 4.0, 0.07559],
        [-3.88832, 3.56185, 4.0, 0.07559],
        [-4.17121, 3.66864, 4.0, 0.07559],
        [-4.45426, 3.77501, 4.0, 0.07559],
        [-4.73748, 3.88093, 4.0, 0.0756],
        [-5.0209, 3.98635, 3.65718, 0.08268],
        [-5.30454, 4.09117, 3.18609, 0.09491],
        [-5.58844, 4.19532, 2.81498, 0.10742],
        [-5.87266, 4.29862, 2.50763, 0.1206],
        [-6.15648, 4.40055, 2.22878, 0.13531],
        [-6.43801, 4.4959, 1.96641, 0.15116],
        [-6.71574, 4.57968, 1.6, 0.18131],
        [-6.98825, 4.64765, 1.6, 0.17554],
        [-7.25385, 4.69544, 1.6, 0.16866],
        [-7.51088, 4.71971, 1.6, 0.16136],
        [-7.75759, 4.71759, 1.6, 0.1542],
        [-7.99195, 4.68634, 1.6, 0.14777],
        [-8.21111, 4.62249, 1.6, 0.14267],
        [-8.41043, 4.52068, 1.6, 0.13989],
        [-8.57491, 4.36542, 1.77987, 0.12708],
        [-8.70264, 4.17046, 1.98104, 0.11765],
        [-8.79276, 3.94837, 2.18193, 0.10985],
        [-8.84648, 3.7096, 2.40807, 0.10163],
        [-8.86734, 3.46207, 2.58799, 0.09599],
        [-8.85823, 3.21089, 2.808, 0.08951],
        [-8.82291, 2.95956, 2.97217, 0.08539],
        [-8.76387, 2.7104, 3.01593, 0.0849],
        [-8.68303, 2.46513, 3.01593, 0.08563],
        [-8.57904, 2.22613, 3.30915, 0.07876],
        [-8.45596, 1.99397, 3.5166, 0.07472],
        [-8.31608, 1.76899, 3.57698, 0.07406],
        [-8.16108, 1.55139, 3.57698, 0.07469],
        [-7.98998, 1.34265, 3.90502, 0.06912],
        [-7.80543, 1.14214, 4.0, 0.06813],
        [-7.60933, 0.94925, 4.0, 0.06877],
    ],
)  # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)


if __name__ == "__main__":

    def get_test_params():
        return {
            "x": -8.45596,
            "y": 1.99397,
            "heading": -62.06962,
            "track_width": 1,
            "is_reversed": False,
            "is_offtrack": False,
            "steering_angle": -1.0,
            "all_wheels_on_track": True,
            "is_left_of_center": True,
            "distance_from_center": 0.3,
            "progress": 90,
            "speed": 2.63745,
        }

    params = get_test_params()

    reward = reward_function(params)

    print("test_reward: {}".format(reward))
