import math


class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = 0
        self.verbose = verbose

    def reward_function(self, params):

        ################## HELPER FUNCTIONS ###################

        def dist_2_points(x1, x2, y1, y2):
            return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5

        def closest_2_racing_points_index(racing_coords, car_coords):

            # Calculate all distances to racing points
            distances = []
            for i in range(len(racing_coords)):
                distance = dist_2_points(
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

        def dist_to_racing_line(
            closest_coords, second_closest_coords, car_coords
        ):

            # Calculate the distances between 2 closest racing points
            a = abs(
                dist_2_points(
                    x1=closest_coords[0],
                    x2=second_closest_coords[0],
                    y1=closest_coords[1],
                    y2=second_closest_coords[1],
                )
            )

            # Distances between car and closest and second closest racing point
            b = abs(
                dist_2_points(
                    x1=car_coords[0],
                    x2=closest_coords[0],
                    y1=car_coords[1],
                    y2=closest_coords[1],
                )
            )
            c = abs(
                dist_2_points(
                    x1=car_coords[0],
                    x2=second_closest_coords[0],
                    y1=car_coords[1],
                    y2=second_closest_coords[1],
                )
            )

            # Calculate distance between car and racing line (goes through 2 closest racing points)
            # try-except in case a=0 (rare bug in DeepRacer)
            try:
                distance = abs(
                    -(a**4)
                    + 2 * (a**2) * (b**2)
                    + 2 * (a**2) * (c**2)
                    - (b**4)
                    + 2 * (b**2) * (c**2)
                    - (c**4)
                ) ** 0.5 / (2 * a)
            except:
                distance = b

            return distance

        # Calculate which one of the closest racing points is the next one and which one the previous one
        def next_prev_racing_point(
            closest_coords, second_closest_coords, car_coords, heading
        ):

            # Virtually set the car more into the heading direction
            heading_vector = [
                math.cos(math.radians(heading)),
                math.sin(math.radians(heading)),
            ]
            new_car_coords = [
                car_coords[0] + heading_vector[0],
                car_coords[1] + heading_vector[1],
            ]

            # Calculate distance from new car coords to 2 closest racing points
            distance_closest_coords_new = dist_2_points(
                x1=new_car_coords[0],
                x2=closest_coords[0],
                y1=new_car_coords[1],
                y2=closest_coords[1],
            )
            distance_second_closest_coords_new = dist_2_points(
                x1=new_car_coords[0],
                x2=second_closest_coords[0],
                y1=new_car_coords[1],
                y2=second_closest_coords[1],
            )

            if (
                distance_closest_coords_new
                <= distance_second_closest_coords_new
            ):
                next_point_coords = closest_coords
                prev_point_coords = second_closest_coords
            else:
                next_point_coords = second_closest_coords
                prev_point_coords = closest_coords

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(
            closest_coords, second_closest_coords, car_coords, heading
        ):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(
                closest_coords, second_closest_coords, car_coords, heading
            )

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0]
            )

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list
        # (start index is included, end index is not)
        def indexes_cyclical(start, end, array_len):

            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count - 1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(
                first_index, closest_index, len(times_list)
            )

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum(
                [times_list[i] for i in indexes_traveled]
            )

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (
                    current_actual_time / current_expected_time
                ) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the Spain track
        # Each row: [x,y,speed,timeFromPreviousPoint]
        racing_track = [
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
        ]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params["all_wheels_on_track"]
        x = params["x"]
        y = params["y"]
        distance_from_center = params["distance_from_center"]
        is_left_of_center = params["is_left_of_center"]
        heading = params["heading"]
        progress = params["progress"]
        steps = params["steps"]
        speed = params["speed"]
        steering_angle = params["steering_angle"]
        track_width = params["track_width"]
        waypoints = params["waypoints"]
        closest_waypoints = params["closest_waypoints"]
        is_offtrack = params["is_offtrack"]

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(
            racing_track, [x, y]
        )

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = (
                0  # this is just for testing purposes
            )
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist / (track_width * 0.5)))
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2] - speed)
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1
        STANDARD_TIME = 37
        FASTEST_TIME = 27
        times_list = [row[3] for row in racing_track]
        projected_time = projected_time(
            self.first_racingpoint_index, closest_index, steps, times_list
        )
        try:
            steps_prediction = projected_time * 15 + 1
            reward_prediction = max(
                1e-3,
                (
                    -REWARD_PER_STEP_FOR_FASTEST_TIME
                    * (FASTEST_TIME)
                    / (STANDARD_TIME - FASTEST_TIME)
                )
                * (steps_prediction - (STANDARD_TIME * 15 + 1)),
            )
            steps_reward = min(
                REWARD_PER_STEP_FOR_FASTEST_TIME,
                reward_prediction / steps_prediction,
            )
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading
        )
        if direction_diff > 30:
            reward = 1e-3

        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2] - speed
        if speed_diff_zero > 0.5:
            reward = 1e-3

        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = (
            1500  # should be adapted to track length and other rewards
        )
        STANDARD_TIME = 37  # seconds (time that is easily done by model)
        FASTEST_TIME = 27  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(
                1e-3,
                (
                    -REWARD_FOR_FASTEST_TIME
                    / (15 * (STANDARD_TIME - FASTEST_TIME))
                )
                * (steps - STANDARD_TIME * 15),
            )
        else:
            finish_reward = 0
        reward += finish_reward

        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################

        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print(
                "=== Distance reward (w/out multiple): %f ==="
                % (distance_reward)
            )
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)

        #################### RETURN REWARD ####################

        # Always return a float value
        return float(reward)


reward_object = (
    Reward()
)  # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)
