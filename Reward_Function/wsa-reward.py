import math
import numpy as np


def reward_function(params):
    """
    rocketansky: drive like a pro

    parts of this are shamelessly copied from reward functions i found on the
    web, and the original authors, to my shame, are not credited.

    the basic idea is to have a collection of objectives that give a reward
    about in the range [0, 1], and then weight the reward according to
    how important the objective is.

    the important stuff is after all the helper functions, natch.

    for the weird names, r_goatifi is named after an F1 driver that DNFs
    a lot, so it's the 'don't DNF' reward, and r_verstappen is so named
    because max verstappen busts track limits a lot.

    the best strategy i've used so far is with the following reward priority:

    reward = (
        (30 * r_point_car)
        + (10 * r_verstappen)
        + (10 * r_input)
        + (10 * r_prog)
        + r_goatifi
    )

    and a discrete action space such that (abs(steering_angle), speed) is
    in {(30, 1.2), (20, 1.8), (10, 3), (0, 4)}
    """

    # Parameters
    FUTURE_STEP = 7
    MID_STEP = 4
    TURN_THRESHOLD = 10  # degrees
    DIST_THRESHOLD = 1.2  # metres
    SPEED_THRESHOLD = 1.8  # m/s

    def dist(point1, point2):
        return (
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        ) ** 0.5

    # thanks to https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    def rect(r, theta):
        """
        theta in degrees
        returns tuple; (float, float); (x,y)
        """

        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        return x, y

    # thanks to https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    def polar(x, y):
        """
        returns r, theta(degrees)
        """

        r = (x**2 + y**2) ** 0.5
        theta = math.degrees(math.atan2(y, x))
        return r, theta

    def angle_mod_360(angle):
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

    def get_waypoints_ordered_in_driving_direction(params):
        # waypoints are always provided in counter clock wise order
        if params["is_reversed"]:  # driving clock wise.
            return list(reversed(params["waypoints"]))
        else:  # driving counter clock wise.
            return params["waypoints"]

    def up_sample(waypoints, factor):
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

    def get_target_point(params):
        waypoints = up_sample(
            get_waypoints_ordered_in_driving_direction(params), 20
        )

        car = [params["x"], params["y"]]

        distances = [dist(p, car) for p in waypoints]
        min_dist = min(distances)
        i_closest = distances.index(min_dist)

        n = len(waypoints)

        waypoints_starting_with_closest = [
            waypoints[(i + i_closest) % n] for i in range(n)
        ]

        r = params["track_width"] * 0.9

        is_inside = [dist(p, car) < r for p in waypoints_starting_with_closest]
        i_first_outside = is_inside.index(False)

        if (
            i_first_outside < 0
        ):  # this can only happen if we choose r as big as the entire track
            return waypoints[i_closest]

        return waypoints_starting_with_closest[i_first_outside]

    def get_target_steering_degree(params):
        tx, ty = get_target_point(params)
        car_x = params["x"]
        car_y = params["y"]
        dx = tx - car_x
        dy = ty - car_y
        heading = params["heading"]

        _, target_angle = polar(dx, dy)

        steering_angle = target_angle - heading

        return angle_mod_360(steering_angle)

    def score_steer_to_point_ahead(params):
        best_stearing_angle = get_target_steering_degree(params)
        steering_angle = params["steering_angle"]

        error = (
            steering_angle - best_stearing_angle
        ) / 60.0  # 60 degree is already really bad

        score = 1.0 - abs(error)

        return max(
            score, 0.01
        )  # optimizer is rumored to struggle with negative numbers and numbers too close to zero

    def identify_corner(waypoints, closest_waypoints, future_step):
        # Identify next waypoint and a further waypoint
        point_prev = waypoints[closest_waypoints[0]]
        point_next = waypoints[closest_waypoints[1]]
        point_future = waypoints[
            min(len(waypoints) - 1, closest_waypoints[1] + future_step)
        ]

        # Calculate headings to waypoints
        heading_current = math.degrees(
            math.atan2(
                point_prev[1] - point_next[1], point_prev[0] - point_next[0]
            )
        )
        heading_future = math.degrees(
            math.atan2(
                point_prev[1] - point_future[1], point_prev[0] - point_future[0]
            )
        )

        # Calculate the difference between the headings
        diff_heading = abs(heading_current - heading_future)

        # Check we didn't choose the reflex angle
        if diff_heading > 180:
            diff_heading = 360 - diff_heading

        # Calculate distance to further waypoint
        dist_future = np.linalg.norm(
            [point_next[0] - point_future[0], point_next[1] - point_future[1]]
        )

        return diff_heading, dist_future

    def select_speed(waypoints, closest_waypoints, future_step, mid_step):

        # Identify if a corner is in the future
        diff_heading, dist_future = identify_corner(
            waypoints, closest_waypoints, future_step
        )

        if diff_heading < TURN_THRESHOLD:
            # If there's no corner encourage going faster
            go_fast = True
        else:
            if dist_future < DIST_THRESHOLD:
                # If there is a corner and it's close encourage going slower
                go_fast = False
            else:
                # If the corner is far away, re-assess closer points
                diff_heading_mid, dist_mid = identify_corner(
                    waypoints, closest_waypoints, mid_step
                )

                if diff_heading_mid < TURN_THRESHOLD:
                    # If there's no corner encourage going faster
                    go_fast = True
                else:
                    # If there is a corner and it's close encourage going slower
                    go_fast = False

        return go_fast

    # Read input parameters
    all_wheels_on_track = params["all_wheels_on_track"]
    closest_waypoints = params["closest_waypoints"]
    distance_from_center = params["distance_from_center"]
    is_offtrack = params["is_offtrack"]
    progress = params["progress"]
    speed = params["speed"]
    steps = params["steps"]
    track_width = params["track_width"]
    waypoints = params["waypoints"]
    theta = params["heading"]
    steering_angle = params["steering_angle"]

    if not all_wheels_on_track:
        reward = 1e-4
        return float(reward)

    # point in track direction
    r_point_car = score_steer_to_point_ahead(params)

    # gotta go fast
    go_fast = select_speed(waypoints, closest_waypoints, FUTURE_STEP, MID_STEP)

    r_speed = 1e-6
    if go_fast:
        r_speed = (speed**12) / (4.0**12)
    else:
        r_speed = (speed**2) / (4.0**2)

    # finish in as few setps as possible, and actually finish
    r_prog = (progress / 100) ** 3
    r_prog_rate = progress / steps

    r_goatifi = 1e-6
    if r_prog > 0.95:
        r_goatifi = 1e4

    # try to do as little steering as possible
    r_input = max(
        float(1e-6),
        float(((-7.1 * (abs(steering_angle) ** (1.0 / 3.0))) + 22.0) / 22.0),
    )

    # actually stay on the track
    r_verstappen = max(
        float(1e-6),
        float(-((distance_from_center / (track_width / 2.0)) ** 32) + 1),
    )

    print(
        r_point_car,
        r_verstappen,
        r_input,
        r_speed,
        r_prog,
        r_prog_rate,
        r_goatifi,
    )

    # add together
    reward = (
        (30 * r_point_car)
        + (10 * r_verstappen)
        + (15 * r_input)
        + (20 * r_speed)
        + (10 * r_prog)
        + (30 * r_prog_rate)
        + r_goatifi
    )

    return float(reward)


if __name__ == "__main__":

    def get_test_params():
        return {
            "x": 0.7,
            "y": 1.05,
            "heading": 160.0,
            "track_width": 0.45,
            "is_reversed": False,
            "is_offtrack": False,
            "steering_angle": -3.0,
            "all_wheels_on_track": True,
            "distance_from_center": 0.5,
            "progress": 3,
            "speed": 2.1,
            "steps": 31,
            "waypoints": [
                [0.75, -0.7],
                [1.0, 0.0],
                [0.7, 0.52],
                [0.58, 0.7],
                [0.48, 0.8],
                [0.15, 0.95],
                [-0.1, 1.0],
                [-0.7, 0.75],
                [-0.9, 0.25],
                [-0.9, -0.55],
            ],
            "closest_waypoints": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }

    params = get_test_params()

    reward = reward_function(params)

    print("test_reward: {}".format(reward))
