import math
import matplotlib.pyplot as plt
import numpy as np

from image import generate_database_1, get_aerial_image, get_fov
import itertools


def cart_to_polar(x, y, center_x, center_y):
    X = x - center_x
    Y = y - center_y
    r = math.sqrt(X*X + Y*Y)
    theta = math.atan2(Y, X)
    return r, theta


def angle_pi_to_pi(angle):
    if angle > math.pi:
        return angle_pi_to_pi(angle - 2*math.pi)
    elif angle < -math.pi:
        return angle_pi_to_pi(angle + 2*math.pi)
    else:
        return angle


def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # Non intersecting
    if d > r0 + r1:
        print("Non intersecting")
        return None
    # One circle within other
    if d < abs(r0 - r1):
        print("One circle within other")
        return None
    # Coincident circles
    if d == 0 and r0 == r1:
        print("Coincident circles")
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        res = (x3, y3, x4, y4)
        return res


def find_position(database, image):
    # parameters
    min_num_obj = 6
    tol_theta = 0.2  # theta tolerance to determine match
    tol_r = 0.2   # relative distance tolerance to determine match
    tol_r_perc = 0.1  # don't use points with distance to the center less than this percentage
    tol_theta_circle = 0.0001  # theta tolerance to determine circle intersection
    max_iter = 200
    sig_circle = 5

    center_x = image.size_x / 2
    center_y = image.size_y / 2

    max_match = 0
    min_error_std = 1000
    best_x = None
    best_y = None

    result = {'valid': False, 'x': None, 'y': None, 'num_matches': 0}

    if len(image.objects) < min_num_obj:
        return result

    image_object_indices = range(len(image.objects))
    object_indices = range(len(database.objects))
    image_object_idx_pairs = itertools.combinations(image_object_indices, 2)
    object_idx_pairs = itertools.combinations(object_indices, 2)
    for image_idx_pair in image_object_idx_pairs:
        img_obj1 = image.objects[image_idx_pair[0]]
        img_obj2 = image.objects[image_idx_pair[1]]

        r1, theta1 = cart_to_polar(img_obj1.x, img_obj1.y, center_x, center_y)
        r2, theta2 = cart_to_polar(img_obj2.x, img_obj2.y, center_x, center_y)

        if r1 < tol_r_perc * image.size_x or r2 < tol_r_perc * image.size_x:
            continue

        rn1 = 1.0  # normalized distance
        rn2 = r2 / r1
        theta_12 = angle_pi_to_pi(theta1 - theta2)

        for idx_pair in object_idx_pairs:
            obj1 = database.objects[idx_pair[0]]
            obj2 = database.objects[idx_pair[1]]

            # 2 circle intersection
            d = math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)

            R1_min = d / (1 + rn2)
            if abs(1 - rn2) < 0.01:
                R1_max = database.size_x
            else:
                R1_max = min(database.size_x, d / abs(1 - rn2))

            R1 = (R1_min + R1_max) / 2
            R1_min_test = R1_min
            R1_max_test = R1_max

            count_iter = 0
            found = False
            while count_iter < max_iter:
                res = get_intersections(obj1.x, obj1.y, R1, obj2.x, obj2.y, R1 * rn2)
                assert res is not None

                x3, y3, x4, y4 = res
                R1_3, Theta1_3 = cart_to_polar(obj1.x, obj1.y, x3, y3)
                R2_3, Theta2_3 = cart_to_polar(obj2.x, obj2.y, x3, y3)

                Theta_12_3 = angle_pi_to_pi(Theta1_3 - Theta2_3)
                if theta_12 * Theta_12_3 >= 0 or abs(Theta_12_3 - math.pi) <= tol_theta:  # same sign
                    Theta_12 = Theta_12_3
                    x_cand = x3
                    y_cand = y3
                else:
                    R1_4, Theta1_4 = cart_to_polar(obj1.x, obj1.y, x4, y4)
                    R2_4, Theta2_4 = cart_to_polar(obj2.x, obj2.y, x4, y4)
                    Theta_12_4 = angle_pi_to_pi(Theta1_4 - Theta2_4)
                    assert theta_12 * Theta_12_4 > 0, "{}, {}".format(theta_12, Theta_12_4)
                    Theta_12 = Theta_12_4
                    x_cand = x4
                    y_cand = y4

                if abs(Theta_12 - theta_12) < tol_theta_circle:
                    found = True
                    break
                elif abs(Theta_12) > abs(theta_12):
                    R1_min_test = R1
                    if abs(R1 - R1_max_test) < 0.0001:
                        break
                    R1 = (R1 + R1_max_test) / 2  # increase R1
                else:
                    R1_max_test = R1
                    if abs(R1 - R1_min_test) < 0.0001:
                        break
                    R1 = (R1 + R1_min_test) / 2  # decrease R1

                count_iter += 1

            if found:
                test_xy_list = [(x_cand, y_cand)]
                # test_xy_list.append((x_cand+sig_circle*np.random.normal(), y_cand+sig_circle*np.random.normal()))
                # test_xy_list.append(
                #     (x_cand + sig_circle * np.random.normal(), y_cand + sig_circle * np.random.normal()))
                # test_xy_list.append(
                #     (x_cand + sig_circle * np.random.normal(), y_cand + sig_circle * np.random.normal()))

                for xy in test_xy_list:
                    x_cand = xy[0]
                    y_cand = xy[1]

                    # compare points
                    image_object_indices_set = set(image_object_indices)
                    image_object_indices_set.remove(image_idx_pair[0])
                    image_object_indices_set.remove(image_idx_pair[1])

                    count_match = 2
                    match_error_list = []
                    for image_idx in image_object_indices_set:
                        object_indices_set = set(object_indices)
                        object_indices_set.remove(idx_pair[0])
                        object_indices_set.remove(idx_pair[1])

                        img_obj = image.objects[image_idx]
                        r, theta = cart_to_polar(img_obj.x, img_obj.y, center_x, center_y)
                        if r < tol_r_perc * image.size_x:
                            continue
                        del_theta = angle_pi_to_pi(theta1 - theta)
                        r_norm = r / r1

                        object_indices_list = list(object_indices_set)
                        possible_objects = {}
                        for idx in object_indices_list:
                            obj = database.objects[idx]
                            R, Theta = cart_to_polar(obj.x, obj.y, x_cand, y_cand)
                            if R < tol_r_perc * database.size_x:
                                continue
                            del_Theta = angle_pi_to_pi(theta1 - Theta)
                            R_norm = R / R1
                            if abs(del_Theta - del_theta) < tol_theta and abs(r_norm - R_norm) < tol_r:
                                possible_objects[idx] = abs(del_Theta - del_theta)/math.pi + abs(r_norm - R_norm)/r_norm

                        if len(possible_objects) > 0:
                            min_idx = min(possible_objects, key=possible_objects.get)
                            object_indices_set.remove(min_idx)
                            count_match += 1
                            match_error_list.append(possible_objects[min_idx])

                    if count_match >= max_match and count_match >= min_num_obj:
                        error_std = np.std(np.array(match_error_list))
                        if error_std < min_error_std:
                            max_match = count_match
                            min_error_std = error_std
                            best_x = x_cand
                            best_y = y_cand

    print("#. matched points: {}/{}".format(max_match, len(image.objects)))
    print("min_theta_std: {0:.2f}".format(min_error_std))

    result['num_matches'] = max_match
    result['x'] = best_x
    result['y'] = best_y

    if result['num_matches'] > len(image.objects) * 0.5 and result['num_matches'] >= min_num_obj and result['x'] is not None and result['y'] is not None:
        result['valid'] = True
    else:
        result['valid'] = False

    return result


def run_simulation():
    database = generate_database_1(size_x=250, size_y=150)
    # database.resize(250, 150)

    num_samples = 500
    image_size_x = 640
    image_size_y = 480
    fov_x_deg = 45
    z_true = 100
    fov_x, fov_y = get_fov(image_size_x, image_size_y, z_true, fov_x_deg)
    print("fov_x: {0:.2f} m, fov_y: {1:.2f} m.".format(fov_x, fov_y))
    x_samples = np.random.uniform(fov_x, database.size_x - fov_x, num_samples)
    y_samples = np.random.uniform(fov_y, database.size_y - fov_y, num_samples)

    errors = []
    num_image_obj = []
    num_fp = 0
    num_matched = 0
    for x_true, y_true in zip(x_samples, y_samples):
        image = get_aerial_image(database=database, x=x_true, y=y_true, z=z_true, size_x=image_size_x, size_y=image_size_y,
                                 fov_x_deg=fov_x_deg, attitude_error_std_rad=0.05*math.pi/180, pixel_error_std=3)
        num_image_obj.append(len(image.objects))
        res = find_position(database, image)

        if res['valid']:
            error = np.linalg.norm(np.array(res['x'] - x_true, res['y'] - y_true))
            print("error: {0:.2f} m.".format(error))
            num_matched += 1
            if error > 15:
                num_fp += 1
            else:
                errors.append(error)

    # print("Average error: {0:.2f} m.".format(np.average(np.array(errors))))
    print("Error std: {0:.2f} m.".format(np.std(np.array(errors))))
    print("Largest error: {0:.2f} m.".format(max(errors)))
    print("Matched images: {}/{}.".format(num_matched, num_samples))
    print("Num. false positives: {}/{}".format(num_fp, num_matched))
    print("Average image objects: {}.".format(np.average(np.array(num_image_obj))))


def test_simulation():
    # generate database meta image
    database = generate_database_1(size_x=250, size_y=150)
    database.resize(250, 150)

    x_true = 120.4
    y_true = 92.2
    z_true = 100
    fov_x_deg = 45
    image = get_aerial_image(database=database, x=x_true, y=y_true, z=z_true, size_x=640, size_y=480, fov_x_deg=fov_x_deg, attitude_error_std_rad=0, pixel_error_std=0)
    fov_x, fov_y = get_fov(image.size_x, image.size_y, z_true, fov_x_deg)

    find_position(database, image)

    fig, (sub1, sub2) = plt.subplots(2, 1)
    for obj in database.objects:
        sub1.plot(obj.x, obj.y, 'ko')

    rectangle = plt.Rectangle((x_true - fov_x, y_true - fov_y), fov_x*2, fov_y*2, fc=(0,0,0,0) , ec="red")
    sub1.add_patch(rectangle)
    sub1.set_xlim([0, 250])
    sub1.set_ylim([0, 150])
    sub1.set_xlabel('X (m)')
    sub1.set_ylabel('Y (m)')
    sub1.set_aspect('equal')

    sub2.set_xlabel('x (pixel)')
    sub2.set_ylabel('y (pixel)')

    for obj in image.objects:
        sub2.plot(obj.x, obj.y, 'ko')

    sub2.set_xlim([0, 640])
    sub2.set_ylim([0, 480])
    sub2.set_aspect('equal')

    plt.show()



if __name__ == '__main__':
    # test_simulation()
    run_simulation()


