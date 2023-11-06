# Copyright (c) 2021 Youngjoo Kim (MIT License)
# Author: Youngjoo Kim (rhymesg@gmail.com)
# Related work: https://arxiv.org/abs/2107.00689

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
    # Define parameters for the algorithm.
    min_num_obj = 6  # Minimum number of objects required to perform the matching.
    tol_theta = 0.2  # Angular tolerance to determine if two angles are similar.
    tol_r = 0.2  # Relative distance tolerance to determine if two distances are similar.
    tol_r_perc = 0.1  # Minimum relative distance from the center to consider an object.
    tol_theta_circle = 0.0001  # Angular tolerance for determining the intersection of circles.
    max_iter = 200  # Maximum number of iterations for the circle intersection algorithm.
    sig_circle = 5  # Not used in this snippet.

    # Calculate the center coordinates of the image.
    center_x = image.size_x / 2
    center_y = image.size_y / 2

    # Initialize variables to store the best match's information.
    max_match = 0  # Maximum number of matches found.
    min_error_std = 1000  # Minimum error standard deviation.
    best_x = None  # X-coordinate of the best match position.
    best_y = None  # Y-coordinate of the best match position.

    # Initialize a result dictionary to store the output.
    result = {'valid': False, 'x': None, 'y': None, 'num_matches': 0}

    # If there aren't enough objects in the image, return the result early.
    if len(image.objects) < min_num_obj:
        return result

    # Create all combinations of pairs of objects in the image and in the database.
    image_object_indices = range(len(image.objects))
    object_indices = range(len(database.objects))
    image_object_idx_pairs = itertools.combinations(image_object_indices, 2)
    object_idx_pairs = itertools.combinations(object_indices, 2)

    # Iterate over all pairs of image objects.
    for image_idx_pair in image_object_idx_pairs:
        # Get the two image objects for this pair.
        img_obj1 = image.objects[image_idx_pair[0]]
        img_obj2 = image.objects[image_idx_pair[1]]

        # Convert their coordinates from Cartesian to polar with respect to the image center.
        r1, theta1 = cart_to_polar(img_obj1.x, img_obj1.y, center_x, center_y)
        r2, theta2 = cart_to_polar(img_obj2.x, img_obj2.y, center_x, center_y)

        # Skip pairs that are too close to the center based on the tolerance percentage.
        if r1 < tol_r_perc * image.size_x or r2 < tol_r_perc * image.size_x:
            continue

        # Normalize the distances and calculate the angular difference.
        rn1 = 1.0  # The first distance is normalized to 1.
        rn2 = r2 / r1  # Normalize the second distance by the first.
        theta_12 = angle_pi_to_pi(theta1 - theta2)  # Calculate the angular difference and wrap it within [-π, π].

        # Iterate over all pairs of database objects.
        for idx_pair in object_idx_pairs:
            # Get the two database objects for this pair.
            obj1 = database.objects[idx_pair[0]]
            obj2 = database.objects[idx_pair[1]]

            # Compute the Euclidean distance between the two database objects.
            d = math.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)

            # Calculate the minimum and maximum possible distances for the first object based on the geometry.
            R1_min = d / (1 + rn2)
            if abs(1 - rn2) < 0.01:
                R1_max = database.size_x
            else:
                R1_max = min(database.size_x, d / abs(1 - rn2))

            # Start the iterative process to find the intersection of circles.
            R1 = (R1_min + R1_max) / 2  # Initial guess for R1.
            R1_min_test = R1_min  # Lower bound for R1 during testing.
            R1_max_test = R1_max  # Upper bound for R1 during testing.

            count_iter = 0  # Initialize iteration count.
            found = False  # Flag to indicate if a match is found.

            # Begin the iterative process to find circle intersections.
            while count_iter < max_iter:
                # Calculate circle intersections based on current R1 and the normalized distance.
                res = get_intersections(obj1.x, obj1.y, R1, obj2.x, obj2.y, R1 * rn2)
                assert res is not None  # Ensure that we get a result.

                # Extract intersection points from the result.
                x3, y3, x4, y4 = res

                # Convert the coordinates of the first intersection point from Cartesian to polar.
                R1_3, Theta1_3 = cart_to_polar(obj1.x, obj1.y, x3, y3)
                R2_3, Theta2_3 = cart_to_polar(obj2.x, obj2.y, x3, y3)

                # Compute the angular difference for the first intersection.
                Theta_12_3 = angle_pi_to_pi(Theta1_3 - Theta2_3)

                # Determine if the first intersection point's angle has the same sign as the image angle or is pi radians apart.
                if theta_12 * Theta_12_3 >= 0 or abs(Theta_12_3 - math.pi) <= tol_theta:  # Same sign or close to pi radians difference.
                    Theta_12 = Theta_12_3  # Use the angle from the first intersection point.
                    x_cand = x3  # Candidate x-coordinate for the image position.
                    y_cand = y3  # Candidate y-coordinate for the image position.
                else:
                    # Convert the coordinates of the second intersection point from Cartesian to polar.
                    R1_4, Theta1_4 = cart_to_polar(obj1.x, obj1.y, x4, y4)
                    R2_4, Theta2_4 = cart_to_polar(obj2.x, obj2.y, x4, y4)

                    # Compute the angular difference for the second intersection.
                    Theta_12_4 = angle_pi_to_pi(Theta1_4 - Theta2_4)

                    # Ensure the second intersection point's angle has the same sign as the image angle.
                    assert theta_12 * Theta_12_4 > 0, "{}, {}".format(theta_12, Theta_12_4)

                    Theta_12 = Theta_12_4  # Use the angle from the second intersection point.
                    x_cand = x4  # Update the candidate x-coordinate.
                    y_cand = y4  # Update the candidate y-coordinate.

                # Check if the angular difference is within the tolerance for a circle intersection.
                if abs(Theta_12 - theta_12) < tol_theta_circle:
                    found = True  # A match is found.
                    break
                elif abs(Theta_12) > abs(theta_12):
                    # If the angle is too large, adjust the minimum radius and recalculate the middle value for R1.
                    R1_min_test = R1
                    if abs(R1 - R1_max_test) < 0.0001:
                        break  # If the adjustment is too small, exit the loop.
                    R1 = (R1 + R1_max_test) / 2  # Increase R1 to get a smaller angle.
                else:
                    # If the angle is too small, adjust the maximum radius and recalculate the middle value for R1.
                    R1_max_test = R1
                    if abs(R1 - R1_min_test) < 0.0001:
                        break  # If the adjustment is too small, exit the loop.
                    R1 = (R1 + R1_min_test) / 2  # Decrease R1 to get a larger angle.

                count_iter += 1  # Increment the iteration count.


            if found:
                # Set up to compare the additional points in the image with the database.
                # Create a set of image object indices excluding the pair already used.
                image_object_indices_set = set(image_object_indices)
                image_object_indices_set.remove(image_idx_pair[0])
                image_object_indices_set.remove(image_idx_pair[1])

                # Initialize match count and error list.
                # Start with a count of 2, as one pair is already matched.
                count_match = 2
                match_error_list = []
                # Iterate over remaining image objects to find additional matches.
                for image_idx in image_object_indices_set:
                    # Create a set of database object indices excluding the pair already used.
                    object_indices_set = set(object_indices)
                    object_indices_set.remove(idx_pair[0])
                    object_indices_set.remove(idx_pair[1])

                    # Get the next image object to compare.
                    img_obj = image.objects[image_idx]
                    r, theta = cart_to_polar(img_obj.x, img_obj.y, center_x, center_y)
                    # Skip objects too close to the center.
                    if r < tol_r_perc * image.size_x:
                        continue
                    # Calculate the angular difference and normalized distance for the image object.
                    del_theta = angle_pi_to_pi(theta1 - theta)
                    r_norm = r / r1

                    # Create a list from the remaining database object indices.
                    object_indices_list = list(object_indices_set)
                    possible_objects = {}
                    # Iterate through remaining database objects to find potential matches.
                    for idx in object_indices_list:
                        obj = database.objects[idx]
                        R, Theta = cart_to_polar(obj.x, obj.y, x_cand, y_cand)
                        # Skip database objects too close to the candidate center.
                        if R < tol_r_perc * database.size_x:
                            continue
                        # Calculate angular difference and normalized distance for the database object.
                        del_Theta = angle_pi_to_pi(theta1 - Theta)
                        R_norm = R / R1
                        # If the differences are within tolerances, store them as possible matches.
                        if abs(del_Theta - del_theta) < tol_theta and abs(r_norm - R_norm) < tol_r:
                            possible_objects[idx] = abs(del_Theta - del_theta)/math.pi + abs(r_norm - R_norm)/r_norm

                    # If there are possible matches, choose the one with the smallest error.
                    if len(possible_objects) > 0:
                        min_idx = min(possible_objects, key=possible_objects.get)
                        object_indices_set.remove(min_idx)
                        count_match += 1
                        match_error_list.append(possible_objects[min_idx])

                # If the number of matches is sufficient and the standard deviation of errors is low,
                # update the best match position and error metrics.
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

# Run repetitive simulations to analyze errors statistically.
def run_monte_carlo_simulation():
    database = generate_database_1(size_x=250, size_y=150)
    # database.resize(250, 150)

    # Settings
    num_samples = 500   # Number of Monte-Carlo runs
    image_size_x = 640  # True x position
    image_size_y = 480  # True y position
    z_true = 100        # True z position
    fov_x_deg = 45      # Camera field of view (30~45)

    attitude_error_std_deg = 0.05;  # Attitude error (deg)
    pixel_error_std = 1;            # Image processing error (pixel)

    # Generate true position samples over the database.
    fov_x, fov_y = get_fov(image_size_x, image_size_y, z_true, fov_x_deg)
    print("fov_x: {0:.2f} m, fov_y: {1:.2f} m.".format(fov_x, fov_y))
    x_samples = np.random.uniform(fov_x, database.size_x - fov_x, num_samples)
    y_samples = np.random.uniform(fov_y, database.size_y - fov_y, num_samples)

    # Run pattern matching algorithm for each sample
    # and gather results.
    errors = []
    num_image_obj = []
    num_fp = 0
    num_matched = 0
    for x_true, y_true in zip(x_samples, y_samples):
        # Simulate aerial image.
        # by putting pixel error and attitude error.
        image = get_aerial_image(database=database, x=x_true, y=y_true, z=z_true, size_x=image_size_x, size_y=image_size_y,
                                 fov_x_deg=fov_x_deg, attitude_error_std_rad=attitude_error_std_deg*math.pi/180, pixel_error_std=pixel_error_std)
        num_image_obj.append(len(image.objects))

        # Run pattern matching algorithm to estimate position.
        res = find_position(database, image)

        if res['valid']:
            error = np.linalg.norm(np.array(res['x'] - x_true, res['y'] - y_true))
            print("error: {0:.2f} m.".format(error))
            num_matched += 1
            if error > 15:
                num_fp += 1
            else:
                errors.append(error)

    print("")
    # print("Average error: {0:.2f} m.".format(np.average(np.array(errors))))
    print("Error std: {0:.2f} m.".format(np.std(np.array(errors))))
    # print("Largest error: {0:.2f} m.".format(max(errors)))
    print("Matched images: {}/{}.".format(num_matched, num_samples))
    print("Num. false positives: {}/{}".format(num_fp, num_matched))
    print("Average image objects: {}.".format(np.average(np.array(num_image_obj))))


# Run a case and show plot.
def test_a_case():
    # generate database meta image
    database = generate_database_1(size_x=250, size_y=150)
    database.resize(250, 150)

    # Settings
    x_true = 120.4  # True x position
    y_true = 92.2   # True y position
    z_true = 100    # True z position
    fov_x_deg = 30  # Camera field of view (30~45)

    # Simulate aerial image.
    image = get_aerial_image(database=database, x=x_true, y=y_true, z=z_true, size_x=640, size_y=480, fov_x_deg=fov_x_deg, attitude_error_std_rad=0, pixel_error_std=0)
    fov_x, fov_y = get_fov(image.size_x, image.size_y, z_true, fov_x_deg)

    # Run pattern matching algorithm to estimate position.
    find_position(database, image)

    # Draw figure of database and aerial image.
    fig, (sub1, sub2) = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.8)
    
    for obj in database.objects:
        sub1.plot(obj.x, obj.y, 'ko')
    
    rectangle = plt.Rectangle((x_true - fov_x, y_true - fov_y), fov_x*2, fov_y*2, fc=(0,0,0,0) , ec="red")
    sub1.add_patch(rectangle)
    sub1.set_xlim([0, 250])
    sub1.set_ylim([0, 150])
    sub1.set_xlabel('X (m)')
    sub1.set_ylabel('Y (m)')
    sub1.set_aspect('equal')
    sub1.set_title('Database')

    sub2.set_xlabel('x (pixel)')
    sub2.set_ylabel('y (pixel)')
    sub2.set_title('Image')

    for obj in image.objects:
        sub2.plot(obj.x, obj.y, 'ko')

    sub2.set_xlim([0, 640])
    sub2.set_ylim([0, 480])
    sub2.set_aspect('equal')

    plt.show()



if __name__ == '__main__':
    # test_a_case()
    run_monte_carlo_simulation()


