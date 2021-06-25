import math
import copy


class GroundObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Building(GroundObject):
    pass


class Image:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.objects = []

    def add_building(self, x, y):
        self.add_object(Building(x, y))

    def add_object(self, obj):
        assert 0 <= obj.x <= self.size_x
        assert 0 <= obj.y <= self.size_y
        self.objects.append(obj)

    def resize(self, new_size_x, new_size_y):
        scale_x = new_size_x / self.size_x
        scale_y = new_size_y / self.size_y

        for idx in range(len(self.objects)):
            obj = self.objects[idx]
            obj.x = obj.x * scale_x
            obj.y = obj.y * scale_y

        self.size_x = new_size_x
        self.size_y = new_size_y


def get_fov(size_x, size_y, z, fov_x_deg):
    fov_x = z * math.tan(fov_x_deg * math.pi / 180.0)
    fov_y = size_y / size_x * fov_x
    return fov_x, fov_y


def get_aerial_image(database: Image, x, y, z, size_x, size_y, fov_x_deg):
    image = Image(size_x, size_y)
    fov_x, fov_y = get_fov(image.size_x, image.size_y, z, fov_x_deg)

    print("fov_x: {0:.2f} m, fov_y: {1:.2f} m.".format(fov_x, fov_y))

    min_x = x - fov_x
    max_x = x + fov_x
    min_y = y - fov_y
    max_y = y + fov_y

    for obj in database.objects:
        if min_x <= obj.x <= max_x and min_y <= obj.y <= max_y:
            obj_image = copy.deepcopy(obj)

            X = obj.x - x  # origin at center
            Y = obj.y - x

            X_im = X * (image.size_x / 2) / fov_x  # origin at center
            Y_im = Y * (image.size_y / 2) / fov_y

            obj_image.x = X_im + image.size_x / 2  # origin at corner
            obj_image.y = Y_im + image.size_y / 2
            image.add_object(obj_image)

    return image


def generate_database_1(size_x, size_y):
    database_1 = Image(1220, 812)
    database_1.add_building(302, 78)
    database_1.add_building(294, 181)
    database_1.add_building(401, 197)
    database_1.add_building(260, 325)
    database_1.add_building(357, 341)
    database_1.add_building(236, 547)
    database_1.add_building(331, 547)
    database_1.add_building(232, 646)
    database_1.add_building(234, 755)
    database_1.add_building(331, 753)
    database_1.add_building(549, 52)
    database_1.add_building(531, 159)
    database_1.add_building(605, 171)
    database_1.add_building(680, 185)
    database_1.add_building(525, 232)
    database_1.add_building(743, 268)
    database_1.add_building(813, 282)
    database_1.add_building(585, 500)
    database_1.add_building(672, 515)
    database_1.add_building(763, 531)
    database_1.add_building(747, 609)
    database_1.add_building(735, 698)
    database_1.add_building(644, 682)
    database_1.add_building(553, 668)
    database_1.add_building(466, 652)
    database_1.add_building(482, 569)
    database_1.add_building(1011, 54)
    database_1.add_building(995, 131)
    database_1.add_building(981, 214)
    database_1.add_building(963, 300)
    database_1.add_building(1062, 315)
    database_1.add_building(1163, 333)
    database_1.add_building(1201, 761)
    database_1.add_building(1054, 741)
    database_1.add_building(925, 753)
    database_1.add_building(34, 115)
    database_1.add_building(34, 212)
    database_1.add_building(1048, 490)
    database_1.add_building(1040, 537)
    database_1.add_building(1032, 581)
    database_1.resize(size_x, size_y)

    return database_1