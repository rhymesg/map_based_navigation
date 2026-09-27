import unittest
import contextlib
import io
import copy, math
from image import Image
from main import find_position

POINTS = [(22, 30), (35, 18), (62, 25), (80, 38), (72, 72), (48, 80), (25, 70), (40, 45)]
TRUE = (50.0, 50.0)

def fixture(degrees=0, order=None):
    database = Image(100, 100)
    for x, y in POINTS:
        database.add_building(x, y)
    image = Image(200, 200)
    a = math.radians(degrees)
    for i in (order or range(len(POINTS))):
        x, y = POINTS[i]
        dx, dy = 2 * (x - TRUE[0]), 2 * (y - TRUE[1])
        image.add_building(100 + dx * math.cos(a) - dy * math.sin(a), 100 + dx * math.sin(a) + dy * math.cos(a))
    return database, image

class MatchingChecks(unittest.TestCase):
    def test_rotation_and_detection_order(self):
        for degrees, order in [(0,None), (15,None),
                               (15,[4,1,7,0,6,3,2,5]),
                               (0,list(reversed(range(8)))),
                               (15,list(reversed(range(8))))]:
            with self.subTest(degrees=degrees, order=order):
                database, image = fixture(degrees, order)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = find_position(database, image)
                self.assertTrue(result['valid'])
                self.assertEqual(result['num_matches'], 8)
                self.assertLess(math.hypot(result['x']-50, result['y']-50), 0.001)

    def test_repeated_detections_do_not_make_six_matches(self):
        database, image = fixture()
        image.objects = [copy.deepcopy(image.objects[i]) for i in (0,1,2,2,2,2)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = find_position(database, image)
        self.assertFalse(result['valid'])

if __name__ == '__main__':
    unittest.main()
