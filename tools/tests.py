import unittest
import numpy as np
from data import rotate_data

class TestRotateData(unittest.TestCase):
    def test_rotation_180(self):
        data = np.zeros((1, 17, 3))
        for i in range(17):
            data[0, i] = np.array([i, i + 1, 1])

        result = rotate_data(data, angle=180)
        self.assertEqual(result.shape, (2, 17, 3))

        original = result[0]
        rotated = result[1]
        pivot = original[0, :2]
        expected = -(original[:, :2] - pivot) + pivot

        np.testing.assert_allclose(rotated[:, :2], expected, atol=1e-6)
        np.testing.assert_allclose(rotated[:, 2], original[:, 2])

    def test_rotate_data_2d_xy(self):
        # Create a simple test input: one sample with 17 2D points
        data = np.zeros((1, 17, 2))
        for i in range(17):
            data[0, i] = [i, i + 1]

        # Rotate by 180 degrees
        result = rotate_data(data, angle=180)

        # Result shape should be doubled in batch dimension
        assert result.shape == (2, 17, 2)

        original = result[0]
        rotated = result[1]

        # Subtract root joint (index 0) position to get relative coordinates
        pivot = original[0]
        expected = -(original - pivot) + pivot  # 180-degree rotation: flip x and y

        # Assert rotated is as expected (within floating-point tolerance)
        np.testing.assert_allclose(rotated, expected, atol=1e-6)



if __name__ == '__main__':
    unittest.main()
    
