import io
import unittest

import numpy as np
from PIL import Image

import app as service


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class FakeModel:
    def __call__(self, batch, training=False):
        return FakeTensor(np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32))


class PredictRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_model = service.model
        service.model = FakeModel()
        service.app.config.update(TESTING=True)
        cls.client = service.app.test_client()

    @classmethod
    def tearDownClass(cls):
        service.model = cls.original_model

    @staticmethod
    def encoded_image(size):
        output = io.BytesIO()
        Image.new("RGB", size, color=(30, 120, 45)).save(output, format="JPEG")
        return output.getvalue()

    def test_rejects_unreadable_image(self):
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(b"not an image"), "leaf.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a readable image", response.get_json()["error"])

    def test_rejects_image_below_minimum_dimensions(self):
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(self.encoded_image((32, 96))), "leaf.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 64 pixels", response.get_json()["error"])

    def test_returns_compatible_prediction_for_readable_image(self):
        response = self.client.post(
            "/predict",
            data={"image": (io.BytesIO(self.encoded_image((96, 96))), "leaf.jpg")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["label"], "SCAB LEAVES")
        self.assertEqual(len(payload["probabilities"]), 4)
        self.assertTrue(payload["uncertain"])


if __name__ == "__main__":
    unittest.main()
