import tempfile
import unittest
from pathlib import Path

from PIL import Image

from preprocess import discover_dataset


class DiscoverDatasetTests(unittest.TestCase):
    def test_removes_duplicate_pixels_within_a_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class_dir = root / "APPLE"
            class_dir.mkdir()
            image = Image.new("RGB", (96, 96), color=(30, 120, 45))
            image.save(class_dir / "leaf-a.png")
            image.save(class_dir / "leaf-b.png")

            paths, labels, classes = discover_dataset(root)

            self.assertEqual(len(paths), 1)
            self.assertEqual(labels, [0])
            self.assertEqual(classes, ["APPLE"])

    def test_rejects_duplicate_pixels_across_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "APPLE ROT LEAVES"
            second = root / "HEALTHY LEAVES"
            first.mkdir()
            second.mkdir()
            image = Image.new("RGB", (96, 96), color=(30, 120, 45))
            image.save(first / "leaf.png")
            image.save(second / "same-leaf.png")

            with self.assertRaisesRegex(ValueError, "conflicting class labels"):
                discover_dataset(root)

    def test_reports_corrupt_image_instead_of_skipping_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            class_dir = root / "APPLE"
            class_dir.mkdir()
            (class_dir / "broken.jpg").write_bytes(b"not an image")

            with self.assertRaisesRegex(ValueError, "Unreadable image"):
                discover_dataset(root)


if __name__ == "__main__":
    unittest.main()
