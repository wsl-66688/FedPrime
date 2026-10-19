import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa


spec = importlib.util.spec_from_file_location(
    'make_arrow_food101', Path(__file__).with_name('make_arrow_food101.py')
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class Food101ArrowTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / 'raw'
        self.output = Path(self.temporary_directory.name) / 'arrows'
        self.root.mkdir()
        self.samples = [f'apple_pie_{index}.jpg' for index in range(6)]
        self.texts = {
            sample: f'caption {index} 苹果派'
            for index, sample in enumerate(self.samples)
            if index != 2
        }
        self.write_json('class_idx.json', {'apple_pie': 71})
        self.write_json('text.json', self.texts)
        self.write_json('split.json', {
            'train': self.samples[:4],
            'val': [self.samples[4]],
            'test': [self.samples[5]],
        })
        for index, sample in enumerate(self.samples):
            split = 'test' if index == 5 else 'train'
            path = self.root / 'images' / split / 'apple_pie' / sample
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'image-payload-' + sample.encode('ascii'))

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding='utf-8')

    def convert(self):
        with patch.object(converter, 'tqdm', lambda samples: samples):
            with contextlib.redirect_stdout(io.StringIO()):
                converter.make_arrow(str(self.root), str(self.output), batch_size=2)

    def read_output(self, split):
        with pa.OSFile(str(self.output / f'food101_{split}.arrow'), 'rb') as source:
            reader = pa.ipc.RecordBatchFileReader(source)
            return reader.read_all().to_pydict(), reader.num_record_batches

    def test_batches_preserve_payload_labels_text_and_split_order(self):
        self.convert()
        for split, indices, batch_count in (
            ('train', [0, 1, 3, 4], 2),
            ('test', [5], 1),
        ):
            actual, count = self.read_output(split)
            samples = [self.samples[index] for index in indices]
            expected = {
                'image': [b'image-payload-' + sample.encode('ascii') for sample in samples],
                'text': [[self.texts[sample]] for sample in samples],
                'label': [71] * len(samples),
                'image_id': samples,
                'split': [split] * len(samples),
            }
            self.assertEqual(actual, expected)
            self.assertEqual(count, batch_count)
        self.assertFalse((self.output / 'food101_val.arrow').exists())

    def test_missing_image_does_not_replace_existing_output(self):
        self.convert()
        original = {
            split: (self.output / f'food101_{split}.arrow').read_bytes()
            for split in ('train', 'test')
        }
        # Fail after a complete batch has already been written to the temporary file.
        (self.root / 'images/train/apple_pie' / self.samples[4]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.convert()
        for split in ('train', 'test'):
            self.assertEqual(
                (self.output / f'food101_{split}.arrow').read_bytes(), original[split]
            )
        self.assertEqual(list(self.output.glob('*.tmp')), [])

    def test_all_samples_without_text_produce_valid_empty_files(self):
        self.write_json('text.json', {})
        self.convert()
        for split in ('train', 'test'):
            actual, count = self.read_output(split)
            self.assertEqual(actual, {
                'image': [], 'text': [], 'label': [], 'image_id': [], 'split': []
            })
            self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
