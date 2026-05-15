#!/usr/bin/env python3
"""Unit tests and optional dev server launcher for the BixBench explorer."""
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from app import (  # noqa: E402
    _benchmark_metadata_from_row,
    _merge_benchmark_into_v1_5,
    _parse_str_list_field,
    _scalar_or_none,
    app,
    data,
    load_data,
)


class TestBixBenchFieldParsers(unittest.TestCase):
    def test_parse_list_from_python_list(self):
        self.assertEqual(_parse_str_list_field(['a', 'b']), ['a', 'b'])

    def test_parse_list_from_literal_string(self):
        raw = "['7.82E-05', '0.0003', '1.84E-05']"
        self.assertEqual(
            _parse_str_list_field(raw),
            ['7.82E-05', '0.0003', '1.84E-05'],
        )

    def test_parse_list_from_numpy_like(self):
        class Arr:
            def tolist(self):
                return ['x', 'y']

        self.assertEqual(_parse_str_list_field(Arr()), ['x', 'y'])

    def test_parse_list_empty_returns_none(self):
        self.assertIsNone(_parse_str_list_field([]))
        self.assertIsNone(_parse_str_list_field(None))
        self.assertIsNone(_parse_str_list_field(''))

    def test_scalar_or_none(self):
        self.assertIsNone(_scalar_or_none(float('nan')))
        self.assertEqual(_scalar_or_none('  hello '), 'hello')
        self.assertIsNone(_scalar_or_none('   '))


class TestBenchmarkMerge(unittest.TestCase):
    def test_merge_attaches_ideal_and_distractors(self):
        data['v1_5'] = pd.DataFrame([
            {
                'uuid': 'bix-1-q1',
                'question': 'Q?',
                'target': 'C',
                'eval_type': 'mcq-no-refusal',
                'model': 'gpt-4o',
                'correct': True,
                'evaluation_mode': 'str_verifier',
                'predicted': 'C',
                'grade': 1,
            },
        ])
        benchmark = pd.DataFrame([
            {
                'question_id': 'bix-1-q1',
                'ideal': '0.001',
                'distractors': ['0.002', '0.003', '0.004'],
                'hypothesis': 'H',
                'result': 'R',
                'answer': True,
                'paper': 'https://example.com',
            },
        ])
        _merge_benchmark_into_v1_5(benchmark)
        row = data['v1_5'].iloc[0]
        meta = _benchmark_metadata_from_row(row)
        self.assertEqual(meta['ideal'], '0.001')
        self.assertEqual(meta['distractors'], ['0.002', '0.003', '0.004'])


class TestQuestionDetailRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data['original'] = pd.DataFrame()
        data['v1_5'] = pd.DataFrame([
            {
                'uuid': 'bix-1-q1',
                'question': 'Test question?',
                'target': 'C',
                'eval_type': 'mcq-no-refusal',
                'model': 'gpt-4o',
                'correct': True,
                'evaluation_mode': 'str_verifier',
                'predicted': 'C',
                'grade': 1,
                'sure': True,
                'ideal': '0.001',
                'distractors': ['0.002', '0.003', '0.004'],
            },
            {
                'uuid': 'bix-1-q1',
                'question': 'Test question?',
                'target': 'C',
                'eval_type': 'mcq-refusal',
                'model': 'claude-3.5-sonnet',
                'correct': False,
                'evaluation_mode': 'str_verifier',
                'predicted': 'A',
                'grade': 0,
                'sure': False,
                'ideal': '0.001',
                'distractors': ['0.002', '0.003', '0.004'],
            },
        ])

    def test_question_page_renders_mcq_pool(self):
        client = app.test_client()
        resp = client.get('/questions/bix-1-q1')
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn('MCQ answer pool', body)
        self.assertIn('0.002', body)
        self.assertIn('shuffled', body)
        self.assertIn('Insufficient information to answer the question', body)


if __name__ == '__main__':
    if '--serve' in sys.argv:
        print('Loading data...')
        load_data()
        print('Data loaded successfully!')
        port = int(os.environ.get('PORT', '5000'))
        debug = os.environ.get('FLASK_DEBUG') == '1'
        print(f'\nStarting Flask server on http://localhost:{port}')
        app.run(debug=debug, host='0.0.0.0', port=port)
    else:
        unittest.main()
