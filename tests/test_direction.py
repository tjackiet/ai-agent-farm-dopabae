"""方向の出どころ。対照群が契約を満たし、失敗が檻に伝わること。"""

from __future__ import annotations

import dataclasses
import unittest

from dopabae import direction as d
from tests.helpers import load_config


class DirectionTest(unittest.TestCase):
    def test_always_approach(self):
        result = d.always_approach()
        self.assertTrue(result.ok)
        self.assertEqual(result.value, d.APPROACH)

    def test_random_is_reproducible_with_seed(self):
        a = d.random_direction(42, "2026-09-15T09:00:00+09:00")
        b = d.random_direction(42, "2026-09-15T09:00:00+09:00")
        self.assertEqual(a.value, b.value)
        self.assertEqual(a.seed, 42)
        self.assertIn(a.value, d.DIRECTIONS)

    def test_random_without_seed_records_the_seed_it_used(self):
        result = d.random_direction(None, "run")
        self.assertIsNotNone(result.seed)
        again = d.random_direction(result.seed, "run")
        self.assertEqual(result.value, again.value)

    def test_random_covers_all_three_directions(self):
        seen = {d.random_direction(1, f"run-{i}").value for i in range(200)}
        self.assertEqual(seen, set(d.DIRECTIONS))

    def test_resolve_follows_config(self):
        cfg = load_config()
        self.assertEqual(d.resolve(dataclasses.replace(cfg, direction_source="always_approach"), "r").value, d.APPROACH)
        self.assertTrue(d.resolve(dataclasses.replace(cfg, direction_source="random", direction_random_seed=7), "r").ok)

    def test_fly_is_not_implemented_yet_and_fails_safely(self):
        cfg = dataclasses.replace(load_config(), direction_source="fly")
        result = d.resolve(cfg, "r")
        self.assertFalse(result.ok)
        self.assertIsNone(result.value)
        self.assertIn("Phase 3", result.error)

    def test_as_dict_has_every_field(self):
        self.assertEqual(
            set(d.always_approach().as_dict()), {"source", "value", "reason", "error", "seed"}
        )
