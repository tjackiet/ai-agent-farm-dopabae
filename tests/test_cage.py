"""檻。ハエがどう答えても、agent.yaml の値で最悪ケースが決まること。"""

from __future__ import annotations

import dataclasses
import unittest
from decimal import Decimal

from dopabae import cage
from dopabae.direction import APPROACH, AVOID, NONE
from tests import helpers
from tests.helpers import direction, guards, market, open_order, pair_spec, state

NOW = helpers.at(helpers.NOW_ISO)


class CageTest(unittest.TestCase):
    def setUp(self):
        self.cfg = helpers.load_config()

    def decide(self, st, dr, cfg=None, mk=None, gd=None, spec=None, stopped=None):
        return cage.decide(cfg or self.cfg, gd or guards(), mk or market(), spec or pair_spec(), st, dr, NOW, stopped)

    # --- ガード ---

    def test_missing_observation_holds(self):
        result = cage.decide(self.cfg, guards(), None, pair_spec(), state(), direction(), NOW)
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.GUARD))

    def test_exchange_and_circuit_and_stale_data_hold(self):
        for gd, mk in ((guards(exchange="MAINTENANCE"), market()), (guards(circuit="CIRCUIT_BREAK"), market()), (guards(), market(age_sec=600))):
            result = self.decide(state(), direction(), gd=gd, mk=mk)
            self.assertEqual((result.action, result.cage), (cage.HOLD, cage.GUARD))
            self.assertEqual(result.place, ())

    def test_position_mismatch_holds(self):
        result = self.decide(state(position="0.01", avg_cost="14000000", mismatch=True), direction())
        self.assertEqual(result.cage, cage.GUARD)

    # --- 方向が無い ---

    def test_no_direction_holds_and_cancels_stale_orders(self):
        st = state(pending_buy=[open_order("b1", "buy", "14600000", "0.001")])
        result = self.decide(st, direction(value=None, error="見えなかった"))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.NO_DIRECTION))
        self.assertEqual(result.cancel, ("b1",))
        self.assertEqual(result.place, ())

    # --- APPROACH ---

    def test_approach_buys_fixed_budget_at_bid(self):
        result = self.decide(state(), direction(APPROACH))
        self.assertEqual((result.action, result.cage), (cage.BUY, cage.BOUGHT))
        order = result.place[0]
        self.assertEqual((order.side, order.order_type), ("buy", "limit"))
        self.assertEqual(order.price, market().bid)
        budget = Decimal(str(self.cfg.budget_jpy_per_order))
        self.assertLessEqual(order.amount * order.price, budget)
        self.assertGreater(order.amount * order.price, budget - order.price * pair_spec().unit_amount)

    def test_approach_with_position_buys_more_until_cap(self):
        st = state(position="0.03", avg_cost="14000000", cash="580000")
        result = self.decide(st, direction(APPROACH))
        self.assertEqual(result.cage, cage.BOUGHT)

    def test_position_cap_is_a_wall(self):
        # 取得原価 800,000（= 初期資金の 80%）に達している。
        st = state(position="0.05", avg_cost="16000000", cash="200000")
        result = self.decide(st, direction(APPROACH))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.WALL))

    def test_cash_reserve_is_a_wall(self):
        # 使える現金が初期資金の 10% ちょうど。ここから先は使えない（総資産は減っていない）。
        st = state(cash="1000000", cash_available="100000")
        result = self.decide(st, direction(APPROACH))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.WALL))

    def test_budget_never_exceeds_per_order_max(self):
        cfg = dataclasses.replace(self.cfg, budget_jpy_per_order=10_000_000.0)
        result = self.decide(state(), direction(APPROACH), cfg=cfg)
        order = result.place[0]
        self.assertLessEqual(order.amount * order.price, Decimal(str(cfg.per_order_max_jpy)))

    def test_cooldown_stops_approach(self):
        st = state(cooldown_until="2026-09-15T09:10:00+09:00")
        result = self.decide(st, direction(APPROACH))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.COOLDOWN))

    def test_daily_fill_limit_is_a_wall(self):
        st = state(fills_today=self.cfg.max_fills_per_day)
        result = self.decide(st, direction(APPROACH))
        self.assertEqual(result.cage, cage.WALL)

    def test_hibernation_stops_new_buys_but_avoid_can_still_sell(self):
        st = state(position="0.01", avg_cost="14700000", cash="700000", equity="840000")  # -16%
        self.assertEqual(cage.state_name(self.cfg, st), "HIBERNATING")
        walled = self.decide(st, direction(APPROACH))
        self.assertEqual((walled.action, walled.cage), (cage.HOLD, cage.WALL))
        sold = self.decide(st, direction(AVOID))
        self.assertEqual((sold.action, sold.cage), (cage.SELL, cage.SOLD))

    def test_stale_buy_order_is_cancelled_before_replacing(self):
        st = state(pending_buy=[open_order("b1", "buy", "14600000", "0.001")], cash="1000000", cash_available="985400")
        result = self.decide(st, direction(APPROACH))
        self.assertEqual(result.cancel, ("b1",))
        self.assertEqual(result.cage, cage.BOUGHT)

    # --- AVOID ---

    def test_avoid_sells_all_at_ask(self):
        st = state(position="0.01", avg_cost="14000000", cash="860000")
        result = self.decide(st, direction(AVOID))
        self.assertEqual((result.action, result.cage), (cage.SELL, cage.SOLD))
        order = result.place[0]
        self.assertEqual((order.side, order.order_type, order.price), ("sell", "limit", market().ask))
        self.assertEqual(order.amount, Decimal("0.01"))

    def test_avoid_without_position_does_nothing_but_is_recorded(self):
        result = self.decide(state(), direction(AVOID))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.AVOID_FLAT))
        self.assertEqual(result.place, ())

    def test_avoid_reuses_amount_locked_by_stale_sell(self):
        st = state(position="0.01", avg_cost="14000000", cash="860000", base_available="0",
                   pending_sell=[open_order("s1", "sell", "15000000", "0.01")])
        result = self.decide(st, direction(AVOID))
        self.assertEqual(result.cancel, ("s1",))
        self.assertEqual(result.place[0].amount, Decimal("0.01"))

    # --- NONE ---

    def test_none_holds_and_cancels_stale(self):
        st = state(position="0.01", avg_cost="14000000", cash="860000",
                   pending_sell=[open_order("s1", "sell", "15000000", "0.01")])
        result = self.decide(st, direction(NONE))
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.STILL))
        self.assertEqual(result.cancel, ("s1",))

    # --- 檻の諦めは方向より先 ---

    def test_time_stop_exits_regardless_of_direction(self):
        st = state(position="0.01", avg_cost="14000000", cash="860000", age_days=self.cfg.time_stop_days + 0.5)
        for dr in (direction(APPROACH), direction(AVOID), direction(NONE), direction(None, error="x")):
            result = self.decide(st, dr)
            self.assertEqual((result.action, result.cage), (cage.SELL, cage.EXIT))
            self.assertEqual(result.place[0].order_type, "market")

    def test_forced_exit_on_drawdown(self):
        st = state(position="0.01", avg_cost="14700000", cash="600000", equity="740000")  # -26%
        result = self.decide(st, direction(APPROACH))
        self.assertEqual(cage.state_name(self.cfg, st), "HALTED")
        self.assertEqual((result.action, result.cage), (cage.SELL, cage.EXIT))

    def test_long_stop_recovers_by_cancelling_everything(self):
        st = state(pending_buy=[open_order("b1", "buy", "14600000", "0.001")])
        result = self.decide(st, direction(APPROACH), stopped=self.cfg.stale_tick_hours + 1)
        self.assertEqual((result.action, result.cage), (cage.HOLD, cage.RECOVER))
        self.assertEqual(result.cancel, ("b1",))
        self.assertEqual(result.place, ())

    # --- 現物のみ ---

    def test_never_places_a_sell_larger_than_the_position(self):
        st = state(position="0.0003", avg_cost="14000000", cash="990000")
        result = self.decide(st, direction(AVOID))
        self.assertLessEqual(result.place[0].amount, Decimal("0.0003"))
