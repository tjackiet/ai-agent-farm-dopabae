"""1周（観測 → 方向 → 檻 → 発注 → 記録）のオーケストレーション。

エントリポイントはここだけとする。Claude Code から呼ぶのもこの1コマンドで、
`bitbank` コマンドを外側で組み立てない。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from . import cage, cli, config as config_module, direction as direction_module, journal, observe
from . import state as state_module
from . import timeutil
from .orders import Executor, execute


@dataclass
class Cycle:
    """1周の結果。JSON にしてそのまま標準出力へ流す。"""

    run_id: str
    action: str
    state: str
    reason: str
    dry_run: bool
    direction: dict | None
    cage: str | None
    orders: list[dict[str, Any]]
    sources: list[dict[str, str]]
    warnings: list[str]
    error: str | None = None
    status_written: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "action": self.action,
            "state": self.state,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "direction": self.direction,
            "cage": self.cage,
            "orders": self.orders,
            "sources": self.sources,
            "warnings": self.warnings,
            "error": self.error,
            "status_written": self.status_written,
        }


def _observe_account(client: cli.Client, cfg: config_module.Config, now: datetime, last_price, spec):
    """ペーパー口座を観測して状態を導出する。"""
    client.paper_tick()
    assets = client.paper_assets().data
    pnl = client.paper_pnl(cfg.pair).data
    orders = client.paper_active_orders().data
    history = client.paper_trade_history().data
    if not isinstance(assets, list) or not isinstance(orders, list) or not isinstance(history, list):
        raise ValueError("ペーパー口座の応答が配列ではありません")
    derived = state_module.derive(
        config=cfg,
        now=now,
        last_price=last_price,
        assets_rows=assets,
        pnl_report=pnl if isinstance(pnl, dict) else {},
        order_rows=orders,
        history_rows=history,
        unit_amount=spec.unit_amount if spec is not None else None,
    )
    trades = state_module.parse_trades(history, cfg.pair, cfg.timezone)
    return derived, trades


def run_once(
    cfg: config_module.Config,
    client: cli.Client,
    now: datetime,
    *,
    force_dry_run: bool = False,
    repo_root: Path | None = None,
    resolver=None,
) -> Cycle:
    """1周する。`resolver` は方向の出どころの差し替え口（テスト用）。"""
    dry_run = cfg.dry_run or force_dry_run
    run_id = timeutil.to_iso(now)
    started = time.monotonic()

    guards = observe.Guards(exchange_status=None, circuit_mode=None)
    market = None
    spec = None
    derived = None
    trades: tuple[state_module.Trade, ...] = ()
    error: str | None = None
    stopped_hours: float | None = None
    try:
        guards = observe.check_guards(client, cfg)
        market = observe.observe_market(client, cfg, now)
        spec = observe.observe_pair_spec(client, cfg)
        # ペーパー口座に触る前に測る。tick も lazy tick も lastTickAt を進めるため。
        stopped_hours = state_module.stopped_hours(cfg, now, repo_root)
        derived, trades = _observe_account(client, cfg, now, market.last, spec)
    except cli.CliError as exc:
        error = f"{exc}（{exc.cmd}）"
    except (ValueError, KeyError, TypeError) as exc:
        error = f"観測を解釈できません: {exc}"

    # 方向は観測が済んでから諮る。観測に失敗した回は諮らない（記録には null）。
    resolved: direction_module.Direction | None = None
    if error is not None:
        decision = cage.Decision(
            action=cage.HOLD,
            state=cage.state_name(cfg, derived),
            reason=f"観測に失敗したため何もしない: {error}",
            cage=cage.GUARD,
        )
    else:
        elapsed = time.monotonic() - started
        if elapsed > cfg.max_runtime_sec:
            decision = cage.Decision(
                action=cage.HOLD,
                state=cage.state_name(cfg, derived),
                reason=f"観測に {int(elapsed)} 秒かかり、実行時間の上限を超えた",
                cage=cage.GUARD,
            )
        else:
            resolved = (resolver or direction_module.resolve)(cfg, run_id)
            if resolved.error is not None:
                client.warnings.append(f"方向を得られませんでした: {resolved.error}")
            decision = cage.decide(cfg, guards, market, spec, derived, resolved, now, stopped_hours)

    order_records: list[dict[str, Any]] = []
    if error is None and (decision.cancel or decision.place):
        executor = Executor(client=client, pair=cfg.pair, dry_run=dry_run)
        try:
            order_records, aborted = execute(executor, decision.cancel, decision.place)
            if aborted is not None:
                decision = cage.Decision(
                    action=cage.HOLD, state=decision.state, reason=aborted, cage=decision.cage
                )
        except cli.CliError as exc:
            error = f"{exc}（{exc.cmd}）"
            decision = cage.Decision(
                action=cage.HOLD,
                state=decision.state,
                reason=f"発注または取消に失敗した: {error}",
                cage=decision.cage,
            )

    # 実際に注文を出した回は、スナップショットを取り直す。
    if error is None and not dry_run and order_records:
        try:
            derived, trades = _observe_account(client, cfg, now, market.last, spec)
        except (cli.CliError, ValueError, KeyError, TypeError) as exc:
            error = f"発注後の観測に失敗した: {exc}"

    direction_dict = resolved.as_dict() if resolved is not None else None
    record = journal.build_record(
        run_id=run_id,
        config_version=cfg.version,
        fingerprint=config_module.fingerprint(cfg.raw),
        state_label=decision.state,
        pair=cfg.pair,
        market=market,
        position_amount=float(derived.position.amount) if derived else None,
        avg_cost=float(derived.position.avg_cost_jpy)
        if derived and derived.position.avg_cost_jpy is not None
        else None,
        direction=direction_dict,
        cage=decision.cage,
        action=decision.action,
        reason=decision.reason,
        orders=order_records,
        sources=client.sources,
        warnings=client.warnings,
        error=error,
    )
    journal.append(cfg, now, record, root=repo_root)

    status_written = False
    if error is None:
        document = state_module.build_status(
            config=cfg,
            now=now,
            run_id=run_id,
            state_label=decision.state,
            market=market,
            state=derived,
            trades=trades,
            action=decision.action,
            reason=decision.reason,
            direction=direction_dict,
            price_source=market.source_cmd if market is not None else None,
            unit=spec.unit_amount if spec is not None else None,
        )
        root = repo_root if repo_root is not None else config_module.REPO_ROOT
        # 書き出し先は Git 管理外。リポジトリの status.yaml は見本として触らない。
        state_module.write_status(document, root / cfg.status_output)
        status_written = True

    return Cycle(
        run_id=run_id,
        action=decision.action,
        state=decision.state,
        reason=decision.reason,
        dry_run=dry_run,
        direction=direction_dict,
        cage=decision.cage,
        orders=order_records,
        sources=[s.as_dict() for s in client.sources],
        warnings=list(client.warnings),
        error=error,
        status_written=status_written,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m dopabae.run",
        description="ドパバエの判断を1周ぶん実行する（ペーパートレードのみ）",
    )
    parser.add_argument("--config", default=None, help="agent.yaml のパス")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="発注せず、組み立てた注文だけを出力する（agent.yaml より安全側にのみ倒せる）",
    )
    args = parser.parse_args(argv)

    try:
        cfg = config_module.load(args.config)
    except config_module.ConfigError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1

    now = timeutil.now(cfg.timezone)
    client = cli.Client(config=cfg)
    cycle = run_once(cfg, client, now, force_dry_run=args.dry_run)
    print(json.dumps(cycle.as_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
