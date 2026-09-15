"""bitbank コマンドの薄いラッパ。

- 応答は必ず --format=json --machine のエンベロープとして読む
- 取得した値には出典コマンドと取得時刻を添える（memory-policy.md）
- 禁止コマンドは組み立てない（CLAUDE.md / risk-policy.md）
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import Config, REPO_ROOT
from . import timeutil

DEFAULT_TIMEOUT_SEC = 30


class CliError(Exception):
    """CLI が異常終了した、応答を解釈できない、または口座が未初期化。"""

    def __init__(self, message: str, *, cmd: str, not_initialized: bool = False) -> None:
        super().__init__(message)
        self.cmd = cmd
        self.not_initialized = not_initialized


class ForbiddenCommand(Exception):
    """実資金・実績破壊のコマンドを組み立てようとした。"""


@dataclass(frozen=True)
class Source:
    """判断に使った数値の出典。"""

    cmd: str
    at: str

    def as_dict(self) -> dict[str, str]:
        return {"cmd": self.cmd, "at": self.at}


@dataclass(frozen=True)
class Response:
    data: Any
    source: Source
    warnings: tuple[str, ...] = ()


Runner = Callable[[Sequence[str], dict[str, str], int], "subprocess.CompletedProcess[str]"]


def _default_runner(
    argv: Sequence[str], env: dict[str, str], timeout: int
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603
        list(argv),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@dataclass
class Client:
    """1回の実行で使う CLI クライアント。"""

    config: Config
    runner: Runner = _default_runner
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- 低レベル ---

    def _assert_allowed(self, args: Sequence[str]) -> None:
        line = " ".join([self.config.cli_command, *args])
        for forbidden in self.config.forbidden:
            if line.startswith(forbidden):
                raise ForbiddenCommand(f"禁止されたコマンドです: {forbidden}")

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        state = Path(self.config.state_path)
        if not state.is_absolute():
            state = REPO_ROOT / state
        env["BITBANK_PAPER_STATE_PATH"] = str(state)
        return env

    def call(self, *args: str) -> Response:
        """bitbank コマンドを1つ実行し、エンベロープの data を返す。"""
        self._assert_allowed(args)
        argv = [self.config.cli_command, *args, *self.config.global_flags]
        cmd = " ".join(argv)
        try:
            proc = self.runner(argv, self._env(), self.timeout_sec)
        except FileNotFoundError as exc:
            raise CliError(f"{self.config.cli_command} が見つかりません", cmd=cmd) from exc
        except subprocess.TimeoutExpired as exc:
            raise CliError(f"{self.timeout_sec} 秒で応答がありませんでした", cmd=cmd) from exc

        at = timeutil.to_iso(timeutil.now(self.config.timezone))
        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise CliError(f"応答が空です (exit={proc.returncode})", cmd=cmd)
        try:
            envelope = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise CliError(f"応答を JSON として解釈できません: {exc}", cmd=cmd) from exc
        if not isinstance(envelope, dict) or "success" not in envelope:
            raise CliError("応答がエンベロープ形式ではありません", cmd=cmd)
        if not envelope.get("success"):
            message = str(envelope.get("error", "(理由なし)"))
            raise CliError(
                message, cmd=cmd, not_initialized="not initialized" in message.lower()
            )

        source = Source(cmd=cmd, at=at)
        self.sources.append(source)
        warnings = tuple(
            line.strip() for line in (proc.stderr or "").splitlines() if line.strip()
        )
        self.warnings.extend(warnings)
        return Response(data=envelope.get("data"), source=source, warnings=warnings)

    # --- 観測 ---

    def status(self) -> Response:
        return self.call("status")

    def circuit_break(self, pair: str) -> Response:
        return self.call("circuit-break", pair)

    def ticker(self, pair: str) -> Response:
        return self.call("ticker", pair)

    def candles(
        self,
        pair: str,
        candle_type: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> Response:
        args = ["candles", pair, f"--type={candle_type}"]
        # 期間指定は --from と --to を必ず対で渡す（片方だけは CLI がエラーにする）。
        if date_from is not None and date_to is not None:
            args.extend([f"--from={date_from}", f"--to={date_to}"])
        return self.call(*args)

    def pairs(self) -> Response:
        return self.call("pairs")

    # --- 口座 ---

    def paper_assets(self) -> Response:
        return self.call("paper", "assets")

    def paper_pnl(self, pair: str) -> Response:
        return self.call("paper", "pnl", f"--pair={pair}")

    def paper_active_orders(self) -> Response:
        return self.call("paper", "active-orders")

    def paper_trade_history(self) -> Response:
        return self.call("paper", "trade-history")

    # --- 実行 ---

    def paper_tick(self) -> Response:
        # --pair を付けない。部分 tick は lastTickAt を進めないため、
        # 遡り区間が伸び続けて 24 時間の上限に当たる。
        return self.call("paper", "tick")

    def paper_create_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        amount: str,
        price: str | None = None,
    ) -> Response:
        args = [
            "paper",
            "create-order",
            f"--pair={pair}",
            f"--side={side}",
            f"--type={order_type}",
            f"--amount={amount}",
        ]
        if price is not None:
            args.append(f"--price={price}")
        return self.call(*args)

    def paper_cancel_order(self, order_id: str) -> Response:
        return self.call("paper", "cancel-order", f"--id={order_id}")


def is_already_filled(error: CliError) -> bool:
    """取消が「すでに約定していた」で返ったか（risk-policy.md「発注と取消の競合」）。"""
    message = str(error).lower()
    return "not found" in message and "already filled" in message
