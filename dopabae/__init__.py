"""ドパバエ — ペーパートレードのエージェント本体。

方向（APPROACH / AVOID / NONE）はハエ脳（Phase 3 で追加）が出し、
数量・価格・上限・諦めは決定的コード（檻。`cage.py`）が `agent.yaml` の値で決める。
LLM は売買に関与しない（docs/IMPLEMENTATION_PLAN.md Phase 1）。
"""
