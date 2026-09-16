# AI Agent Farm — Dopabae

[![tests](https://github.com/tjackiet/ai-agent-farm-dopabae/actions/workflows/tests.yml/badge.svg)](https://github.com/tjackiet/ai-agent-farm-dopabae/actions/workflows/tests.yml)
[![security](https://github.com/tjackiet/ai-agent-farm-dopabae/actions/workflows/security.yml/badge.svg)](https://github.com/tjackiet/ai-agent-farm-dopabae/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ドパバエ（Dopabae）は、AIエージェントファームの2体目の試験エージェントです。

ショウジョウバエのコネクトーム（配線図）を LIF ニューロンでシミュレートし、
15分足のチャート画像を「目」に見せて、その反応を売買の**方向**（近づく／逃げる）にだけ使います。
いくら買うか・どこで諦めるかは決定的コード（檻）が担い、`bitbank-lab-cli` のペーパートレードで運用します。
**実取引は行いません。現物のみです。**

1体目のナンピノニクス（[tjackiet/ai-agent-farm-nampinonychus](https://github.com/tjackiet/ai-agent-farm-nampinonychus)）を型にした別個体です。
参考にした外部プロジェクトは [nftechie/stonkfly](https://github.com/nftechie/stonkfly) で、
モデル部分だけを参照し、実取引モードは持ち込みません。

## 目的

儲けることが目的ではありません。次の2つを観察します。

1. ハエ脳の反応が、決定的コードの判断に対して何らかの偏りを持つか
2. 報酬学習（ドーパミンによる可塑性）を足したとき、挙動がどう変わるか

ハエの方向が、同じ頻度でランダムに近づく／逃げるを出す対照群と区別できなければ、
ハエは「たまに買って、たまに売る乱数」と同じです。評価計画は `docs/DESIGN_MEMO.md` 4 章にあります。

## 現在の状態

Phase 1（檻と方向インタフェース）、Phase 2（画像の描画と光受容細胞へのマッピング）、
Phase 4 の評価基盤まで実装済み。
観測 → 画像 → 方向 → 檻 → 発注 → 記録が1周します。
方向の出どころは評価用の「常時 APPROACH」と「ランダム」だけで、ハエ脳（Phase 3）はまだありません。
実際のペーパー口座ではまだ回していません。実装の順序は `docs/IMPLEMENTATION_PLAN.md` を参照してください。

## 名前

**ドパバエ**（Dopabae）。2026-09-15 に人間が決めた。

| 要素 | 由来 |
| --- | --- |
| ドパ | ドーパミン。報酬学習で刺激するのは比喩ではなく、コネクトーム上のドーパミン細胞（PAM11）そのもの |
| バエ | ハエ。「映え」にも聞こえる |

命名の型はナンピノニクス（ナンピン＋ -onychus）と同じく「性質＋生物」。
実装の数値がそのままキャラクターになる。報酬の細胞は 15 個、嫌悪の細胞は 2 個なので、
喜びには敏感で痛みには鈍い。学習ルールは抑圧のみなので、ドパが出るたびに覚えるのは「やめかた」ばかり。

キャラクターデザインは `character-design.yaml` で扱う（未着手）。

## ファイル構成

```text
.
├── CLAUDE.md               # 毎回必ず守る不変のルール
├── README.md
├── LICENSE
├── agent.yaml              # エージェント定義。数値パラメータとバージョンの唯一の正（型だけ）
├── personality.md          # 性格・行動原則・話し方
├── strategy.md             # 檻の説明。方向をどう数量・価格・上限・諦めに落とすか
├── risk-policy.md          # リスク制約。性格と矛盾した場合はこちらが優先
├── memory-policy.md        # 記憶の構造と書き込みルール
├── status.yaml             # 状態のスキーマの見本（実行時は var/status.yaml）
├── dopabae/                # エージェント本体（Python 3）
├── scripts/
│   ├── make_arm.py         # 評価の腕（方向の出どころだけが違う設定）を作る
│   ├── run_arms.py         # 腕をまとめて1回ぶん回す（定期実行から呼ぶ）
│   ├── launchd/            # 15分ごとの定期実行（macOS）
│   └── systemd/            # 同（Linux。cron の例も）
├── requirements.txt        # Python の依存。PyYAML のみ
├── tests/                  # agent.yaml が不変ルールを満たすことの検証
├── .claude/settings.json   # 禁止コマンドのハーネス側での二重化
├── .github/
│   ├── workflows/          # テストとセキュリティ点検（GitHub Actions）
│   └── dependabot.yml
└── docs/
    ├── REPOSITORY_PLAN.md      # 担当範囲（実装範囲の唯一の正）
    ├── IMPLEMENTATION_PLAN.md  # 実装順序（Phase 1〜6）
    ├── DESIGN_MEMO.md          # 設計メモ（作業指示ではない）
    └── CONNECTOME_SURVEY.md    # 配線図の調査結果（判断材料。一部は未検証）
```

| ファイル                      | 役割                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `CLAUDE.md`                   | 毎回必ず守る不変のルール                                     |
| `agent.yaml`                  | エージェント定義。**数値パラメータとバージョンの唯一の正**。現在は型だけで、ハエ脳の値は候補値 |
| `personality.md`              | 性格・行動原則・話し方。売買の判断には使わない               |
| `strategy.md`                 | 檻の説明。方向をどう数量・価格・上限・諦めに落とすか         |
| `risk-policy.md`              | リスク制約。性格と矛盾した場合はこちらが優先                 |
| `memory-policy.md`            | 記憶の構造と書き込みルール                                   |
| `dopabae/summary.py` / `narrate.py` | 日次サマリと記録の言語化。売買の判断には関与しない      |
| `status.yaml`                 | 状態のスキーマの見本。実行では書き換えない                   |
| `dopabae/`                    | エージェント本体。観測・画像・マッピング・方向・檻・発注・記録（Phase 1〜2） |
| `scripts/make_arm.py`         | 評価の腕の設定を作る。発注はしない                           |
| `scripts/run_arms.py`         | 腕をまとめて1回ぶん回す。定期実行から呼ぶ                    |
| `scripts/launchd/` / `systemd/` | 15分ごとの定期実行の雛形                                   |
| `requirements.txt`            | Python の依存。PyYAML のみ。シミュレーションの依存は未確認の前提が確認できてから足す |
| `tests/`                      | 檻・方向・1周の流れ・`agent.yaml` の不変ルールの検証。外には出ない |
| `.github/workflows/`          | テスト（`tests.yml`）とセキュリティ点検（`security.yml`）。ナンピノニクスと同じ構成 |
| `docs/REPOSITORY_PLAN.md`     | 本リポジトリで実装してよい範囲                               |
| `docs/IMPLEMENTATION_PLAN.md` | 実装順序。未確認の前提（Python 3.11、C++17、メモリ 16 GB、配線図の選定とライセンス）もここに書く |
| `docs/DESIGN_MEMO.md`         | この個体の設計メモ。ナンピノニクス側から持ち込んだもの       |
| `docs/CONNECTOME_SURVEY.md`   | 配線図の調査結果（2026-09-15）。判断材料であり、一部は未検証 |
| `LICENSE`                     | MIT License                                                  |

## 構成

```text
15分足 (bitbank candles)
   │
   ▼
チャート画像を描く (320×180 RGB, 明るい背景)
   │
   ▼
光受容細胞へ入力 (R1–R6: 輝度, R8: 青/緑)
   │
   ▼
コネクトーム LIF シミュレーション（観測ごとに神経時間 0.5 秒）
   │
   ├──▶ 読み出し: 接近側 − 回避側 の発火率 → APPROACH / AVOID / NONE
   │
   └──▶ キノコ体 KC→MBON の重み（報酬学習で変わる唯一の場所）
              ▲
              │ 決済が確定した回だけ、確定損益 → PAM11（報酬）/ PPL101（嫌悪）
              │
檻 (決定的コード): 方向を受け取り、数量・価格・上限・諦めを agent.yaml で決めて発注する
```

## 守ること

- ペーパートレードのみ。`bitbank trade create-order` / `bitbank trade cancel-order` /
  `bitbank paper reset` は実行も生成もしない
- 判断できないときは HOLD。シミュレーションの失敗・入力欠損も HOLD
- ハエ脳が決めるのは方向だけ（`APPROACH` / `AVOID` / `NONE`）。数量・価格・上限・諦めは
  `agent.yaml` の値で檻が決める
- 現物のみ。売りから入る建玉は作らない
- 学習で変わるのはシナプス重みだけ。読み出しの閾値・対象ニューロン・報酬の閾値は
  `agent.yaml` に置き、ハエが書き換えない
- 観測していない値を書かない。秘密情報を残さない

詳細は `CLAUDE.md` を参照してください。

## セットアップ

必要なのは Python 3.11 以上と PyYAML だけです（シミュレーションの依存はまだ入れていません）。
2026-09-15 に macOS の Python 3.14 と `bitbank-lab-cli` 0.5.0 で 1 周を確認しています。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -t .
```

テストは外に出ません。`bitbank` も `claude` も呼ばず、資格情報も要りません。

### 1周を実行する

```bash
npm i -g bitbank-lab-cli
BITBANK_PAPER_STATE_PATH=var/paper-state.json bitbank paper init --jpy=1000000
.venv/bin/python -m dopabae.run --dry-run
```

`--dry-run` は発注せず、組み立てた注文だけを出力します。`agent.yaml` の `runtime.dry_run` も
既定で `true` です。ペーパー口座へ実際に出すかは、人間が 1 周の出力を見てから決めます。
実資金には、どちらでも影響しません（paper は公開 API しか叩きません）。

### 評価する

ハエの方向がランダムと区別できるかを見ます。過去の足での再実行（replay）はしません。
`bitbank paper` の約定判定を自前で真似ると、真似が本物とずれた分だけ評価が嘘になるためです。
代わりに、**方向の出どころだけが違う設定（腕）を並べて同時に走らせます。**

```bash
python3 scripts/make_arm.py cage-only --source always_approach
python3 scripts/make_arm.py fly       --source fly
python3 -m dopabae.evaluate --config arms/fly.yaml --config arms/cage-only.yaml
```

出るのは方向の頻度と偏り、檻の扱いの内訳、方向のあとの値動き（並べ替え検定つき）、成績です。
**結論は出しません。** 標本が少ないうちは、その旨を添えて数字だけを返します。

### 定期運用する

腕を 15 分ごとにまとめて回します。腕は同じ相場を見る必要があるので、**並行して**起動します。

```bash
.venv/bin/python scripts/run_arms.py          # arms/*.yaml を全部。無ければ agent.yaml
```

macOS は `scripts/launchd/local.dopabae.plist` の `__REPO__` と `__PATH__` を置き換えて
`~/Library/LaunchAgents` へ置きます。Linux は `scripts/systemd/README.md` に cron と
systemd の例があります。

**二重起動は防がれています。** 前の回がまだ走っている腕は、その回を飛ばします
（`skipped` として要約に出ます）。実行しないことによる機会損失は許容し、二重発注は許容しません。

各回の要約は `var/arms.log`、腕ごとの詳細は `var/arms/<腕>/run.log` に残ります。

## 未確認の前提

- Stonkfly の環境要件（Python 3.11、C++17 コンパイラ、メモリ 16 GB 推奨）を
  本リポジトリの環境で満たせるかは未確認
- 配線図のデータソースは **2026-09-16 に MaleCNS v1.0 を選定した**
  （`docs/CONNECTOME_SURVEY.md` 6.3）。ライセンスは同日、一次資料で
  **CC BY 4.0 と確認済み**（同 3.2）。データはリポジトリに含めず、
  取得 URL とチェックサムだけを置く
- シナプス数の閾値、切り出しの範囲、読み出しに使うニューロン群、静止発火率の
  扱いは**未決**。値は `agent.yaml` に置く（同 6 章）

## ライセンス

MIT License
