# Linux で15分ごとに回す

macOS は [`../launchd/local.dopabae.plist`](../launchd/local.dopabae.plist) を使う。
Linux では systemd の timer か cron を使う。二重起動は `dopabae/lock.py` が
腕ごとに防ぐので、どちらでも同じように動く。

## cron

```cron
*/15 * * * * cd /path/to/ai-agent-farm-dopabae && .venv/bin/python scripts/run_arms.py >> var/arms.log 2>> var/arms.err.log
```

cron の `PATH` には npm のグローバル配置先が入らないことが多い。`bitbank` が
見つからないと毎回 HOLD になるので、`crontab` の先頭で `PATH` を明示する。

```cron
PATH=/usr/local/bin:/usr/bin:/bin:/path/to/node/bin
```

## systemd

`~/.config/systemd/user/dopabae.service`:

```ini
[Unit]
Description=ドパバエの腕を1回ぶん回す

[Service]
Type=oneshot
WorkingDirectory=/path/to/ai-agent-farm-dopabae
Environment=PATH=/usr/local/bin:/usr/bin:/bin:/path/to/node/bin
ExecStart=/path/to/ai-agent-farm-dopabae/.venv/bin/python scripts/run_arms.py
```

`~/.config/systemd/user/dopabae.timer`:

```ini
[Unit]
Description=15分ごとにドパバエを回す

[Timer]
OnCalendar=*:0/15
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now dopabae.timer
systemctl --user list-timers dopabae.timer
```

`Persistent=true` は、止まっていた間に過ぎた回を復帰後に1度だけ実行する。
`paper tick` が遡れるのは24時間までなので、それを超えて止まっていた回は
檻が板の指値をすべて取り消してやり直す（`risk-policy.md`「運用を中断したあとの復帰」）。
