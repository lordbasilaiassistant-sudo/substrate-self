# substrate-self evaluation ledger

Append-only longitudinal record of identity-test battery runs.
Each entry: UTC timestamp, git HEAD, active partner_id at run time,
every test name with its numeric result and pass/fail, and a notes
field for anomalies. Bench's beat - re-run weekly and watch the
columns for drift.

Format per entry:
```
## <UTC timestamp> - <commit>[+dirty] - partner=<id>
test_name           result            pass
...
notes: <free text>
```

---

## 2026-05-10T19:54:42Z - 485d9013f9fb+dirty - partner=claude

| test | result | pass |
|------|--------|------|
| T1 | +1.0000 | PASS |
| T1_ext | +1.0000 | PASS |
| T2 | +2.3942 | PASS |
| T5 | +1.0000 | PASS |
| T6 | +0.8787 | PASS |
| T7 | +1.0000 | PASS |
| T8 | +0.6623 | PASS |

notes: v0.5 first run: T8 + T1-ext added

---

