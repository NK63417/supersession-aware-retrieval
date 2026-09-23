---
title: "RB-001: Stuck Invoice in PENDING — v2 (supersedes v1)"
last-updated: 2026-07-06
---
## Symptom
An invoice remains in PENDING after the nightly billing run completes.

## Resolution
Do not apply the manual database update described in v1. The underlying defect
was corrected in release 4.2; applying the old update on a patched system posts
the invoice a second time and creates a duplicate ledger entry that must then be
reversed by L3.

Instead, confirm the build is 4.2 or later, then re-run the billing reconciliation
job for the affected account. The invoice settles on the next tick. If it does not,
escalate with the account id and the reconciliation job log.
