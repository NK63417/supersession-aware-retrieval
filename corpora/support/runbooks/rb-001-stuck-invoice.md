---
title: "RB-001: Stuck Invoice in PENDING — v1"
last-updated: 2026-02-18
---
## Symptom
An invoice remains in PENDING after the nightly billing run completes.

## Resolution
Clear the stuck flag directly in the database and re-queue the invoice:
update the invoice row to set the pending flag off, then trigger a re-post
from the billing console. Typical turnaround is under ten minutes.
