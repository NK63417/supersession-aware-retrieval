---
title: "RB-022: Refund Amount Mismatch"
last-updated: 2026-05-11
---
## Symptom
The refund shown to the customer does not match the ledger total.

## Resolution
Almost always a rounding difference on multi-line refunds. Recalculate from the
line items, correct the ledger total with a data fix, and annotate the ticket with
the recalculated figure. Escalate only if the line items themselves disagree.
