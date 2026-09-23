---
title: "Reference: billing schema — key fields"
last-updated: 2026-08-04
---
## Invoice fields
- `invoice_state` — current lifecycle state (PENDING, POSTED, SETTLED, VOID).
  This is the field to read and filter on.
- `invoice_status_flag` — DEPRECATED as of release 4.2. Retained for historical rows
  only; it is no longer maintained by the billing run. Lookups filtering on this
  field will silently miss recent invoices.
- `account_id` — join key to the accounts table.
