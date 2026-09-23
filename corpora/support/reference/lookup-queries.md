---
title: "Reference: common L2 lookup queries"
last-updated: 2026-01-22
---
## Find pending invoices for an account
Filter the invoice table on the account id and the pending status flag, ordered by
created date. Use this to confirm whether a reported stuck invoice is genuinely
stuck or simply awaiting the next nightly run.

## Find recent failed batch steps
Filter the job history table on a non-zero exit code within the last seven days.
