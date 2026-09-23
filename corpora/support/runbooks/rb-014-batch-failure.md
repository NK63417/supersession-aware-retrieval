---
title: "RB-014: Nightly Batch Job Failure — triage"
last-updated: 2026-06-22
---
## Triage order
1. Check the scheduler dashboard for the failed step and its exit code.
2. Look for a matching known error in the ticket history before investigating fresh.
3. If the failure is data-shaped (a single bad record blocking the run), isolate the
   record and apply the documented data fix.
4. If the failure is code-shaped (stack trace, null dereference, regression after a
   release), escalate to L3 with the trace and the failing record id.
