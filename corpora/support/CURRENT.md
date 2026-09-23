# Current Operational Notes
*Last updated: 2026-08-14*

## Active advisories
- Release 4.2 is fully deployed to production. Manual data-fix workarounds written
  before 2026-06-01 should be checked against the release notes before use.
- `invoice_status_flag` is deprecated. Use `invoice_state` (see reference/schema-billing.md).

## Escalation
L3 escalation goes through the platform queue with a completed impact summary.
Do not escalate without a reproduction or a documented data-only root cause.
