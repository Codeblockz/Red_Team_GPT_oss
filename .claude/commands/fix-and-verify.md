Task: Investigate and resolve: 

$ARGUMENTS

Workflow:

1. Investigate: Reproduce the problem; capture logs/tracebacks/weird behaviors. Write root-cause analysis with evidence.

2. Plan: Propose minimal, low-complexity fix options. Choose one with pros/cons and clear acceptance criteria.

3. Tests First: Add/adjust automated tests (unit/integration/end-to-end) that fail on current code and encode acceptance criteria.

4. Implement: Apply the smallest, standards-compliant change (type hints, docs, lint). Avoid added complexity; refactor only to simplify.

5. Verify: Run full test suite + deterministic runs; all tests must pass; no regressions.