# Engineering Rules

Apply rules to every new or modified file.

## Design

- Follow SOLID strictly: single responsibility, open/closed, substitutable abstractions, small interfaces, dependency inversion.
- Keep layers separated. Inject dependencies. Avoid global state, hidden coupling, and duplicated logic.
- Prefer small composable units over inheritance-heavy designs.

## Code

- Comments are forbidden in production and test code: no inline, block, or explanatory comments.
- Use clear names, types, structure, and tests instead of comments.
- Every method or function must contain at most 20 physical lines.
- Extract helpers when a method exceeds 20 lines. Do not evade the limit with dense statements or compressed formatting.
- Keep formatting, linting, typing, and existing project conventions clean.

## Tests

- Every behavior change requires automated tests.
- Add unit tests for isolated logic and integration tests for component boundaries.
- Cover success paths, failure paths, edge cases, and regressions.
- Run relevant tests before handoff. Report failures with exact command and error.
- Do not weaken, delete, or skip tests to make a change pass.
