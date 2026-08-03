"""The harness: the ReAct loop and the types describing a run.

Owns `Harness`/`AsyncHarness`, the effect protocol they are driven by,
`RunResult`, run logging, and the exception taxonomy.

Deliberately knows nothing about the data domain: no pandas, no `SessionCache`,
no interpreter. What a tool *returns* and what a run's final state *is* are
supplied by whoever builds the harness, through `RunEnvironment`. That is what
makes the loop reusable for a domain other than data analysis.

May import `llm`. May not import `data` or `app`.
"""
