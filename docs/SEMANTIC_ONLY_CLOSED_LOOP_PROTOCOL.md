# Routing-V2 Semantic-only Closed-loop Baseline

This baseline isolates Stage-1 Semantic routing.

For every action chunk, the shared Base-VLM Semantic AE bank ranks all skills currently available in the CL stage, and the Semantic Top-1 skill is executed. Ground-truth task ID is used only by LIBERO to choose the environment/task and by the server for diagnostics.

Dynamics verification is hard-disabled with `delta=0` and `lambda_max=0`. Since `e1 <= e2`, the normalized Semantic gap is non-negative and no sample can activate Stage-2. Thus no Base-WM future or Dynamics AE forward is performed per chunk.

The selected skill executes through its complete task-specific Skill Path, including its task-specific VLM/QFormer/LaWM adapters and Action Expert.
