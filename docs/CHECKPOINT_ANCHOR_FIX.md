# Routing-V2 closed-loop checkpoint-anchor fix

## Root cause

The routing server is materialized from the first candidate checkpoint (`TEMPLATE_CKPT`, T6 in the CL-only bank), and advertises that path in WebSocket metadata. The evaluation client previously used the *true task* checkpoint (`GT_CKPT`) as `--args.pretrained-path`. `ModelClient` therefore rejected CL2/T7 and later cells before rollout because the metadata checkpoint paths differed.

This was not a routing/model error. CL1/T6 and CL2/T6 worked because client and server happened to use the same T6 checkpoint.

## Fix

- Use `TEMPLATE_CKPT` as the eval-client config/normalization/metadata anchor.
- Keep `TASK_ID` unchanged, so LIBERO still evaluates the true task (T7, T8, ...).
- The routing server still selects and executes the actual Skill Path per chunk; the client-side checkpoint is not used for action inference.
- Keep `GT_CKPT` only for diagnostics/path audit.
- Replace raw TCP readiness probing with server-log readiness to remove the harmless WebSocket handshake error.

All V2 skills were previously audited to share the Base normalization/action configuration, so a common client config anchor is appropriate.
