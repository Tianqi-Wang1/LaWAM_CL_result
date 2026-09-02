# B2 closed-loop protocol

For every action chunk:

1. Compute the shared Base semantic anchor and Semantic-AE errors over the current CL skill bank.
2. If `C_sem >= delta`, execute Semantic Top-1 directly.
3. If `C_sem < delta`, take Semantic Top-2 candidates. For each candidate `k`:
   - activate candidate-specific VLM/Query/QFormer to obtain `z_k`;
   - disable candidate LaWM-LoRA and compute `h_hat_k = WM_Base(h_t, z_k)`;
   - score the B2 Dynamics-AE on `[h_t, h_hat_k-h_t]`.
4. Fuse pair-normalized Semantic and Dynamics errors with confidence-adaptive `lambda(C)`.
5. Activate the selected Skill's complete task-specific path and generate the real action chunk.

GT task ID is retained only for LIBERO environment selection and diagnostics; the router never consumes it.
