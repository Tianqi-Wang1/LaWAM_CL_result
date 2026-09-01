#!/usr/bin/env python3
LA_WAM = 2_555_179_000
TEXT_R32 = 19_922_944
LAST8 = 115_433_472
ACTION_R32 = 9_306_112
COND_R128 = 6_651_904
RAW_REPLAY = 2 * 256 * 256 * 3  # two uint8 RGB views; state/action/text are negligible here

rows = [
    ("E1 Last8 Dense + Text LoRA-r32", LAST8 + TEXT_R32),
    ("E3 Text LoRA-r32 + Action DiT16 LoRA-r32", TEXT_R32 + ACTION_R32),
    ("E7 Text LoRA-r32 + enc_vlm/AdaLN Adapter-r128", TEXT_R32 + COND_R128),
]
print("method,params,percent_lawam,bf16_MiB,raw_replay_equiv_samples")
for name,p in rows:
    print(f"{name},{p},{100*p/LA_WAM:.4f},{2*p/2**20:.2f},{2*p/RAW_REPLAY:.1f}")
