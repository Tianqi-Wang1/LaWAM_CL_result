#!/usr/bin/env python3
LAWM_TOTAL = 2_555_179_000
rows = [
    ("A1 Last4 Dense + Conditioning-r128", 64_368_640),
    ("A2 Last8 Dense + Conditioning-r128", 122_085_376),
    ("A3 Last4 Dense + Text-LoRA-r32", 77_639_680),
    ("A4 Last4 Dense + Conditioning-r128 + Text-LoRA-r32", 84_291_584),
    ("A5 Text-LoRA-r128 + Action-DiT16-LoRA-r128", 116_916_224),
]
print(f"{'variant':58s} {'params':>12s} {'LaWAM %':>9s} {'bf16 MiB':>10s}")
for name,n in rows:
    print(f"{name:58s} {n:12,d} {100*n/LAWM_TOTAL:8.3f}% {2*n/1024**2:10.1f}")
