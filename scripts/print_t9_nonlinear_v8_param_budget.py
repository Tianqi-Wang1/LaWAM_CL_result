#!/usr/bin/env python3
TOTAL=2_555_179_000
rows=[
    ("B1 Last4+Cond+NL[0:11]",70_660_096),
    ("B2 B1+TextLoRA-r32",90_583_040),
    ("B3 Last8+Cond+NL[0:7]",126_279_680),
    ("B4 Cond+NL[0:15]",15_040_512),
]
for name,n in rows:
    print(f"{name:34s} {n:12,d} params  {100*n/TOTAL:6.3f}% of LaWAM")
