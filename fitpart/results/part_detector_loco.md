Leave-one-category-out over 15 part names shared by >= 2 categories: base, blade, body, button, cap, door, drawer, handle, leg, lid, screen, seat, spout, switch, wheel

| Detector | queries | acc (IoU>=0.5, part or instance) | mean IoU with GT part |
|---|---|---|---|
| SAM 3 (union score) | 242 | 0.331 | 0.268 |
| geometry | 241 | 0.237 | 0.210 |
| geometry+topology | 241 | 0.286 | 0.237 |
| geometry+topology+chebyshev | 241 | 0.266 | 0.222 |
| all + SAM 3 + SigLIP scores (fusion) | 241 | 0.299 | 0.251 |
| SAM 3, else geometry+topology (SAM 3 max < 0.05) | 241 | 0.390 | 0.333 |
| SAM 3, else geometry+topology (SAM 3 max < 0.15) | 241 | 0.382 | 0.316 |
| SAM 3, else geometry+topology (SAM 3 max < 0.3) | 241 | 0.361 | 0.300 |

| Part | n | SAM 3 (union score) | geometry | geometry+topology | geometry+topology+chebyshev | all + SAM 3 + SigLIP scores (fusion) | SAM 3, else geometry+topology (SAM 3 max < 0.05) | SAM 3, else geometry+topology (SAM 3 max < 0.15) | SAM 3, else geometry+topology (SAM 3 max < 0.3) |
|---|---|---|---|---|---|---|---|---|---|
| button | 40 | 0.10 | 0.03 | 0.07 | 0.15 | 0.05 | 0.10 | 0.07 | 0.07 |
| handle | 37 | 0.46 | 0.24 | 0.24 | 0.22 | 0.41 | 0.49 | 0.46 | 0.41 |
| lid | 35 | 0.46 | 0.54 | 0.63 | 0.63 | 0.60 | 0.57 | 0.63 | 0.60 |
| door | 32 | 0.03 | 0.00 | 0.16 | 0.09 | 0.12 | 0.16 | 0.19 | 0.16 |
| wheel | 19 | 0.47 | 0.26 | 0.21 | 0.16 | 0.26 | 0.42 | 0.42 | 0.42 |
| leg | 13 | 0.77 | 0.23 | 0.38 | 0.38 | 0.46 | 0.77 | 0.62 | 0.54 |
| switch | 11 | 0.00 | 0.36 | 0.27 | 0.09 | 0.27 | 0.27 | 0.27 | 0.27 |
| body | 10 | 0.00 | 0.20 | 0.20 | 0.20 | 0.00 | 0.20 | 0.20 | 0.20 |
| blade | 8 | 0.62 | 0.25 | 0.25 | 0.25 | 0.38 | 0.62 | 0.62 | 0.62 |
| base | 8 | 0.88 | 0.62 | 0.75 | 0.50 | 0.62 | 0.88 | 0.88 | 0.75 |
| seat | 8 | 0.38 | 0.62 | 0.75 | 0.62 | 0.50 | 0.38 | 0.38 | 0.50 |
| spout | 8 | 0.88 | 0.12 | 0.12 | 0.12 | 0.25 | 0.88 | 0.88 | 0.88 |
| screen | 8 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| drawer | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cap | 2 | 0.00 | 0.00 | 0.50 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
