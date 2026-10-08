| Method | queries | accuracy | accuracy where SAM 3 is empty | SAM 3 empty share |
|---|---|---|---|---|
| SAM 3 alone | 323 | 0.381 | 0.102 | 0.49 |
| SAM 3 alone (same 242 queries) | 242 | 0.393 | 0.110 | 0.49 |
| SAM 3, else Z2 pseudo-label (leave-category-out) | 242 | 0.393 | 0.110 | 0.49 |
| SAM 3, else supervised (leave-category-out, reference) | 242 | 0.413 | 0.153 | 0.49 |
| SAM 3, else Z2 pseudo-label (leave-object-out) | 323 | 0.384 | 0.108 | 0.49 |
| SAM 3, else Z1 ShapeNetPart-trained | 101 | 0.554 | 0.222 | 0.27 |
| SAM 3 alone (Z1 types) | 101 | 0.545 | 0.185 | 0.27 |
| SAM 3, else Z1 (non-ShapeNet categories) | 72 | 0.569 | 0.240 | 0.35 |
| SAM 3 alone (Z1 types, non-ShapeNet categories) | 72 | 0.556 | 0.200 | 0.35 |

SAM 3, else Z1 ShapeNetPart-trained vs SAM 3 alone (Z1 types): +0.010 [-0.020, +0.040] (queries changed: +2 / -1)

SAM 3, else Z1 (non-ShapeNet categories) vs SAM 3 alone (Z1 types, non-ShapeNet categories): +0.014 [-0.028, +0.069] (queries changed: +2 / -1)

SAM 3, else supervised (leave-category-out, reference) vs SAM 3 alone (same 242 queries): +0.021 [-0.017, +0.058] (queries changed: +13 / -8)
