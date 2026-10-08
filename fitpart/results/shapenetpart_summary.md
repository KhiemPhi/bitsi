67 objects, 16 categories, theta=0.9 (class-mean)

| Variant | recall | miou_oracleK | miou_kmeans | miou_theta | n_seg_theta | sem_iou | sem_iou_sel | sel_acc | sem_iou_sel_sam3 | sem_iou_selmulti_sam3 | sel_acc_sam3 | sem_iou_sam3direct | sel_acc_sam3direct | sem_iou_sam3direct@0.3 | sel_acc_sam3direct@0.3 | acc:sem_iou_sel | acc:sem_iou_selmulti_sam3 | acc:sem_iou_sam3direct | acc:sem_iou_sam3direct@0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.753 | 0.582 | nan | 0.507 | 3.473 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:none | 0.775 | 0.590 | nan | 0.517 | 2.230 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:jump | 0.778 | 0.589 | nan | 0.519 | 2.292 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:single | 0.746 | 0.580 | nan | 0.505 | 2.516 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:extrema | 0.776 | 0.588 | nan | 0.538 | 2.438 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:axis | 0.774 | 0.573 | nan | 0.511 | 3.982 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo:global | 0.752 | 0.600 | nan | 0.414 | 26.587 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| sem only (lam=1) | 0.775 | 0.590 | nan | 0.560 | 4.156 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| geo only (lam=0) | 0.425 | 0.304 | nan | 0.282 | 1.406 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| mask finest | 0.764 | 0.562 | nan | 0.533 | 6.230 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| no long-range | 0.636 | 0.361 | nan | 0.317 | 12.281 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| combine: product | 0.752 | 0.569 | nan | 0.509 | 13.953 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| combine: min | 0.736 | 0.549 | nan | 0.515 | 11.382 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| product, geo:none | 0.775 | 0.590 | nan | 0.560 | 4.156 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |

Class-agnostic mIoU (mean #segments) vs merge threshold theta:

| Variant | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.93 | 0.96 |
|---|---|---|---|---|---|---|---|
| full | 0.295 (1.3) | 0.285 (1.4) | 0.306 (1.8) | 0.398 (2.2) | 0.507 (3.5) | 0.522 (6.2) | 0.481 (18.8) |
| geo:none | 0.284 (1.0) | 0.284 (1.0) | 0.286 (1.0) | 0.342 (1.4) | 0.517 (2.2) | 0.574 (3.2) | 0.545 (4.9) |
| geo:jump | 0.284 (1.0) | 0.284 (1.0) | 0.286 (1.0) | 0.342 (1.4) | 0.519 (2.3) | 0.573 (3.2) | 0.535 (5.1) |
| sem only (lam=1) | 0.342 (1.4) | 0.433 (1.7) | 0.517 (2.2) | 0.579 (3.0) | 0.560 (4.2) | 0.536 (5.2) | 0.379 (12.9) |
| combine: product | 0.387 (2.1) | 0.466 (2.8) | 0.506 (3.5) | 0.533 (6.1) | 0.509 (14.0) | 0.451 (19.7) | 0.303 (34.9) |
| product, geo:none | 0.342 (1.4) | 0.433 (1.7) | 0.517 (2.2) | 0.579 (3.0) | 0.560 (4.2) | 0.536 (5.2) | 0.379 (12.9) |
