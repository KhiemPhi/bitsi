180 objects, 45 categories, theta=0.9 (class-mean)

| Variant | recall | miou_oracleK | miou_kmeans | miou_theta | n_seg_theta | sem_iou | sem_iou_sel | sel_acc | sem_iou_sel_sam3 | sem_iou_selmulti_sam3 | sel_acc_sam3 | sem_iou_sam3direct | sel_acc_sam3direct | sem_iou_sam3direct@0.3 | sel_acc_sam3direct@0.3 | acc:sem_iou_sel | acc:sem_iou_selmulti_sam3 | acc:sem_iou_sam3direct | acc:sem_iou_sam3direct@0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.760 | 0.408 | nan | 0.399 | 5.683 | 0.176 | 0.155 | 0.108 | 0.283 | 0.316 | 0.237 | 0.280 | 0.177 | 0.315 | 0.237 | 0.106 | 0.247 | 0.153 | 0.207 |
| geo:none | 0.763 | 0.417 | nan | 0.409 | 4.022 | 0.184 | 0.151 | 0.099 | 0.284 | 0.318 | 0.239 | 0.280 | 0.177 | 0.315 | 0.237 | 0.103 | 0.247 | 0.153 | 0.207 |
| geo:jump | 0.754 | 0.412 | nan | 0.400 | 5.528 | 0.182 | 0.151 | 0.094 | 0.282 | 0.319 | 0.233 | 0.280 | 0.177 | 0.315 | 0.237 | 0.104 | 0.250 | 0.153 | 0.207 |
| geo:single | 0.754 | 0.400 | nan | 0.402 | 4.428 | 0.183 | 0.152 | 0.102 | 0.286 | 0.320 | 0.241 | 0.280 | 0.177 | 0.315 | 0.237 | 0.108 | 0.249 | 0.153 | 0.207 |
| geo:extrema | 0.758 | 0.418 | nan | 0.408 | 4.094 | 0.186 | 0.152 | 0.099 | 0.284 | 0.318 | 0.239 | 0.280 | 0.177 | 0.315 | 0.237 | 0.104 | 0.247 | 0.153 | 0.207 |
| geo:axis | 0.753 | 0.401 | nan | 0.390 | 7.028 | 0.174 | 0.163 | 0.104 | 0.278 | 0.317 | 0.238 | 0.280 | 0.177 | 0.315 | 0.237 | 0.109 | 0.250 | 0.153 | 0.207 |
| geo:global | 0.737 | 0.385 | nan | 0.294 | 80.144 | 0.170 | 0.159 | 0.111 | 0.278 | 0.316 | 0.239 | 0.280 | 0.177 | 0.315 | 0.237 | 0.104 | 0.241 | 0.153 | 0.207 |
| sem only (lam=1) | 0.763 | 0.417 | nan | 0.345 | 9.950 | 0.131 | 0.151 | 0.099 | 0.284 | 0.318 | 0.239 | 0.280 | 0.177 | 0.315 | 0.237 | 0.103 | 0.247 | 0.153 | 0.207 |
| geo only (lam=0) | 0.364 | 0.224 | nan | 0.228 | 2.322 | 0.172 | 0.192 | 0.032 | 0.210 | 0.241 | 0.116 | 0.280 | 0.177 | 0.315 | 0.237 | 0.118 | 0.170 | 0.153 | 0.207 |
| mask finest | 0.750 | 0.411 | nan | 0.389 | 7.133 | 0.150 | 0.151 | 0.117 | 0.278 | 0.309 | 0.235 | 0.280 | 0.177 | 0.315 | 0.237 | 0.107 | 0.241 | 0.153 | 0.207 |
| no long-range | 0.716 | 0.332 | nan | 0.360 | 5.450 | 0.188 | 0.139 | 0.076 | 0.261 | 0.297 | 0.212 | 0.280 | 0.177 | 0.315 | 0.237 | 0.085 | 0.225 | 0.153 | 0.207 |
| combine: product | 0.758 | 0.417 | nan | 0.335 | 12.344 | 0.146 | 0.154 | 0.105 | 0.279 | 0.314 | 0.232 | 0.280 | 0.177 | 0.315 | 0.237 | 0.107 | 0.243 | 0.153 | 0.207 |
| combine: min | 0.757 | 0.410 | nan | 0.337 | 11.333 | 0.150 | 0.159 | 0.099 | 0.279 | 0.312 | 0.234 | 0.280 | 0.177 | 0.315 | 0.237 | 0.108 | 0.241 | 0.153 | 0.207 |
| product, geo:none | 0.763 | 0.417 | nan | 0.345 | 9.950 | 0.131 | 0.151 | 0.099 | 0.284 | 0.318 | 0.239 | 0.280 | 0.177 | 0.315 | 0.237 | 0.103 | 0.247 | 0.153 | 0.207 |
| sem: PartField | 0.776 | 0.407 | nan | 0.335 | 12.372 | 0.151 | 0.170 | 0.109 | 0.288 | 0.329 | 0.242 | 0.280 | 0.177 | 0.315 | 0.237 | 0.112 | 0.258 | 0.153 | 0.207 |
| sem: PartField, geo:none | 0.782 | 0.412 | nan | 0.345 | 11.183 | 0.157 | 0.173 | 0.111 | 0.291 | 0.331 | 0.245 | 0.280 | 0.177 | 0.315 | 0.237 | 0.113 | 0.264 | 0.153 | 0.207 |
| sem: lifted+PartField | 0.785 | 0.411 | nan | 0.363 | 11.722 | 0.157 | 0.172 | 0.124 | 0.292 | 0.319 | 0.234 | 0.280 | 0.177 | 0.315 | 0.237 | 0.116 | 0.254 | 0.153 | 0.207 |
| sem: lifted+PartField, geo:none | 0.795 | 0.410 | nan | 0.378 | 10.194 | 0.142 | 0.162 | 0.112 | 0.296 | 0.320 | 0.242 | 0.280 | 0.177 | 0.315 | 0.237 | 0.114 | 0.253 | 0.153 | 0.207 |
| sem: lifted+PartField, geo:jump | 0.793 | 0.410 | nan | 0.369 | 11.644 | 0.153 | 0.162 | 0.112 | 0.294 | 0.318 | 0.233 | 0.280 | 0.177 | 0.315 | 0.237 | 0.115 | 0.253 | 0.153 | 0.207 |
| raw cosine | 0.760 | 0.408 | nan | 0.399 | 5.683 | 0.176 | 0.140 | 0.084 | 0.283 | 0.316 | 0.237 | 0.280 | 0.177 | 0.315 | 0.237 | 0.097 | 0.247 | 0.153 | 0.207 |
| baseline: PartField | 0.781 | 0.454 | 0.408 | nan | nan | nan | nan | nan | 0.288 | 0.340 | 0.236 | nan | nan | nan | nan | nan | 0.269 | nan | nan |

Per-category (full):

| Category | n | sem mIoU (segments) | sem mIoU (SigLIP sel.) | sel acc (SigLIP) | sem mIoU (SAM 3 sel.) | sel acc (SAM 3) |
|---|---|---|---|---|---|---|
| Bottle | 4 | 0.000 | 0.172 | 0.250 | 0.492 | 0.500 |
| Box | 4 | 0.028 | 0.223 | 0.250 | 0.478 | 0.500 |
| Bucket | 4 | 0.234 | 0.400 | 0.250 | 0.647 | 0.750 |
| Camera | 4 | 0.126 | 0.024 | 0.000 | 0.088 | 0.167 |
| Cart | 4 | 0.188 | 0.266 | 0.000 | 0.582 | 0.500 |
| Chair | 4 | 0.103 | 0.069 | 0.100 | 0.052 | 0.250 |
| Clock | 4 | 0.250 | 0.024 | 0.000 | 0.000 | 0.000 |
| CoffeeMachine | 4 | 0.188 | 0.017 | 0.000 | 0.286 | 0.182 |
| Dishwasher | 4 | 0.019 | 0.132 | 0.125 | 0.130 | 0.125 |
| Dispenser | 4 | 0.211 | 0.111 | 0.000 | 0.139 | 0.000 |
| Display | 4 | 0.121 | 0.199 | 0.167 | 0.372 | 0.333 |
| Door | 4 | 0.083 | 0.133 | 0.091 | 0.488 | 0.455 |
| Eyeglasses | 4 | 0.250 | 0.057 | 0.000 | 0.189 | 0.000 |
| Faucet | 4 | 0.107 | 0.136 | 0.125 | 0.416 | 0.375 |
| FoldingChair | 4 | 0.000 | 0.498 | 0.750 | 0.498 | 0.750 |
| Globe | 4 | 0.058 | 0.379 | 0.250 | 0.930 | 1.000 |
| Kettle | 4 | 0.305 | 0.092 | 0.111 | 0.656 | 0.667 |
| Keyboard | 4 | 0.250 | 0.001 | 0.000 | 0.000 | 0.000 |
| KitchenPot | 4 | 0.149 | 0.303 | 0.375 | 0.525 | 0.375 |
| Knife | 4 | 0.000 | 0.080 | 0.000 | 0.348 | 0.250 |
| Lamp | 4 | 0.251 | 0.033 | 0.000 | 0.399 | 0.500 |
| Laptop | 4 | 0.233 | 0.233 | 0.000 | 0.200 | 0.000 |
| Lighter | 4 | 0.394 | 0.190 | 0.000 | 0.250 | 0.000 |
| Microwave | 4 | 0.128 | 0.204 | 0.000 | 0.067 | 0.000 |
| Mouse | 4 | 0.167 | 0.120 | 0.000 | 0.265 | 0.100 |
| Oven | 4 | 0.522 | 0.407 | 0.000 | 0.500 | 0.000 |
| Pen | 4 | 0.159 | 0.241 | 0.250 | 0.125 | 0.000 |
| Phone | 4 | 0.250 | 0.031 | 0.000 | 0.230 | 0.333 |
| Pliers | 4 | 0.116 | 0.057 | 0.000 | 0.158 | 0.000 |
| Printer | 4 | 0.002 | 0.151 | 0.250 | 0.146 | 0.250 |
| Refrigerator | 4 | 0.083 | 0.190 | 0.125 | 0.000 | 0.000 |
| Remote | 4 | 0.040 | 0.036 | 0.000 | 0.015 | 0.000 |
| Safe | 4 | 0.251 | 0.092 | 0.000 | 0.257 | 0.000 |
| Scissors | 4 | 0.167 | 0.025 | 0.000 | 0.344 | 0.000 |
| Stapler | 4 | 0.154 | 0.045 | 0.000 | 0.026 | 0.000 |
| StorageFurniture | 4 | 0.169 | 0.006 | 0.000 | 0.091 | 0.000 |
| Suitcase | 4 | 0.500 | 0.190 | 0.250 | 0.491 | 0.500 |
| Switch | 4 | 0.000 | 0.034 | 0.000 | 0.026 | 0.000 |
| Table | 4 | 0.538 | 0.349 | 0.300 | 0.503 | 0.300 |
| Toaster | 4 | 0.000 | 0.051 | 0.000 | 0.085 | 0.143 |
| Toilet | 4 | 0.253 | 0.128 | 0.000 | 0.285 | 0.429 |
| TrashCan | 4 | 0.667 | 0.338 | 0.333 | 0.630 | 0.667 |
| USB | 4 | 0.187 | 0.000 | 0.000 | 0.125 | 0.000 |
| WashingMachine | 4 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| Window | 4 | 0.007 | 0.528 | 0.500 | 0.199 | 0.250 |


Class-agnostic mIoU (mean #segments) vs merge threshold theta:

| Variant | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.93 | 0.96 |
|---|---|---|---|---|---|---|---|
| full | 0.235 (1.0) | 0.238 (1.1) | 0.258 (1.3) | 0.363 (2.8) | 0.399 (5.7) | 0.373 (9.4) | 0.307 (12.9) |
| geo:none | 0.235 (1.0) | 0.237 (1.0) | 0.248 (1.2) | 0.357 (1.8) | 0.409 (4.0) | 0.393 (7.6) | 0.328 (10.4) |
| geo:jump | 0.241 (2.0) | 0.244 (2.2) | 0.260 (2.4) | 0.362 (3.1) | 0.400 (5.5) | 0.380 (9.2) | 0.325 (11.8) |
| sem only (lam=1) | 0.357 (1.8) | 0.392 (2.6) | 0.409 (4.0) | 0.406 (6.8) | 0.345 (9.9) | 0.319 (10.2) | 0.276 (13.3) |
| sem: PartField | 0.353 (3.2) | 0.382 (4.7) | 0.406 (6.8) | 0.385 (9.5) | 0.335 (12.4) | 0.313 (13.0) | 0.310 (20.1) |
| sem: PartField, geo:none | 0.349 (2.9) | 0.380 (4.3) | 0.404 (6.4) | 0.392 (9.4) | 0.345 (11.2) | 0.316 (10.7) | 0.313 (16.7) |
| sem: lifted+PartField | 0.274 (1.3) | 0.315 (2.0) | 0.393 (3.3) | 0.404 (6.8) | 0.363 (11.7) | 0.317 (13.0) | 0.303 (15.2) |
| sem: lifted+PartField, geo:none | 0.272 (1.2) | 0.314 (1.8) | 0.383 (2.9) | 0.414 (5.3) | 0.378 (10.2) | 0.335 (12.0) | 0.311 (10.5) |

Task instructions (full; 'rewrite' = oracle part phrase standing in for LLM rewriting):

| Query | n | acc (IoU>=0.5) | mean IoU |
|---|---|---|---|
| siglip_raw | 122 | 0.041 | 0.065 |
| siglip_rewrite | 122 | 0.082 | 0.111 |
| sam3_raw | 122 | 0.016 | 0.047 |
| sam3_rewrite | 122 | 0.180 | 0.219 |

Graspable-part queries (handles, knobs, shafts, necks, grips; metric scale from category priors, G = 8 cm):

| Variant | Engine | Selection | n queries | mean IoU | correct (IoU>=0.5) | graspable (any selection) | GPR given correct | correct and graspable |
|---|---|---|---|---|---|---|---|---|
| full | SigLIP | relevance | 44 | 0.120 | 0.091 | 0.818 | 1.000 | 0.091 |
| full | SigLIP | feasible | 44 | 0.119 | 0.091 | 0.864 | 1.000 | 0.091 |
| full | SAM 3 | relevance | 44 | 0.253 | 0.182 | 0.932 | 1.000 | 0.182 |
| full | SAM 3 | feasible | 44 | 0.249 | 0.182 | 0.955 | 1.000 | 0.182 |
| full | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| full | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| geo:none | SigLIP | relevance | 44 | 0.098 | 0.068 | 0.773 | 1.000 | 0.068 |
| geo:none | SigLIP | feasible | 44 | 0.097 | 0.068 | 0.818 | 1.000 | 0.068 |
| geo:none | SAM 3 | relevance | 44 | 0.228 | 0.159 | 0.909 | 1.000 | 0.159 |
| geo:none | SAM 3 | feasible | 44 | 0.229 | 0.159 | 0.932 | 1.000 | 0.159 |
| geo:none | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| geo:none | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| geo:jump | SigLIP | relevance | 44 | 0.096 | 0.045 | 0.795 | 1.000 | 0.045 |
| geo:jump | SigLIP | feasible | 44 | 0.095 | 0.045 | 0.818 | 1.000 | 0.045 |
| geo:jump | SAM 3 | relevance | 44 | 0.233 | 0.159 | 0.909 | 1.000 | 0.159 |
| geo:jump | SAM 3 | feasible | 44 | 0.237 | 0.159 | 0.955 | 1.000 | 0.159 |
| geo:jump | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| geo:jump | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| baseline: PartField | SAM 3 | relevance | 44 | 0.249 | 0.182 | 0.909 | 1.000 | 0.182 |
| baseline: PartField | SAM 3 | feasible | 44 | 0.248 | 0.182 | 0.955 | 1.000 | 0.182 |
| sem: PartField | SigLIP | relevance | 44 | 0.088 | 0.023 | 0.864 | 1.000 | 0.023 |
| sem: PartField | SigLIP | feasible | 44 | 0.086 | 0.023 | 0.841 | 1.000 | 0.023 |
| sem: PartField | SAM 3 | relevance | 44 | 0.243 | 0.159 | 0.932 | 1.000 | 0.159 |
| sem: PartField | SAM 3 | feasible | 44 | 0.233 | 0.159 | 0.932 | 1.000 | 0.159 |
| sem: PartField | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: PartField | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: PartField, geo:none | SigLIP | relevance | 44 | 0.088 | 0.045 | 0.864 | 1.000 | 0.045 |
| sem: PartField, geo:none | SigLIP | feasible | 44 | 0.086 | 0.045 | 0.841 | 1.000 | 0.045 |
| sem: PartField, geo:none | SAM 3 | relevance | 44 | 0.238 | 0.159 | 0.909 | 1.000 | 0.159 |
| sem: PartField, geo:none | SAM 3 | feasible | 44 | 0.241 | 0.159 | 0.955 | 1.000 | 0.159 |
| sem: PartField, geo:none | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: PartField, geo:none | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField | SigLIP | relevance | 44 | 0.090 | 0.023 | 0.795 | 1.000 | 0.023 |
| sem: lifted+PartField | SigLIP | feasible | 44 | 0.090 | 0.023 | 0.773 | 1.000 | 0.023 |
| sem: lifted+PartField | SAM 3 | relevance | 44 | 0.246 | 0.159 | 0.864 | 1.000 | 0.159 |
| sem: lifted+PartField | SAM 3 | feasible | 44 | 0.251 | 0.159 | 0.909 | 1.000 | 0.159 |
| sem: lifted+PartField | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField, geo:none | SigLIP | relevance | 44 | 0.083 | 0.023 | 0.795 | 1.000 | 0.023 |
| sem: lifted+PartField, geo:none | SigLIP | feasible | 44 | 0.082 | 0.023 | 0.773 | 1.000 | 0.023 |
| sem: lifted+PartField, geo:none | SAM 3 | relevance | 44 | 0.252 | 0.159 | 0.886 | 1.000 | 0.159 |
| sem: lifted+PartField, geo:none | SAM 3 | feasible | 44 | 0.253 | 0.159 | 0.955 | 1.000 | 0.159 |
| sem: lifted+PartField, geo:none | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField, geo:none | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField, geo:jump | SigLIP | relevance | 44 | 0.075 | 0.023 | 0.841 | 1.000 | 0.023 |
| sem: lifted+PartField, geo:jump | SigLIP | feasible | 44 | 0.074 | 0.023 | 0.818 | 1.000 | 0.023 |
| sem: lifted+PartField, geo:jump | SAM 3 | relevance | 44 | 0.250 | 0.159 | 0.886 | 1.000 | 0.159 |
| sem: lifted+PartField, geo:jump | SAM 3 | feasible | 44 | 0.251 | 0.159 | 0.955 | 1.000 | 0.159 |
| sem: lifted+PartField, geo:jump | SAM 3 lifted, no hierarchy | relevance | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
| sem: lifted+PartField, geo:jump | SAM 3 lifted, no hierarchy | feasible | 44 | 0.211 | 0.250 | 0.273 | 1.000 | 0.250 |
