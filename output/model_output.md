## Model output comparison



| Model name | batch size | optimizer | volume | train acc | val acc | epoch time |
| --- | --- | --- | --- | --- |--- |--- |
| Model_88_BN | 64 | sgd(lr = 1e-3) | 551,722 | 0.91 | 0.90 | 17s (21s/step) |
| Model_88 | 64 | sgd(lr = 1e-3) | 550,570 | 0.89 | 0.88 | 14s (18s/step) |
| Model_88_GAP | 64 | sgd(lr = 1e-3) | 288,298 | 0.91 | 0.89 | 14s (18s/step) |
| Model_88_GAP_v1(1*1) | 64 | sgd(lr = 1e-3) | 207,610 | 0.87 | 0.83 | 14s (18s/step) |
| Model_88_GAP_v1_9_blocks | 64 | sgd(lr = ~) | 139,850 | 0.86 | 0.82 | 14s (18s/step) |
| Model_88_GAP_v1_8_blocks-wider | 64 | sgd(lr = ~) | 194,442 | 0.86 | 0.82 | 14s (18s/step) |
| Model_88_GAP_v2 | 64 | sgd(lr = 1e-3) | 224,202 | 0.89 | 0.86 | 14s (18s/step) |
| ResNet20_v1 | 32 | adm(lr = ~) | 273,066 | 0.99 | 0.91 | 24s (30s/step) |
| ResNet20_v2 | 32 | adm(lr = ~) | 574,090 |  |  | 24s (30s/step) |
