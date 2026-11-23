model mlx-community/Qwen3-30B-A3B-4bit, activations model, 8-row verify pass vs first k rows alone

| context | k | rows identical | max abs logit diff | argmax agrees | full pass repeat identical |
|---|---|---|---|---|---|
| 0 | 1 | 0/1 | 0.191 | 1/1 | True |
| 0 | 2 | 0/2 | 0.734 | 2/2 | True |
| 0 | 3 | 0/3 | 0.734 | 3/3 | True |
| 0 | 4 | 0/4 | 0.734 | 4/4 | True |
| 0 | 5 | 5/5 | 0 | 5/5 | True |
| 0 | 6 | 6/6 | 0 | 6/6 | True |
| 0 | 7 | 7/7 | 0 | 7/7 | True |
| 1 | 1 | 0/1 | 0.656 | 1/1 | True |
| 1 | 2 | 0/2 | 1.19 | 2/2 | True |
| 1 | 3 | 0/3 | 1.19 | 3/3 | True |
| 1 | 4 | 0/4 | 1.19 | 4/4 | True |
| 1 | 5 | 5/5 | 0 | 5/5 | True |
| 1 | 6 | 6/6 | 0 | 6/6 | True |
| 1 | 7 | 7/7 | 0 | 7/7 | True |
| 2 | 1 | 0/1 | 0.484 | 1/1 | True |
| 2 | 2 | 0/2 | 0.52 | 2/2 | True |
| 2 | 3 | 0/3 | 0.625 | 3/3 | True |
| 2 | 4 | 0/4 | 2.69 | 4/4 | True |
| 2 | 5 | 5/5 | 0 | 5/5 | True |
| 2 | 6 | 6/6 | 0 | 6/6 | True |
| 2 | 7 | 7/7 | 0 | 7/7 | True |

total: 54/84 kept rows bit-identical, 84/84 same argmax
