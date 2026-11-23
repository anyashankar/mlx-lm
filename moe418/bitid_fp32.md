model /Users/anyashankar/models/granite-3b, activations float32, 8-row verify pass vs first k rows alone

| context | k | rows identical | max abs logit diff | argmax agrees | full pass repeat identical |
|---|---|---|---|---|---|
| 0 | 1 | 0/1 | 7.51e-06 | 1/1 | True |
| 0 | 2 | 2/2 | 0 | 2/2 | True |
| 0 | 3 | 3/3 | 0 | 3/3 | True |
| 0 | 4 | 4/4 | 0 | 4/4 | True |
| 0 | 5 | 5/5 | 0 | 5/5 | True |
| 0 | 6 | 6/6 | 0 | 6/6 | True |
| 0 | 7 | 7/7 | 0 | 7/7 | True |
| 1 | 1 | 0/1 | 1.14e-05 | 1/1 | True |
| 1 | 2 | 2/2 | 0 | 2/2 | True |
| 1 | 3 | 3/3 | 0 | 3/3 | True |
| 1 | 4 | 4/4 | 0 | 4/4 | True |
| 1 | 5 | 5/5 | 0 | 5/5 | True |
| 1 | 6 | 6/6 | 0 | 6/6 | True |
| 1 | 7 | 7/7 | 0 | 7/7 | True |
| 2 | 1 | 0/1 | 7.63e-06 | 1/1 | True |
| 2 | 2 | 2/2 | 0 | 2/2 | True |
| 2 | 3 | 3/3 | 0 | 3/3 | True |
| 2 | 4 | 4/4 | 0 | 4/4 | True |
| 2 | 5 | 5/5 | 0 | 5/5 | True |
| 2 | 6 | 6/6 | 0 | 6/6 | True |
| 2 | 7 | 7/7 | 0 | 7/7 | True |
| 3 | 1 | 0/1 | 8.11e-06 | 1/1 | True |
| 3 | 2 | 2/2 | 0 | 2/2 | True |
| 3 | 3 | 3/3 | 0 | 3/3 | True |
| 3 | 4 | 4/4 | 0 | 4/4 | True |
| 3 | 5 | 5/5 | 0 | 5/5 | True |
| 3 | 6 | 6/6 | 0 | 6/6 | True |
| 3 | 7 | 7/7 | 0 | 7/7 | True |
| 4 | 1 | 0/1 | 8.11e-06 | 1/1 | True |
| 4 | 2 | 2/2 | 0 | 2/2 | True |
| 4 | 3 | 3/3 | 0 | 3/3 | True |
| 4 | 4 | 4/4 | 0 | 4/4 | True |
| 4 | 5 | 5/5 | 0 | 5/5 | True |
| 4 | 6 | 6/6 | 0 | 6/6 | True |
| 4 | 7 | 7/7 | 0 | 7/7 | True |
| 5 | 1 | 0/1 | 1.12e-05 | 1/1 | True |
| 5 | 2 | 2/2 | 0 | 2/2 | True |
| 5 | 3 | 3/3 | 0 | 3/3 | True |
| 5 | 4 | 4/4 | 0 | 4/4 | True |
| 5 | 5 | 5/5 | 0 | 5/5 | True |
| 5 | 6 | 6/6 | 0 | 6/6 | True |
| 5 | 7 | 7/7 | 0 | 7/7 | True |

total: 162/168 kept rows bit-identical, 168/168 same argmax
