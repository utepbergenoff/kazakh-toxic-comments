# Final dataset report

## Annotation status of the 600 sampled main comments

| status                      |   n |
|:----------------------------|----:|
| not_annotated               | 292 |
| used                        | 285 |
| excluded_UNCLEAR            |  10 |
| excluded_UNCLEAR_incomplete |   8 |
| excluded_NOTKKRU            |   5 |

Rule applied: offensive or hate = 1 implies toxic = 1 (29 toxic labels changed 0 -> 1).

## Usable comments and positives by language

| lang   |   comments |   toxic |   offensive |   hate |
|:-------|-----------:|--------:|------------:|-------:|
| KK     |        124 |      33 |          23 |      5 |
| RU     |        117 |      43 |          34 |     11 |
| CS     |         44 |      18 |          10 |      2 |
| All    |        285 |      94 |          67 |     18 |

## 5-fold grouped cross-validation (grouped by thread, stratified by toxic)

|   fold |   comments |   toxic |   offensive |   hate |   CS |
|-------:|-----------:|--------:|------------:|-------:|-----:|
|      0 |         57 |      19 |          14 |      4 |    5 |
|      1 |         57 |      19 |          13 |      6 |   10 |
|      2 |         57 |      19 |          15 |      3 |    9 |
|      3 |         57 |      18 |          12 |      4 |   14 |
|      4 |         57 |      19 |          13 |      1 |    6 |
