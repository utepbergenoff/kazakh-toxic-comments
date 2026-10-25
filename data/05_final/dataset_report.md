# Final dataset report

## Annotation status of the 600 sampled main comments

| status                      |   n |
|:----------------------------|----:|
| used                        | 554 |
| excluded_UNCLEAR            |  25 |
| excluded_UNCLEAR_incomplete |  10 |
| excluded_NOTKKRU            |   9 |
| not_annotated               |   2 |

Rule applied: offensive or hate = 1 implies toxic = 1 (52 toxic labels changed 0 -> 1).

## Usable comments and positives by language

| lang   |   comments |   toxic |   offensive |   hate |
|:-------|-----------:|--------:|------------:|-------:|
| KK     |        225 |      74 |          48 |      8 |
| RU     |        230 |      88 |          61 |     12 |
| CS     |         99 |      34 |          20 |      2 |
| All    |        554 |     196 |         129 |     22 |

## 5-fold grouped cross-validation (grouped by thread, stratified by toxic)

|   fold |   comments |   toxic |   offensive |   hate |   CS |
|-------:|-----------:|--------:|------------:|-------:|-----:|
|      0 |        111 |      39 |          26 |      6 |   16 |
|      1 |        112 |      40 |          22 |      6 |   17 |
|      2 |        111 |      39 |          26 |      2 |   18 |
|      3 |        110 |      39 |          23 |      7 |   19 |
|      4 |        110 |      39 |          32 |      1 |   29 |
