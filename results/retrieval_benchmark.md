Queries: 150 per setting · corpus: 289 summaries

|                                                       |   recall@1 |   recall@5 |   recall@10 |   hit@5 |   mrr |   ndcg@10 |   mrr_ci_low |   mrr_ci_high |
|:------------------------------------------------------|-----------:|-----------:|------------:|--------:|------:|----------:|-------------:|--------------:|
| ('standard', 'BM25')                                  |      0.173 |      0.462 |       0.556 |   0.687 | 0.475 |     0.426 |        0.412 |         0.539 |
| ('standard', 'dense (nomic-embed-text)')              |      0.186 |      0.407 |       0.500 |   0.627 | 0.452 |     0.402 |        0.388 |         0.519 |
| ('standard', 'hybrid (nomic-embed-text)')             |      0.172 |      0.426 |       0.578 |   0.640 | 0.454 |     0.427 |        0.390 |         0.514 |
| ('standard', 'dense (mxbai-embed-large)')             |      0.159 |      0.414 |       0.495 |   0.627 | 0.429 |     0.387 |        0.369 |         0.493 |
| ('standard', 'hybrid (mxbai-embed-large)')            |      0.164 |      0.412 |       0.524 |   0.647 | 0.444 |     0.400 |        0.385 |         0.510 |
| ('hard (names masked)', 'BM25')                       |      0.167 |      0.365 |       0.441 |   0.573 | 0.431 |     0.357 |        0.367 |         0.487 |
| ('hard (names masked)', 'dense (nomic-embed-text)')   |      0.090 |      0.246 |       0.302 |   0.427 | 0.290 |     0.228 |        0.233 |         0.349 |
| ('hard (names masked)', 'hybrid (nomic-embed-text)')  |      0.117 |      0.329 |       0.423 |   0.533 | 0.360 |     0.307 |        0.303 |         0.422 |
| ('hard (names masked)', 'dense (mxbai-embed-large)')  |      0.070 |      0.196 |       0.280 |   0.340 | 0.240 |     0.195 |        0.190 |         0.293 |
| ('hard (names masked)', 'hybrid (mxbai-embed-large)') |      0.130 |      0.316 |       0.414 |   0.527 | 0.359 |     0.303 |        0.302 |         0.429 |