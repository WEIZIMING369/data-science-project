# Results and interpretation

The downloaded UCI data contains 150 observations, 0 missing cells and 3 identical rows (retained).

| Feature pair | Five-fold mean accuracy | Individual fold accuracies |
|---|---:|---|
| Sepal | 81.3% | 86.7%, 76.7%, 80.0%, 93.3%, 70.0% |
| Petal | 95.3% | 100.0%, 96.7%, 90.0%, 100.0%, 90.0% |

Petal measurements exceed sepal measurements by 14.0 percentage points under this fixed evaluation. The plots support the same descriptive pattern: setosa separates clearly in petal space, while the other two species overlap. The balanced-class majority baseline is 33.3%.

Both feature sets use identical five-fold stratified splits (shuffle=True, random_state=42). Each fold tests 30 flowers; a single prediction changes its accuracy by 3.33 percentage points. Scaling is fitted on training data only. This is an exploratory within-dataset comparison, not an independent external validation or a statistical significance claim.

The dataset is small and historical, so results should not be generalized to newly collected flowers without further evaluation. No model tuning or causal inference is performed. Exact version and checksum information is in `summary.json`.
