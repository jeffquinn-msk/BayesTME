IMAGE=c5d78cd54214e

docker run -v $(pwd)/melanoma_results:/data \
  -v $(pwd)/data:/input \
  "${IMAGE}" \
  filter_bleed \
  --count-mat /input/ST_mel1_rep2_counts.tsv \
  --data-dir /data \
  --n-gene 1000 \
  --filter-type spots

docker run -v $(pwd)/melanoma_results:/data \
  "${IMAGE}" \
  deconvolve \
  --data-dir /data \
  --n-gene 1000 \
  --n-components 4 \
  --lam2 10000 \
  --n-samples 100 \
  --n-burnin 500 \
  --n-thin 5

docker run -v $(pwd)/melanoma_results:/data \
  "${IMAGE}" \
  spatial_expression \
  --data-dir /data \
  --n-spatial-patterns 10 \
  --n-samples 100 \
  --n-burn 100 \
  --n-thin 2 \
  --simple

docker run -v $(pwd)/melanoma_results:/data "${IMAGE}" \
  grid_search \
  --data-dir /data/k_fold/jobs/data \
  --config /data/k_fold/setup/config/melanoma/config_1.cfg \
  --output-dir /data/k_fold/setup/outputs/1