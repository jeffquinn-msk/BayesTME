from . import bayestme

melanoma = bayestme.BayesTME(exp_name='melanoma')
melanoma_stdata = melanoma.load_data_from_count_mat('data/ST_mel1_rep2_counts.tsv')
melanoma_stdata.filter(n_gene=1000, filter_type='spots')
melanoma_kfold = melanoma_stdata.k_fold('melanoma_results/')

melanoma_deconvolve = melanoma.deconvolve(melanoma_stdata, n_gene=1000, n_components=4, lam2=10000,
                                          n_samples=100, n_burnin=500, n_thin=5)

melanoma_spatial = melanoma.spatial_expression(melanoma_deconvolve, n_spatial_patterns=10,
                                               n_samples=100, n_burn=100, n_thin=2, simple=True)
