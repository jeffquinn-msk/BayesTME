import bayestme

zebrafish_A1 = bayestme.BayesTME(exp_name='zebrafish_A1', storage_path='zebrafish_A1_data')
zebrafish_A1_stdata = zebrafish_A1.load_data_from_spaceranger('A1_spaceranger_output')
zebrafish_A1_stdata.filter(n_gene=1000, filter_type='ribosome', verbose=True)
zebrafish_A1_stdata.bleeding_correction()