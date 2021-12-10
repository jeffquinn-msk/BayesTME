import numpy as np
from model_bkg import GraphFusedMultinomial
from scipy.stats import multinomial
import utils
import matplotlib.pyplot as plt
from bayestme_data import RawSTData, CleanedSTData, DeconvolvedSTData
import pandas as pd
import scipy.io as io

class BayesTME:
    def __init__(self, exp_name='BayesTME'):
        # set up experiment name
        self.exp_name = exp_name

    def load_from_spaceranger(self, data_path, layout=1):
        '''
        Load data from spaceranger /outputs folder
        Inputs:
            data_path:  /path/to/spaceranger/outs
                        should contain at least 1) /raw_feature_bc_matrix for raw count matrix
                                                2) /filtered_feature_bc_matrix for filtered count matrix
                                                3) /spatial for position list
            layout:     Visim(hex)  1
                        ST(square)  2
        '''
        raw_count_path = data_path + 'raw_feature_bc_matrix/matrix.mtx.gz'
        filtered_count_path = data_path + 'filtered_feature_bc_matrix/matrix.mtx.gz'
        features_path = data_path + 'raw_feature_bc_matrix/features.tsv.gz'
        barcodes_path = data_path + 'raw_feature_bc_matrix/barcodes.tsv.gz'
        positions_path = data_path + 'spatial/tissue_positions_list'

        try:
            positions_list = pd.read_csv(positions_path+'.csv', header=None, index_col=0, names=None)
        except:
            positions_list = pd.read_csv(positions_path+'.txt', sep=',', header=None, index_col=0, names=None)
        raw_count = np.array(io.mmread(raw_count_path).todense())
        filtered_count = np.array(io.mmread(filtered_count_path).todense())
        features = pd.read_csv(features_path, header=None, sep='\t')
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')
        # positions_list = pd.read_csv(positions_path, header=None, index_col=0, names=None)
        n_spots = raw_count.shape[1]
        n_genes = raw_count.shape[0]
        print('detected {} spots, {} genes'.format(n_spots, n_genes))
        pos = np.zeros((n_spots, 3))
        for i in range(n_spots):
            pos[i] = np.array(positions_list.loc[barcodes[0][i]][:3])
        tissue_mask = pos[:, 0] == 1
        positions_tissue = pos[tissue_mask][:, 1:]
        positions = pos[:, 1:]
        n_spot_in = tissue_mask.sum()
        print('\t {} spots in tissue sample'.format(n_spot_in))
        all_counts = raw_count.sum()
        tissue_counts = filtered_count.sum()
        print('\t {:.3f}% UMI counts bleeds out'.format((1 - tissue_counts/all_counts) * 100))

        return RawSTData(raw_count.T, filtered_count.T, tissue_mask, positions_tissue, positions, features, layout, self.exp_name)

    def cleaning_data(self, RawSTData):
        '''

        '''
        return CleanedSTData(RawSTData.raw_count, RawSTData.Reads, RawSTData.tissue_mask, RawSTData.positions_tissue, RawSTData.positions, RawSTData.features, RawSTData.layout, RawSTData.data_name)

    def hyperparam_auto_tuning(self, STData, n_folds):
        '''
        Auto-tuning 1) number of cell-types         K
                    2) spatial smoothing parameter  lam
        '''
        return CrossValidationSTData(STData, n_folds)

    def deconvolve(self, STData, n_gene=None, n_components=None, lam2=None, n_samples=100, n_burnin=1000, n_thin=10, random_seed=0, bkg=False, lda=False, cv=False, save_trace=False):
        '''

        Inputs:
            data:           either (1) RawSTData
                                or (2) CleanedSTData
            n_gene:         int or list
                            number or list of indices of the genes to look at
            n_componets:    int 
                            number of celltypes to segment (if known)
                            otherwise can be determined by cross validation
            lam2:           real positive number 
                            parameter controls the degree of spatial smoothing
                            recommend range (1e-2, 1e6) the less lam2 the more smoothing
                            otherwise can be determined by cross validation
            n_sample:       int
                            number of posterior samples
            n_burnin:       int
                            number of burn-in samples
            n_thin:         int
                            number of thinning
            random_seed:    int
                            random state
            bkg:            boolean
                            if fit with background noise
            lda:            boolean
                            if initialize model with LDA, converges faster but no garantee of correctness
                            recommend set to False
        '''
        # load position, and spatial layout
        self.pos = STData.positions_tissue
        self.layout = STData.layout
        # generate edge graph from spot positions and ST layout
        self.edges = utils.get_edges(self.pos, self.layout)
        self.n_components = n_components
        self.lam2 = lam2
        # detetermine the number of spots
        self.n_nodes = STData.Reads.shape[0]

        # load the count matrix
        if n_gene is None:
            self.n_gene = STData.Reads.shape[1]
            Observation = STData.Reads
        elif isinstance(n_gene, (list, np.ndarray)):
            self.n_gene = len(n_gene)
            Observation = STData.Reads[:, n_gene]
        elif isinstance(n_gene, int) and n_gene <= STData.Reads.shape[1]:
            self.n_gene = n_gene
            top = np.argsort(np.std(np.log(1+STData.Reads), axis=0))[::-1]
            Observation = STData.Reads[:, top[:self.n_gene]]
        else:
            raise ValueException('n_gene must be a integer less or equal to total number of gene ({}) or a list of indices of genes'.format(Observation.shape[1]))

        np.random.seed(random_seed)

        # initialize the model
        if n_components is not None:
            self.n_components = n_components
        else:
            raise Exception('use cv to determine number of cell_types')
        if lam2 is not None:
            self.lam2 = lam2
        else:
            raise Exception('use cv to determine spatial smoothing parameter')
        gfm = GraphFusedMultinomial(n_components=n_components, edges=self.edges, Observations=Observation, n_gene=self.n_gene, lam_psi=self.lam2, 
                                    background_noise=bkg, lda_initialization=lda)


        cell_prob_trace = np.zeros((n_samples, self.n_nodes, self.n_components+1))
        cell_num_trace = np.zeros((n_samples, self.n_nodes, self.n_components+1))
        expression_trace = np.zeros((n_samples, self.n_components, self.n_gene))
        beta_trace = np.zeros((n_samples, self.n_components))
        # reads_trace = np.zeros((n_samples, n_nodes, n_gene, n_components))
        if cv:
            loglhtest_trace = np.zeros(n_samples)
        loglhtrain_trace = np.zeros(n_samples)
        total_samples = n_samples*n_thin+n_burnin
        for step in range(total_samples):
            print(f'Step {step}/{total_samples} ...', end='\r')
            # perform Gibbs sampling
            gfm.sample(Observation)
            # save the trace of GFMM parameters
            if step >= n_burnin and (step - n_burnin) % n_thin == 0:
                idx = (step - n_burnin) // n_thin
                cell_prob_trace[idx] = gfm.probs
                expression_trace[idx] = gfm.phi
                beta_trace[idx] = gfm.beta
                cell_num_trace[idx] = gfm.cell_num
                # reads_trace[idx] = gfnb.reads[:, :, :-1]
                # save_npz('results/{}_reads_{}_{}_{}_{}'.format(args.exp_name, idx, args.n_components, args.n_gene, args.n_fold), csc_matrix(gfnb.reads[:, :, :-1].reshape(n_nodes, -1)))
                rates = (gfm.cell_num[:, 1:][:, :, None] * (gfm.beta[:, None] * gfm.phi)[None])
                nb_probs = rates.sum(axis=1) / rates.sum()
                if cv:
                    loglhtest_trace[idx] = multinomial.logpmf(test.flatten(), test.sum(), nb_probs.flatten())
                loglhtrain_trace[idx] = multinomial.logpmf(Observation.flatten(), Observation.sum(), nb_probs.flatten())
                # print('{}'.format(loglhtrain_trace[idx]))
        print('Done')
        if save_trace:
            np.save('results/{}_cell_prob_{}_{}.npy'.format(self.exp_name, self.n_components, self.lam_psi), cell_prob_trace)
            np.save('results/{}_phi_post_{}_{}.npy'.format(self.exp_name, self.n_components, self.lam_psi), expression_trace)
            np.save('results/{}_beta_post_{}_{}.npy'.format(self.exp_name, self.n_components, self.lam_psi), beta_trace)
            # np.save('results/{}_cell_num_{}_{}_{}_lda{}.npy'.format(args.exp_name, args.n_components, args.lam_psi, args.n_fold, args.lda), cell_num_trace)
            np.save('results/{}_train_likelihood_{}_{}.npy'.format(self.exp_name, self.n_components, self.lam_psi), loglhtrain_trace)
            np.save('results/{}_test_likelihood_{}_{}.npy'.format(self.exp_name, self.n_components, self.lam_psi), loglhtest_trace)
        return DeconvolvedSTData(Observation, self.pos, STData.features, self.layout, self.exp_name, cell_prob_trace, expression_trace, beta_trace, cell_num_trace, self.lam2)


    def spatial_expression(self):
        pass


