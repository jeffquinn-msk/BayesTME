import numpy as np
import matplotlib.pyplot as plt
import bayestme_plot as st_plt
import bleeding_correction as bleed

class RawSTData:
    def __init__(self, raw_count, filtered_count, tissue_mask, positions_tissue, positions, features, layout, data_name):
        '''
        data_path: /path/to/spaceranger/outs
                   should contain at least /raw_feature_bc_matrix for raw count matrix
                                           /filtered_feature_bc_matrix for filtered count matrix
                                           /spatial for position list
        '''
        self.raw_count = raw_count
        self.Reads = filtered_count
        self.tissue_mask = tissue_mask
        self.positions_tissue = positions_tissue.astype(int).T
        self.positions = positions.astype(int).T
        self.features = features
        self.n_spot_in = tissue_mask.sum()
        self.layout = layout
        self.data_name = data_name

    def plot_bleeding(self, gene, cmap='jet'):
        gene_idx = np.argwhere(np.array(self.features[1]) == gene)[0][0]
        print('Gene: {}'.format(gene))
        raw_filtered_align = (self.raw_count[gene_idx][self.tissue_mask] == self.Reads[gene_idx]).sum()
        print('\t UMI counts {}/{} matches'.format(raw_filtered_align, self.n_spot_in))
        if raw_filtered_align == self.n_spot_in:
            print('\t no bleeding filtering performed')
        all_counts = self.raw_count[gene_idx].sum()
        tissue_counts = self.Reads[gene_idx].sum()
        print('\t {:.3f}% bleeds out'.format((1-tissue_counts/all_counts) * 100))
        fig, ax = plt.subplots(1, 3, figsize=(6*3, 8))
        v_min = np.nanpercentile(self.raw_count[gene_idx], 5)
        v_max = np.nanpercentile(self.raw_count[gene_idx], 95)
        im = st_plt.plot_spots(ax[0], self.raw_count[gene_idx], self.positions.T, s=10, cmap=cmap, v_min=v_min, v_max=v_max)
        plt.colorbar(im, ax=ax[0])
        ax[0].invert_xaxis()
        ax[0].invert_yaxis()
        ax[0].set_title('raw UMI counts')
        plot_braf = self.raw_count[gene_idx].copy().astype(float)
        plot_braf[~self.tissue_mask] = np.nan
        v_min = np.nanpercentile(plot_braf, 5)
        v_max = np.nanpercentile(plot_braf, 95)
        im = st_plt.plot_spots(ax[1], plot_braf, self.positions.T, s=10, cmap=cmap, v_min=v_min, v_max=v_max)
        plt.colorbar(im, ax=ax[1])
        ax[1].invert_xaxis()
        ax[1].invert_yaxis()
        ax[1].set_title('UMI counts inside ({:.3f}%)'.format((tissue_counts/all_counts) * 100))
        blead_braf = self.raw_count[gene_idx].copy().astype(float)
        blead_braf[self.tissue_mask] = np.nan
        v_min = np.nanpercentile(blead_braf, 5)
        v_max = np.nanpercentile(blead_braf, 95)
        im = st_plt.plot_spots(ax[2], blead_braf, self.positions.T, s=10, cmap=cmap, v_min=v_min, v_max=v_max)
        plt.colorbar(im, ax=ax[2])
        ax[2].invert_xaxis()
        ax[2].invert_yaxis()
        ax[2].set_title('UMI counts outside ({:.3f}%)'.format((1-tissue_counts/all_counts) * 100))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        plt.savefig('{}_{}_bleeding.pdf'.format(self.data_name, gene))


class CleanedSTData(RawSTData):
    def __init__(self, raw_count, filtered_count, tissue_mask, positions_tissue, positions, features, layout, data_name):
        RawSTData.__init__(self, raw_count, filtered_count, tissue_mask, positions_tissue, positions, features, layout, data_name)
        self.clean_bleed()

    def clean_bleed(self):
        basis_idxs, basis_mask = bleed.build_basis_indices(self.positions)
        global_rates, fit_Rates, basis_functions, Weights = bleed.decontaminate_spots(self.Reads, self.tissue_mask, basis_idxs, basis_mask)
        self.Observation = fit_Rates

    def plot_basis_functions(self):
        pass

    def plot_gene_cleanup(self):
        pass

    def plot_before_after_cleanup(self):
        pass

class DeconvolvedSTData(RawSTData):
    def __init__(self, Observation, pos, features, layout, exp_name, cell_prob_trace, expression_trace, beta_trace, cell_num_trace, lam):
        self.Reads = Observation
        self.locations = pos
        self.genes = features
        self.layout = layout
        self.exp_name = exp_name
        self.cell_prob_trace = cell_prob_trace
        self.expression_trace = expression_trace
        self.beta_trace = beta_trace
        self.cell_num_trace = cell_num_trace
        self.lam = lam
        self.n_gene = self.expression_trace.shape[2]
        self.n_components = self.expression_trace.shape[1]

    def detect_communities(self, min_clusters, max_clusters, assignments_ref=None, alignment=False):
        best_clusters, best_assignments, scores = communities_from_posteriors(self.cell_prob_trace[:, :, 1:], self.edges, min_clusters=min_clusters, max_clusters=max_clusters, cluster_score=gaussian_aicc_bic_mixture)
        if alignment and assignments_ref is not None:
            best_assignments, _, _ = align_clusters(assignments_ref, best_assignments)
        return best_assignments

    def detect_marker_genes(self):
        score = (self.expression_trace == np.amax(self.expression_trace, axis=1)[:, None]).sum(axis=0)
        score /= self.expression_trace.shape[0]
        marker_gene = [self.features[score[k] > 0.95] for k in range(self.n_components)]
        return marker_gene

    def plot_expression_profiles(self):
        pass

    def plot_deconvolution(self):
        pass

    def plot_communities(self):
        pass

    def plot_marker_genes(self):
        pass

class CrossValidationSTData(RawSTData):
    def __init__(self, raw_count, filtered_count, tissue_mask, positions_tissue, positions, features, layout, data_name):
        RawSTData.__init__(self, raw_count, filtered_count, tissue_mask, positions_tissue, positions, features, layout, data_name)
        self.read_loc = np.zeros((self.Reads.sum(), 2)).astype(int)
        read_id = 0
        for i in range(self.Reads.shape[0]):
            for j in range(self.Reads.shape[1]):
                n_read = self.Reads[i, j]
                read_loc[read_id:read_id+n_read+1] = np.array([i, j])
                read_id += n_read
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        self.folds = kf.split(self.read_loc)

    def save_folds(self, kf_path):
        k=0
        for train, test in self.folds:
            print('fold {}'.format(k))
            hold_out_count = np.zeros_like(self.Reads)
            for i, j in self.read_loc[test]:
                hold_out_count[i, j] += 1
            np.save(kf_path+'{}_fold{}'.format(exp_name, k), self.Reads - hold_out_count)
            np.save(kf_path+'{}_test{}'.format(exp_name, k), hold_out_count)
            k+=1

    def create_slurm_jobs():
        pass

    def create_condor_jobs():
        pass

    def plot_cv_results():
        pass

    def best_settings():
        pass


class SpatialVaryingDeconvolvedSTData:
    def __init__(self):
        pass


    def plot_spatial_expression(self):
        pass

    def plot_transcriptional_programs(self):
        pass







        