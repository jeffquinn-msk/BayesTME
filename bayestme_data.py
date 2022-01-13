import numpy as np
import matplotlib.pyplot as plt
import bayestme_plot as bp
import bleeding_correction as bleed
import re
import os
import warnings
from sklearn.model_selection import KFold
import utils
import configparser

class RawSTData:
    def __init__(self, data_name, load=None, raw_count=None, positions=None, tissue_mask=None, gene_names=None, layout=None, storage_path='./', 
                    x_y_swap=False, invert=[0, 0], **kwargs):
        '''
        Inputs:
            load        /path/to/stored/data, if want to load from stored data
            raw_count   gene counts matrix of all spots in the spatial transcriptomics sample
                        (including spots outside tissue if possible)
            position    spatial coordinated of all spots in the spatial transcriptomics sample
                        (including spots outside tissue if possible)
            tissue_mask mask of in-tissue spots
            gene_names  gene names of sequenced genes in the spatial transcriptomics sample
            layout      Visim(hex)  1
                        ST(square)  2 
        '''
        if load:
            self.data_name = data_name
            self.load(load)
            self.n_spot_in = self.Reads.shape[0]
            self.n_gene = self.Reads.shape[1]
        else:
            self.data_name = data_name
            # clean up storage path
            if storage_path[-1] != '/':
                storage_path += '/'
            self.storage_path = storage_path
            if not os.path.isdir(storage_path):
                os.mkdir(storage_path)

            # store raw_count and position
            np.save(storage_path+'raw_count.npy', raw_count)
            np.save(storage_path+'all_spots_position.npy', positions)

            # get gene reads and spatial coordinates of in-tissue spots
            self.Reads = raw_count[tissue_mask]
            self.positions_tissue = positions[:, tissue_mask].astype(int)

            # set up other parameters
            self.tissue_mask = tissue_mask
            self.gene_names = gene_names
            self.layout = layout
            self.storage_path = storage_path

            # set up plotting parameter
            self.x_y_swap = x_y_swap
            self.invert = invert

            self.n_spot_in = self.Reads.shape[0]
            self.n_gene = self.Reads.shape[1]
            self.filtering = np.zeros(self.n_gene).astype(bool)
            self.filter_genes = np.array([])
            self.selected_gene_idx = np.arange(self.n_gene)
            self.save()

    def set_plot_param(self, x_y_swap=False, invert=[0, 0], save=True):
        self.x_y_swap = x_y_swap
        self.invert = invert
        if save:
            self.save()

    def filter(self, n_gene=None, filter_type='ribo', pattern=None, filter_idx=None, spot_threshold=0.95, verbose=False, save=True):
        '''
        data preprocessing
        1.  narrow down number of genes to look at for cell-typing
            select top N gene by the standard deviation across spots
        2.  filter out confounding genes
            built-in filters:
            1)  'spots': universial genes that are observed in more than n% percent of the sample (defualt 95%)
            2)  'ribosome': ribosome genes, i.e. rpl and rps genes
            user can also pass in custom pattern or select gene idx for filtering
        '''
        # order genes by the standard deviation across spots
        top = np.argsort(np.std(np.log(1+self.Reads), axis=0))[::-1]
        # apply n_gene filter
        if n_gene:
            n_gene_filter = min(n_gene, self.n_gene)
            print('filtering top {} genes from original {} genes...'.format(n_gene_filter, self.n_gene))
            n_gene_filter = top[:n_gene_filter]
            self.Reads = self.Reads[:, n_gene_filter]
            self.gene_names = self.gene_names[n_gene_filter]
        else:
            n_gene_filter = top
            self.Reads = self.Reads[:, n_gene_filter]
            self.gene_names = self.gene_names[n_gene_filter]

        # define confounding genes filter
        if filter_type == 'spots':
            # built-in spots filter
            self.filtering = (self.Reads>0).sum(axis=0) >= int(self.Reads.shape[0]*spot_threshold)
            print('filtering out genes observed in {}% spots'.format(spot_threshold*100))
        else:
            if filter_type == 'ribosome':
                # built_in ribosome filter
                pattern = '[Rr][Pp][SsLl]'
                self.filtering = np.array([bool(re.match(pattern, g)) for g in self.gene_names])
            else:
                filter_type = filter_type if filter_type else 'custom'
                if pattern:
                    # user-defined pattern filter
                    self.filtering = np.array([bool(re.match(pattern, g)) for g in self.gene_names])
                elif filter_idx:
                    # user defined gene idx filter
                    self.filtering = np.zeros(self.n_gene).astype(bool)
                    self.filtering[filter_idx] = True
                else:
                    self.selected_gene_idx = n_gene_filter
                    return
            print('filtering out {} genes...'.format(filter_type))

        # apply confounding genes filter
        self.Reads = self.Reads[:, ~self.filtering]
        filtered_genes = self.gene_names[self.filtering]
        self.gene_names = self.gene_names[~self.filtering]
        self.selected_gene_idx = n_gene_filter[~self.filtering]
        self.n_spot_in = self.Reads.shape[0]
        self.n_gene = self.Reads.shape[1]
        print('\t {} genes filtered out'.format(self.filtering.sum()))
        if verbose:
            print(filtered_genes)
        np.save(self.storage_path+'filtered_genes', filtered_genes)
        print('Resulting dataset: {} spots, {} genes'.format(self.n_spot_in, self.n_gene))
        if save:
            self.save()

    def save(self):
        print('Data saved in {}'.format(self.storage_path))
        np.save(self.storage_path+'tissue_mask', self.tissue_mask)
        np.save(self.storage_path+'gene_names', self.gene_names)
        np.save(self.storage_path+'Reads', self.Reads)
        params = np.array([self.layout, self.x_y_swap, self.invert[0], self.invert[1]])
        np.save(self.storage_path+'param', params)
        np.save(self.storage_path+'filtering', self.filtering)
        np.save(self.storage_path+'filter_genes', self.filter_genes)
        np.save(self.storage_path+'selected_gene_idx', self.selected_gene_idx)

    def load(self, load_path, storage_path=None):
        print('Loading data from {}'.format(load_path))
        if not storage_path:
            storage_path = load_path
        else:
            if storage_path[-1] != '/':
                storage_path += '/'
            self.storage_path = storage_path
            if not os.path.isdir(storage_path):
                os.mkdir(storage_path)
        self.storage_path = storage_path
        # loading data
        raw_count = np.load(load_path+'raw_count.npy')
        positions = np.load(load_path+'all_spots_position.npy')
        tissue_mask = np.load(load_path+'tissue_mask.npy')
        gene_names = np.load(load_path+'gene_names.npy', allow_pickle=True)
        param = np.load(load_path+'param.npy')

        self.Reads = np.load(load_path+'Reads.npy')
        self.positions_tissue = positions[:, tissue_mask].astype(int)

        # set up other parameters
        self.tissue_mask = tissue_mask
        self.gene_names = gene_names
        self.layout = param[0]

        # set up plotting parameter
        self.x_y_swap = param[1].astype(bool)
        self.invert = param[-2:]

        # load gene filters
        self.filtering = np.load(self.storage_path+'filtering.npy')
        self.filter_genes = np.load(self.storage_path+'filter_genes.npy')
        self.selected_gene_idx = np.load(self.storage_path+'selected_gene_idx.npy')


    def plot_bleeding(self, gene, cmap='jet', save=False):
        '''
        Plot the raw reads, effective reads, and bleeding (if there is any) of a given gene
        where gene can be selected either by gene name or gene index
        '''
        if isinstance(gene, int):
            gene_idx = gene
        elif isinstance(gene, str):
            gene_idx = np.argwhere(self.gene_names == gene)[0][0]
        else:
            raise Exception('`gene` should be either a gene name(str) or the index of some gene(int)')
        print('Gene: {}'.format(self.gene_names[gene_idx]))
        # load raw reads
        raw_count = np.load(self.storage_path+'raw_count.npy')[:, self.selected_gene_idx[gene_idx]]
        pos = np.load(self.storage_path+'all_spots_position.npy')
        raw_filtered_align = (raw_count[self.tissue_mask] == self.Reads[:, gene_idx]).sum()
        # determine if any bleeding filtering is performed
        if raw_filtered_align == self.n_spot_in:
            print('\t no bleeding filtering performed')
        # calculate bleeding ratio
        all_counts = raw_count.sum()
        tissue_counts = self.Reads[:, gene_idx].sum()
        bleed_ratio = 1-tissue_counts/all_counts
        print('\t {:.3f}% bleeds out'.format(bleed_ratio * 100))

        # plot
        plot_intissue = np.ones_like(raw_count) * np.nan
        plot_intissue[self.tissue_mask] = self.Reads[:, gene_idx]
        plot_outside = raw_count.copy().astype(float)
        plot_outside[self.tissue_mask] = np.nan
        if bleed_ratio == 0:
            plot_data = np.vstack([raw_count, plot_intissue])
            plot_titles = ['Raw Read', 'Reads']
        else:
            plot_data = np.vstack([raw_count, plot_intissue, plot_outside])
            plot_titles = ['Raw Read', 'Reads', 'Bleeding']
        v_min = np.nanpercentile(plot_data, 5, axis=1)
        v_max = np.nanpercentile(plot_data, 95, axis=1)
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10
        if save:
            save = self.storage_path+'gene_bleeding_plots/'
            if not os.path.isdir(save):
                os.mkdir(save)
        bp.st_plot(plot_data, pos, unit_dist=size, cmap=cmap, layout=marker, x_y_swap=self.x_y_swap, invert=self.invert, v_min=v_min, v_max=v_max, subtitles=plot_titles, 
                    name='{}_bleeding_plot'.format(self.gene_names[gene_idx]), save=save)

    def bleeding_correction(self, n_top=50, max_steps=5, n_gene=None):
        return CleanedSTData(stdata=self, n_top=n_top, max_steps=max_steps, n_gene=n_gene)

    def k_fold(self, cluster_storage, n_fold=5, n_splits=15, n_samples=100, n_burn=2000, n_thin=5, lda=0):
        return CrossValidationSTData(stdata=self, cluster_storage=cluster_storage, n_fold=n_fold, n_splits=n_splits, n_samples=n_samples, n_burn=n_burn, n_thin=n_thin, lda=lda)

class CleanedSTData(RawSTData):
    def __init__(self, stdata=None, n_top=50, max_steps=5, n_gene=None, load_path=None):
        if load_path:
            self.load_cleaned(load_path)
        else:
            super().__init__(stdata.data_name, load=stdata.storage_path)
            self.positions = np.load(self.storage_path+'all_spots_position.npy').T
            self.raw_Reads = np.load(self.storage_path+'raw_count.npy')[:, self.selected_gene_idx]
            self.clean_bleed(n_top=n_top, max_steps=max_steps, n_gene=None)
            self.save()
        self.clean_data_plots = self.storage_path + 'cleaned_data_plots/'
        if not os.path.isdir(self.clean_data_plots):
            os.mkdir(self.clean_data_plots)

    def clean_bleed(self, n_top=50, max_steps=5, n_gene=None):
        if not n_gene:
            n_gene = self.n_gene
        basis_idxs, basis_mask = bleed.build_basis_indices(self.positions)
        self.global_rates, fit_Rates, self.basis_functions, self.Weights = bleed.decontaminate_spots(self.raw_Reads[:, :n_gene], self.tissue_mask, basis_idxs, basis_mask, n_top=n_top, max_steps=max_steps)
        self.corrected_Reads = np.round(fit_Rates / fit_Rates.sum(axis=0, keepdims=True) * self.raw_Reads[:, :n_gene].sum(axis=0, keepdims=True))
        self.Reads = self.corrected_Reads[tissue_mask]

    def plot_basis_functions(self):
        basis_names = ['North', 'South', 'West', 'East']
        for d in range(self.basis_functions.shape[0]):
            plt.plot(np.arange(self.basis_functions.shape[1]), self.basis_functions[d], label=basis_names[d])
        plt.xlabel('Distance along cardinal direction')
        plt.ylabel('Relative bleed probability')
        plt.legend(loc='upper right')
        plt.savefig(self.clean_data_plots+'A1_basis_functions.pdf', bbox_inches='tight')
        plt.close()

    def plot_before_after_cleanup(self, gene, cmap='jet', save=False):
        if isinstance(gene, int):
            gene_idx = gene
        elif isinstance(gene, str):
            gene_idx = np.argwhere(self.gene_names == gene)[0][0]
        else:
            raise Exception('`gene` should be either a gene name(str) or the index of some gene(int)')
        print('Gene: {}'.format(self.gene_names[gene_idx]))

        # plot
        plot_data = np.vstack([self.raw_Reads[:, gene_idx], self.corrected_Reads[:, gene_idx]])
        plot_titles = ['Raw Read', 'Corrected Reads']
        v_min = np.nanpercentile(plot_data, 5, axis=1)
        v_max = np.nanpercentile(plot_data, 95, axis=1)
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10
        if save:
            save = self.clean_data_plots+'gene_bleeding_plots/'
            if not os.path.isdir(save):
                os.mkdir(save)
        bp.st_plot(plot_data,  self.positions.T, unit_dist=size, cmap=cmap, layout=marker, x_y_swap=self.x_y_swap, invert=self.invert, v_min=v_min, v_max=v_max, subtitles=plot_titles, 
                    name='{}_bleeding_plot'.format(self.gene_names[gene_idx]), save=save)

    def save(self):
        np.save(self.storage_path+'corrected_Reads', self.corrected_Reads)
        np.save(self.storage_path+'Reads', self.Reads)
        np.save(self.storage_path+'global_rates', self.global_rates)
        np.save(self.storage_path+'basis_functions', self.basis_functions)
        np.save(self.storage_path+'Weights', self.Weights)

    def load_cleaned(self, load_path):
        super().load(load_path)
        self.corrected_Reads = np.load(self.storage_path+'corrected_Reads.npy')
        self.global_rates = np.load(load_path+'global_rates.npy')
        self.basis_functions = np.load(load_path+'basis_functions.npy')
        self.Weights = np.load(load_path+'Weights.npy')
        self.positions = np.load(self.storage_path+'all_spots_position.npy').T
        self.raw_Reads = np.load(self.storage_path+'raw_count.npy')[:, self.selected_gene_idx]

class DeconvolvedSTData(RawSTData):
    def __init__(self, load_path=None, stdata=None, cell_prob_trace=None, expression_trace=None, beta_trace=None, 
                    cell_num_trace=None, lam=None):
        super().__init__(stdata.data_name, load=stdata.storage_path)
        if load_path:
            self.load_deconvolved(load_path)
        else:
            self.cell_prob_trace = cell_prob_trace
            self.expression_trace = expression_trace
            self.beta_trace = beta_trace
            self.cell_num_trace = cell_num_trace
            self.lam = lam
            self.n_components = self.expression_trace.shape[1]
            self.save()

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

    def plot_deconvolution(self, plot_type='cell_prob', cmap='jet', seperate_pdf=False):
        '''
        plot the deconvolution results
        '''
        if self.layout == 1:
            marker = 'H'
            size = 5
        else:
            marker = 's'
            size = 10

        if plot_type == 'cell_prob':
            plot_object = self.cell_prob_trace[:, :, 1:].mean(axis=0)
        elif plot_type == 'cell_num':
            plot_object = self.cell_num_trace[:, :, 1:].mean(axis=0)
        else:
            raise Exception("'plot_type' can only be either 'cell_num' for cell number or 'cell_prob' for cell-type probability")

        if seperate_pdf:
            for i in range(self.n_components):
                bp.st_plot(plot_object[i].T, self.positions_tissue, unit_dist=size, cmap=cmap, x_y_swap=self.x_y_swap, invert=self.invert)
        else:
            bp.st_plot(plot_object.T, self.positions_tissue, unit_dist=size, cmap=cmap, x_y_swap=self.x_y_swap, invert=self.invert)

    def plot_marker_genes(self, n_top=5):
        gene_expression = self.expression_trace.mean(axis=0)
        difference = np.zeros_like(gene_expression)
        n_components = gene_expression.shape[0]
        for k in range(n_components):
            max_exp = gene_expression.max(axis=0)
            difference[k] = gene_expression[k] / max_exp

        fig, ax = plt.subplots(1, 1, figsize=(8, 20))
        for i in range(self.n_components):
            ax.barh(np.arange(n_top*self.n_components)[::-1]+0.35-i*0.1, gene_expression[i][marker_gene_idx], height=0.1, label='cell_type{}'.format(i))
        for i in range(self.n_components-1):
            ax.axhline(ref_gene.flatten().shape[0]/7 * (i+1)-0.45, ls='--', alpha=0.5)
        #     ax.axvline(0)
        ax.set_yticks(np.arange(ref_gene.flatten().shape[0])[::-1])
        ax.set_yticklabels(ref_gene.flatten(), fontsize=20)
        #     ax.set_xlim(-0.01, 0.02)
        ax.margins(x=0.1, y=0.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc=4, fontsize=12)
        ax.set_title('Marker genes from the filtered results', fontsize=20)
        plt.tight_layout()
        plt.savefig('marker_gene_filtered.pdf')
        plt.close()

    def save(self):
        results_path = self.storage_path+'results/'
        print('Saved to {}'.format(results_path))
        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        np.save(results_path+'cell_prob_trace.npy', self.cell_prob_trace)
        np.save(results_path+'expression_trace.npy', self.expression_trace)
        np.save(results_path+'beta_trace.npy', self.beta_trace)
        np.save(results_path+'cell_num_trace.npy', self.cell_num_trace)
        np.save(results_path+'lam.npy', np.array([self.lam]))

    def load_deconvolved(self, load_path):
        self.cell_prob_trace = np.load(load_path+'cell_prob_trace.npy')
        self.expression_trace = np.load(load_path+'expression_trace.npy')
        self.beta_trace = np.load(load_path+'beta_trace.npy')
        self.cell_num_trace = np.load(load_path+'cell_num_trace.npy')
        self.lam = np.load(load_path+'lam.npy')[0]
        self.n_components = self.expression_trace.shape[1]

class CrossValidationSTData(RawSTData):
    def __init__(self, stdata, cluster_storage, n_fold=5, n_splits=15, n_samples=100, n_burn=2000, n_thin=5, lda=0):
        super().__init__(stdata.data_name, stdata.storage_path)
        self.k_fold_path = self.storage_path+'k_fold/'
        if not os.path.isdir(self.k_fold_path):
            os.mkdir(self.k_fold_path)
        self.k_fold_jobs = self.k_fold_path+'jobs/'
        if not os.path.isdir(self.k_fold_jobs):
            os.mkdir(self.k_fold_jobs)
        self.n_fold = n_fold
        self.exc_file = 'grid_search_cfg.py'
        self.save_folds(n_fold=5, n_splits=15)
        self.create_lsf_jobs(cluster_storage, n_samples, n_burn, n_thin, lda)

    def save_folds(self, n_fold=5, n_splits=15):
        self.k_fold_data = self.k_fold_jobs+'data/'
        if not os.path.isdir(self.k_fold_data):
            os.mkdir(self.k_fold_data)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        edges = utils.get_edges(self.positions_tissue, layout=self.layout)
        n_neighbours = np.zeros(self.n_spot_in)
        if self.layout == 1:
            edge_threshold = 5
        else:
            edge_threshold = 3
        for i in range(self.n_spot_in):
            n_neighbours[i] = (edges[:, 0] == i).sum() + (edges[:, 1] == i).sum()
        splits = kf.split(np.arange(self.n_spot_in)[n_neighbours>edge_threshold])
        fig, ax = plt.subplots(1, n_fold, figsize=(6*(n_fold+1), 6))
        for k in range(n_fold):
            _, heldout = next(splits)
            mask = np.array([i in np.arange(self.n_spot_in)[n_neighbours>edge_threshold][heldout] for i in range(self.n_spot_in)])
            train = self.Reads.copy()
            test = self.Reads.copy()
            train[mask] = 0
            test[~mask] = 0
            np.save(self.k_fold_data+'{}_mask_fold{}'.format(self.data_name, k), mask)
            np.save(self.k_fold_data+'{}_fold{}'.format(self.data_name, k), train.astype(int))
            np.save(self.k_fold_data+'{}_test{}'.format(self.data_name, k), test.astype(int))
            bp.plot_spots(ax[k], n_neighbours, self.positions_tissue, s=5, cmap='viridis')
            ax[k].scatter(self.positions_tissue[0, mask], self.positions_tissue[1, mask], s=5, c='r')
        plt.savefig(self.k_fold_path+'{}_masks.pdf'.format(self.data_name))
        np.save(self.k_fold_data+'{}_pos'.format(self.data_name), self.positions_tissue)

    def write_cgf(self, n_samples, n_burn, n_thin, cluster_storage, lda, spatial, folds=5, n_comp_min=2, n_comp_max=15, lams=[1, 1e1, 1e2, 1e3, 1e4, 1e5], max_ncell=120, n_genes=[1000]):
        setup_path = self.k_fold_path + 'setup/'
        if not os.path.isdir(setup_path):
            os.mkdir(setup_path)
        config_root = setup_path + 'config/'

        if not os.path.isdir(config_root):
            os.mkdir(config_root)
        self.config_path = config_root + '{}/'.format(self.data_name)

        results_root = setup_path + 'results/'
        if not os.path.isdir(results_root):
            os.mkdir(results_root)
        self.results_path = results_root + '{}/'.format(self.data_name)
        self.likelihood_path = results_root + '{}/likelihoods/'.format(self.data_name)
        print('results at {}'.format(self.results_path))

        # job log/error outputs storage path
        self.outputs_path = setup_path + 'outputs'
        if not os.path.isdir(self.outputs_path):
            os.mkdir(self.outputs_path)
        print('log/error at {}'.format(self.outputs_path))

        if not os.path.isdir(self.config_path):
            os.mkdir(self.config_path)
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        if not os.path.isdir(self.likelihood_path):
            os.mkdir(self.likelihood_path)

        config = configparser.ConfigParser()
        
        config['setup'] = {
            'n_samples': n_samples,
            'n_burn': n_burn,
            'n_thin': n_thin,
            'exp_name': self.data_name,
            'lda': lda,
            'spatial': spatial,
            'max_ncell': max_ncell,
            'storage_path': cluster_storage+'results/{}/'.format(self.data_name)
        }

        idx = 1
        for n_fold in range(folds):
            for n_comp in range(n_comp_min, n_comp_max+1, 1):
                for lam2 in lams:
                    for n_gene in n_genes:
                        config['exp'] = {
                            'lam_psi': lam2,
                            'n_components': n_comp,
                            'n_fold': n_fold,
                            'n_gene': n_gene
                        }

                        with open(self.config_path+'config_{}.cfg'.format(idx), 'w') as configfile:
                            config.write(configfile)
                        idx += 1
        print('{} jobs generated'.format(idx-1))
        print('\t {} cv folds'.format(folds))
        print('\t {} n_comp grid: {}'.format(n_comp_max-n_comp_min+1, np.arange(n_comp_min, n_comp_max+1, 1)))
        print('\t {} lambda grid: {}'.format(len(lams), lams))
        print('\t {} n_gene grid: {}'.format(len(n_genes), n_genes))
        return idx-1

    def create_lsf_jobs(self, cluster_storage, n_samples=100, n_burn=2000, n_thin=5, lda=0, time_limit=96, mem_req=24):
        n_exp = self.write_cgf(n_samples, n_burn, n_thin, cluster_storage, lda=lda, spatial=self.layout, folds=self.n_fold, n_genes=[self.n_gene])
        jobsfile = self.k_fold_jobs+'{}.sh'.format(self.data_name)
        f = open(jobsfile, 'w')
        job = """#!/usr/bin/env bash
#BSUB -W {6}:00
#BSUB -R rusage[mem={7}]
#BSUB -J {5}[1-{0}]
#BSUB -e {4}/{5}_%I.err
#BSUB -eo {4}/{5}_%I.out

cd $LS_SUBCWD
python {2} --config {1}config_${{LSB_JOBINDEX}}.cfg
"""
        f.write(job.format(n_exp, cluster_storage+'config/{}/'.format(self.data_name), self.exc_file, self.results_path, cluster_storage+'outputs', self.data_name, time_limit, mem_req))







        