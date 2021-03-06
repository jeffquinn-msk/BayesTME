import numpy as np
# from CRT_rvs import crt_rvs
from scipy.stats import binom
from sklearn.decomposition import LatentDirichletAllocation


from . import utils
from .HMM_fast import HMM, transition_mat_vec
from .gfbt_multinomial import GraphFusedBinomialTree


def transition_mat(phi, n_max, coeff, ifsigma=False):
    # get the binomial transition matrix
    T = np.zeros((n_max, n_max))
    if ifsigma:
        p = utils.ilogit(phi)
    else:
        p = phi
    p_s = p ** np.arange(n_max)
    p_f = (1 - p) ** np.arange(n_max)
    for n in range(n_max):
        T[n, :(n + 1)] = coeff[n] * p_s[:(n + 1)] * p_f[:(n + 1)][::-1]
    return T


class GraphFusedMultinomial:
    def __init__(self, n_components, edges, Observations, n_gene=300, n_max=120, background_noise=False, random_seed=0,
                 mask=None,
                 c=4, D=30, tf_order_psi=0, lam_psi=1e-2, lda_initialization=False, known_cell_num=None,
                 known_spots=None,
                 Truth_expression=None, Truth_prob=None, Truth_cellnum=None, Truth_reads=None, Truth_beta=None,
                 **kwargs):
        np.random.seed(random_seed)
        self.n_components = n_components
        self.n_max = n_max
        self.n_gene = n_gene
        self.edges = edges
        self.bkg = background_noise
        self.HMM = HMM(self.n_components, self.n_max)
        self.gtf_psi = GraphFusedBinomialTree(self.n_components + 1, edges, lam2=lam_psi)
        self.mask = mask
        self.n_nodes = self.gtf_psi.n_nodes

        # initialize cell-type probs
        if Truth_prob is not None:
            self.probs = Truth_prob
        else:
            self.probs = np.ones(self.gtf_psi.probs.shape) * 1 / self.n_components
            self.probs[:, 0] = 0.5

        # initialize gene expression profile
        self.alpha = np.ones(self.n_gene)
        if Truth_expression is not None:
            self.phi = Truth_expression
        elif lda_initialization:
            print('Initializing with lda')
            lda = LatentDirichletAllocation(n_components=self.n_components)
            lda.fit(Observations)
            self.phi = lda.components_ / lda.components_.sum(axis=1)[:, None]
            self.probs[:, 1:] = lda.transform(Observations)
        else:
            self.phi = np.random.dirichlet(self.alpha, size=n_components)
        if self.bkg:
            bkg = np.ones(self.n_gene) / self.n_gene
            self.phi = np.vstack((self.phi, bkg))

        if self.bkg:
            self.cell_num = np.zeros((self.n_nodes, self.n_components + 2)).astype(int)
        else:
            self.cell_num = np.zeros((self.n_nodes, self.n_components + 1)).astype(int)
        if Truth_cellnum is not None:
            self.cell_num[:, :-1] = Truth_cellnum
            self.cell_num[:, -1] = 1
        else:
            self.cell_num[:, 0] = np.random.binomial(self.n_max, self.probs[:, 0])
            if self.bkg:
                self.cell_num[:, -1] = 1
                self.cell_num[:, 1:-1] = utils.multinomial_rvs(self.cell_num[:, 0], p=self.probs[:, 1:])
            else:
                self.cell_num[:, 1:] = utils.multinomial_rvs(self.cell_num[:, 0], p=self.probs[:, 1:])

        if mask is not None:
            spot_count = Observations[~mask].sum(axis=1)
        else:
            spot_count = Observations.sum(axis=1)
        mu = spot_count.sum() / (self.n_nodes * D)
        L = np.percentile(spot_count, 5) / D
        U = np.percentile(spot_count, 95) / D
        s = max(mu - L, U - mu)
        self.a_beta = c ** 2 * mu ** 2 / s ** 2
        self.b_beta = c ** 2 * mu / s ** 2
        if Truth_beta is not None:
            self.beta = Truth_beta
        else:
            self.beta = np.random.gamma(self.a_beta, 1 / self.b_beta, size=n_components)
        if self.bkg:
            self.beta = np.concatenate([self.beta, [np.min([Observations.sum(axis=1).min(), 100])]])

        if Truth_reads is not None:
            self.reads = Truth_reads

        # get the transition matrices
        self.Transition = np.zeros((self.n_nodes, self.n_components - 1, self.n_max + 1, self.n_max + 1))
        self.expression = self.beta[:, None] * self.phi

    def sample_reads(self, Observations):
        '''
        sample cell-type-wise reads of each gene at each spot, R_igk
        reads:            N*G*K   R_igk
        Observation:      N*G     R_ig      observed data, gene reads at each spot
        assignment_probs: N*G*K   xi_igk    multinational prob
        cell_num:         N*K     d_ik      cell-type-wise cell count in each spot, d_ik, N*K
        betas:            K       beta_k    expected cell-type-wise total gene expression of individual cells
        '''
        self.expression = self.beta[:, None] * self.phi
        expected_counts = self.cell_num[:, 1:, None] * self.expression[None]
        self.assignment_probs = expected_counts / np.clip(expected_counts.sum(axis=1, keepdims=True), 1e-20, None)
        self.assignment_probs = np.transpose(self.assignment_probs, [0, 2, 1])
        # multinational draw for all spots
        self.reads = utils.multinomial_rvs(Observations, self.assignment_probs)

    def sample_phi(self):
        '''
        sample cell-type-wise gene expression profile, phi_kg
        '''
        phi_posteriors = self.alpha[None] + self.reads.sum(axis=0).T
        self.phi = np.array([np.random.dirichlet(c) for c in phi_posteriors])

    def sample_cell_num(self):
        '''
        sample the cell-type-wise cell count 
        cell_num[i] = [D_i, d_i1, d_i2, ..., d_iK], where d_i1 + d_i2 + ... + d_iK = D_i
        D_i     total cell number in spot i
        d_ik    cell number of cell-type k in spot i
        '''
        with np.errstate(divide='ignore'):
            prob = 1 - utils.ilogit(self.gtf_psi.Thetas[:, 1:])
            # print(np.argwhere(np.isinf(np.exp(-self.gtf_psi.Thetas[:, 1:]))))
            start_prob = np.array(
                [binom.logpmf(np.arange(self.n_max + 1), self.n_max, p=self.probs[i, 0]) for i in range(self.n_nodes)])
            self.Transition = transition_mat_vec(prob, self.n_max + 1)
            if self.bkg:
                self.cell_num[:, :-1] = self.HMM.ffbs(np.transpose(self.reads[:, :, :-1], [0, 2, 1]), start_prob,
                                                      LogTransition=np.log(self.Transition),
                                                      expression=self.expression[:-1])
            else:
                self.cell_num = self.HMM.ffbs(np.transpose(self.reads, [0, 2, 1]), start_prob,
                                              LogTransition=np.log(self.Transition), expression=self.expression)

    def sample_probs(self):
        '''
        sample cell-type probability psi_ik with spatial smoothing
        '''
        # clean up the GFTB input cell num
        if self.bkg:
            cell_num = self.cell_num[:, :-1].copy()
        else:
            cell_num = self.cell_num.copy()
        cell_num[:, 0] = self.n_max - cell_num[:, 0]
        # GFTB sampling
        if self.mask is not None:
            cell_num[self.mask] = 0
        self.gtf_psi.resample(cell_num)
        # clean up the cell-type prob
        self.probs = self.gtf_psi.probs
        self.probs[:, 0] = 1 - self.probs[:, 0]
        self.probs[:, 1:] /= self.probs[:, 1:].sum(axis=1, keepdims=True)

    def sample_beta(self):
        '''
        sample expected cell-type-wise total cellular gene expression
        '''
        R_k = self.reads.sum(axis=(0, 1))
        d_k = self.cell_num[:, 1:].sum(axis=0)
        self.beta = np.random.gamma(R_k + self.a_beta, 1 / (d_k + self.b_beta))

    def sample(self, Obs):
        self.sample_reads(Obs)
        self.sample_phi()
        self.sample_cell_num()
        self.sample_probs()
        self.sample_beta()

    def load_model(self, load_dir=''):
        # load model parameters
        self.cell_num = np.load(load_dir + 'cell_num.npy')
        self.beta = np.load(load_dir + 'beta.npy')
        self.phi = np.load(load_dir + 'phi.npy')
        self.probs = np.load(load_dir + 'probs.npy')
        self.reads = np.load(load_dir + 'reads.npy')
        # load spatial smoothing parameters
        self.gtf_psi.Thetas = np.save(save_dir + 'checkpoint_Thetas')
        self.gtf_psi.Omegas = np.save(save_dir + 'checkpoint_Omegas')
        self.gtf_psi.Tau2 = np.save(save_dir + 'checkpoint_Tau2')
        self.gtf_psi.Tau2_a = np.save(save_dir + 'checkpoint_Tau2_a')
        self.gtf_psi.Tau2_b = np.save(save_dir + 'checkpoint_Tau2_b')
        self.gtf_psi.Tau2_c = np.save(save_dir + 'checkpoint_Tau2_c')

    def save_model(self, save_dir=''):
        np.save(save_dir + 'checkpoint_cell_num.npy', self.cell_num)
        np.save(save_dir + 'checkpoint_beta.npy', self.beta)
        np.save(save_dir + 'checkpoint_phi.npy', self.phi)
        np.save(save_dir + 'checkpoint_probs.npy', self.probs)
        np.save(save_dir + 'checkpoint_reads.npy', self.reads)
        np.save(save_dir + 'checkpoint_Thetas', self.gtf_psi.Thetas)
        np.save(save_dir + 'checkpoint_Omegas', self.gtf_psi.Omegas)
        np.save(save_dir + 'checkpoint_Tau2', self.gtf_psi.Tau2)
        np.save(save_dir + 'checkpoint_Tau2_a', self.gtf_psi.Tau2_a)
        np.save(save_dir + 'checkpoint_Tau2_b', self.gtf_psi.Tau2_b)
        np.save(save_dir + 'checkpoint_Tau2_c', self.gtf_psi.Tau2_c)
