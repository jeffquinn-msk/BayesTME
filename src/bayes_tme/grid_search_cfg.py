import numpy as np
import argparse
import configparser
import os
import pathlib
from scipy.stats import multinomial

from . import utils
from .model_bkg import GraphFusedMultinomial


parser = argparse.ArgumentParser(description='GFMM modeling on st data')
parser.add_argument('--config', type=str, default='semi_syn_1.cfg',
                    help='configration file')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--output-dir', type=str,
                    help='output data dir')


def main():
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    lam_psi = float(config['exp']['lam_psi'])
    n_samples = int(config['setup']['n_samples'])
    n_burn = int(config['setup']['n_burn'])
    n_thin = int(config['setup']['n_thin'])
    n_components = int(config['exp']['n_components'])
    exp_name = config['setup']['exp_name']
    n_fold = int(config['exp']['n_fold'])
    lda = int(config['setup']['lda'])
    spatial = int(config['setup']['spatial'])
    max_ncell = int(config['setup']['max_ncell'])
    n_gene_raw = int(config['exp']['n_gene'])
    storage_path = config['setup']['storage_path']

    pos_ss = np.load(os.path.join(args.data_dir, '{}_pos.npy'.format(exp_name)))
    test = np.load(os.path.join(args.data_dir, '{}_test{}.npy'.format(exp_name, n_fold)))
    train = np.load(os.path.join(args.data_dir, '{}_fold{}.npy'.format(exp_name, n_fold)))
    n_gene = min(train.shape[1], n_gene_raw)
    top = np.argsort(np.std(np.log(1 + train), axis=0))[::-1]
    train = train[:, top[:n_gene]]
    test = test[:, top[:n_gene]]
    mask = np.load(os.path.join(args.data_dir, '{}_mask_fold{}.npy'.format(exp_name, n_fold)))
    if spatial == 0:
        edges = utils.get_edges(pos_ss, layout=2)
    else:
        edges = utils.get_edges(pos_ss, layout=1)

    # gene_filter = ~(train>train.mean()).all(axis=0)
    # train = train[:, gene_filter]
    # test = test[:, gene_filter]
    n_nodes = train.shape[0]

    heldout_spots = np.argwhere(mask).flatten()
    train_spots = np.argwhere(~mask).flatten()
    if len(heldout_spots) == 0:
        mask = None

    print('experiment: {}, lambda {}, {} components, fold {}, {} spots heldout'.format(exp_name, lam_psi, n_components,
                                                                                       n_fold, len(heldout_spots)))
    print('\t {} lda, {} spatial, {} max cells, {}({}) gene'.format(lda, spatial, max_ncell, n_gene, n_gene_raw))
    print('sampling: {} burn_in, {} samples, {} thinning'.format(n_burn, n_samples, n_thin))
    print('storage: {}'.format(storage_path))

    gfnb = GraphFusedMultinomial(n_components=n_components, edges=edges, Observations=train, n_gene=n_gene,
                                 lam_psi=lam_psi,
                                 background_noise=False, lda_initialization=False, mask=mask, n_max=max_ncell)

    cell_prob_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    cell_num_trace = np.zeros((n_samples, n_nodes, n_components + 1))
    expression_trace = np.zeros((n_samples, n_components, n_gene))
    beta_trace = np.zeros((n_samples, n_components))
    # reads_trace = np.zeros((n_samples, n_nodes, n_gene, n_components))
    loglhtest_trace = np.zeros(n_samples)
    loglhtrain_trace = np.zeros(n_samples)

    pathlib.Path(os.path.join(args.output_dir, 'likelihoods')).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_cell_prob_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), cell_prob_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_phi_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), expression_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_beta_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), beta_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_cell_num_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), cell_num_trace)
    np.save(os.path.join(args.output_dir,
                         'likelihoods/{}_{}_{}_train_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                     n_components, lam_psi, n_fold)),
            loglhtrain_trace)
    np.save(os.path.join(args.output_dir,
                         'likelihoods/{}_{}_{}_test_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                    n_components, lam_psi, n_fold)),
            loglhtest_trace)
    for step in range(n_samples * n_thin + n_burn):
        if step % 10 == 0:
            print(f'Step {step}')
        # perform Gibbs sampling
        gfnb.sample(train)
        # save the trace of GFMM parameters
        if step >= n_burn and (step - n_burn) % n_thin == 0:
            idx = (step - n_burn) // n_thin
            cell_prob_trace[idx] = gfnb.probs
            expression_trace[idx] = gfnb.phi
            beta_trace[idx] = gfnb.beta
            cell_num_trace[idx] = gfnb.cell_num
            rates = (gfnb.probs[:, 1:][:, :, None] * (gfnb.beta[:, None] * gfnb.phi)[None])
            nb_probs = rates.sum(axis=1) / rates.sum(axis=(1, 2))[:, None]
            loglhtest_trace[idx] = np.array(
                [multinomial.logpmf(test[i], test[i].sum(), nb_probs[i]) for i in heldout_spots]).sum()
            loglhtrain_trace[idx] = np.array(
                [multinomial.logpmf(train[i], train[i].sum(), nb_probs[i]) for i in train_spots]).sum()
            print('{}, {}'.format(loglhtrain_trace[idx], loglhtest_trace[idx]))

    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_cell_prob_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), cell_prob_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_phi_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), expression_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_beta_post_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components,
                                                                  lam_psi, n_fold)), beta_trace)
    np.save(os.path.join(args.output_dir,
                         '{}_{}_{}_cell_num_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell, n_components, lam_psi,
                                                                 n_fold)), cell_num_trace)
    np.save(os.path.join(args.output_dir,
                         'likelihoods/{}_{}_{}_train_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                     n_components, lam_psi, n_fold)),
            loglhtrain_trace)
    np.save(os.path.join(args.output_dir,
                         'likelihoods/{}_{}_{}_test_likelihood_{}_{}_{}.npy'.format(exp_name, n_gene_raw, max_ncell,
                                                                                    n_components, lam_psi, n_fold)),
            loglhtest_trace)


if __name__ == '__main__':
    main()
