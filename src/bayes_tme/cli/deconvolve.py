import argparse
from bayes_tme import bayestme
from bayes_tme import bayestme_data

parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--n-gene', type=int,
                    help='number of genes')
parser.add_argument('--n-components', type=int,
                    help='number of components')
parser.add_argument('--lam2', type=int,
                    help='lam2')
parser.add_argument('--n-samples', type=int,
                    help='number of samples')
parser.add_argument('--n-burnin', type=int,
                    help='burnin iterations')
parser.add_argument('--n-thin', type=int,
                    help='thin iterations')


def main():
    args = parser.parse_args()

    obj = bayestme.BayesTME(storage_path=args.data_dir)
    stdata = bayestme_data.CleanedSTData(load_path=args.input_dir)

    obj.deconvolve(
        stdata,
        n_gene=args.n_gene,
        n_components=args.n_components,
        lam2=args.lam2,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        n_thin=args.n_thin)
