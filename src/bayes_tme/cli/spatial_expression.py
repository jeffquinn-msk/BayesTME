import argparse
from bayes_tme import bayestme
from bayes_tme import bayestme_data

parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--n-spatial-patterns', type=int,
                    help='number of spatial patterns')
parser.add_argument('--n-samples', type=int,
                    help='number of samples')
parser.add_argument('--n-burnin', type=int,
                    help='burnin iterations')
parser.add_argument('--n-thin', type=int,
                    help='thin iterations')
parser.add_argument('--simple', action='store_true',
                    help='simple mode')


def main():
    args = parser.parse_args()

    obj = bayestme.BayesTME(storage_path=args.data_dir)

    stdata = bayestme_data.DeconvolvedSTData(load_path=args.data_dir)

    obj.spatial_expression(
        stdata,
        n_spatial_patterns=args.n_spatial_patterns,
        n_samples=args.n_samples,
        n_burn=args.n_burnin,
        n_thin=args.n_thin,
        simple=args.simple)
