import argparse
from bayes_tme import bayestme

parser = argparse.ArgumentParser(description='Filter data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--count-mat', type=str,
                    help='count mat file')
parser.add_argument('--output-dir', type=str,
                    help='output data dir')
parser.add_argument('--n-gene', type=int,
                    help='number of genese')
parser.add_argument('--filter-type', type=str,
                    help='filter type')


def main():
    args = parser.parse_args()

    reader = bayestme.BayesTME(storage_path=args.data_dir)
    stdata = reader.load_data_from_count_mat(args.count_mat)
    stdata.filter(n_gene=args.n_gene, filter_type=args.filter_type)
    stdata.bleeding_correction()

