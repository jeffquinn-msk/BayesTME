import argparse
from . import bayestme

parser = argparse.ArgumentParser(description='Deconvolve data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--count-mat', type=str,
                    help='count mat file')


def main():
    args = parser.parse_args()

    reader = bayestme.BayesTME(storage_path=args.data_dir)
    stdata = reader.load_data_from_count_mat(args.count_mat)
    stdata.k_fold(args.data_dir)
