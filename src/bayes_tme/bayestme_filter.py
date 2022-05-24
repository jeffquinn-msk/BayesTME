import argparse
from . import bayestme

parser = argparse.ArgumentParser(description='Filter data')
parser.add_argument('--data-dir', type=str,
                    help='input data dir')
parser.add_argument('--output-dir', type=str,
                    help='output data dir')
parser.add_argument('--n-gene', type=int,
                    help='number of genese')
parser.add_argument('--filter-type', type=str,
                    help='filter type')

