#!/usr/bin/env python3

#    Identification of potential RNA G-quadruplexes by G4RNA screener.
#    Copyright (C) 2018  Jean-Michel Garant
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import pickle
import argparse
import warnings
import pandas as pd
import utils
import g4base

# Suppress warnings for numpy dependencies
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def apply_network(ann, fasta, columns, wdw_len, wdw_step, bedgraph=None, verbose=False):
    """
    Apply the ANN object to the sequences given in a FASTA file or FASTA string.
    """
    # Define columns if "all" is specified
    if "all" in columns:
        columns = [
            'gene_symbol', 'mrnaAcc', 'protAcc', 'gene_stable_id',
            'transcript_stable_id', 'full_name', 'HGNC_id', 'identifier',
            'source', 'genome_assembly', 'chromosome', 'start', 'end', 'strand',
            'length', 'sequence', 'cGcC', 'G4H', 'G4NN'
        ]

    columns_to_drop = []
    # Ensure essential columns are included
    for essential in ['length', 'sequence', 'g4']:
        if essential not in columns:
            columns.append(essential)
            columns_to_drop.append(essential)

    # Handle FASTA input (string or file)
    if isinstance(fasta, str):
        RNome_df = g4base.gen_G4RNA_df(
            utils.fasta_str_fetcher(fasta, verbose=verbose),
            columns, 1, int(wdw_len), int(wdw_step), verbose=verbose
        )
    else:
        RNome_df = g4base.gen_G4RNA_df(
            utils.fasta_fetcher(fasta, 0, 0, verbose=verbose),
            columns, 1, int(wdw_len), int(wdw_step), verbose=verbose
        )

    # Load ANN and apply transformations if G4NN is in columns
    if 'G4NN' in columns:
        ann = pickle.load(ann)
        RNome_trans_df = utils.trimer_transfo(RNome_df, 'sequence', verbose=verbose)
        RNome_df = g4base.submit_seq(
            ann, RNome_trans_df.drop('G4NN', axis=1),
            [c for c in columns if c != 'G4NN'], "G4NN", verbose=verbose
        )

    # Write BedGraph header if requested
    if bedgraph:
        chromosome = RNome_df['chromosome'].dropna().iloc[0]
        start_min = RNome_df[RNome_df.chromosome == chromosome].start.min()
        end_max = RNome_df[RNome_df.chromosome == chromosome].end.max()
        sys.stdout.write(f'browser position {chromosome}:{start_min}-{end_max}\n')
        sys.stdout.write(f'track type=bedGraph name={RNome_df.drop(columns_to_drop, axis=1).columns[-1]} '
                         'visibility=full color=200,0,0\n')

    return RNome_df.drop(columns_to_drop, axis=1)


def screen_usage(error_value=False, error_message=None):
    """
    Provide the user with instructions to use screen.py.
    """
    print("Usage: PATH/TO/screen.py [OPTIONS...]")
    print("Use -? or --help to show this message")
    print("Use -V or --version to show program version\n")
    print("Apply options:")
    print("  -a, --ann       \tSupply a pickled ANN (.pkl format)")
    print("  -f, --fasta     \tSupply a FASTA file (.fa .fas format)")
    print("  -w, --window    \tWindow length")
    print("  -s, --step      \tStep length between windows")
    print("  -b, --bedgraph  \tDisplay output as BedGraph, user must provide columns")
    print("  -c, --columns   \tColumns to display: gene_symbol, sequence, ...")
    print("                  \tTo browse available columns use: -c list\n")

    if error_value and error_message:
        sys.stderr.write(f"UsageError: {error_message}\n\n")
        sys.exit(error_value)
    else:
        sys.exit(0)


def arguments():
    """
    Manage command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Identification of potential RNA G-quadruplexes",
        epilog="G4RNA screener Copyright (C) 2018 Jean-Michel Garant. "
               "This program comes with ABSOLUTELY NO WARRANTY. This is free "
               "software, and you are welcome to redistribute it under certain "
               "conditions <http://www.gnu.org/licenses/>."
    )

    parser.add_argument('FASTA', type=argparse.FileType('r'), help='FASTA file (.fa .fas .fasta), - for STDIN')
    parser.add_argument("-a", "--ann", type=argparse.FileType('rb'),
                        default=os.path.join(os.path.dirname(__file__), "G4RNA_2016-11-07.pkl"),
                        help="Supply a pickled ANN (default: G4RNA_2016-11-07.pkl)")
    parser.add_argument("-w", "--window", type=int, default=60, help="Window length (default: 60)")
    parser.add_argument("-s", "--step", type=int, default=10, help="Step length between windows (default: 10)")
    parser.add_argument("-c", "--columns", nargs="+", default=["description", "sequence", "start", "cGcC", "G4H", "G4NN"],
                        help="Columns to display (default: description sequence start cGcC G4H G4NN)")
    parser.add_argument("-b", "--bedgraph", action="store_true", help="Display output as BedGraph")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-V", "--version", action="version", version="G4RNA screener 0.3")
    parser.add_argument("-e", "--error", action="store_true", help="Raise errors and exceptions")

    return parser


def main():
    """
    Main function to handle arguments and execute the program.
    """
    parser = arguments()
    args = parser.parse_args()

    # Validate BedGraph options
    if args.bedgraph and (
            len(args.columns) != 4 or
            args.columns[:3] != ['chromosome', 'start', 'end'] or
            args.columns[-1] not in ['cGcC', 'G4H', 'G4NN']):
        parser.print_usage()
        sys.stderr.write(f"{parser.prog}: error: BedGraph format requires 4 ordered columns: "
                         "chromosome start end [SCORE], where [SCORE] is either cGcC, G4H, or G4NN\n")
        sys.exit(1)

    try:
        apply_network(
            args.ann, args.FASTA, args.columns, args.window, args.step,
            args.bedgraph, args.verbose
        ).to_csv(
            path_or_buf=sys.stdout, sep='\t',
            index=not args.bedgraph, header=not args.bedgraph
        )
    except Exception as e:
        if args.error:
            raise
        else:
            parser.print_usage()
            sys.stderr.write(f"{parser.prog}: error: {str(e)}\n")
            sys.exit(1)


if __name__ == '__main__':
    main()