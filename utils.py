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

import sys
import regex
import argparse
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from Bio import SeqIO


class Formatter(argparse.HelpFormatter):
    """
    Extended HelpFormatter class to correct the greediness of --columns
    that includes the last positional argument. This extension of HelpFormatter
    brings the positional argument to the beginning of the command and the
    optional arguments are sent to the end.
    """
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage: '
        if usage is not None:
            usage = usage % dict(prog=self._prog)
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)
            action_usage = self._format_actions_usage(actions, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])
        return f'{prefix}{usage}\n\n'


def verbosify(verbose, message, flush=False):
    """
    Handle verbosity for the user.

    Supports both Boolean and numerical levels of verbosity.
    """
    if verbose and not flush:
        sys.stdout.write(message + "\n")
    elif flush:
        sys.stdout.write(message + "..." + " " * (77 - len(message)) + "\r")
        sys.stdout.flush()


def fasta_fetcher(fasta_file, number_to_fetch, seq_size, verbose=False):
    """
    Fetch sequences from a FASTA file and return a defined number of
    random sequences or random windows from random sequences if seq_size is
    not 0.

    number_to_fetch = 0 takes everything
    seq_size = 0 takes full-length sequences

    Returns a dictionary: {Description: sequence}
    """
    fas_dic = OrderedDict()
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        if len(seq.seq) > seq_size and seq_size != 0:
            r_int = np.random.randint(0, len(seq.seq) - seq_size)
            fas_dic[seq.description] = str(seq.seq)[r_int:r_int + seq_size]
        else:
            fas_dic[seq.description] = str(seq.seq)

    verbosify(verbose, "File fetched")
    if number_to_fetch == 0:
        return fas_dic
    else:
        randomize = np.random.permutation(len(fas_dic))
        return {list(fas_dic.keys())[i]: fas_dic[list(fas_dic.keys())[i]].strip('N').strip('n')
                for i in randomize[:number_to_fetch]}


def fasta_str_fetcher(fasta_string, verbose=False):
    """
    Fetch sequences from a FASTA string.

    Returns a dictionary: {Description: sequence}
    """
    fas_dic = OrderedDict()
    for instance in regex.split(r'\r\n>|\\n>|>', fasta_string)[1:]:
        description, seq = regex.split(r'\r\n|\n', instance, maxsplit=1)
        fas_dic[description] = regex.sub(r'\r\n|\n', '', seq)
    return fas_dic


def kmer_transfo(df_, depth, sequence_column, verbose=False):
    """
    Define sequences by their k-mer proportions and return a larger
    dataframe containing it.

    depth: Length of k-mers (e.g., 1 for monomers, 2 for dimers, etc.)
    Returns a pandas DataFrame.
    """
    df = df_.copy()
    nts = ['A', 'U', 'C', 'G']
    kmers = []

    # Generate k-mers
    def generate_kmers(depth, prefix=""):
        if depth == 0:
            kmers.append(prefix)
            return
        for nt in nts:
            generate_kmers(depth - 1, prefix + nt)

    generate_kmers(depth)

    # Initialize k-mer columns
    for kmer in kmers:
        df[kmer] = 0.0

    # Calculate k-mer frequencies
    for row in df.index:
        sequence = df.loc[row, sequence_column].upper().replace('T', 'U')
        kmer_counts = Counter(regex.findall(f'.{{{depth}}}', sequence, overlapped=True))
        total_kmers = sum(kmer_counts.values())
        for kmer, count in kmer_counts.items():
            if "N" not in kmer:
                df.loc[row, kmer] = count / total_kmers

    verbosify(verbose, "K-mer transformation completed")
    return df


def trimer_transfo(df_, sequence_column, verbose=False):
    """
    Define sequences by their trimer proportions and return a larger
    dataframe containing it.

    This version always considers overlapping trimers.
    Returns a pandas DataFrame.
    """
    df = df_.copy()
    nts = ['A', 'U', 'C', 'G']
    trimers = [f"{nt1}{nt2}{nt3}" for nt1 in nts for nt2 in nts for nt3 in nts]

    for trimer in trimers:
        pattern = f"(?P<{trimer}>{trimer[0]}(?={trimer[1]}{trimer[2]}))"
        df[trimer] = df[sequence_column].str.upper().str.replace(
            'T', 'U').str.count(pattern) / (df[sequence_column].str.len() - 2)

    verbosify(verbose, "Trimer transformation completed")
    return df