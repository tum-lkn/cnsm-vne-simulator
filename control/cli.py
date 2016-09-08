# -*- coding: utf-8 -*-
# TODO add description

import argparse

__authors__ = "Johannes Zerwas, Andreas Blenk"


def create_cl_argparser():
    """
    Creates a command line argument parser. The options can be used to overwrite values given in the configuration file
    Returns:
        Instance of ArgumentParser
    """
    # TODO extend or add config file support
    # TODO think about default values
    # TODO think about types

    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('-c', '--configfile',
                        default='control/configuration_default.ini')
    parser.add_argument('-t', '--simtime',
                        help='maximum simulation time?!')
    parser.add_argument('--rmin',
                        help='min ids of runs to solve')
    parser.add_argument('--rmax',
                        help='max ids of runs to solve')

    # Network model configuration
    parser.add_argument('-s', '--substrate',
                        help="substrate that should be run")
    parser.add_argument('-v', '--vnets',
                        help='num of vnets to run')
    parser.add_argument('-mcs' '--multi-controller-switches',
                        help='num of sdn++ switches')
    parser.add_argument('--hypervisor-split',
                        help='split hypervisor paths')

    # Algorithms configuration
    parser.add_argument('-algo', '--algorithm',
                        help='the problem specific algorithm')
    parser.add_argument('-kc', '--kcontroller',
                        help="maximum number of controllers")
    parser.add_argument('-kh', '--khypervisors',
                        help="maximum number of hypervisors")
    parser.add_argument('--mipgap',
                        help="adjust the precision of the solver")
    parser.add_argument('--ctropt',
                        help='position of vsdn ctrs optimized')
    parser.add_argument('--arrproc',
                        help='Name of the arrival process')
    parser.add_argument('-objs', '--objectives',
                        help='objectives to run simulation for')
    parser.add_argument("--ignore_ctr",
                        action='store_true',
                        default=False,
                        dest='ignore_ctr',
                        help='?')
    parser.add_argument("--preselection",
                        default='',
                        help='?')

    return parser
