#!/usr/bin/python

import os
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <fn>'.format(sys.argv[0])
        sys.exit()


    fn = sys.argv[1]
    dict_nodes = {}

    with open(fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            for node in parts:
                if node not in dict_nodes:
                    dict_nodes[node] = 1
    print 'vocab size', len(dict_nodes)
