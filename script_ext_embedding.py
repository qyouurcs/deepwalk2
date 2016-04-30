#!/usr/bin/python

import os
import sys
import cPickle
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <model_fn>'.format(sys.argv[0])
        sys.exit()

    model_fn = sys.argv[1]

    vals = cPickle.load(open(model_fn))
    emb_vals = vals['emb_values'][0]

    save_fn = os.path.splitext(model_fn)[0] + '.fea'

    with open(save_fn,'w') as fid:
        for i in xrange(emb_vals.shape[0]):
            i_fea = ' '.join([ str(val) for val in emb_vals[i,:] ])
            print >>fid, i, i_fea

    print 'Done with', save_fn

    
