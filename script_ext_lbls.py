'''
>>> print scipy.sparse.find(a['group'][1,:])
(array([0], dtype=int32), array([7], dtype=int32), array([ 1.]))
>>> 
'''

import sys
import os
import scipy.io, scipy.sparse
import numpy

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage:{0} <dta_fn>'.format(sys.argv[0])
        sys.exit()

    dta_fn = sys.argv[1]
    save_fn = os.path.splitext(dta_fn)[0] + '.lbl'

    dta = scipy.io.loadmat(dta_fn)

    lbl = scipy.sparse.find(dta['group'])

    with open(save_fn,'w') as fid:
        for idx, l in zip(lbl[0], lbl[1]):
            print>>fid, idx, l

    print 'Done with', save_fn



    
