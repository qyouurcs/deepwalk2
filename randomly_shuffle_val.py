#!/usr/bin/python
#randomly shuffle the lines of the list file and output one as train files list and the other one as validation files list.

import os
import sys
import numpy as np
import pdb

if len(sys.argv)< 4:
    print 'len(sys.argv) = ', len(sys.argv)
    ratio = 0.1
    if len(sys.argv) < 3:
        print 'Usage: python randomly_shuffle_val.py <file_list> <save_prefix> [ratio]\n'
        sys.exit()
else:
    ratio = float(sys.argv[3])

root_dir = ''
fid = open(sys.argv[1],'r')
prefix = sys.argv[2]
fn_list = fid.readlines()
fid.close()

rand_perm = range(len(fn_list))
np.random.shuffle(rand_perm)

val_num = int(len(fn_list) * ratio)
train_num = len(fn_list) - val_num


print 'A total of {0} val file and {1} train file'.format(val_num, train_num)

fn_array = np.array(fn_list)

val_list = fn_array[rand_perm[0:val_num]]
train_list = fn_array[rand_perm[val_num:-1]]


print 'len of val {0} and len of train {1}'.format(len(val_list), len(train_list))


fid = open(prefix + '_train.txt','w')
for train_fn in train_list:
    train_fn = train_fn.replace('\\', '/')
    fid.write(root_dir + train_fn)
fid.close()



fid = open(prefix + '_val.txt', 'w')

for val_fn in val_list:
    val_fn = val_fn.replace('\\', '/')
    fid.write(root_dir + val_fn)
fid.close()
