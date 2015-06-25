from __future__ import division
import sys

args = sys.argv[1:]

if not args:
    print './cmd label1 label2'
    sys.exit(-1)

label1, label2 = args

with open(label1) as f, open(label2) as g:
    l1 = [int(i) for i in f.read().split()]
    l2 = [int(i) for i in g.read().split()]

    eqs = 0
    ueqs = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            eqs += 1
        else:
            ueqs += 1

print 'error:\t', ueqs / (eqs + ueqs)

