#!/usr/bin/env python2
import sys

for line in sys.stdin:
    for word in line.strip().split():
        print('%s\t1' % word)  # Output: word \t 1
