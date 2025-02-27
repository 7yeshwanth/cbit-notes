#!/usr/bin/env python2
import sys

current_word, current_count = None, 0

for line in sys.stdin:
    word, count = line.strip().split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue  # Ignore invalid counts

    if word == current_word:
        current_count += count
    else:
        if current_word is not None:
            print('%s\t%s' % (current_word, current_count))
        current_word, current_count = word, count

# Print last word count
if current_word:
    print('%s\t%s' % (current_word, current_count))
