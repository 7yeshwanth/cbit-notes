#!/usr/bin/env python3
import sys

def reducer():
    current_date, max_temp = None, float('-inf')

    for line in sys.stdin:
        date, temp = line.strip().split('\t')

        try:
            temp = float(temp)
        except ValueError:
            continue

        if date != current_date:
            if current_date is not None:
                print(f"{current_date}\t{max_temp}")

            current_date, max_temp = date, temp
        else:
            max_temp = max(max_temp, temp)

    if current_date is not None:
        print(f"{current_date}\t{max_temp}")

if __name__ == "__main__":
    reducer()
