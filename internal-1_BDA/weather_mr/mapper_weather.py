#!/usr/bin/env python3
import sys

def mapper():
    for line in sys.stdin:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        date = parts[2].split('_')[0]  # Extract YYYYMMDD
        try:
            temp = float(parts[3])  # Extract temperature
            print(f"{date}\t{temp}")
        except ValueError:
            continue

if __name__ == "__main__":
    mapper()
