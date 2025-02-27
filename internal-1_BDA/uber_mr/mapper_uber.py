import sys
import csv

def mapper():
    for line in sys.stdin:
        data = list(csv.reader([line.strip()]))[0]

        if len(data) < 4 or data[0].lower() == 'date/time':  
            continue

        print(f"{data[0].split(' ')[0]}\t1")

if __name__ == "__main__":
    mapper()
