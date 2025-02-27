import sys

def reducer():
    current_date, current_count = None, 0

    for line in sys.stdin:
        date, count = line.strip().split('\t')

        try:
            count = int(count)
        except ValueError:
            continue

        if date == current_date:
            current_count += count
        else:
            if current_date:
                print(f"{current_date}\t{current_count}")

            current_date, current_count = date, count

    if current_date:
        print(f"{current_date}\t{current_count}")

if __name__ == "__main__":
    reducer()
