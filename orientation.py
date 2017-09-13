import numpy as np
import csv
import spiceypy as sp
import itertools

# Leap Seconds Kernel
sp.furnsh('/home/nathan/lib/lsk.tls')

# Useful stuff
time_format = '%Y-%m-%dT%H:%M:%S.%f'
flyby_start = sp.str2et('2015-07-14T10:00:00.0000')
flyby_end   = sp.str2et('2015-07-14T19:00:00.0000')

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), ..."""
    a, b = itertools.tee(iterable)# Make two independent iterables from the original
    next(b, None)
    return itertools.izip(a, b)

def isolate_timeslice(newfilename, t1, t2):
    """Cut down the file so it starts with the last row before t1
    and ends with the first row after t2"""
    with open(newfilename, 'wb') as newfile:
        with open('swap_angles.csv') as oldfile:
            writer = csv.writer(newfile)
            reader = csv.reader(oldfile)
            # First write the field names
            writer.writerow(next(reader))

            # Then record the entries from the correct timeslice
            for row, next_row in pairwise(reader):
                t = float(row[3]) # START_ET field
                next_t = float(next_row[3])

                if next_t < t1:
                    continue
                if t > t2:
                    writer.writerow(row)
                    break

                writer.writerow(row)

if __name__ == "__main__":
    isolate_timeslice('flyby_angles.csv', flyby_start-6000, flyby_end+6000)

