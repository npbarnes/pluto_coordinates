import numpy as np
import csv
from astropy.io import fits
import spiceypy as sp
import SortedSearch as ss
from matplotlib.ticker import FuncFormatter

# Leap Seconds Kernel
sp.furnsh('/home/nathan/lib/lsk.tls')

# Useful stuff
rehersal_start = sp.str2et('2015-07-12T12:00:00.0000')
rehersal_end   = sp.str2et('2015-07-12T17:00:00.0000')

#flyby_start = sp.str2et('2015-07-14T10:00:00.0000')
#flyby_end   = sp.str2et('2015-07-14T19:00:00.0000')
flyby_start = sp.str2et('2015-07-14T11:00:00.0000')
flyby_end   = sp.str2et('2015-07-14T16:00:00.0000')

close_start = sp.str2et('2015-07-14T11:10:00.0000')
close_end   = sp.str2et('2015-07-14T14:20:00.0000')

stupid_close_start = sp.str2et('2015-07-14T13:02:00.0000')
stupid_close_end   = sp.str2et('2015-07-14T13:10:00.0000')

@FuncFormatter
def et_formatter(tickval, tickpos):
    return sp.timout(tickval, 'HR:MN ::UTC')


@np.vectorize
def convert_JDTDB_ET(jdtdb):
    return sp.unitim(jdtdb, 'JDTDB', 'ET')

# Helpful stuff for JPL Horizons type 3 CSV no object header
csv.register_dialect('horizons', skipinitialspace=True)
header_rows = 23
fields = ['JDTDB', 'TDB', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'LT', 'RG', 'RR']
end_str = '$$EOE'
table_dtype = zip(fields, [np.float64,'S30']+[np.float64]*9)

def read_horizons(filename):
    """Read a type 3 csv file output by the JPL Horizons system (no object header)
    return data as a structured numpy array. 
    """
    with open(filename) as f:
        # Skip the header
        for _ in xrange(header_rows):
            next(f)

        # Define csv reader
        reader = csv.reader(f, dialect='horizons')

        table = []
        for row in reader:
            if row[0] == end_str:
                break

            table.append(tuple(row[:-1]))

        return np.array(table, dtype=table_dtype)

def read_horizons_et(filename):
    """read_horizons and convert times to ET"""
    h = read_horizons(filename)

    # Convert times
    h['JDTDB'] = convert_JDTDB_ET(h['JDTDB'])

    # Rename the column
    h.dtype.names = ['ET'] + fields[1:]

    return h

def read_NH_orientation(filename):
    """Read the NH orientation file that Heather Elliott provided"""
    angles    = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            angles.append((float(row['START_ET']),
                           float(row['THETA_SUN']),
                           float(row['PHI_SUN']),
                           float(row['SPIN_ANGLE'])))
        
    angle_dtype = [('ET', np.float64), ('THETA', np.float64), ('PHI', np.float64), ('SPIN', np.float64)]
    angles = np.array(angles, dtype=angle_dtype)

    return angles

def interpolate(key, arr, key_field, value_field, reverse=False):
    l_index, l_key = ss.find_le(arr[key_field], key, reverse=reverse)
    r_index, r_key = ss.find_ge(arr[key_field], key, reverse=reverse)

    bin_delta = r_key-l_key

    if bin_delta == 0.0:
        return arr[value_field][l_index]

    key_delta = key-l_key

    fraction = key_delta / bin_delta

    # The interpolation step is complicated by the fact that we want to support lists of value_fields
    if hasattr(value_field, 'capitalize'):# Check if it's a string
        return fraction*arr[value_field][r_index] + (1-fraction)*arr[value_field][l_index]
    else:
        return tuple(np.add(np.multiply(fraction,tuple(arr[value_field][r_index])),
                            np.multiply((1-fraction),tuple(arr[value_field][l_index]))))

def get_NH_pluto_coords():
    if not hasattr(get_NH_pluto_coords, 'NH'):
        get_NH_pluto_coords.NH = read_horizons_et('/home/nathan/Code/coordinates/NH_Sun_J2000.txt')
        get_NH_pluto_coords.Pluto = read_horizons_et('/home/nathan/Code/coordinates/Pluto_Sun_J2000.txt')

    NH = get_NH_pluto_coords.NH 
    Pluto = get_NH_pluto_coords.Pluto

    assert np.all(NH['ET'] == Pluto['ET'])

    # vectors from the sun to NH for each time
    vec_n = np.empty((NH.shape[0],3), dtype=np.float64)
    vec_n[:,0] = NH['X']
    vec_n[:,1] = NH['Y']
    vec_n[:,2] = NH['Z']

    # vectors from the sun to Pluto for each time
    vec_p = np.empty((Pluto.shape[0],3), dtype=np.float64)
    vec_p[:,0] = Pluto['X']
    vec_p[:,1] = Pluto['Y']
    vec_p[:,2] = Pluto['Z']

    # vectors from Pluto to NH for each time
    vec_nprime = vec_n - vec_p
    
    # Pluto coordinate unit vectors at each time
    ux = -vec_p/np.sqrt(np.sum(vec_p**2,axis=1))[:,np.newaxis]
    uy = -np.cross(ux,[0,0,1])
    uz = np.cross(ux,uy)

    # Setup the return array
    ret_dtype = [('ET', np.float64), ('X', np.float64), ('Y', np.float64), ('Z', np.float64)]
    ret = np.empty(NH.shape, dtype=ret_dtype)
    ret['ET'] = NH['ET']

    # These einsums are doing row-wise dot products i.e. multiply cooresponding entries and sum over columns.
    # Same as np.sum(a*b, axis=1) but a little faster
    ret['X'] = np.einsum('ij,ij->i',vec_nprime,ux)
    ret['Y'] = np.einsum('ij,ij->i',vec_nprime,uy)
    ret['Z'] = np.einsum('ij,ij->i',vec_nprime,uz)
    
    return ret


def get_NH_pluto_x():
    """return a structured array of times and cooresponding x-coordinates."""

    if not hasattr(get_NH_pluto_x, 'NH'):
        get_NH_pluto_x.NH = read_horizons_et('/home/nathan/Code/coordinates/NH_Pluto_J2000.txt')
        get_NH_pluto_x.Sun = read_horizons_et('/home/nathan/Code/coordinates/Sun_Pluto_J2000.txt')

    NH = get_NH_pluto_x.NH 
    Sun = get_NH_pluto_x.Sun 

    assert np.all(NH['ET'] == Sun['ET'])

    ret_dtype = [('ET', np.float64), ('X', np.float64)]
    ret = np.empty(NH.shape, dtype=ret_dtype)
    ret['ET'] = NH['ET']

    Sun_dir = np.array([Sun['X']/Sun['RG'],
                        Sun['Y']/Sun['RG'],
                        Sun['Z']/Sun['RG']])

    # Dot product of NH position vector and Sun direction unit vector gives pluto x-coordinate
    ret['X'] = NH['X']*Sun_dir[0] + NH['Y']*Sun_dir[1] + NH['Z']*Sun_dir[2]

    return ret

def pos_at_time(et):
    x_table = get_NH_pluto_x()
    return interpolate(et, x_table, key_field='ET', value_field='X')

def coordinate_at_time(et):
    table = get_NH_pluto_coords()
    return interpolate(et, table, key_field='ET', value_field=['X','Y','Z'])

def pos_at_time_utc(utc=None):
    et = sp.unitim(utc, 'UTC', 'ET')
    return pos_at_time()

def time_at_pos(x):
    """Return the interpolated SPICE ET time (TDB seconds since J2000) when NH
    was at the given pluto x-coordinate
    """
    NH_x = get_NH_pluto_x()
    return interpolate(x, NH_x, key_field='X', value_field='ET', reverse=True)

def orientation_at_time(et):
    """Return the (Theta,Phi,Spin) tuple when NH is at the given SPICE ET (TDB
    seconds since J2000)
    """
    # The orientation file should only need to be read once per session
    if not hasattr(orientation_at_time, 'NH_angles'):
        orientation_at_time.NH_angles = read_NH_orientation('/home/nathan/Code/coordinates/flyby_angles.csv')

    return interpolate(et, orientation_at_time.NH_angles, key_field='ET', value_field=['THETA','PHI','SPIN'])

def orientation_at_pos(x):
    """Return the (Theta,Phi,Spin) tuple when NH is at the given pluto x-coordinate"""
    t = time_at_pos(x)
    return orientation_at_time(t)

def trajectory2(t1, t2, dt):
    """Read in New Horizons trajectory and orientation data"""
    times = np.arange(t1,t2,dt)

    pos = np.empty((len(times), 3), dtype=np.float64)
    o   = np.empty((len(times), 3), dtype=np.float64)

    for i,t in enumerate(times):
        pos[i,:] = coordinate_at_time(t)
        o[i,:]   = orientation_at_time(t)

    return pos, o, times

def trajectory(t1, t2, dt):
    """Read in New Horizons trajectory and orientation data"""
    pos, o, times = trajectory2(t1, t2, dt)
    return pos, o

if __name__ == "__main__":
    get_NH_pluto_coords()
