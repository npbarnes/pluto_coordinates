import numpy as np
import spiceypy as sp
from math import sin, cos, radians
from datetime import datetime
from matplotlib.dates import date2num, num2date

# Leap Seconds Kernel
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/lsk/naif0012.tls')

#SP-Kernels (ephemerides)
#2012-12-31T23:58:52.816 - 2014-12-08T05:12:10.384 (contains rehearsal)
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/spk/nh_recon_od117_v01.bsp')
#2014-12-08T05:12:10.384 - 2015-09-27T06:12:36.818 (contains flyby)
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/spk/nh_recon_pluto_od122_v01.bsp')

# C-Kernels (spacecraft orientations)
# 2013 is when the flyby rehearsal was. Load it first so will be searched last.
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ck/merged_nhpc_2013_v001.bc')
# Load 2015 C-Kernel for the actual flyby.
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ck/merged_nhpc_2015_v001.bc')

# Spacecraft Clock Kernel (SCLK)
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/sclk/new_horizons_1454.tsc')

# Frames Kernel
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/fk/nh_v220.tf')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/fk/heliospheric_v004u.tf')
sp.furnsh('/home/nathan/Code/coordinates/pluto_frames_v001.tf')

# Planetary Constants Kernel (PCK)
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/pck/pck00010.tpc')

# Instrument Kernels (instrument FOV and mounting information)
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_alice_v120u.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_lorri_v201.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_pepssi_v110.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_ralph_v100u.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_rex_v100.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_sdc_v101.ti')
sp.furnsh('/home/nathan/Code/coordinates/NH_SPICE_KERNELS/data/ik/nh_swap_v200.ti')

############################################################################

# Define some times in ET that may be useful
# Rehersal_start/end and flyby_start/end are aligned so that flyby maneuvers
# are sychronized reasonably well.
rehearsal_start   = sp.str2et('2013-07-12T11:58:22.0000')
rehearsal_end     = sp.str2et('2013-07-12T16:58:32.0000')
rehearsal_lmargin = sp.str2et('2014-07-12T11:00:00.0000')
rehearsal_rmargin = sp.str2et('2014-07-12T18:00:00.0000')

flyby_start = sp.str2et('2015-07-14T10:59:51.0000')
flyby_end   = sp.str2et('2015-07-14T16:00:01.0000')
flyby_lmargin = sp.str2et('2015-07-14T10:00:00.0000')
flyby_rmargin   = sp.str2et('2015-07-14T17:00:00.0000')
nh_in_wake   = sp.str2et('2015-07-14T12:30:00.0000')

# Other commonly used variables
pluto_id_code = sp.bodn2c('Pluto')
nh_body_code = sp.bodn2c('New Horizons')
nh_inst_code = -98000
swap_inst_code = sp.bodn2c('NH_SWAP')

def et2pydatetime(et):
    utc_str = sp.timout(et, 'Mon DD,YYYY  HR:MN:SC ::UTC')
    return datetime.strptime(utc_str, '%b %d,%Y  %H:%M:%S')

flyby_start_pydatetime = et2pydatetime(flyby_start)
flyby_end_pydatetime = et2pydatetime(flyby_end)
rehearsal_start_pydatetime = et2pydatetime(rehearsal_start)
rehearsal_end_pydatetime = et2pydatetime(rehearsal_end)

def pydatetime2et(dt):
    utc_str = dt.strftime('%Y %B %d, %H:%M:%S (UTC)')
    return sp.str2et(utc_str)

def met2et(met):
    return scs2e(nh_body_code, met)

def met2pydatetime(met):
    """Takes in a full MET string, outputs a datetime object at that time.
    """
    return et2pydatetime( met2et(met) )

def mpldate2et(d):
    return pydatetime2et(num2date(d))

def et2mpldate(et):
    return date2num(et2pydatetime(et))

"""Pasive rotation matricies used to change coordinate systems. These are only for 
backward compatability. Use SPICE C-Matricies instead.
"""
def Rx(a):
    a = radians(a)
    return np.array([   [1.0,0.0,0.0],
                        [0.0,cos(a),sin(a)],
                        [0.0,-sin(a),cos(a)]])
def Ry(a):
    a = radians(a)
    return np.array([   [cos(a),0.0,-sin(a)],
                        [0.0,1.0,0.0],
                        [sin(a),0.0,cos(a)]])
def Rz(a):
    a = radians(a)
    return np.array([   [cos(a),sin(a),0.0],
                        [-sin(a),cos(a),0.0],
                        [0.0,0.0,1.0]])

def R1(o):
    """Given an orientation o = (theta, phi, spin), R1(o) will be the rotation matrix
    to convert pluto coordinates into spacecraft coordinates.
    """
    #return np.linalg.multi_dot([Rz(o[1]),Rx(o[0]),Ry(o[2]),Rz(-90.)])
    return np.linalg.multi_dot([Rz(o[1]),Rx(-o[0]),Ry(o[2]),Rz(-90.)])

def R2(o):
    """Given an orientation o = (theta, phi, spin), R1(o) will be the rotation matrix
    to convert pluto coordinates into spacecraft coordinates.
    """
    return np.linalg.multi_dot([Rz(o[1]),Rx(o[0]),Rz(-90.)])

def look_vectors(v, cmat):
    """Converts velocity vectors to spacecraft look directions."""
    # Negative of the velocity vector in NH coordinates.
    # The einsum just applies the appropriate rotation matrix to each v.
    # We take the negative since it's the look direction; i.e. the
    # direction to look to see that particle coming in.
    return -np.einsum('ij,kj->ki', cmat, v)

def fixall(xs, high, low):
    ret = np.empty_like(xs)
    for i,x in enumerate(xs):
        ret[i] = fix(x,high,low)
    return ret

def fix(x, high, low):
    assert high > low

    while x > high:
        x -= high-low
    while x < low:
        x += high-low

    assert x > low
    assert x <= high

    return x

def vec2tp(l):
    """Convert a look vector into a theta phi pair"""
    _, lon, lat = sp.reclat(l)
    lat = sp.convrt(lat, 'RADIANS', 'DEGREES')
    lon = sp.convrt(lon, 'RADIANS', 'DEGREES')

    return lat, fix( 90.0 - lon , 180.0, -180.0 )

def look_directions(v, cmat):
    """Computes the SWAP look direction phi and theta for each of
    a collection of ion velocities. Returns array of [theta, phi]
    pairs. The cmat argument may be optionally be a [theta, phi, spin]
    triple instead of a SPICE C-Matrix. C-Matricies are prefered;
    t,p,s angles are only supported for backward compatability.
    """
    if cmat.shape == (3,):
        print "Warning: Using (Theta, Phi, Spin) instead of a SPICE C-Matrix"
        cmat = R1(cmat)

    look = look_vectors(v,cmat)
    ret = np.empty((look.shape[0],2), dtype=np.float64)

    for i,l in enumerate(look):
        ret[i,(0,1)] = vec2tp(l)

    return ret

def time_at_pos(targ, crdsys='RECTANGULAR', coord='X', beg_end=None, n_intervals=20, step=60*60, mccomas=False):
    """This is a slightly simplified interface to the SPICE function gfposc.
    See documentation on gfposc for help. This function is only looking for
    individual times and raises and exception if an interval is found or if
    there are multiple solutions. Call gfposc directly if you need either
    of those things.

    targ: position in kilometers
    beg_end: Time interval to search for targ given as a tuple of ET times. 
             Defaults to (flyby_start, flyby_end).
    mccomas: Specifies whether the position is in McComas coordinates
    """
    if mccomas:
        frame = 'PLUTO_MCCOMAS'
    else:
        frame = 'HYBRID_SIMULATION_INTERNAL'

    if beg_end is None:
        beg_end = (flyby_start, flyby_end)

    confine = sp.utils.support_types.SPICEDOUBLE_CELL(2)
    result  = sp.utils.support_types.SPICEDOUBLE_CELL(2*n_intervals)
    sp.wninsd(beg_end[0], beg_end[1], confine)

    sp.gfposc('New Horizons', frame, 'NONE', 'Pluto', crdsys, coord, '=', targ, 0., step, n_intervals, confine, result)

    if result.card != 2:
        raise ValueError('x={} could not be found in the given search interval {}'.format(targ, beg_end))

    endpoints = sp.wnfetd(result,0)
    if endpoints[0] != endpoints[1]:
        raise RuntimeError('A time interval was found instead of a single time.')

    return endpoints[0]

def mpl_date_at_pos(*args, **kwargs):
    """Same as time at position but returns the time as an mpl date instead of ET.
    Note that the beg_end argument is still in ET time.
    """
    return et2mpldate(time_at_pos(*args, **kwargs))

def coordinate_at_time(et, mccomas=False):
    """Get the position coordinate at a certain et time
    et: The time when position is desired
    mccomas: True to output McComas coordinates and False to output internal coordinates
    """
    if mccomas:
        frame = 'PLUTO_MCCOMAS'
    else:
        frame = 'HYBRID_SIMULATION_INTERNAL'

    return sp.spkpos('New Horizons', et, frame, 'NONE', 'Pluto')[0]

def pos_at_time(et, mccomas=False):
    """Get the x coordinate at a certain et time.
    This is included for compatablility. Coordinate_at_time() gives all three components so it
    supersedes this function.

    et: The time when x coordinate is desired
    mccomas: True to output McComas coordinates and False to output internal coordinates
    """
    return coordinate_at_time(et, mccomas)[0]

def pos_at_pydatetime(dt, mccomas=False):
    return pos_at_time(pydatetime2et(dt), mccomas=mccomas)

def pos_at_mpl_date(d, mccomas=False):
    return pos_at_time(mpldate2et(d), mccomas=mccomas)

def pos_at_time_utc(utc):
    et = sp.unitim(utc, 'UTC', 'ET')
    return pos_at_time(et)

def orientation_at_time(et):
    """Return the (Theta,Phi,Spin) tuple when NH is at the given SPICE ET (TDB
    seconds since J2000)
    """
    raise NotImplemented

def cmat_at_time(et, mccomas=False):
    """Get the C-Matrix that converts vectors from either McComas or internal coordinates
    to the New Horizons body frame.
    et: Epoch
    mccomas: True if the vectors to be transformed are in McComas coordinates, False if the
    vectors to be transformed are in internal coordinates
    """
    if mccomas:
        frame = 'PLUTO_MCCOMAS'
    else:
        frame = 'HYBRID_SIMULATION_INTERNAL'

    sclk = sp.sce2c(nh_body_code, et)
    cmat, clkout = sp.ckgp(nh_inst_code, sclk, 0, frame)

    return cmat

def is_sunview(et):
    """Broken for some reason. The function fovtrg() doesn't work for SWAP. I think it's
    getting confused since the FOV is wider than pi/2 radians.
    """
    return sp.fovtrg('NH_SWAP', 'Sun', 'POINT', '', 'NONE', 'New Horizons', et)

def trajectory(t1, t2, dt, mccomas=False):
    """Read in New Horizons trajectory and orientation data
    t1: Start time
    t2: End time
    dt: Time step
    mccomas: True to output positions in McComas coordinates and cmats that transform from McComas coordinates.
        False to output positions in internal coordinates and cmats that transform from internal coordinates.
    """
    times = np.arange(t1,t2,dt)

    pos = np.empty((len(times), 3), dtype=np.float64)
    cmat   = np.empty((len(times), 3,3), dtype=np.float64)

    for i,t in enumerate(times):
        pos[i,:] = coordinate_at_time(t, mccomas=mccomas)
        cmat[i,:,:] = cmat_at_time(t, mccomas=mccomas)

    return pos, cmat, times

