import numpy as np
import spiceypy as sp
import timing as t

# Leap Seconds Kernel
sp.furnsh('./NH_SPICE_KERNELS/data/lsk/naif0012.tls')

#SP-Kernels (ephemerides)
#2012-12-31T23:58:52.816 - 2014-12-08T05:12:10.384 (contains rehearsal)
sp.furnsh('./NH_SPICE_KERNELS/data/spk/nh_recon_od117_v01.bsp')
#2014-12-08T05:12:10.384 - 2015-09-27T06:12:36.818 (contains flyby)
sp.furnsh('./NH_SPICE_KERNELS/data/spk/nh_recon_pluto_od122_v01.bsp')

# C-Kernels (spacecraft orientations)
# 2013 is when the flyby rehearsal was. Load it first so will be searched last.
sp.furnsh('./NH_SPICE_KERNELS/data/ck/merged_nhpc_2013_v001.bc')
# Load 2015 C-Kernel for the actual flyby.
sp.furnsh('./NH_SPICE_KERNELS/data/ck/merged_nhpc_2015_v001.bc')

# Spacecraft Clock Kernel (SCLK)
sp.furnsh('./NH_SPICE_KERNELS/data/sclk/new_horizons_1454.tsc')

# Frames Kernel (reference frames for spacecraft and each instrument) 
sp.furnsh('./NH_SPICE_KERNELS/data/fk/nh_v220.tf')

# Instrument Kernels (instrument FOV and mounting information)
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_alice_v120u.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_lorri_v201.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_pepssi_v110.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_ralph_v100u.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_rex_v100.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_sdc_v101.ti')
sp.furnsh('./NH_SPICE_KERNELS/data/ik/nh_swap_v200.ti')

############################################################################

# Define some times in ET that may be useful
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

def NH_J2000(et):
    position, _ = sp.spkpos('New Horizons', et, 'Pluto', 'NONE', 'Pluto')
    return position

if __name__ == '__main__':
    print np.linalg.norm( NH_J2000(t.flyby_start) )/1187
