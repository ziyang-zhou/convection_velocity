from copy import copy
import os
import numpy as np
from antares import *
import h5py
import pdb
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import pandas as pd

# ---------------------
# Defined functions
# ---------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mov_avg(X,k):
    X_new = X
    for i in range(k//2,X_new.size-k//2):
        X_new[i] = sum(X[i-k//2:i+k//2])/k
    return X_new

#############################################################################################
#                                        Load Inputs                                        #
#############################################################################################

#Declare parameter
settings = pd.read_csv("setting.csv", index_col= 0)
start_inst = int(eval(settings.at["start_inst", settings.columns[0]]))
ninst = int(eval(settings.at["ninst", settings.columns[0]]))
chunk_size = int(eval(settings.at["chunk_size", settings.columns[0]]))
max_del_star = eval(settings.at["max_del_star", settings.columns[0]])

var = 'pressure'
time_list = []

#Probe location
c=0.1356
loc = int(eval(settings.at["loc", settings.columns[0]]))

streamwise_range_ndim = eval(settings.at["streamwise_range_ndim", settings.columns[0]])
streamwise_range_res = int(eval(settings.at["streamwise_range_res", settings.columns[0]]))

if loc==21:
    probe = [-0.0192,0.00794] #probe21 location 
    delta_star_NS = 0.00133789468758723 #[m]
    Ue = 17.5997632176
elif loc==24:
    probe = [-0.01066,0.00474] #probe24 location 
    delta_star_NS = 0.00176700515795414 #[m]
    Ue = 16.9780903136
elif loc==9:
    probe = [-0.467*c,0.019] #[m]   #probe9 location 
    delta_star_NS = 0.000435456945922567 #[m]
    Ue = 20.8394223712
elif loc==7:
    probe = [-0.080961,0.02210] #[m]   #probe7 location    
    delta_star_NS = 0.000352361395373204  #[m]  
    Ue = 21.0653551168;                                             

delta_star_NS = max_del_star
       
grid_size = streamwise_range_ndim*delta_star_NS/streamwise_range_res #grid size calculated using max del star to be consistent with struc grid generation
streamwise_range = streamwise_range_ndim*delta_star_NS
stencil_size = int(streamwise_range/grid_size)#number of grid points in streamwise extent of the stencil
print('streamwise_range_ndim',streamwise_range_ndim)
print('delta_star_NS',delta_star_NS)
print('stencil size',stencil_size)

# ------------------
# Reading the files
# ------------------

#Read the CFD data set
file = '../../sherfwh/tr-meas-surface.hdf5'
fd = h5py.File(file,'r')

x = fd['Geometry']['X']
y = fd['Geometry']['Y']
z = fd['Geometry']['Z']

x_mask = (np.array(x) < probe[0] + 1.1*streamwise_range/2) & (np.array(x) > probe[0] - 1.1*streamwise_range/2)

base_cfd = Base()
base_cfd.init()
base_cfd[0].shared['x']=x[x_mask]
base_cfd[0].shared['y']=y[x_mask]
base_cfd[0].shared['z']=z[x_mask]

# ----------------------------------------
# Import the structured grid from cfd data set
# ----------------------------------------

myr = Reader('hdf_antares')
myr['filename'] = '../interpolation_2d_grid.h5'
base_grid = myr.read()

x_grid = base_grid[0][0]['x']
y_grid = base_grid[0][0]['y']
z_grid = base_grid[0][0]['z']

indx = int(find_nearest(x_grid[:,0],probe[0]))
indz = int(find_nearest(z_grid[0,:],0)) #find midspan location
n_col_z_grid = len(z_grid[0,:])
n_row_x_grid = len(x_grid[:,0])

stencil_max_x = min(n_row_x_grid,int(indx + stencil_size/2))
stencil_min_x = max(0,int(indx - stencil_size/2))
stencil_max_z = min(n_col_z_grid,int(indz + stencil_size/4)) 
stencil_min_z = max(0,int(indz - stencil_size/4))

end_inst = start_inst + ninst
n_chunk = int(ninst/chunk_size)

for n in range(n_chunk):
# ----------------------------------------------------
# Load the timeframes in the CFD data
# ----------------------------------------------------
	for i in range(chunk_size):
	  current_inst = i + start_inst + n*chunk_size
	  base_cfd[0]['{0:06d}'.format(current_inst)] = Instant()
	  time_list.append(fd['Frame_{0:06d}'.format(current_inst)]['time'][0])
	  base_cfd[0]['{0:06d}'.format(current_inst)][var] = fd['Frame_{0:06d}'.format(current_inst)][var][x_mask].flatten()
	  print('instant',i,'of',ninst,'read')
	  print('size of import domain',len(x))
	  print('size of truncated stencil',len(np.array(base_cfd[0][0]['x'])))
	del base_cfd[0][0]
# ----------------------------------------------------
# Create the new structured base to contain cfd data
# ----------------------------------------------------
	base_st = Base()
	base_st.init()
	base_st[0].shared['x']=x_grid[int(indx - stencil_size/2) : int(indx + stencil_size/2),int(indz - stencil_size/4) : int(indz + stencil_size/4)]
	base_st[0].shared['y']=y_grid[int(indx - stencil_size/2) : int(indx + stencil_size/2),int(indz - stencil_size/4) : int(indz + stencil_size/4)]
	
	base_st[0].shared['z']=z_grid[int(indx - stencil_size/2) : int(indx + stencil_size/2),int(indz - stencil_size/4) : int(indz + stencil_size/4)]
	
	for i in range(chunk_size):
	  current_inst = i + start_inst + n*chunk_size
	  base_st[0]['{0:06d}'.format(current_inst)] = Instant()
	del base_st[0][0]
# ----------------------------------------------------
# Do the interpolation and write the interpolated file
# ----------------------------------------------------
	print('interpolating chunk {}...'.format(n))
	myt = Treatment('interpolation')
	myt['source'] = base_cfd
	myt['target'] = base_st
	base_cfd_on_grid = myt.execute()

	print('writing interpolated data of chunk {}...'.format(n))
	myw = Writer('hdf_antares')
	myw['filename'] = '../probe_{}_structured/interpolated_meas_surface_pressure_{}_{}'.format(loc,start_inst+n*chunk_size,start_inst+(n+1)*(chunk_size))
	myw['base'] = base_cfd_on_grid
	myw.dump()
	
	del base_st
	
