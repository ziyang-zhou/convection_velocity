from copy import copy
import os
import numpy as np
from antares import *
import h5py
import pdb
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import math
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

#Declare parameter
settings = pd.read_csv("setting.csv", index_col= 0)
start_inst = int(eval(settings.at["start_inst", settings.columns[0]]))
ninst = int(eval(settings.at["ninst", settings.columns[0]]))
chunk_size = int(eval(settings.at["chunk_size", settings.columns[0]]))
loc = int(eval(settings.at["loc", settings.columns[0]]))
filter_freq = eval(settings.at["filter_freq", settings.columns[0]])

var = 'pressure'
chord = 0.1356

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

# ------------------
# Reading the files
# ------------------

dt = 6.83583e-06
fs = 1/dt 
ts = 1/fs
loc = int(eval(settings.at["loc", settings.columns[0]]))
streamwise_range_res = int(eval(settings.at["streamwise_range_res", settings.columns[0]]))
max_del_star = eval(settings.at["max_del_star", settings.columns[0]])


end_inst = start_inst + ninst
n_chunk = int(ninst/chunk_size)

for j in range(n_chunk):
	myr = Reader('hdf_antares')
	myr['filename'] = '../probe_{}_structured/interpolated_meas_surface_pressure_{}_{}.h5'.format(loc,start_inst+j*chunk_size,start_inst+(j+1)*(chunk_size))
	base_cfd_on_grid = myr.read()
	
	if j == 0:
		xcoor = base_cfd_on_grid[0][0]['x'][:,0]
		ycoor = base_cfd_on_grid[0][0]['y'][:,0]
		zcoor = base_cfd_on_grid[0][0]['z'][:,0]

	for n,i in enumerate(base_cfd_on_grid[0].keys()):
		pressure_append = np.array(base_cfd_on_grid[0][i][var])
		if (n == 0) and (j == 0):
			pressure = pressure_append
		elif (n == 1) and (j == 0):
			pressure = np.concatenate((pressure[np.newaxis,:,:], pressure_append[np.newaxis,:,:]), axis=0)
		else:
			pressure = np.concatenate((pressure, pressure_append[np.newaxis,:,:]), axis=0)
	print('chunk {} read'.format(j))
indx = int(len(xcoor)/2)

# --------------------------
# Initialize probe location
# --------------------------

if loc==21:
    probe = [-0.0192,0.00794] #probe21 location 
    delta_star_NS = 0.00133789468758723 #[m]
    Ue = 17.5997632176
elif loc==24:
    probe = [-0.01066,0.00474] #probe24 location 
    delta_star_NS = 0.00176700515795414 #[m]
    Ue = 16.9780903136
elif loc==9:
    probe = [-0.467*chord,0.019] #[m]   #probe9 location 
    delta_star_NS = 0.000435456945922567 #[m]
    Ue = 20.8394223712
elif loc==7:
    probe = [-0.080961,0.02210] #[m]   #probe7 location    
    delta_star_NS = 0.000352361395373204  #[m]  
    Ue = 21.0653551168;                                                    

# ------------------------------
# Cross-correlation  calculation
# ------------------------------

nbi = pressure.shape[0]
meanpressure = pressure.mean(axis=0,dtype=np.float64)                                         

#Compute the arithmetic mean along the specified axis.
pressurefluc = pressure - np.tile(meanpressure,(nbi,1,1))   
pfluc = pressurefluc
RxtA = np.zeros((math.ceil(nbi*2-1),np.shape(pfluc)[1],np.shape(pfluc)[2]))
XI = np.zeros((np.shape(pfluc)[1]))
xi_x=np.zeros_like(XI)

for ki in range(0,np.shape(pfluc)[2]):                                          #spanwise index_point                                      
    for l in range(0,np.shape(pfluc)[1]):                                       #streamwise index_point
        p1 = butter_bandpass_filter(pfluc[:,l,ki],filter_freq[0], filter_freq[1], fs, order=2)
        p0 = butter_bandpass_filter(pfluc[:,indx,ki],filter_freq[0], filter_freq[1], fs, order=2)
        coeff = np.sqrt(np.max(np.correlate(p1,p1,"full"))*np.max(np.correlate(p0,p0,"full")))
        c = np.correlate(p0,p1,"full")/coeff                                    
        #argmax:Returns the indices of the maximum values along an axis.
        RxtA[:,l,ki] = c
        XI[l] = np.sqrt((xcoor[indx] - xcoor[l])**2 + (ycoor[indx] - ycoor[l])**2)#distance between correlated points
        xi_x[l] = XI[l]
        
Rxt_spectrum = np.mean(RxtA,axis=2) #R(xi_x,tau)
dt = (np.linspace(0,c.size-1,c.size) - c.size/2)*ts
xi_X = np.zeros_like(XI)
xi_X[int(streamwise_range_res/2):] = XI[int(streamwise_range_res/2):]
xi_X[0:int(streamwise_range_res/2)] = -XI[:int(streamwise_range_res/2)]
xi_x = xi_X

## PLOTS
fig, ax = plt.subplots()
ax.plot(xi_x/delta_star_NS,np.max(Rxt_spectrum,0),label='Sensor %d'%loc)
ax.set_xlabel(r'$\xi_x/\delta^{*}$')
ax.set_ylabel(r'$R(\xi_x,\tau_{max})$')   
ax.set_xlim(0,22)
plt.legend(loc='upper right')
plt.title('Peaks of the longitudinal space-time correlation')
plt.savefig('peak_correlation')


### === contourplot 
for ind in range(0,np.shape(Rxt_spectrum)[0]):
    for ind2 in range(0,np.shape(Rxt_spectrum)[1]):
        if Rxt_spectrum[ind,ind2]<0:
            Rxt_spectrum[ind,ind2]=0
    
X,Y =np.meshgrid(dt,xi_x)
fig,ax = plt.subplots(figsize=(5,8))

CS = ax.contourf(X*16/delta_star_NS, Y/delta_star_NS, np.flip((Rxt_spectrum.T),1),cmap='Greys')
ax.set_xlabel(r'$\Delta t U_0/\delta^{*}$', fontsize=22)
ax.set_ylabel(r'$\xi_x/\delta^{*}$', fontsize=22)

if loc==7 or loc==9:
    ax.set_xlim(-32,32) 
    ax.set_ylim(-28,28)
else:
    ax.set_xlim(-8,8)
    ax.set_ylim(-5.5,5.5)

interval = 20
levels = np.linspace(-25, 25, 51)

troubleshoot = False
if troubleshoot:
	for i in range(0, pfluc.shape[0], interval):
    		plt.figure()
    		plt.contourf(pfluc[i, :, :],levels=levels, extend='both')
    		plt.title(f'Contour Plot at Index {i}')
    		plt.colorbar()
    		plt.savefig(f'troubleshoot/contour_plot_{i}.png')  # Save the figure with a unique name
    		plt.close()  # Close the current figure to prevent overlap in subsequent plots

plt.savefig('contour_convection_velocity')

# Create a new HDF5 file
with h5py.File('probe_{}_xcorr_{}Hz_to_{}Hz.h5'.format(loc,filter_freq[0],filter_freq[1]), 'w') as hf:
    # Save X
    hf.create_dataset('X', data=X)

    # Save Y
    hf.create_dataset('Y', data=Y)

    # Save R_xt_spectrum.T
    hf.create_dataset('Rxt_spectrum', data=Rxt_spectrum)
