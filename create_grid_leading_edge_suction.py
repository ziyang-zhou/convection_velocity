#!/usr/bin/env python
# coding: utf-8

# In[1]:

from antares import *

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sintp
import scipy.ndimage as sim
import h5py
import pdb

#############################################################################################
#                                        Load Inputs                                        #
#############################################################################################
# Load the settings dataframe:
settings = pd.read_csv("setting.csv", index_col= 0)
le_cut = eval(settings.at["le_cut", settings.columns[0]])
te_cut = eval(settings.at["te_cut", settings.columns[0]])
include_pressure_side = settings.at["include_pressure_side", settings.columns[0]]
refinement_factor = float(settings.at["refinement_factor", settings.columns[0]])
max_del_star = eval(settings.at["max_del_star", settings.columns[0]])
streamwise_range_ndim = eval(settings.at["streamwise_range_ndim", settings.columns[0]])
streamwise_range_res = int(eval(settings.at["streamwise_range_res", settings.columns[0]]))
span_size = eval(settings.at["span_size", settings.columns[0]])
loc = int(eval(settings.at["loc", settings.columns[0]]))
#############################################################################################
#                                        Perform geometry extraction                        #
#############################################################################################
angle = '8' #attack angle of the CD airfoil
span_size = span_size*0.1356 #dimensionalize the span size
file = '../tr-meas-surface_first_70511.hdf5'
b = h5py.File(file,'r')
#Investigate content of file
print('zones in base b',b.keys())
print('instants in base b',b['Geometry'].keys())
# In[4]:
xvals=[-0.0307167,-0.0293987,-0.000383135,-0.00220858,-0.0307167]
yvals=[0.0120501,0.0166085,0.00613008,0.00174928,0.0120501]
# In[5]:
nskip = 10000
print('Reading x coordinate list')
x_coord_list = list(b['Geometry']['X'])[::nskip]
print('Reading y coordinate list')
y_coord_list = list(b['Geometry']['Y'])[::nskip]
print('Coordinates have been read successfully')
# Convert lists to NumPy arrays
x_coord = np.array(x_coord_list)
y_coord = np.array(y_coord_list)
#Cut off the LE and TE radii to simplify the geometry
mask = (x_coord >= le_cut) & (x_coord <= te_cut)
x_coord = x_coord[mask]
y_coord = y_coord[mask]
print('x_coord min',min(x_coord))
print('y_coord max',max(y_coord))


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


#Define a piecewise mean camber line to differentiate suction points from pressure points
if angle == '15':
	def f(x):
		# Define the points for the first line segment
		x1 = -0.13
		y1 = 0.027
		x2 = -0.06
		y2 = 0.015
		# Define the points for the second line segment
		x3 = -0.06
		y3 = 0.015
		x4 = 0.0
		y4 = -0.009  
		# Check which line segment to use based on the value of x
		if x <= x2:
			# Equation of the first line segment (y = mx + b)
			m=(y2-y1)/(x2-x1)  
			return m*x+0.005
		else:
			m=(y4-y3)/(x4-x3)
			# Handle values of x outside the defined segments
			return  m*x-0.009# You can choose to return a default value or raise an error
elif angle == '8':
	def f(x):
		# Define the points for the first line segment
		x1 = -0.14
		y1 = 0.019
		x2 = -0.08
		y2 = 0.019
		# Define the points for the second line segment
		x3 = -0.08
		y3 = 0.019
		x4 = 0.0
		y4 = 0.0  
		# Check which line segment to use based on the value of x
		if x <= x2:
			# Equation of the first line segment (y = mx + b)
			return y2
		else:
			m=(y4-y3)/(x4-x3)
			# Handle values of x outside the defined segments
			return  m*x# You can choose to return a default value or raise an error

#Sort all points into two lists based on whether they are above or below the mean camber line
# Create empty lists for the two subgroups
x_coord_suction = []
y_coord_suction = []
x_coord_pressure = []
y_coord_pressure = []
# Iterate through the coordinates and sort them into subgroups
for i in range(len(x_coord)):
    x = x_coord[i]
    y = y_coord[i]
    mean_camber = f(x)  # Calculate the mean camber for the current x
    print('mean_camber',mean_camber)
    print('y',y)
    if y > mean_camber:
        # If y is greater than the mean camber, add to the greater subgroup
        x_coord_suction.append(x)
        y_coord_suction.append(y)
    else:
        # If y is smaller than or equal to the mean camber, add to the smaller subgroup
        x_coord_pressure.append(x)
        y_coord_pressure.append(y)
# Convert the lists to NumPy arrays if needed

x_coord_pressure = np.array(x_coord_pressure)
y_coord_pressure = np.array(y_coord_pressure)
x_coord_suction = np.array(x_coord_suction)
y_coord_suction = np.array(y_coord_suction)

#Sort all points according to order of ascending x-coordinate
sort_indices = np.argsort(x_coord_pressure)
sort_indices_descending = sort_indices[::-1]
x_coord_pressure_sorted_descending = x_coord_pressure[sort_indices_descending]
y_coord_pressure_sorted_descending = y_coord_pressure[sort_indices_descending]
# Get the indices that would sort x_coord_suction in ascending order
sort_indices_suction = np.argsort(x_coord_suction)
x_coord_suction_ascending = x_coord_suction[sort_indices_suction]
y_coord_suction_ascending = y_coord_suction[sort_indices_suction]

print('x_coord_pressure_sorted_descending min',min(x_coord_pressure_sorted_descending))
print('y_coord_pressure_sorted_descending max',max(y_coord_pressure_sorted_descending))

print('x_coord_suction_ascending min',min(x_coord_suction_ascending))
print('y_coord_suction_ascending max',max(y_coord_suction_ascending))

if include_pressure_side is True:
	x_coord = np.concatenate((x_coord_pressure_sorted_descending, x_coord_suction_ascending),axis=0)
	y_coord = np.concatenate((y_coord_pressure_sorted_descending, y_coord_suction_ascending),axis=0)
else:
	x_coord = x_coord_suction_ascending
	y_coord = y_coord_suction_ascending

plt.figure()
plt.plot(x_coord_suction_ascending,y_coord_suction_ascending)
plt.savefig('domain_plot_suction_side.png')
print('full domain saved')

plt.figure()
plt.plot(x_coord_pressure_sorted_descending,y_coord_pressure_sorted_descending)
plt.savefig('domain_plot_pressure_side.png')
print('full domain saved')

# In[6]:

xmin = le_cut
xmax = te_cut

keep=(x_coord>xmin)*(x_coord<xmax)
plt.figure()
plt.plot(xvals,yvals,linestyle='dashed')
plt.plot(x_coord[keep],y_coord[keep])
plt.axis('equal')
plt.savefig('domain_extent.png')
print('extracted domain saved')


# In[7]:

#creation of interpolation function which takes streamwise coordinate as input and outputs cartesian coordinates
xprof = x_coord[keep]
yprof = y_coord[keep]
ds = np.sqrt((xprof[1:]-xprof[:-1])**2 + (yprof[1:]-yprof[:-1])**2)
sprof = np.zeros(ds.size+1,)
sprof[1:] = np.cumsum(ds)
ls = sprof[-1]

fx = sintp.interp1d(sprof,xprof)
fy = sintp.interp1d(sprof,yprof)


# In[8]:

#declaration of dr - step size in new curvilinear array of streamwise coordinate.


zmin = max(-streamwise_range_ndim*max_del_star/2,-span_size/2)
zmax = min(streamwise_range_ndim*max_del_star/2,span_size/2)

dr = streamwise_range_ndim*max_del_star/streamwise_range_res

# In[9]:
#resample the curvilinear coordinates to make them equidistant
#create a corresponding set of x and y coordinates
vec_s = np.arange(0,ls,dr)
npts_prof = vec_s.size

vec_x_prof = fx(vec_s)
vec_y_prof = fy(vec_s)

vec_t_prof = np.zeros((npts_prof,2))
#create a new array of vectors vec_t_prof which contains unit vectors of the displacement between neighbouring points
for iz in range(1,npts_prof-1):
    tx_dn = vec_x_prof[iz+1]-vec_x_prof[iz]
    ty_dn = vec_y_prof[iz+1]-vec_y_prof[iz]
    tnorm = np.sqrt(tx_dn**2+ty_dn**2)
    tx_dn = tx_dn/tnorm
    ty_dn = ty_dn/tnorm

    tx_up = vec_x_prof[iz]-vec_x_prof[iz-1]
    ty_up = vec_y_prof[iz]-vec_y_prof[iz-1]
    tnorm = np.sqrt(tx_up**2+ty_up**2)
    tx_up = tx_up/tnorm
    ty_up = ty_up/tnorm
    
    vec_t_prof[iz,0] = 0.5 * (tx_up + tx_dn)
    vec_t_prof[iz,1] = 0.5 * (ty_up + ty_dn)

tx_dn = vec_x_prof[1]-vec_x_prof[0]
ty_dn = vec_y_prof[1]-vec_y_prof[0]
tnorm = np.sqrt(tx_dn**2+ty_dn**2)
vec_t_prof[0,0] = tx_dn/tnorm
vec_t_prof[0,1] = ty_dn/tnorm

tx_up = vec_x_prof[-1]-vec_x_prof[-2]
ty_up = vec_y_prof[-1]-vec_y_prof[-2]
tnorm = np.sqrt(tx_up**2+ty_up**2)
vec_t_prof[-1,0] = tx_up/tnorm
vec_t_prof[-1,1] = ty_up/tnorm

vec_n_prof = np.zeros((npts_prof,3))
vec_n_prof[:,0] = -sim.gaussian_filter1d(vec_t_prof[:,1],sigma=10, order=0, mode='nearest')
vec_n_prof[:,1] = sim.gaussian_filter1d(vec_t_prof[:,0],sigma=10, order=0, mode='nearest')

plt.figure(figsize=(15,10))
plt.plot(vec_x_prof,vec_n_prof[:,1])
plt.savefig('surface_vector.png')

# In[10]:
plt.figure()
plt.plot(vec_x_prof,vec_y_prof,'.')
plt.quiver(vec_x_prof,vec_y_prof,vec_n_prof[:,0],vec_n_prof[:,1])
plt.plot(xvals,yvals,linestyle='dashed')
plt.plot(x_coord[keep],y_coord[keep])
plt.axis('equal')
plt.savefig('surface_vector_2.png')

# In[11]:

dn0 = 20e-6
dn_max = 120e-6
dn_q = 1.03
N = 50
Nn = 73
dn = np.zeros(Nn,)
for idx in range(Nn):
    dn[idx] = min(dn0*dn_q**idx,dn_max)

vec_n = np.zeros(Nn+1,)
vec_n[1:] = np.cumsum(dn)
    
plt.figure()
plt.plot(vec_n,'o')
plt.savefig('normal_vector.png')


# In[12]:


Xmat = np.zeros((npts_prof,Nn+1))
Ymat = np.zeros((npts_prof,Nn+1))
for idx,nv in enumerate(vec_n):
    Xmat[:,idx] = vec_x_prof + nv*vec_n_prof[:,0]
    Ymat[:,idx] = vec_y_prof + nv*vec_n_prof[:,1]
    
plt.figure()
plt.contourf(Xmat,Ymat,np.ones_like(Xmat),linestyles='solid')
plt.axis('equal')
plt.savefig('surface_grid')

# In[13]:

bi = Base()
bi.init()
bi[0][0]['x']=Xmat
bi[0][0]['y']=Ymat
w=Writer('bin_tp')
w['base']=bi
w['filename']='interpolation_grid.plt'
w.dump()

# In[14]:

vec_z = np.arange(zmin+dr/2,zmax+dr/2,dr)
Nz = vec_z.size
print((zmax - zmin )/dr)
print(vec_z[-1])
print(Nz)

plt.plot([0,0],[zmin,zmin],'o')
plt.plot([Nz,Nz],[zmax,zmax],'s')
plt.plot(vec_z)
plt.show()

# In[14.5]:
#Create structured surface grid of the airfoil
X_surf = vec_x_prof
Y_surf = vec_y_prof
Z_surf = vec_z
Nx = vec_x_prof.size

X_surface = np.repeat(X_surf[:,np.newaxis],Nz,axis=1)
Y_surface = np.repeat(Y_surf[:,np.newaxis],Nz,axis=1)
Z_surface = np.repeat(Z_surf[np.newaxis,:],Nx,axis=0)

bs = Base()
bs.init()
bs[0][0]['x']=X_surface
bs[0][0]['y']=Y_surface
bs[0][0]['z']=Z_surface

w=Writer('hdf_antares')
w['filename']='interpolation_2d_grid'
w['base']=bs
w.dump()


