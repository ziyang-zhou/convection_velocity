import h5py
import numpy as np
import pandas as pd
import pdb
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit


settings = pd.read_csv("setting.csv", index_col= 0)
loc = int(eval(settings.at["loc", settings.columns[0]]))
delta_star_list = eval(settings.at["displacement_thickness", settings.columns[0]])#probe 7 displacement thickness
U_ref_list = eval(settings.at["U_ref_list", settings.columns[0]])
filter_freq = eval(settings.at["filter_freq", settings.columns[0]])

U_inf = U_ref_list[2] # #0:DNS 1:DNS-SLR 2:DNS-SLRT
delta_star = delta_star_list[2] #0:DNS 1:DNS-SLR 2:DNS-SLRT

def linear_regression(X,m,n):
    return m*X+n

def read_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if the variables exist in the file
            if 'X' in f.keys() and 'Y' in f.keys() and 'Rxt_spectrum' in f.keys():
                # Read the variables
                X = f['X'][:,:]
                Y = f['Y'][:,:]
                R_xt_spectrum = f['Rxt_spectrum'][:,:]
                
                return X, Y, R_xt_spectrum
            else:
                print("Variables 'X', 'Y', or 'R_xt_spectrum' not found in the HDF5 file.")
                return None, None, None
    except Exception as e:
        print("Error reading HDF5 file:", e)
        return None, None, None

def calculate_polynomial_regression(X, Y, R_xt_spectrum):
	max_R_x_index = [] #index along time axis of the max correlation
	max_R_x = [] #time delay of the max correlation
	y_value_list = [] #offset of the max correlation
	R_xt_spectrum = np.flip(R_xt_spectrum.T,1)
	for i,y_value in enumerate(Y[:,0]):
		index = np.argmax(abs(R_xt_spectrum[i,:]))
		if abs(R_xt_spectrum[i,index]) > 0.05:	
			max_R_x_index.append(index)
			max_R_x.append(X[i,index])
			y_value_list.append(y_value)
	y = np.array(y_value_list)
	x = max_R_x
	
	params, covariance = curve_fit(linear_regression, x,y)
	x_pred_linear = np.linspace(np.min(X),np.max(X),101)
	y_pred_linear = linear_regression(x_pred_linear, params[0],params[1])
	print('Linear slope:',params[0])  
	center = np.abs(x).argmin()

	derivative = (y[center+1]-y[center-1])/(x[center+1]-x[center-1])
	intercept = y[center] - derivative*x[center]
	x_pred_derivative = np.linspace(np.min(X),np.max(X),101)
	y_pred_derivative = linear_regression(x_pred_derivative, derivative,intercept)
	print('Derivative slope:',derivative)
	return x_pred_derivative,y_pred_derivative,derivative,x,y

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



################################ Example usage:######################################3
file_path = 'probe_{}_xcorr_{}Hz_to_{}Hz.h5'.format(loc,filter_freq[0],filter_freq[1])

X, Y, R_xt_spectrum = read_h5_file(file_path)
if X is not None:
    print("X:", X)
    print("Y:", Y)
    print("R_xt_spectrum:", R_xt_spectrum)

x_pred_derivative,y_pred_derivative,derivative,x_pt,y_pt = calculate_polynomial_regression(X, Y, R_xt_spectrum)
print("Gradient at minimum X value:", derivative)
print("normalized convection velocity:",derivative/U_inf)
###################################Plot the graph#############################################3
for i in range(R_xt_spectrum.shape[0]):
    for j in range(R_xt_spectrum.shape[1]):
        if abs(R_xt_spectrum[i][j]) < 0.18:
            R_xt_spectrum[i][j] = 0

fig,ax = plt.subplots(figsize=(6,8))
CS = ax.contour(X*16/delta_star, Y/delta_star, np.flip(R_xt_spectrum.T,1),12,colors='black')
plt.plot(x_pred_derivative*16/delta_star,y_pred_derivative/delta_star,'-g',label='Linear regression')
ax.set_xlabel(r'$\Delta t U_0/\delta^{*}$', fontsize=15)
ax.set_ylabel(r'$\xi_x/\delta^{*}$', fontsize=15)

ax.set_xlim(-28,28)
ax.set_ylim(-28,28)

plt.savefig('contour_convection_velocity')
