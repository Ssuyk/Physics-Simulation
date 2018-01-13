import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import scipy.optimize 
from scipy.integrate import odeint

def FS_2_AU(t):
	"""t en fs """
	return t/2.419e-2

def omega_2_AU(w):
	"""w en rad.fs-1 """
	return w*2.419e-2

def I_2_AU(I):
	"""I en W/cm2"""
	return I/3.5094e16

def I_AU_2_E_AU(I):
	return np.sqrt(I)

def Elec_velocity(t,params):

	A1 = params['A1']
	A2 = params['A2']
	omega1 = params['omega1']
	omega2 = params['omega2']
	phi0 = params['phi0']

	t_i = params['t_i'] 
	# Y = -A1/omega1*(np.sin(omega1*t)-np.sin(omega1*t_i))
	
	Y = -A1/omega1*(np.sin(omega1*t)-np.sin(omega1*t_i))\
	-A2/omega2*(np.sin(omega2*t+phi0)-np.sin(omega2*t_i+phi0))
	
	return Y

def Elec_position(t,params):

	A1 = params['A1']
	A2 = params['A2']
	omega1 = params['omega1']
	omega2 = params['omega2']
	phi0 = params['phi0']
	t_i = params['t_i']  

	# Y = A1/omega1**2*(np.cos(omega1*t)-np.cos(omega1*t_i))\
	#  + A1/omega1*np.sin(omega1*t_i)*(t-t_i)
	Y = A1/omega1**2*(np.cos(omega1*t)-np.cos(omega1*t_i))\
	 + A1/omega1*np.sin(omega1*t_i)*(t-t_i)\
	 + A2/omega2**2*(np.cos(omega2*t+phi0)-np.cos(omega2*t_i+phi0))\
	 + A2/omega2*np.sin(omega2*t_i+phi0)*(t-t_i)
		
	return Y

def E_field(t,params):

	A1 = params['A1']
	A2 = params['A2']
	omega1 = params['omega1']
	omega2 = params['omega2']
	DeltaT1 = params['DeltaT1'] 
	DeltaT2 = params['DeltaT2'] 
	phi0 = params['phi0']

	return -A1*np.cos(omega1*(t))*np.exp(-(2*t/DeltaT1)**2)\
			- A2*np.cos(omega2*(t)+phi0)*np.exp(-(2*t/DeltaT2)**2)\

	# return -A1*(np.cos(omega1*t))		
	
def Ode_func(y,t,params):

	x = y[0]
	v = y[1]

	A1 = params['A1'] 
	A2 = params['A2'] 
	omega1 = params['omega1'] 
	omega2 = params['omega2'] 
	DeltaT1 = params['DeltaT1'] 
	DeltaT2 = params['DeltaT2'] 
	t_i = params['t_i']
	phi0 = params['phi0']

	derivs = [v,
			-A1*np.cos(omega1*(t))*np.exp(-(2*t/DeltaT1)**2)\
			- A2*np.sin(omega2*(t)+phi0)*np.exp(-(2*t/DeltaT2)**2)\
			]
	return derivs

Npts= 1025*4
c =2.99e8
Lambda1 = 800
T1= FS_2_AU(Lambda1*1e+6/c) 
omega1 =omega_2_AU(2*np.pi*c/(Lambda1*1e-9)*1e-15)
start = FS_2_AU(-100)
stop = FS_2_AU(100)#2*T1
t = np.linspace(start, stop, Npts)
t_forUp = np.linspace(0, T1, Npts)
A1 = I_AU_2_E_AU(I_2_AU(2.5e14))

Up = A1**2/(4*omega1**2)

font = {'size': 12}
matplotlib.rc('font', **font)

fig = plt.figure(1)
plt.clf()

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan = 2)

N_ti = 2

ti_vect = np.zeros(N_ti)
tr_vect = np.zeros(N_ti)
E_c_test_vect = np.zeros(N_ti)
Up_vect = np.zeros(N_ti)

for ind in range(N_ti):
	t_i = 0.1*ind*T1+start+30*T1
	ti_vect[ind] = t_i

	for i, j in enumerate(t):
		if j >= t_i:
			index_ti = int(i)
			break

	data = {'A1': A1,
			'omega1' : omega1,
			'DeltaT1': FS_2_AU(50),
			't_i' : t_i, 
			'A2': 0.3*A1,
			'omega2' : 2*omega1,
			'DeltaT2': FS_2_AU(70),
			'phi0' : 1*np.pi
			}
	
	t_ode = np.linspace(t_i, stop, Npts)		
	x_startode = [0, 0]
	x_ode =  odeint(Ode_func, x_startode, t_ode, args = (data,))	

	x = Elec_position(t, data)
	# index_tr = np.where(x[index_ti:] >= 0)[0][0]+index_ti
	# tr_vect[ind] = t[index_tr]
	# x[t<t_i] = 0

	E_c = 0.5*Elec_velocity(t,data)**2/Up
	E_c[t<t_i] = 0
	E_c_ode = 0.5*x_ode[:,1]**2/Up
	# E_c_test = 0.5*Elec_velocity(t[index_tr],data)**2/Up
	# E_c_test_vect[ind] = E_c_test

	Up_vect[ind] = 0.5/T1*np.trapz(Elec_velocity(t_forUp, data)**2, t_forUp)/Up

	# Traj, = plt.plot(t[index_ti-1:index_tr]/T1, x[index_ti-1:index_tr])
	# Traj_ode, = ax1.plot(t/T1, x)
	Traj_ode, = ax1.plot(t_ode/T1, np.array(x_ode)[:,0])
	# plt.subplot(122)
	# velocity, = ax3.plot(t[index_ti-1:index_tr]/T1, E_c)
	

	# Kin, = ax3.plot(t/T1, E_c)
	Kin_ode, = ax3.plot(t_ode/T1, E_c_ode)



tr_vect[0] = T1
norm =1000/np.max(np.abs(E_field(t,data)))

Field, = ax1.plot(t/T1,E_field(t,data)*norm,\
		'--',lw =2, color = 'red', label = 'rectangular E. field')
# Field_ref, =plt.plot(t/T1,-A1*(np.cos(omega1*t))*norm,\
		# '--',lw =1, color = 'blue', label = 'cos E. field')
# plt.plot([0, t[-1]/T1], [0, 0],'--',lw=2, color='k')
ax1.set_xlabel('Time (period)')
ax1.set_ylabel('Electron trajectory (atomic unit)')
ax1.set_xlim(t_i/T1, 5)
ax1.set_ylim(-2000, 2000)
ax1.set_title('x(t)')
ax1.grid()
# plt.set_aspect('equal', adjustable = 'box')

# plt.legend(handles=[Field])#, Field_ref])


ax2.plot(tr_vect/T1, E_c_test_vect,'--x', lw = 3)
ax2.plot(ti_vect/T1, E_c_test_vect,'--x', lw = 3)
ax2.plot(ti_vect/T1,Up_vect,'--x', lw = 2)

ax2.plot([0, t[-1]/T1], [3.17, 3.17],'--',lw=2, color='k')
ax2.set_title('$E_k$(t)')
ax2.set_xlabel('Time (period)')
ax2.set_ylabel('Maximum kinetic energy (Up)')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(0, 4)
ax2.grid()

ax3.set_xlabel('Time (period)')
ax3.set_ylabel('Maximum kinetic energy (Up)')
# ax3.set_xlim(-0.05, 1.05)
# plt.ylim(0.75, 1.25)
ax3.grid()


plt.show()
