from numpy import sqrt
"""
* Lambda is in [m]
* axis must be x, y or z, otherwise, an error will popup
* the final result is the refractive index for the given wavelength and 
axis
"""
def sellmeier(Lambda, crystal, axis):
	if crystal == 'BBO':
		if axis == 'x' or axis =='y':
			A = 2.7359
			B = 0.01878
			C = 0.01822
			D = 0.01354
		elif axis == 'z':
		    A = 2.3753
		    B = 0.01224
		    C = 0.01667
		    D = 0.01516
		else:
			raise ValueError("unvalid axis, must be x, y or z")
	else: 
		raise ValueError("unvalid crystal")		

	return sqrt(A + B/ ((Lambda*1e6)**2 - C)  - D*(Lambda*1e6)**2)