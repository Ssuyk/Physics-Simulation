from numpy import zeros
def nonlinearCoefficientBBO():
	d31 = -0.04 #pm/V
	d22 = 2.16
	A = zeros((3,6))
	A[0][4] = d31
	A[1][3] = d31
	A[2][0] = d31
	A[2][1] = d31
	A[0][5] = -d22
	A[1][0] = -d22
	A[1][1] = d22
	return A
