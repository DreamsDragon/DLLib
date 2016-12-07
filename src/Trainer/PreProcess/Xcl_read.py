import xlrd
import numpy as np
import Normalisation

def check_int(val):
	try:
		return float(val)
	except:
		return None
def xl_cnv(path,in_no = None,out_no = None):
	x = []
	y = []






	book = xlrd.open_workbook(path)

	first_sheet = book.sheet_by_index(0)

	rows =  first_sheet.nrows
	cols =  first_sheet.ncols

	if in_no == None:
		ins = cols-1
	else:
		ins = in_no

	if out_no == None:
		outs = 1
	else:
		outs = out_no

	val = int(rows*1)



	for i in range(0,val):
		in_row = []
		for j in range(ins):
 			cell = first_sheet.cell(i,j)
 			if check_int(cell.value) == None:
 				continue
 			in_row.append(check_int(cell.value))
  		
  		if len(in_row)!=0:
 			in_row = np.array(in_row)
 			x.append(np.array([in_row]))
 		
 		out_row = []
 		for j in range(ins,ins+outs):
 			cell = first_sheet.cell(i,j)
 			if check_int(cell.value) == None:
 				continue
 			out_row.append(check_int(cell.value))
 
  		if len(out_row)!=0:
	 		out_row = np.array(out_row)
	 		y.append(np.array([out_row]))



	for i in range(len(x))	:
		x[i] = x[i].T 
		y[i] = y[i].T


	x = np.array(x)
	y = np.array(y)
 	return (x,y)




def get_min_max_med(array):
	xmin = np.min(array)
	xmax = np.max(array)
	xmed = np.median(array)

	return xmin,xmax,xmed

def forawrd_sig_convert(x_data):
	 data = []
	 coeffs = ([],[],[])
	 for i in range(len(x_data)):
 		for j in range(len(x_data[i])):
		 	if i == 0:
	 			data.append([])
	 			continue
	 		else:
	 			data[j].append(x_data[i][j])

	 for i in data:
	 	mini,maxi,med = get_min_max_med(i)
	 	coeffs[0].append(mini)#Minimum
	 	coeffs[1].append(maxi)#Maximum
	 	coeffs[2].append(med)#Median


	 ret_data = []
	 for i in range(len(x_data)):
	 	normal_dat = []
	 	for j in range(len(x_data[i])):
	 		local_min,local_max,local_med = coeffs[0][j],coeffs[1][j],coeffs[2][j]
	 		A = np.array([[1,local_min,local_min**2],[1,local_med,local_med**2],[1,local_max,local_max**2]])
	 		B = np.array([-1,0,1])
	 		[a0,a1,a2] = np.linalg.solve(A,B)
	 		x_normal = a0+a1*(x_data[i][j])+a2*(x_data[i][j]**2)
	 		normal_dat.append(x_normal)
	 	ret_data.append(normal_dat)

	 return ret_data