import xlwt


def xl_wrt(file_name,inputs,actual,predicted):

	book = xlwt.Workbook()
	
	sh = book.add_sheet("Data")

	sh.write(0,0,"Inputs")

	sh.write(0,len(inputs[0])+1,"Actual")

	sh.write(0,len(inputs[0])+len(actual[0])+2,"Predicted")

	for i in range(len(inputs)):
		for j in range(len(inputs[i])):
			sh.write(i+1,j,inputs[i][j][0])
		for j in range(len(actual[i])):
			sh.write(i+1,j+len(inputs[0])+1,actual[i][j][0])

		for j in range(len(predicted[i])):
			sh.write(i+1,j+len(inputs[0])+len(actual[0])+2,predicted[i][j][0])


	book.save(file_name+".xls")