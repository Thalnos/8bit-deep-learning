import csv
import numpy as np
import matplotlib.pyplot as plt

with open('plotdata.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for i in range(11):
		row = spamreader.next()
		bench32fl = 78.2299995422363
		if i == 3:		
			train16 = np.array(row[:6], dtype = 'float')
		if i == 4:		
			test16 = np.array(row[:6], dtype = 'float')
		if i == 9:
			train8 = np.array(row[:3], dtype = 'float')
		if i == 10:
			test8 = np.array(row[:3], dtype = 'float')

	
	x16 = range(3,9)
	x8 = range(3,6)
	
	plt.figure(1)
	plt.subplot(211)	
	plt.plot(x8, train8, label='8 bit')
	plt.plot(x16, train16, label='16 bit')
	plt.plot(x16, np.ones(len(x16))*bench32fl, label='32 float benchmark')
	plt.ylabel('accuracy', fontsize=16)
	plt.title('Final trainepoch')
	plt.legend(bbox_to_anchor=(0.6, 0.3), loc=2, borderaxespad=0.)
	plt.axis([2, 9, 20, 100])

	plt.subplot(212)
	plt.plot(x8, test8, label='8 bit')
	plt.plot(x16, test16, label='16 bit')
	plt.plot(x16, np.ones(len(x16))*bench32fl, label='32 float benchmark')
	plt.xlabel('fractiebits', fontsize=18)
	plt.ylabel('accuracy', fontsize=16)
	plt.title('Test results')
	plt.legend(bbox_to_anchor=(0.6, 0.3), loc=2, borderaxespad=0.)
	plt.axis([2, 9, 20, 100])
	plt.show()