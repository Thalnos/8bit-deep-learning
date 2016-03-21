import csv
import numpy as np
import matplotlib.pyplot as plt

with open('plotdata.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for i in range(11):
		row = spamreader.next()
		
		if i == 3:		
			train16 = np.array(row[:6], dtype = 'float')
		if i == 4:		
			test16 = np.array(row[:6], dtype = 'float')
		if i == 9:
			train8 = np.array(row[:6], dtype = 'float')
		if i == 10:
			test8 = np.array(row[:6], dtype = 'float')
				
	
	x16 = range(3,9)
	x8 = range(3,9)
	bench32fl = 80.6600004434586
	plt.figure(1)
	ax = plt.subplot(211)	
	
	plt.plot(x8, train8, linewidth=2, label='8 bit')
	plt.plot(x16, train16, linewidth=2, label='16 bit')
	plt.plot(x16, np.ones(len(x16))*bench32fl, linewidth=2, label='32 float', color=(0,0,0))
	plt.ylabel('accuracy', fontsize=16)
	plt.title('Validation result')
	plt.legend(bbox_to_anchor=(1.0, 0.7), loc=2, borderaxespad=0.)
	plt.axis([3, 8, 0, 100])
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)

	bench32fl = 78.58000010252
	plt.subplot(212)
	plt.plot(x8, test8, linewidth=2, label='8 bit')
	plt.plot(x16, test16, linewidth=2, label='16 bit')
	plt.plot(x16, np.ones(len(x16))*bench32fl, linewidth=2, label='32 float', color=(0,0,0))
	plt.xlabel('bits after point', fontsize=18)
	plt.ylabel('accuracy', fontsize=16)
	plt.title('Test results')
	plt.legend(bbox_to_anchor=(1.0, 0.7), loc=2, borderaxespad=0.)
	plt.axis([3, 8, 0, 100])
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.show()