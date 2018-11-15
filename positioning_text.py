import os
import glob
import random
import textwrap
import math
import numpy
import markovify

##Positioning: in: positioning data from all the extraction
##out: locations (3 positions, 1 for each panel) of words for asw comic

##if the inputs are 
#TEST = [(5, 5), (5, 5), (5, 5)]
#PANEL_COORDS = [(13, 38), (254, 35), (491, 37)]


def positioning(positions):
	n = len(positions)
	all_x = []
	all_y = []
	for x,y in positions:
		all_x.append(x)
		all_y.append(y)
	
	average_x = numpy.mean(all_x)
	average_y = numpy.mean(all_y)
	x_stddev = numpy.std(all_x)
	y_stddev = numpy.std(all_y)
	#print(average_x, average_y, x_stddev, y_stddev)

	##Make random distribution centered around averages, choose randomly
	new_x = numpy.random.normal(average_x, x_stddev)
	new_y = numpy.random.normal(average_y, y_stddev)
	return (new_x, new_y)




