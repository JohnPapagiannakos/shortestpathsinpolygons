#   ~   Computational Geometry <COMP 416> - (2020)  ~
#   
#   * Topic: Given two points inside a polygon, the goal is to find 
#     the shortest path lying inside the polygon between the 
#     two points. *
#     
#   * Author's Name  : Ioannis Marios Papagiannakos
# 

import os

from project_libs import findShortestPathInPolygon, point

# Main 

projectDir = os.path.dirname(__file__)
inputDataDir = os.path.join('Data','in')
outputDataDir = os.path.join('Data', 'out')
outputFileName = ['LayerOfInterest.shp', 'MonotonizedLayer.shp', 'TriangulatedLayer.shp', 'Path.shp', 'ShortestPath.shp']

# Open first shapefile
inputFileName = os.path.join('GSHHS_shp', 'c', 'GSHHS_c_L1.shp')
# takes time...!
# inputFileName = os.path.join('GSHHS_shp', 'f', 'GSHHS_f_L1.shp')
inputFileName = os.path.join(projectDir, inputDataDir, inputFileName)

# Shapefile format
driverName = "ESRI Shapefile"

# Input Format
# point1 = point(<Longitude>, <Latitude>)

# Madagascar
input_starting_point = point(49.162, -12.424)
input_ending_point = point(44.706, -16.295)

# # Indonesia
# input_starting_point = point(119.978, -5.539)
# input_ending_point = point(124.534, 1.002)

# # Nunavut (Canada)
# input_starting_point = point(-77.10, 65.16)
# input_ending_point = point(-87.999, 70.381)

findShortestPathInPolygon(input_starting_point, input_ending_point,
                         inputFileName, outputFileName, driverName, outputDataDir)

