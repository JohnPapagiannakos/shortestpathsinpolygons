import os
import sys

from osgeo import gdal, osr, ogr

from dcel import *

# Compute Distance using Haversine formula
def haversineDistance(point1, point2):
    R = 6371 # mean(Earth radius)
    dLat = math.radians(point2.y) - math.radians(point1.y)
    dLon = math.radians(point2.x) - math.radians(point1.x)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(point1.y)) * math.cos(math.radians(point2.y)) * math.sin(dLon/2) * math.sin(dLon/2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Fetch the schema information for this layer (https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html).
def addGeometryToLayer(input_layer, input_polygon):
    featureDefn = input_layer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(input_polygon)
    input_layer.CreateFeature(outFeature)
    outFeature = None

# Create the polygon from dataset
def createPolygon(dataset, starting_point, ending_point):
    Polygon = None
    layer = dataset.GetLayer()
    for feature in layer:
        geom = feature.GetGeometryRef()

        if geom.Contains(starting_point):
            if not geom.Contains(ending_point):
                print("There exists no path between the two given points!")
                return
            # Convert a geometry into well known binary format.
            wkb = geom.ExportToWkb()
            Polygon = ogr.CreateGeometryFromWkb(wkb)
    return Polygon

# Implementation of Funnel Algorithm and miscellaneous functions.
# based on link[2] https://github.com/mmmovania/poly2tri.as3/blob/master/src/org/poly2tri/utils/NewFunnel.as
#          link[3] https://gamedev.stackexchange.com/questions/68302/how-does-the-simple-stupid-funnel-algorithm-work/68305
# 
# Computes and returns the Euclidean distance between points a and b.
def vdistsqr(a, b):
    x = b.x - a.x
    y = b.y - a.y
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

# Boolean function. Returns true if |a - b| < 1e-6 \approx "0"
def vequal(a, b):
    eq = math.pow(0.001, 2)

    return vdistsqr(a, b) < eq

# Computes and returns the cross_product(u,v), where u = b-a, v = c-a,
# or equivalently the Area A of a triangle \Delta(a,b,c) times 2.
def triAreaX2(a, b, c):
    ax = b.x - a.x
    ay = b.y - a.y
    bx = c.x - a.x
    by = c.y - a.y
    return (bx * ay - ax * by)

def funnel(starting_point, ending_point, diagonal_list):
    if diagonal_list is None:
        path = [starting_point, ending_point]
        return path

    leftList = list()
    rightList = list()

    for e in diagonal_list:
        origin = e.getOrigin()
        op = point(origin.getCoordinates()[0], origin.getCoordinates()[1])
        tail = e.getTail()
        tp = point(tail.getCoordinates()[0], tail.getCoordinates()[1])
        leftList.append(op)
        rightList.append(tp)

    leftList.append(ending_point)
    rightList.append(ending_point)

    path = [starting_point]

    rightNode = rightList[0]
    leftNode = leftList[0]
    leftIdx = 0
    rightIdx = 0
    apex = starting_point

    i = 0
    while i<len(diagonal_list):
        i += 1
        nextRight = rightList[i]
        nextLeft = leftList[i]
        
        # Update right vertex.
        if triAreaX2(apex, rightNode, nextRight) <= 0:
            if vequal(apex, rightNode) or triAreaX2(apex, leftNode, nextRight) > 0:
                # Tighten the funnel.
                rightNode = nextRight
                rightIdx = i
            else:
                # Right over left, insert left to path and restart scan from portal left point.
                path.append(leftNode)
                apex = leftNode
                apexIndex = leftIdx
                # Reset funnel
                leftNode = apex
                rightNode = apex
                rightIdx = apexIndex
                # Restart scan.
                i = apexIndex
                continue

        # Update left vertex
        if triAreaX2(apex, leftNode, nextLeft) >= 0:
            if vequal(apex, leftNode) or triAreaX2(apex, rightNode, nextLeft) < 0:
                # Tighten the funnel.
                leftNode = nextLeft
                leftIdx = i
            else:
                # Left over right, insert right to path and restart scan from portal right point.
                path.append(rightNode)
                # Make current right the new apex.
                apex = rightNode
                apexIndex = rightIdx
                # Reset portal.
                leftNode = apex
                rightNode = apex
                leftIdx = apexIndex
                rightIdx = apexIndex
                # Restart scan.
                i = apexIndex
                continue
            
    path.append(ending_point)

    return path

# Finds and returns the intersected diagonals.
def findIntersectedDiagonals(Dcel, shortest_path):
    intersectedDiagonals = list()
    for ii in range(0, len(shortest_path) - 1):
        face = Dcel.faces[shortest_path[ii].fid]
        s = face.getOuterEdge()
        if s.getTwin().getFace().getId() == shortest_path[ii + 1].fid:
            intersectedDiagonals.append(s)
        e = s.getNext()
        while e != s:
            if e.getTwin().getFace().getId() == shortest_path[ii + 1].fid:
                intersectedDiagonals.append(e)
                break
            e = e.getNext()
    
    return intersectedDiagonals

# ---Main Routine---
# Given two points inside a polygon, this function returns the shortest path 
# between the two points, lying inside the polygon.
# Input: * input_starting_point, the given starting point in format (x,y).
#        * input_ending_point, the given ending point in format (x,y).
#        * inputFileName, the input shape file (.shp).
#        * outputFileName, a list of the output shape files (.shp).
#           (can be an empty list)    
#        * driverName, the shapefile format.
#        * outputDataDir, the directory where the produced files will be saved.
# Output: void
def findShortestPathInPolygon(input_starting_point, input_ending_point, inputFileName, outputFileName, driverName, outputDataDir):
    if inputFileName is None:
        sys.exit("Empty inputFileName!")

    if len(outputFileName) > 5:
        print('Ignoring extra filenames')
    if len(outputFileName) < 5:
        print('Using default output filenames')
        outputFileName = list()
        for i in range(5):
            outputFileName.append('output_' + str(i) + '.shp')

    starting_point = ogr.Geometry(ogr.wkbPoint)
    starting_point.AddPoint(input_starting_point.x, input_starting_point.y)

    ending_point = ogr.Geometry(ogr.wkbPoint)
    ending_point.AddPoint(input_ending_point.x, input_ending_point.y)

    # Open the specified shape file
    dataset = gdal.OpenEx(inputFileName, gdal.OF_VECTOR)
    if not dataset:
        sys.exit("Specified project directory does not exist or is empty! Terminating...")
    else:
        print("Data files have been read successfully!")

    # Create Polygon from dataset
    Polygon = createPolygon(dataset, starting_point, ending_point)
    
    if Polygon is None:
        print("Empty Polygon! There is no polygon that contains input starting-ending points.")
        return

    # Close dataset
    dataset = None

    drv = ogr.GetDriverByName(driverName)
    outputFileName[0] = os.path.join(outputDataDir, outputFileName[0])
    if os.path.exists(outputFileName[0]):
        drv.DeleteDataSource(outputFileName[0])
    outDataSource = drv.CreateDataSource(outputFileName[0])
    outLayer = outDataSource.CreateLayer(outputFileName[0], geom_type=ogr.wkbPolygon)
    addGeometryToLayer(outLayer, Polygon)

    # Build the dcel structure from the polygon
    newDcel = Dcel()
    newDcel.buildFromPolygon(Polygon)
    # Monotonize Polygon P (this step is used only to visualize Monotonized P separetelly...)
    print("Monotonizing : Polygon P")
    newDcel.makeMonotone()

    # Create a new shapefile...
    # Write the monotonized polygons...
    outSHPfn = os.path.join(outputDataDir, outputFileName[1])
    if os.path.exists(outSHPfn):
        drv.DeleteDataSource(outSHPfn)
    outDataSource = drv.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPolygon)

    for i in range(1, newDcel.num_faces):
        poly = getPolygonByFace(newDcel.faces[i])
        addGeometryToLayer(outLayer, poly)

    ####
    # Replace prev shapefile
    newDcel = Dcel()
    newDcel.buildFromPolygon(Polygon)
    # Triangulate the monotonized polygon
    print("Monotonizing + Triangulating : Polygon P")
    newDcel.makeMonotone()
    newDcel.triangulate()

    # Create a new shapefile...
    # Write the triangulated & monotonized polygons...
    outSHPfn = os.path.join(outputDataDir, outputFileName[2])
    if os.path.exists(outSHPfn):
        drv.DeleteDataSource(outSHPfn)
    outDataSource = drv.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPolygon)

    for i in range(1, newDcel.num_faces):
        poly = getPolygonByFace(newDcel.faces[i])
        if poly.Contains(starting_point):
            start_face = newDcel.faces[i]
        if poly.Contains(ending_point):
            end_face = newDcel.faces[i]

        addGeometryToLayer(outLayer, poly)

    # Find the shortest path between the given starting and end points
    extractedGraph = newDcel.extractGraph()

    starting_node = extractedGraph.getNodeByFaceId(start_face.getId())
    ending_node = extractedGraph.getNodeByFaceId(end_face.getId())

    shortest_path = extractedGraph.startTraversal(starting_node, ending_node)

    outSHPfn = os.path.join(outputDataDir, outputFileName[3])
    if os.path.exists(outSHPfn):
        drv.DeleteDataSource(outSHPfn)
    outDataSource = drv.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString)

    # Create a LineString
    path_line = ogr.Geometry(ogr.wkbLineString)

    for p in shortest_path:
        path_line.AddPoint(p.getCoordinates()[0], p.getCoordinates()[1])
    addGeometryToLayer(outLayer, path_line)

    # Find the intersected diagonals.
    intersectedDiagonals = findIntersectedDiagonals(newDcel, shortest_path)

    # Funnel Algorithm
    actual_path = funnel(input_starting_point, input_ending_point, intersectedDiagonals)
    outSHPfn = os.path.join(outputDataDir, outputFileName[4])
    if os.path.exists(outSHPfn):
        drv.DeleteDataSource(outSHPfn)
    outDataSource = drv.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString)
    
    # Create a LineString
    path_line = ogr.Geometry(ogr.wkbLineString)

    # Calculate total distance of shortest path (Ellipsoidal formula)
    distance = 0
    idx = 0
    for p in actual_path:
        path_line.AddPoint(p.x, p.y)
        if idx > 0:
            distance += haversineDistance(actual_path[idx-1], actual_path[idx])
        idx+=1
    addGeometryToLayer(outLayer, path_line)

    print("Total length  (Cartesian) " +  " = " + " {:.2f}  deg".format(path_line.Length()))
    print("Total length (Ellipsoidal)" + u" \u2248 " + " {:.1E}  km".format(distance))
