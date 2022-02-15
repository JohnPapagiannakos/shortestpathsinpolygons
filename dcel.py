# based on link[1] https://github.com/anglyan/dcel/blob/master/dcel/dcel.py
#          Book[1] Computational Geometry - Algorithms and Applications, 3rd Ed.
#          Book[2] GIS Algorithms, Ningchuan Xiao, 9781473933460, 2015

import sys, math
from gdal import ogr
from shapely import geometry
from binarySearchTree import *
import graphs

# SET ENUMERATED GLOBAL VARIABLES
start   = 'start'
end     = 'end'
split   = 'split'
merge   = 'merge'
regular = 'regular'


class point:
    def __init__(self, x_coordinate, y_coordinate):
        self.x = x_coordinate
        self.y = y_coordinate

class Node:
    def __init__(self, nid=None, coordinates=None, edge=None):
        # Node id 
        self.nid = nid

        # The coordinate vector of node [x, y]
        self.coordinates = coordinates

        # Pointer to an edge that starts from the node
        self.edge = edge

    def setId(self, nid):
        self.nid = nid
    
    def getId(self):
        return self.nid

    def setCoordinates(self, coordinates):
        self.coordinates = coordinates

    def getCoordinates(self):
        return self.coordinates

    def setEdge(self, e):
        self.edge = e

    def getEdge(self):
        return self.edge

    def isAbove(self, node):
        if self.coordinates[1] > node.coordinates[1]:
            return True
        if self.coordinates[1] == node.coordinates[1]:
            if self.coordinates[0] < node.coordinates[0]:
                return True

        return False

    def isBelow(self, node):
        return not self.isAbove(node)


class Face:

    def __init__(self, fid=None, outer=None, inner=None):
        self.fid = fid

        # Associate each face with an edge
        self.outer = outer
        self.inner = inner

    def getCentroid(self):

        node_list = list()

        if not self.outer:
            s = self.inner
        else:
            s = self.outer

        e = s.getNext()
        node_list.append(s.getOrigin())
        while e != s:
            node_list.append(e.getOrigin())
            e = e.getNext()

        total_coords = [0, 0]
        centroid = [0, 0]

        for n in node_list:
            total_coords[0] += n.getCoordinates()[0]
            total_coords[1] += n.getCoordinates()[1]

        centroid[0] = total_coords[0] / len(node_list)
        centroid[1] = total_coords[1] / len(node_list)

        return centroid

    def getId(self):
        return self.fid

    def setInnerEdge(self, edge):
        self.inner = edge

    def getInnerEdge(self):
        return self.inner

    def setOuterEdge(self, edge):
        self.outer = edge
    
    def getOuterEdge(self):
        return self.outer


class Edge:

    def __init__(self, eid=None, origin=None, tail=None, twin=None, face=None, next=None, prev=None):
        # Edge id
        self.eid = eid
        # Pointer to the origin node
        self.origin = origin
        # Pointer to the tail node
        self.tail = tail
        # Pointer to the twin edge
        self.twin = twin
        # Pointer to the associated face
        self.face = face
        # Pointer to the next edge
        self.next = next
        # Pointer to the previous edge
        self.prev = prev

        self.helperVertex = None

    def getId(self):
        return self.eid

    def setTail(self, tail):
        self.tail = tail

    def setTwin(self, twin):
        self.twin = twin

    def setNext(self, next):
        self.next = next

    def setPrevious(self, prev):
        self.prev = prev

    def setFace(self, face):
        self.face = face

    def getOrigin(self):
        return self.origin

    def getTail(self):
        return self.tail

    def getFace(self):
        return self.face

    def getTwin(self):
        return self.twin

    def getNext(self):
        return self.next

    def getPrevious(self):
        return self.prev

    def setHelper(self, vertex):
        self.helperVertex = vertex

    def getHelper(self):
        if self.helperVertex:
            return self.helperVertex
        else:
            print("Helper vertex not set")

class Dcel:
    def __init__(self):
        self.nodes = list()
        self.edges = list()
        self.faces = list()
        self.num_nodes = 0
        self.num_edges = 0
        self.num_faces = 0
        self.isMonotone = False

    # Use an ogr.Geometry polygon to build the list
    def buildFromPolygon(self, polygon):
        self.__init__()

        # Fetch pointer to feature geometry.
        pntr = polygon.GetGeometryRef(0)

        # Get the number of points in this path's array of data points.
        count = pntr.GetPointCount()

        # The last pointed == first pointed
        count = count - 1

        # Add the first node
        p = pntr.GetPoint(0)
        self.nodes.append(Node(0, [p[0], p[1], None]))

        # Add two faces
        self.faces.append(Face(0, None, None))
        self.faces.append(Face(1, None, None))

        # Add the first edge.
        edge = Edge(0, self.nodes[0], None, None, self.faces[1], None, None)
        self.edges.append(edge)

        for i in range(1, count):
            p = pntr.GetPoint(i)
            self.nodes.append(Node(i, [p[0], p[1], None]))
            # Edge( id, origin, tail, twin, face, next, prev):
            edge = Edge(i, self.nodes[i], self.nodes[i], None, self.faces[1], None, self.edges[i - 1])
            self.edges.append(edge)
            self.edges[i - 1].setNext(self.edges[i])
            self.edges[i - 1].setTail(self.nodes[i])
            self.nodes[i - 1].setEdge(self.edges[i - 1])

        self.edges[0].setPrevious(self.edges[count - 1])
        self.edges[0].getPrevious().setNext(self.edges[0])
        self.edges[0].getPrevious().setTail(self.nodes[0])
        self.nodes[-1].setEdge(self.edges[0].getPrevious())

        # For each edge, add its twin in the edge list.
        e = self.edges[0]

        twin = Edge(count, e.getTail(), e.getOrigin(), e, self.faces[0], None, None)
        e.setTwin(twin)
        self.edges.append(twin)
        for i in range(1, count):
            e = self.edges[i]
            twin = Edge(count + i, e.getTail(), e.getOrigin(), e, self.faces[0], None, None)
            twin.setNext(self.edges[count + i - 1])
            twin.getNext().setPrevious(twin)
            self.edges[i].setTwin(twin)
            self.edges.append(twin)

        twin.setPrevious(self.edges[0].getTwin())

        e = self.edges[0]
        twin = e.getTwin()
        twin.setNext(e.getPrevious().getTwin())
        twin.setPrevious(e.getNext().getTwin())

        # Lastly set the associated edges to the faces
        self.faces[1].setOuterEdge(self.edges[0])
        self.faces[0].setInnerEdge(self.edges[0].getTwin())

        self.num_nodes = count
        self.num_faces = 2
        self.num_edges = 2 * count

    def getNodeById(self, nid):
        for n in self.nodes:
            if n.getId() == nid:
                return n
        return None

    def getEdge(self, origin, tail):
        for e in self.edges:
            if e.getOrigin() == origin and e.getTail() == tail:
                return e
        return None

    def addInnerFace(self, edge):
        self.faces.append(Face(self.num_faces, edge, None))
        self.num_faces += 1

    def getAllEdgesByOrigin(self, origin):
        edges = list()
        for e in self.edges:
            if e.getOrigin() == origin:
                edges.append(e)
        return edges

    ####################################################################
    # Implementation of algorithms for each type of event -- based on Book[1]. 
    def handleStartVertex(self, node, T):
        e = node.getEdge().getTwin().getNext().getTwin()
        e.setHelper(node)
        T.insert(e)

    def handleEndVertex(self, node, T):
        e = node.getEdge()
        h = e.getHelper()
        if self.identifyVertex(h) == merge:
            self.insertDiagonal(node, h)
        T.remove(e)

    def handleSplitVertex(self, node, T):
        left_edge = T.getLeftEdge(node)
        h = left_edge.getHelper()
        self.insertDiagonal(h, node)
        left_edge.setHelper(node)
        e = node.getEdge().getTwin().getNext().getTwin()
        T.insert(e)
        e.setHelper(node)

    def handleMergeVertex(self, node, T):
        e = node.getEdge()
        h = e.getHelper()
        # If mergeVertex
        if self.identifyVertex(h) == merge:
            self.insertDiagonal(h, node)
        T.remove(e)
        left_edge = T.getLeftEdge(node)
        h = left_edge.getHelper()
        # If mergeVertex
        if self.identifyVertex(h) == merge:
            self.insertDiagonal(h, node)
        left_edge.setHelper(node)

    def handleRegularVertex(self, node, T):
        e = node.getEdge()
        if (e.getTail().getCoordinates()[1] > node.getCoordinates()[1]) or (
                e.getTail().getCoordinates()[1] == node.getCoordinates()[1] and e.getTail().getCoordinates()[0] < node.getCoordinates()[0]):
            h = e.getHelper()

            if self.identifyVertex(h) == merge:
                self.insertDiagonal(h, node)
            T.remove(e)
            e = node.getEdge().getTwin().getNext().getTwin()
            T.insert(e)
            e.setHelper(node)
        else:
            left_edge = T.getLeftEdge(node)
            h = left_edge.getHelper()

            if self.identifyVertex(h) == merge:
                self.insertDiagonal(h, node)
            left_edge.setHelper(node)

    ####################################################################

    # Splits a polygon in y-monotone polygons.
    def makeMonotone(self):
        sorted_nodes = self.nodes[:]

        # Sort nodes from top to bottom, if two nodes have the same y-coordinate, the leftmost has priority.
        sorted_nodes = sorted(sorted_nodes, key=lambda x: (-x.getCoordinates()[1], x.getCoordinates()[0]))

        # Initialize an empty binary search tree Tree.
        Tree = BST()

        while len(sorted_nodes) > 0:
            top = sorted_nodes.pop(0)

            vertex_type = self.identifyVertex(top)
            # print("vertex_type=" + vertex_type)
            if vertex_type == start:
                self.handleStartVertex(top, Tree)
            elif vertex_type == end:
                self.handleEndVertex(top, Tree)
            elif vertex_type == split:
                self.handleSplitVertex(top, Tree)
            elif vertex_type == merge:
                self.handleMergeVertex(top, Tree)
            elif vertex_type == regular:
                self.handleRegularVertex(top, Tree)
            else:
                print("Unknown vertex type! Terminating...")
                sys.exit()
        
        self.isMonotone=True
            
    # Triangulates a monotone polygon.
    def triangulate(self):
        if self.isMonotone:
            poly_list = list()
            for f in self.faces[1:]:
                poly = list()
                s = f.getOuterEdge()
                poly.append(s)
                e = s.getNext()
                while e != s:
                    poly.append(e)
                    e = e.getNext()
                poly_list.append(poly)

            for poly in poly_list:
                self.triangulateMonotonePolygon(poly)
        else:
            print("Polygon must first be monotonized! Monotonizing...")
            self.makeMonotone()
            self.triangulate()

    def identifyVertex(self, node):
        rightNode = self.getRightNode(node)
        leftNode = self.getLeftNode(node)

        if node.isAbove(rightNode) and node.isAbove(leftNode):
            if checkDirection(rightNode, node, leftNode) == 1:
                return start
            else:
                return split

        if node.isBelow(rightNode) and node.isBelow(leftNode):
            if checkDirection(rightNode, node, leftNode) == 1:
                return end
            else:
                return merge

        return regular

    # Adds an edge between two nodes.
    def insertDiagonal(self, origin, tail):
        if origin == tail:
            return
        # If the diagonal to be inserted does already exist, then return
        if self.getEdge(origin, tail):
            return

        # Add the edge to our edge list.
        newEdge = Edge(self.num_edges, origin, tail, None, None, None, None)
        newTwin = Edge(self.num_edges + 1, tail, origin, newEdge, None, None, None)
        newEdge.setTwin(newTwin)

        self.edges.append(newEdge)
        self.edges.append(newTwin)

        self.num_edges += 2

        adjacentEdges = self.getAllEdgesByOrigin(origin)
        adjacentEdges = sortInCCW(origin, adjacentEdges)

        for i in range(0, len(adjacentEdges)):
            if newEdge == adjacentEdges[i]:
                le = (i + 1) % len(adjacentEdges)
                re = (i - 1) % len(adjacentEdges)
                break

        le = adjacentEdges[le]
        re = adjacentEdges[re]

        re.getTwin().setNext(newEdge)
        newEdge.setPrevious(re.getTwin())

        le.setPrevious(newTwin)
        newTwin.setNext(le)

        # Two more edges to go
        adjacentEdges = self.getAllEdgesByOrigin(tail)
        adjacentEdges = sortInCCW(tail, adjacentEdges)

        for i in range(0, len(adjacentEdges)):
            if newTwin == adjacentEdges[i]:
                le = (i + 1) % len(adjacentEdges)
                re = (i - 1) % len(adjacentEdges)
                break

        le = adjacentEdges[le]
        re = adjacentEdges[re]

        le.setPrevious(newEdge)
        newEdge.setNext(le)

        re.getTwin().setNext(newTwin)
        newTwin.setPrevious(re.getTwin())

        # Add a new face and fix the values of faces on the edges.
        e = newEdge.getNext()
        e.getFace().setOuterEdge(newEdge)
        newEdge.setFace(e.getFace())
        self.addInnerFace(newEdge)

        self.faces[-1].setOuterEdge(newTwin)
        newTwin.setFace(self.faces[-1])
        e = newTwin.getNext()
        while e != newTwin:
            e.setFace(self.faces[-1])
            e = e.getNext()

    def getRightNode(self, node):
        e = node.getEdge().getTwin().getNext()
        return e.getTail()

    def getLeftNode(self, node):
        return node.getEdge().getTail()

    # This Algorithm triangulates a monotone polygon. 
    # -- Implementation of "TriangulateMonotonePolygon(P)" --
    # Input: A list of edges at the boundaries of the polygon P.
    # Output: A partitioning of P into y-monotone pieces stored in Dcel.
    def triangulateMonotonePolygon(self, edge_list):
        sorted_list = list()

        # Extract the node list from the edge list,
        for e in edge_list:
            sorted_list.append(e.getOrigin())

        # Sort the new list by descending order of y coordinate (top to bottom) of the nodes
        # * If two nodes have the same y coordinate, the one with smaller
        #   x-coordinate has higher priority. *

        sorted_list = sorted(sorted_list, key=lambda x: (-x.getCoordinates()[1], x.getCoordinates()[0]))

        upperChain = [sorted_list[0]]
        lowerChain = list()

        for node in sorted_list[1:]:
            isUpper = False
            for e in edge_list:
                if e.getOrigin() == upperChain[-1] and e.getTail() == node:
                    isUpper = True
                    break
            if isUpper:
                upperChain.append(node)
            else:
                lowerChain.append(node)

        # Initialize an empty stack S, and push u[0] and u[1] onto it.
        stack = [sorted_list[0], sorted_list[1]]

        for i in range(2, len(sorted_list) - 1):
            top = stack[-1]
            if not self.getEdge(sorted_list[i], top):
                while len(stack) > 1:
                    e = stack.pop()
                    self.insertDiagonal(e, sorted_list[i])
                stack.pop()
                stack.append(top)
                stack.append(sorted_list[i])
            else:
                last = stack.pop()
                e = stack[-1]
                if upperChain.count(last):
                    while checkDirection(e, last, sorted_list[i]) == +1:
                        self.insertDiagonal(e, sorted_list[i])
                        last = stack.pop()
                        if len(stack) > 0:
                            e = stack[-1]
                        else:
                            break
                else:
                    while checkDirection(e, last, sorted_list[i]) == -1:
                        self.insertDiagonal(e, sorted_list[i])
                        last = stack.pop()
                        if len(stack) > 0:
                            e = stack[-1]
                        else:
                            break
                stack.append(last)
                stack.append(sorted_list[i])

        stack.pop()
        del stack[0]

        # Add diagonals from u[end] to all stack vertices except the first and the last one.
        for node in stack:
            self.insertDiagonal(node, sorted_list[-1])

    # Extracts an undirected graph from a Dcel structure...
    def extractGraph(self):
        myGraph = graphs.Graph()

        outerFace = self.faces[0]

        for face in self.faces[1:]:
            myGraph.addNode(face, face.getId())

        myGraph.sortById()

        for edge in self.edges[0::2]:
            if edge.getFace() != outerFace and edge.getTwin().getFace() != outerFace:
                myGraph.makeLink(edge.getFace(), edge.getTwin().getFace())

        return myGraph

#########################################################################
# Returns type <ogr.polygon> extracted from a Dcel face.
def getPolygonByFace(face):
    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)

    if face.getId() == 0:
        face_ = face.getInnerEdge()
    else:
        face_ = face.getOuterEdge()

    coords = face_.getOrigin().getCoordinates()
    ring.AddPoint(coords[0], coords[1])
    e = face_.getNext()

    while e != face_:
        coords = e.getOrigin().getCoordinates()
        ring.AddPoint(coords[0], coords[1])
        e = e.getNext()

    coords = face_.getOrigin().getCoordinates()
    ring.AddPoint(coords[0], coords[1])

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    # Create & return the polygon
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon

# Determines if the angle between pair of nodes p, q and q, m, respectively, is counter clockwise (ccw) or not (cw).
def checkDirection(p, q, r):
    # https://gis.stackexchange.com/questions/298290/checking-if-vertices-of-polygon-are-in-clockwise-or-anti-clockwise-direction-in
    ring = geometry.LinearRing([(p.getCoordinates()[0], p.getCoordinates()[1]), 
                                (q.getCoordinates()[0], q.getCoordinates()[1]), 
                                (r.getCoordinates()[0], r.getCoordinates()[1])])

    if ring.is_ccw:
        return -1
    else:
        ring = geometry.LinearRing([(r.getCoordinates()[0], r.getCoordinates()[1]),
                                (q.getCoordinates()[0], q.getCoordinates()[1]),
                                (p.getCoordinates()[0], p.getCoordinates()[1])])
        if ring.is_ccw:
            return +1
        else:
            print("Error! Can not decide if the angle between the given nodes is CCW or CW!")
            return 0

# Sort the given edges and their origin in CCW order.
def sortInCCW(origin, edges):
    # Coordinates of origin.
    p = origin.getCoordinates()

    rhs = list()
    lhs = list()
    for e in edges:
        # Translate every edge to a vector with origin (0,0)
        tailCoords = e.getTail().getCoordinates()
        v = [tailCoords[0] - p[0], tailCoords[1] - p[1]]
        if v[0] >= 0 and not (v[0] == 0 and v[1] < 0):
            if math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2)) == 0:
                costheta = 1
            else:
                costheta = (0 * v[0] + 1 * v[1]) / math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2))

            rhs.append([e, costheta])
        else:
            costheta = (0 * v[0] + 1 * v[1]) / math.sqrt(math.pow(v[0], 2) + math.pow(v[1], 2))
            lhs.append([e, costheta])

    # Sort both two lists and append the respective edge ids
    rhs.sort(key=lambda x: x[1], reverse=False)
    lhs.sort(key=lambda x: x[1], reverse=True)

    sortedEdges = list()

    for i in lhs:
        sortedEdges.append(i[0])

    for i in rhs:
        sortedEdges.append(i[0])

    return sortedEdges
