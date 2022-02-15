import math
from collections import deque


class Node:
    def __init__(self, fid, face, link=None):
        # Face id
        self.fid = fid

        self.face = face

        # Coordinates of node, [x, y]
        self.coords = face.getCentroid()

        # Pointer to other node in the graph
        self.link = list()

        self.visited = False

    def makeLink(self, toNode):
        # Distance between nodes
        p = self.getCoordinates()
        q = toNode.getCoordinates()

        distance = math.sqrt(math.pow((q[0] - p[0]), 2) + math.pow((q[1] - p[1]),2))

        self.link.append(Link(self, toNode, distance))

    def getId(self):
        return self.fid

    def getCoordinates(self):
        return self.coords


class Link:
    def __init__(self, origin, tail, distance):
        # Points to the origin node
        self.origin = origin
        # Points to the tail node
        self.tail = tail
        # Eucl. distance between two nodes
        self.distance = distance

    def getTail(self):
        return self.tail


class Graph:
    def __init__(self, fid=None, node=None):
        self.head = None
        self.list_nodes = list()
        self.num_nodes = 0

    def sortById(self):
        self.list_nodes = sorted(self.list_nodes, key=lambda x: (x.fid))

    def addNode(self, newNode, fid):
        newNode = Node(fid, newNode, None)
        self.list_nodes.append(newNode)

        if self.head is None:
            self.head = newNode

        self.num_nodes += 1
        return newNode

    def makeLink(self, n1, n2):
        n1 = self.list_nodes[n1.getId() - 1]
        n2 = self.list_nodes[n2.getId() - 1]

        # link n1->n2 && n2->n1
        n1.makeLink(n2)
        n2.makeLink(n1)

    def getNodeByFaceId(self, fid):
        return self.list_nodes[fid - 1]

    def DFS(self, path, end):

        last_visited = path[-1]

        for e in last_visited.link:
            t = e.getTail()
            if t.visited == False:
                t.visited = True
                path.append(t)
                if t == end:
                    return path
                if self.DFS(path, end):
                    return path
                path.pop()

        return 

    # Depth-First Search Non-Recursive Function <=> Breadth-First Search 
    # https://stackoverflow.com/questions/8922060/how-to-trace-the-path-in-a-breadth-first-search
    # In python iterative approach performs better than reccursive.
    def BFS(self, start, end):
        # maintain a queue of paths
        queue = []
        # push the first path into the queue
        queue.append([start])
        while queue:
            # get the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            # path found
            if node == end:
                return path
            # enumerate all adjacent nodes, construct a new path and push it into the queue
            for adjacent in node.link:
                t = adjacent.getTail()
                if t.visited == False:
                    t.visited = True
                    new_path = list(path)
                    new_path.append(t)
                    queue.append(new_path)  

    def startTraversal(self, start, end):
        print("Starting to find shortest path ...")

        start.visited = True
        path = [start]

        if start == end:
            return None

        path = self.BFS(start, end)
        print(len(path))
        print("Found shortest path!")

        print("\n")

        return path

