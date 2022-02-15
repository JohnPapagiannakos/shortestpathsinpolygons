# using binarySearchTree.py 
# based on link[4]: https://gist.github.com/jakemmarsh/8273963

class Node:
    def __init__(self, val):
        self.val = val
        self.leftChild = None
        self.rightChild = None

    def get(self):
        return self.val

    def set(self, val):
        self.val = val

    # def getChildren(self):
    #     children = []
    #     if(self.leftChild != None):
    #         children.append(self.leftChild)
    #     if(self.rightChild != None):
    #         children.append(self.rightChild)
    #     return children

class BST:
    def __init__(self):
        self.data = list()

    # def setRoot(self, val):
    #     self.root = Node(val)

    # def insert(self, val):
    #     if(self.root is None):
    #         self.setRoot(val)
    #     else:
    #         self.insertNode(self.root, val)

    def insert(self, edge):
        self.data.append(edge)

    # def insertNode(self, currentNode, val):
    #     if(val <= currentNode.val):
    #         if(currentNode.leftChild):
    #             self.insertNode(currentNode.leftChild, val)
    #         else:
    #             currentNode.leftChild = Node(val)
    #     elif(val > currentNode.val):
    #         if(currentNode.rightChild):
    #             self.insertNode(currentNode.rightChild, val)
    #         else:
    #             currentNode.rightChild = Node(val)

    # def find(self, val):
    #     return self.findNode(self.root, val)

    # def findNode(self, currentNode, val):
    #     if(currentNode is None):
    #         return False
    #     elif(val == currentNode.val):
    #         return True
    #     elif(val < currentNode.val):
    #         return self.findNode(currentNode.leftChild, val)
    #     else:
    #         return self.findNode(currentNode.rightChild, val)

    def remove(self, instance):
        self.data.remove(instance)

    def getLeftEdge(self, node):
        y = node.getCoordinates()[1]
        tmpList = list()

        for edge in self.data:
            p = edge.getOrigin().getCoordinates()
            q = edge.getOrigin().getCoordinates()
            
            if p[0] == q[0]:
                x = q[0]
            elif p[1] == q[1]:
                if p[0] > q[0]:
                    x = p[0]
                else:
                    x = q[0]
            else:
                # Slope:= a = (y_2 - y_1)/(x_2 - x_1)
                a = float(q[1] - p[1]) / (q[0] - p[0] + 1E-6) # add a small positive constant in order to avoid division by zero 
                b = p[0] - a * p[1]
                # x = (y-b)/a
                x = float(y - b) / a

            tmpList.append([edge, x])

        tmpList = sorted(tmpList, key=lambda x: (x[1]))

        i = 0
        while i < len(tmpList) and tmpList[i][1] < node.getCoordinates()[0]:
            i += 1

        if i > 0:
            i -= 1

        return tmpList[i][0]
