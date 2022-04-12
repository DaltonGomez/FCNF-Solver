import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

points = np.array([[0, 0], [0, 2], [3, 2], [2, 1], [3, 0]])

tri = Delaunay(points)
print(tri)
print(type(tri))
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.show()

edgeSet = set()  # Declare a new dictionary for the edge set
for simplex in range(tri.nsimplex):  # Iterate over each simplex
    print("Simplex Number " + str(simplex))
    firstNode = -1
    for vertex in range(3):  # Iterate over the three points in a simplex
        print(tri.points[tri.simplices[simplex, vertex]])  # Prints the (x,y) coordinates of the nodes in the simplex
        if vertex == 0:
            firstNode = tri.simplices[simplex, vertex]  # Store the first vertex in the simplex
        print(tri.simplices[simplex, vertex])  # Prints the node id of the nodes in the simplex
        if vertex != 2:
            edgeList = [tri.simplices[simplex, vertex], tri.simplices[simplex, vertex + 1]]  # Makes bi-directional edge
            forwardEdge = tuple(sorted(edgeList, reverse=False))  # Makes unidirectional forward edge
            edgeSet.add(forwardEdge)  # Adds to set for deduplication
            backwardEdge = tuple(sorted(edgeList, reverse=True))  # Makes unidirectional backward edge
            edgeSet.add(backwardEdge)  # Adds to set for deduplication
        else:  # Logic for the edge connecting the last point to the first
            edgeList = (tri.simplices[simplex, vertex], firstNode)
            forwardEdge = tuple(sorted(edgeList, reverse=False))
            edgeSet.add(forwardEdge)
            backwardEdge = tuple(sorted(edgeList, reverse=True))
            edgeSet.add(backwardEdge)

print(edgeSet)
print(len(edgeSet))
