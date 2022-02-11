class AlphaPath:
    """Class that stores and manipulates paths within a FCFN"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, nodes: list, edges: list, FCFN):
        """Constructor of a Path instance
        NOTE: Paths are not 100% accurate representation of flow!"""
        self.nodes = nodes
        self.edges = edges
        self.start = nodes[0]
        self.end = nodes[-1]
        # Check if the path is from single source to single sink (i.e. not just a segment)
        if nodes[0][0] == "s" and nodes[-1][0] == "t":
            self.truePath = True
        else:
            self.truePath = False
        # Best guess at the flow of a path- the amount leaving the start node at the first edge on the path
        self.flow = FCFN.edgesDict[self.edges[0]].flow
        # Compute the total routing cost and routing cost per unit flow metric
        self.routingCost = 0
        for edge in self.edges:
            edgeCost = FCFN.edgesDict[edge].totalCost
            self.routingCost += edgeCost
        self.routingCostPerFlow = round(self.routingCost / self.flow)
        # The source cost data is accurate but the sink is widely inaccurate- would not use
        self.startCost = FCFN.nodesDict[self.start].variableCost * FCFN.edgesDict[self.edges[0]].flow
        self.endCost = FCFN.nodesDict[self.end].variableCost * FCFN.edgesDict[self.edges[-1]].flow
        self.totalCost = self.startCost + self.endCost + self.routingCost
        self.totalCostPerFlow = round(self.totalCost / self.flow)

    def printPathData(self):
        """Prints all relevant data for a path"""
        if self.truePath is True:
            print("=============== TRUE PATH ===============")
            print("Source/Start = " + self.start)
            print("Sink/End = " + str(self.nodes[-1]))
            print("Nodes:")
            print(self.nodes)
            print("Edges:")
            print(self.edges)
            print("Flow = " + str(self.flow))
            print("Routing Cost = " + str(self.routingCost))
            print("Cost Per Unit Flow = " + str(self.routingCostPerFlow))
        else:
            print("=============== ARC SEGMENT ===============")
            print("Source/Start = " + self.start)
            print("Sink/End = " + str(self.nodes[-1]))
            print("Nodes:")
            print(self.nodes)
            print("Edges:")
            print(self.edges)
            print("Flow = " + str(self.flow))
            print("Routing Cost = " + str(self.routingCost))
            print("Cost Per Unit Flow = " + str(self.routingCostPerFlow))

    def printPathTopology(self):
        """Prints only the nodes and edges of a path"""
        print("=============== PATH ===============")
        print("Nodes:")
        print(self.nodes)
        print("Edges:")
        print(self.edges)
