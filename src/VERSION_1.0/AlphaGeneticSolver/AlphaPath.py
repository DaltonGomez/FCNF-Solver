class AlphaPath:
    """Class that stores opened paths within an alpha individual and computes path-based data"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, nodes: list, edges: list, alphaIndividual):
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
        self.flow = alphaIndividual.openedEdgesDict[self.edges[0]][0]
        # Compute the total routing cost and routing cost per unit flow metric
        self.routingCost = 0
        for edge in self.edges:
            edgeCost = alphaIndividual.openedEdgesDict[edge][1]
            self.routingCost += edgeCost
        self.flowPerCostDensity = self.flow / self.routingCost
        # The source cost data is accurate but the sink is widely inaccurate- would not use
        self.startCost = alphaIndividual.openedNodesDict[self.start][1]
        self.endCost = alphaIndividual.openedNodesDict[self.end][1]
        self.totalCost = self.startCost + self.endCost + self.routingCost
        self.totalFlowPerCostDensity = self.flow / self.totalCost

    def printPathData(self) -> None:
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
            print("Flow Per Cost Density = " + str(self.flowPerCostDensity))
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
            print("Flow Per Cost Density = " + str(self.flowPerCostDensity))

    def printPathTopology(self) -> None:
        """Prints only the nodes and edges of a path"""
        print("=============== PATH ===============")
        print("Nodes:")
        print(self.nodes)
        print("Edges:")
        print(self.edges)
