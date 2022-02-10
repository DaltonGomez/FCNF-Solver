class AlphaPath:
    """Class that stores and manipulates paths within a FCFN"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, nodes: list, edges: list, singlePath: bool, FCFN):
        """Constructor of a Path instance"""
        self.singlePath = singlePath  # Denotes if the path is from single source to single sink (i.e. no branching)
        self.tree = not self.singlePath
        self.nodes = nodes
        self.edges = edges
        self.source = nodes[0]
        self.sinks = []
        for node in self.nodes:
            if node[0] == "t":
                self.sinks.append(node)
        self.flow = FCFN.nodesDict[self.source].flow
        # Compute total true cost and cost per flow metric
        sourceCost = FCFN.nodesDict[self.source].totalCost
        sinkCost = 0
        for sink in self.sinks:
            incomingEdges = FCFN.nodesDict[sink].incomingEdges
            for edge in incomingEdges:
                if edge in self.edges:
                    sinkCost += FCFN.nodesDict[sink].variableCost * FCFN.edgesDict[edge].flow
        routingCost = 0
        for edge in edges:
            edgeCost = FCFN.edgesDict[edge].totalCost
            routingCost += edgeCost
        self.trueCost = sourceCost + sinkCost + routingCost
        self.costPerFlow = round(self.trueCost / self.flow)

    def printPathData(self):
        """Prints all relevant data for a path"""
        if self.singlePath is True:
            print("=============== PATH ===============")
            print("Source = " + self.source)
            print("Sink = " + str(self.nodes[-1]))
            print("Nodes:")
            print(self.nodes)
            print("Edges:")
            print(self.edges)
            print("Flow = " + str(self.flow))
            print("True Cost = " + str(self.trueCost))
            print("Cost Per Unit Flow = " + str(self.costPerFlow))
        elif self.tree is True:
            print("=============== Tree ===============")
            print("Source/Root = " + self.source)
            print("Sink/Leaves = " + str(self.sinks))
            print("Nodes:")
            print(self.nodes)
            print("Edges:")
            print(self.edges)
            print("Flow = " + str(self.flow))
            print("True Cost = " + str(self.trueCost))
            print("Cost Per Unit Flow = " + str(self.costPerFlow))
