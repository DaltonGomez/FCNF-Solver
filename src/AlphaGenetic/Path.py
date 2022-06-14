class Path:
    """Class that stores opened paths/pathlets within an individual and computes path-based data"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, visitedNodes: list, visitedEdges: list, pathFlowAndCosts: tuple):
        """Constructor of a Path instance"""
        # TODO - Decide if pathing is to remain or be removed
        # Path topology fields
        self.nodes = visitedNodes
        self.edges = visitedEdges
        self.length = len(self.edges)
        # Path flow and cost fields
        self.pathFlow = pathFlowAndCosts[0]
        self.pathFixedCost = pathFlowAndCosts[1]
        self.pathVariableCost = pathFlowAndCosts[2]
        self.pathRoutingCost = self.pathFixedCost + self.pathVariableCost
        self.srcSinkCost = pathFlowAndCosts[3]
        self.totalCost = self.pathRoutingCost + self.srcSinkCost
        # Additional metrics
        self.flowPerRoutingCost = self.pathFlow / self.pathRoutingCost
        self.flowPerTotalCost = self.pathFlow / self.totalCost

    def printPathData(self) -> None:
        """Prints the data of the path"""
        print("========= PATH =========")
        print("Nodes: " + str(self.nodes))
        print("Edges: " + str(self.edges))
        print("Path Flow: " + str(round(self.pathFlow, 1)))
        print("Routing Cost: " + str(round(self.pathRoutingCost, 1)))
        print("Path Fixed Cost: " + str(round(self.pathFixedCost, 1)))
        print("Path Variable Cost: " + str(round(self.pathVariableCost, 1)))
        print("Source/Sink Cost: " + str(round(self.srcSinkCost, 1)))
        print("Total Cost: " + str(round(self.totalCost, 1)))
        print("Flow per Routing Cost: " + str(round(self.flowPerRoutingCost, 4)))
        print("Flow per Total Cost: " + str(round(self.flowPerTotalCost, 4)))
