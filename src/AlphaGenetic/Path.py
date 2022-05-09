from Network.FlowNetwork import FlowNetwork


class Path:
    """Class that stores opened paths/pathlets within an individual and computes path-based data"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, network: FlowNetwork, visitedNodes: list, visitedEdges: list):
        """Constructor of a Path instance"""
        self.network = network  # TODO - Decide if this is a needed attribute
        self.nodes = visitedNodes
        self.arcs = visitedEdges

        # TODO - Determine the additional metrics after deciding how paths are computed
        # Compute the total routing cost and routing cost per unit flow metric
        self.routingCost = 0
        """
        for arc in self.arcFlows.keys():
            edgeCost = network.openedEdgesDict[edge][AntDemo]
            self.routingCost += edgeCost
        self.flowPerCostDensity = self.flow / self.routingCost
        self.totalCost = self.startCost + self.endCost + self.routingCost
        self.totalFlowPerCostDensity = self.flow / self.totalCost
        """
