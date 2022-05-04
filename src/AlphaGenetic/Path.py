from Network.FlowNetwork import FlowNetwork


class Path:
    """Class that stores opened paths/pathlets within an individual and computes path-based data"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, arcFlows: dict, network: FlowNetwork):
        """Constructor of a Path instance"""
        self.arcFlows = arcFlows  # Dictionary keyed on all opened arcs in the path, with value of flow amount

        # TODO - Determine the additional metrics after deciding how paths are computed
        # Compute the total routing cost and routing cost per unit flow metric
        self.routingCost = 0
        """
        for arc in self.arcFlows.keys():
            edgeCost = network.openedEdgesDict[edge][1]
            self.routingCost += edgeCost
        self.flowPerCostDensity = self.flow / self.routingCost
        self.totalCost = self.startCost + self.endCost + self.routingCost
        self.totalFlowPerCostDensity = self.flow / self.totalCost
        """
