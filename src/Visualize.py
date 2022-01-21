import networkx as nx
from pyvis.network import Network as netVis

from src.FixedChargeFlowNetwork import FixedChargeFlowNetwork


class Visualize:
    """Class that defines a Fixed Charge Flow Network"""

    def __init__(self, FCFN: FixedChargeFlowNetwork):
        """Constructor of a Visualize instance with NetworkX and PyVis dependencies"""
        self.FCFN = FCFN
        self.nx = nx.DiGraph()
        self.populateGraph()

    def populateGraph(self):
        """Populates a NetworkX instance with the FCFN data"""
        for node in self.FCFN.nodeDict:
            self.nx.add_node(node)
        for edge in self.FCFN.edgeDict:
            edgeObj = self.FCFN.edgeDict[edge]
            self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode)

    def drawGraph(self):
        """Displays the FCNF using PyVis"""
        visual = netVis("500px", "500px", directed=True)
        visual.from_nx(self.nx)
        visual.show(str(self.FCFN.name) + ".html")
        print("Showing...")
