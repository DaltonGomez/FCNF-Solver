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
        for node in self.FCFN.nodesDict:
            nodeObj = self.FCFN.nodesDict[node]
            if node[0] == "s":
                self.nx.add_node(node, value=nodeObj.flow, color="blue")
            elif node[0] == "t":
                self.nx.add_node(node, value=nodeObj.flow, color="red")
            elif node[0] == "n":
                self.nx.add_node(node, value=nodeObj.flow, color="black")
        for edge in self.FCFN.edgesDict:
            edgeObj = self.FCFN.edgesDict[edge]
            self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="black")

    def drawGraph(self, name: str):
        """Displays the FCNF using PyVis"""
        displayName = name + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        # Layouts
        visual.barnes_hut()
        # visual.force_atlas_2based()
        # visual.hrepulsion()

        # Other options
        # visual.show_buttons()
        # visual.toggle_physics(False)
        # visual.toggle_stabilization(False)

        visual.from_nx(self.nx)
        visual.show(displayName)
