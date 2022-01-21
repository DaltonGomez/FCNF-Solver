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

    def drawGraphUiOptions(self, name: str):
        """Displays the FCNF using PyVis"""
        displayName = name + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)

        # Starting Layouts
        visual.barnes_hut()
        # visual.force_atlas_2based()
        # visual.hrepulsion()

        # Other options
        visual.show_buttons()

        # Display for UI option customization
        visual.from_nx(self.nx)
        visual.show(displayName)

    def drawGraphHardcodeOptions(self, name: str):
        """Displays the FCNF using PyVis"""
        displayName = name + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        # Sets visualization options using a JSON format (see vis.js documentation)
        visual.set_options("""
            var options = {
                "layout": { 
                    "randomSeed": 2 
                },
                "configure": {
                    "enabled": true
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "smooth": {
                        "enabled": false,
                        "type": "continuous"
                    }
                },
                "interaction": {
                    "dragNodes": true,
                    "hideEdgesOnDrag": false,
                    "hideNodesOnDrag": false
                },
                "physics": {
                    "barnesHut": {
                        "avoidOverlap": 0,
                        "centralGravity": 0.3,
                        "damping": 0.09,
                        "gravitationalConstant": -80000,
                        "springConstant": 0.001,
                        "springLength": 250
                    },
                    "enabled": true,
                    "stabilization": {
                        "enabled": true,
                        "fit": true,
                        "iterations": 1000,
                        "onlyDynamicEdges": false,
                        "updateInterval": 50
                    }
                }
            }
            """)
        visual.from_nx(self.nx)
        visual.show(displayName)

    def drawGraphHardcodeOptionsTwo(self, name: str):
        """Displays the FCNF using PyVis"""
        displayName = name + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        # Sets visualization options using a JSON format (see vis.js documentation)
        visual.set_options("""
            var options = {
                "layout": { 
                    "randomSeed": 2 
                },
                "configure": {
                    "enabled": true
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "smooth": {
                        "enabled": false,
                        "type": "continuous"
                    }
                },
                "interaction": {
                    "dragNodes": true,
                    "hideEdgesOnDrag": false,
                    "hideNodesOnDrag": false
                },
                "physics": {
                    "barnesHut": {
                        "avoidOverlap": 0,
                        "centralGravity": 0.3,
                        "damping": 0.09,
                        "gravitationalConstant": -80000,
                        "springConstant": 0.001,
                        "springLength": 250
                    },
                    "enabled": true,
                    "stabilization": {
                        "enabled": true,
                        "fit": true,
                        "iterations": 1000,
                        "onlyDynamicEdges": false,
                        "updateInterval": 50
                    }
                }
            }
            """)
        visual.from_nx(self.nx)
        visual.show(displayName)
