import networkx as nx
from pyvis.network import Network as netVis

from src.FixedCostNetworkFlowSolver.FCNF import FCNF


class Visualize:
    """Class that allows visualizations of a FCFN"""

    def __init__(self, FCFNinstance: FCNF):
        """Constructor of a Visualize instance with NetworkX and PyVis dependencies"""
        self.FCNF = FCFNinstance
        self.nx = nx.DiGraph()
        self.populateGraph()

    def populateGraph(self):
        """Populates a NetworkX instance with the FCFN data"""
        addedTotalCost = False
        for node in self.FCNF.nodesDict:
            nodeObj = self.FCNF.nodesDict[node]
            if node[0] == "s":
                if addedTotalCost is False:
                    self.nx.add_node(node, value=nodeObj.flow, color="blue",
                                     label="Total Cost= " + str(round(self.FCNF.totalCost)))
                    addedTotalCost = True
                else:
                    self.nx.add_node(node, value=nodeObj.flow, color="blue")
            elif node[0] == "t":
                self.nx.add_node(node, value=nodeObj.flow, color="red")
            elif node[0] == "n":
                if nodeObj.opened is True:
                    self.nx.add_node(node, value=nodeObj.flow, color="black")
                elif nodeObj.opened is False:
                    self.nx.add_node(node, value=nodeObj.flow, color="grey")
        for edge in self.FCNF.edgesDict:
            edgeObj = self.FCNF.edgesDict[edge]
            if edgeObj.opened is True:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="black",
                                 label=str(round(edgeObj.flow)))
            elif edgeObj.opened is False:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="grey")

    def drawGraph(self, name: str):
        """Displays the FCNF using PyVis and a set of hardcoded options"""
        displayName = name + str(self.FCNF.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        visual.set_options("""
            var options = {
                "layout": { 
                    "randomSeed":""" + str(self.FCNF.visSeed) + "," +
                           """
                    "improvedLayout": true
                },
                "autoResize": true,
                "nodes": {
                    "borderWidth": 2,
                    "borderWidthSelected": 2,
                    "font": {
                        "color": "rgba(221,212,0,1)",
                        "size": 30,
                        "strokeWidth": 5,
                        "strokeColor": "rgba(0,0,0,1)"
                    },
                    "labelHighlightBold": false,
                    "physics": false,
                    "shadow": {
                        "enabled": true
                    },
                    "size": 5
                },
                "configure": {
                    "enabled": false
                },
                "edges": {
                    "color": {
                        "inherit": true
                    },
                    "font": {
                        "color": "rgba(12, 224, 54, 1)",
                        "size": 30,
                        "strokeWidth": 5,
                        "strokeColor": "rgba(0,0,0,1)"
                    },
                    "smooth": {
                        "enabled": false,
                        "type": "continuous"
                    },
                    "shadow": {
                        "enabled": true
                    }
                },
                "interaction": {
                    "dragNodes": false,
                    "selectable": false,
                    "selectConnectedEdges": false,
                    "hoverConnectedEdges": false,
                    "hideEdgesOnDrag": false,
                    "hideNodesOnDrag": false
                },
                "physics": {
                    "barnesHut": {
                        "avoidOverlap": 10,
                        "centralGravity": 0.3,
                        "damping": 0.09,
                        "gravitationalConstant": -80000,
                        "springConstant": 0.001,
                        "springLength": 250
                    },
                    "enabled": true
                }
            }
            """)
        visual.show(displayName)

    def drawGraphUiOptions(self, name: str):
        """Displays the FCNF using PyVis and provides a UI for customizing options, which can be copied in JSON"""
        displayName = name + str(self.FCNF.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        visual.from_nx(self.nx)
        visual.barnes_hut()  # Starting Layouts
        # visual.force_atlas_2based()
        # visual.hrepulsion()
        visual.show_buttons()  # Display for UI option customization
        visual.show(displayName)
