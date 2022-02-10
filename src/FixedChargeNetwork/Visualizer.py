import networkx as nx
from pyvis.network import Network as netVis


class Visualizer:
    """Class that allows visualizations of a FCFN using PyVis and NetworkX"""

    def __init__(self, FCFNinstance):
        """Constructor of a Visualizer instance with NetworkX and PyVis dependencies
        NOTE: FCFNinstance must be of type FixedChargeFlowNetwork (Not type hinted to prevent circular import)"""
        self.FCFN = FCFNinstance
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
                if nodeObj.opened is True:
                    self.nx.add_node(node, value=nodeObj.flow, color="black")
                elif nodeObj.opened is False:
                    self.nx.add_node(node, value=nodeObj.flow, color="grey")
        for edge in self.FCFN.edgesDict:
            edgeObj = self.FCFN.edgesDict[edge]
            if edgeObj.opened is True:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="black")
            elif edgeObj.opened is False:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="grey")

    def drawGraph(self, name: str):
        """Displays the FCNF using PyVis and a set of hardcoded options"""
        displayName = name + "_Cost=" + str(round(self.FCFN.totalCost)) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        visual.set_options("""
                    var options = {
                        "autoResize": true,
                        "width": "1000px",
                        "height": "800px",
                        "layout": { 
                            "randomSeed":""" + str(self.FCFN.visSeed) + "," +
                           """
                    "improvedLayout": true
                },
                "configure": {
                    "enabled": false
                },
                "nodes": {
                    "physics": true,
                    "size": 6,
                    "borderWidth": 3,
                    "color": {
                        "inherit": true
                    },
                    "font": {
                        "size": 0,
                        "color": "rgba(0,0,0,1)",
                        "strokeWidth": 0,
                        "strokeColor": "rgba(0,0,0,1)"
                    },
                    "scaling": {
                        "min": 10,
                        "max": 60
                    },
                    "shadow": {
                        "enabled": true,
                        "size": 15,
                        "color": "rgba(0,0,0,0.5)"
                    }
                },
                "edges": {
                    "physics": true,
                    "color": {
                        "inherit": true
                    },
                    "font": {
                        "size": 0,
                        "color": "rgba(0,0,0,1)",
                        "strokeWidth": 0,
                        "strokeColor": "rgba(0,0,0,1)"
                    },
                    "arrowStrikethrough": false,
                    "arrows": {
                        "to": {
                            "scaleFactor": 2
                        }
                    },
                    "scaling": {
                        "min": 1,
                        "max": 25
                    },
                    "smooth": {
                        "enabled": false
                    },
                    "shadow": {
                        "enabled": true,
                        "size": 15,
                        "color": "rgba(0,0,0,0.5)"
                    }
                },
                "interaction": {
                    "dragView": true,
                    "zoomView": true,
                    "dragNodes": false,
                    "selectable": false,
                    "selectConnectedEdges": false,
                    "hoverConnectedEdges": false,
                    "hideEdgesOnDrag": false,
                    "hideNodesOnDrag": false
                },
                "physics": {
                    "enabled": true,
                    "stabilization": {
                        "enabled": true,
                        "fit": true
                    },
                    "barnesHut": {
                        "avoidOverlap": 1,
                        "centralGravity": 0.2,
                        "damping": 0.90,
                        "gravitationalConstant": -100000,
                        "springConstant": 0.001,
                        "springLength": 500
                    }
                }
            }
            """)
        visual.show(displayName)

    def drawGraphUiOptions(self, name: str):
        """Displays the FCNF using PyVis and provides a UI for customizing options, which can be copied as JSON"""
        displayName = name + str(self.FCFN.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        visual.from_nx(self.nx)
        visual.barnes_hut()  # Starting Layouts
        # visual.force_atlas_2based()
        # visual.hrepulsion()
        visual.show_buttons()  # Display for UI option customization
        visual.show(displayName)
