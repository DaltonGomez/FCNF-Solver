import networkx as nx
from pyvis.network import Network as netVis


class AlphaVisualizer:
    """Class that allows visualizations of an alpha-relaxed FCFN"""

    def __init__(self, individual):
        """Constructor of a AlphaVisualizer instance with NetworkX and PyVis dependencies
        NOTE: individual must be of type Individual (Not type hinted to prevent circular import)"""
        self.individual = individual
        self.nx = nx.DiGraph()
        self.populateGraph()

    def populateGraph(self) -> None:
        """Populates a NetworkX instance with the AlphaIndividual data"""
        for node in self.individual.FCFN.nodesDict:
            nodeObj = self.individual.FCFN.nodesDict[node]
            if nodeObj.nodeID[0] == "s":
                self.nx.add_node(node, value=nodeObj.flow, color="blue")
            elif nodeObj.nodeID[0] == "t":
                self.nx.add_node(node, value=nodeObj.flow, color="red")
            elif nodeObj.nodeID[0] == "n":
                if nodeObj.opened is True:
                    self.nx.add_node(node, value=nodeObj.flow, color="black")
                elif nodeObj.opened is False:
                    self.nx.add_node(node, value=nodeObj.flow, color="grey")
        for edge in self.individual.FCFN.edgesDict:
            edgeObj = self.individual.FCFN.edgesDict[edge]
            if edgeObj.opened is True:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="black")
            elif edgeObj.opened is False:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeObj.flow, color="grey")

    def drawGraph(self, name: str) -> None:
        """Displays the FCNF using PyVis and a set of hardcoded options"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + "_Target=" + str(
            round(self.individual.minTargetFlow)) + ".html"
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
                            "randomSeed":""" + str(self.individual.FCFN.visSeed) + "," +
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
                    "selectable": true,
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

    def drawGraphUiOptions(self, name: str) -> None:
        """Displays the FCNF using PyVis and provides a UI for customizing options, which can be copied as JSON"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "800px", directed=True)
        visual.from_nx(self.nx)
        visual.barnes_hut()  # Starting Layouts
        # visual.force_atlas_2based()
        # visual.hrepulsion()
        visual.show_buttons()  # Display for UI option customization
        visual.show(displayName)

    def drawSmallGraph(self, name: str) -> None:
        """Displays a SMALL FCNF using PyVis and a set of hardcoded options"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        visual.set_options("""
            var options = {
                "layout": { 
                    "randomSeed":""" + str(self.individual.FCFN.visSeed) + "," +
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
