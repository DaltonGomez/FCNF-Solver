from pyvis.network import Network as netVis

from src.Network.FlowNetwork import FlowNetwork


class NetworkVisualizer:
    """Class that allows visualizations of a Network using PyVis and/or NetworkX"""

    def __init__(self, network: FlowNetwork):
        """Constructor of a Visualizer instance with NetworkX and PyVis dependencies"""
        self.network = network
        self.netVis = netVis()
        self.populateGraph()

    def populateGraph(self) -> None:
        """Populates a NetworkX instance with the network data"""
        for node in self.network.nodesDict.values():
            if node.nodeType == 0:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="blue", x=int(node.xPos * 8),
                                     y=int(node.yPos * 8))
            elif node.nodeType == 1:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="red", x=int(node.xPos * 8),
                                     y=int(node.yPos * 8))
            elif node.nodeType == 2:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="black", x=int(node.xPos * 8),
                                     y=int(node.yPos * 8))
        for edge in self.network.edgesArray:
            self.netVis.add_edge(int(edge[0]), int(edge[1]), color="black")

    def drawGraph(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.network.name + ".html"
        print("Drawing " + displayName + "...")
        # Sets visualization options using a JSON format (see vis.js documentation)
        self.netVis.set_options("""
                    var options = {
                        "autoResize": true,
                        "width": "1000px",
                        "height": "1000px",
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
                            "fixed":{
                                "x": true,
                                "y": true
                            },
                            "font": {
                                "size": 20,
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
        self.netVis.show(displayName)
