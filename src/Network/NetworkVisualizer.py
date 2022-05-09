from pyvis.network import Network as netVis

from src.Network.FlowNetwork import FlowNetwork


class NetworkVisualizer:
    """Class that allows visualizations of a Network using PyVis"""

    def __init__(self, network: FlowNetwork, directed=False, supers=False):
        """Constructor of a Visualizer instance with PyVis dependency"""
        self.network = network
        self.netVis = netVis(directed=directed)
        self.positionScalar = 8
        self.populateGraph(supers=supers)

    def populateGraph(self, supers=False) -> None:
        """Populates a PyVis instance with the network data"""
        for node in self.network.nodesDict.values():
            if node.nodeType == 0:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="blue",
                                     x=int(node.xPos * self.positionScalar),
                                     y=int(node.yPos * self.positionScalar))
            elif node.nodeType == 1:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="red",
                                     x=int(node.xPos * self.positionScalar),
                                     y=int(node.yPos * self.positionScalar))
            elif node.nodeType == 2:
                self.netVis.add_node(node.nodeID, label=node.nodeID, color="black",
                                     x=int(node.xPos * self.positionScalar),
                                     y=int(node.yPos * self.positionScalar))
        for edge in self.network.edgesArray:
            self.netVis.add_edge(int(edge[0]), int(edge[1]), label=self.network.edgesDict[(int(edge[0]), int(edge[1]))],
                                 color="black")
        if supers is True:
            self.netVis.add_node(-1, label="Super-Source", color="green", x=100 * self.positionScalar,
                                 y=100 * self.positionScalar)
            for source in self.network.sourcesArray:
                self.netVis.add_edge(-1, int(source), color="black")
                self.netVis.add_edge(int(source), -1, color="black")
            self.netVis.add_node(-2, label="Super-Sink", color="yellow",
                                 x=0, y=0)
            for sink in self.network.sinksArray:
                self.netVis.add_edge(int(sink), -2, color="black")

    def drawUnlabeledGraph(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.network.name + ".html"
        print("Drawing " + displayName + "...")
        # Sets visualization options using a JSON format (see vis.js documentation)
        self.netVis.set_options("""
                            var options = {
                                "autoResize": true,
                                "width": "100%",
                                "height": "100%",
                                "configure": {
                                    "enabled": false
                                },
                                "nodes": {
                                    "size": 10,
                                    "borderWidth": 3,
                                    "color": {
                                        "inherit": true
                                    },
                                    "fixed":{
                                        "x": true,
                                        "y": true
                                    },
                                    "font": {
                                        "size": 0,
                                        "color": "rgba(0, 0, 200, 0)",
                                        "strokeWidth": 2,
                                        "strokeColor": "rgba(0, 200, 0, 0)"
                                    },
                                    "scaling": {
                                        "min": 5,
                                        "max": 20
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
                                    }
                                },
                                "edges": {
                                    "color": {
                                        "inherit": true
                                    },
                                    "font": {
                                        "size": 0,
                                        "color": "rgba(235, 190, 0, 0)",
                                        "strokeWidth": 3,
                                        "strokeColor": "rgba(255, 0, 0, 0)"
                                    },
                                    "arrowStrikethrough": false,
                                    "arrows": {
                                        "to": {
                                            "scaleFactor": AntDemo
                                        }
                                    },
                                    "scaling": {
                                        "min": 2,
                                        "max": 16
                                    },
                                    "smooth": {
                                        "enabled": false,
                                        "type": "curvedCW",
                                        "roundness": 0.10
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
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
                                    "enabled": false,
                                    "stabilization": {
                                        "enabled": true,
                                        "fit": true
                                    },
                                    "barnesHut": {
                                        "avoidOverlap": AntDemo,
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

    def drawGraphWithLabels(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.network.name + ".html"
        print("Drawing " + displayName + "...")
        # Sets visualization options using a JSON format (see vis.js documentation)
        self.netVis.set_options("""
                            var options = {
                                "autoResize": true,
                                "width": "100%",
                                "height": "100%",
                                "configure": {
                                    "enabled": false
                                },
                                "nodes": {
                                    "size": 10,
                                    "borderWidth": 3,
                                    "color": {
                                        "inherit": true
                                    },
                                    "fixed":{
                                        "x": true,
                                        "y": true
                                    },
                                    "font": {
                                        "size": 25,
                                        "color": "rgba(0, 0, 200, AntDemo)",
                                        "strokeWidth": 2,
                                        "strokeColor": "rgba(0, 200, 0, AntDemo)"
                                    },
                                    "scaling": {
                                        "min": 5,
                                        "max": 20
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
                                    }
                                },
                                "edges": {
                                    "color": {
                                        "inherit": true
                                    },
                                    "font": {
                                        "size": 30,
                                        "color": "rgba(235, 190, 0, AntDemo)",
                                        "strokeWidth": 3,
                                        "strokeColor": "rgba(255, 0, 0, AntDemo)"
                                    },
                                    "arrowStrikethrough": false,
                                    "arrows": {
                                        "to": {
                                            "scaleFactor": AntDemo
                                        }
                                    },
                                    "scaling": {
                                        "min": 2,
                                        "max": 16
                                    },
                                    "smooth": {
                                        "enabled": false,
                                        "type": "curvedCW",
                                        "roundness": 0.10
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
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
                                    "enabled": false,
                                    "stabilization": {
                                        "enabled": true,
                                        "fit": true
                                    },
                                    "barnesHut": {
                                        "avoidOverlap": AntDemo,
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

    def drawBidirectionalGraphWithSmoothedLabeledEdges(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.network.name + ".html"
        print("Drawing " + displayName + "...")
        # Sets visualization options using a JSON format (see vis.js documentation)
        self.netVis.set_options("""
                            var options = {
                                "autoResize": true,
                                "width": "100%",
                                "height": "100%",
                                "configure": {
                                    "enabled": false
                                },
                                "nodes": {
                                    "size": 10,
                                    "borderWidth": 3,
                                    "color": {
                                        "inherit": true
                                    },
                                    "fixed":{
                                        "x": true,
                                        "y": true
                                    },
                                    "font": {
                                        "size": 25,
                                        "color": "rgba(0, 0, 200, AntDemo)",
                                        "strokeWidth": 2,
                                        "strokeColor": "rgba(0, 200, 0, AntDemo)"
                                    },
                                    "scaling": {
                                        "min": 5,
                                        "max": 20
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
                                    }
                                },
                                "edges": {
                                    "color": {
                                        "inherit": true
                                    },
                                    "font": {
                                        "size": 20,
                                        "color": "rgba(235, 190, 0, AntDemo)",
                                        "strokeWidth": 3,
                                        "strokeColor": "rgba(255, 0, 0, AntDemo)"
                                    },
                                    "arrowStrikethrough": false,
                                    "arrows": {
                                        "to": {
                                            "scaleFactor": AntDemo
                                        }
                                    },
                                    "scaling": {
                                        "min": 2,
                                        "max": 16
                                    },
                                    "smooth": {
                                        "enabled": true,
                                        "type": "curvedCW",
                                        "roundness": 0.10
                                    },
                                    "shadow": {
                                        "enabled": true,
                                        "size": 15,
                                        "color": "rgba(0, 0, 0, 0.25)"
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
                                    "enabled": false,
                                    "stabilization": {
                                        "enabled": true,
                                        "fit": true
                                    },
                                    "barnesHut": {
                                        "avoidOverlap": AntDemo,
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

    def drawBidirectionalGraph(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.network.name + ".html"
        print("Drawing " + displayName + "...")
        # Sets visualization options using a JSON format (see vis.js documentation)
        self.netVis.set_options("""
                                    var options = {
                                        "autoResize": true,
                                        "width": "100%",
                                        "height": "100%",
                                        "configure": {
                                            "enabled": false
                                        },
                                        "nodes": {
                                            "size": 10,
                                            "borderWidth": 3,
                                            "color": {
                                                "inherit": true
                                            },
                                            "fixed":{
                                                "x": true,
                                                "y": true
                                            },
                                            "font": {
                                                "size": 0,
                                                "color": "rgba(0, 0, 200, AntDemo)",
                                                "strokeWidth": 2,
                                                "strokeColor": "rgba(0, 200, 0, AntDemo)"
                                            },
                                            "scaling": {
                                                "min": 5,
                                                "max": 20
                                            },
                                            "shadow": {
                                                "enabled": true,
                                                "size": 15,
                                                "color": "rgba(0, 0, 0, 0.25)"
                                            }
                                        },
                                        "edges": {
                                            "color": {
                                                "inherit": true
                                            },
                                            "font": {
                                                "size": 0,
                                                "color": "rgba(235, 190, 0, AntDemo)",
                                                "strokeWidth": 3,
                                                "strokeColor": "rgba(255, 0, 0, AntDemo)"
                                            },
                                            "arrowStrikethrough": false,
                                            "arrows": {
                                                "to": {
                                                    "scaleFactor": AntDemo
                                                }
                                            },
                                            "scaling": {
                                                "min": 2,
                                                "max": 16
                                            },
                                            "smooth": {
                                                "enabled": true,
                                                "type": "curvedCW",
                                                "roundness": 0.10
                                            },
                                            "shadow": {
                                                "enabled": true,
                                                "size": 15,
                                                "color": "rgba(0, 0, 0, 0.25)"
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
                                            "enabled": false,
                                            "stabilization": {
                                                "enabled": true,
                                                "fit": true
                                            },
                                            "barnesHut": {
                                                "avoidOverlap": AntDemo,
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
