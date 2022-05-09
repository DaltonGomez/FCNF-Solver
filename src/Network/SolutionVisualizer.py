from pyvis.network import Network as netVis

from src.Network.Solution import Solution


class SolutionVisualizer:
    """Class that allows visualizations of a Solution using PyVis"""

    def __init__(self, solution: Solution):
        """Constructor of a Visualizer instance with PyVis dependency"""
        self.solution = solution
        self.netVis = netVis(directed=True)
        self.positionScalar = 8
        self.populateGraph()

    def populateGraph(self) -> None:
        """Populates a PyVis instance with the solution data"""
        # Add source nodes
        for s in range(self.solution.network.numSources):
            srcID = self.solution.network.sourcesArray[s]
            srcObj = self.solution.network.nodesDict[srcID]
            self.netVis.add_node(srcObj.nodeID, label=srcObj.nodeID, color="blue",
                                 value=self.solution.sourceFlows[s],
                                 x=int(srcObj.xPos * self.positionScalar),
                                 y=int(srcObj.yPos * self.positionScalar))
        # Add sink nodes
        for t in range(self.solution.network.numSinks):
            sinkID = self.solution.network.sinksArray[t]
            sinkObj = self.solution.network.nodesDict[sinkID]
            self.netVis.add_node(sinkObj.nodeID, label=sinkObj.nodeID, color="red",
                                 value=self.solution.sinkFlows[t],
                                 x=int(sinkObj.xPos * self.positionScalar),
                                 y=int(sinkObj.yPos * self.positionScalar))
        # Add intermediate nodes
        for n in range(self.solution.network.numInterNodes):
            nodeID = self.solution.network.interNodesArray[n]
            nodeObj = self.solution.network.nodesDict[nodeID]
            flow = 0
            for edge in nodeObj.incomingEdges:
                edgeIndex = self.solution.network.edgesDict[edge]
                for arc in range(self.solution.network.numArcCaps):
                    flow += self.solution.arcFlows[(edgeIndex, arc)]
            if flow > 0:
                self.netVis.add_node(nodeObj.nodeID, label=nodeObj.nodeID, color="black", value=flow,
                                     x=int(nodeObj.xPos * self.positionScalar),
                                     y=int(nodeObj.yPos * self.positionScalar))
            else:
                self.netVis.add_node(nodeObj.nodeID, label=nodeObj.nodeID, color="rgba(155, 155, 155, 0.35)",
                                     value=flow,
                                     x=int(nodeObj.xPos * self.positionScalar),
                                     y=int(nodeObj.yPos * self.positionScalar))
        # Add edges
        for e in range(self.solution.network.numEdges):
            edge = self.solution.network.edgesArray[e]
            backEdge = self.solution.network.edgesDict[(edge[1], edge[0])]
            flow = 0
            backFlow = 0
            for arc in range(self.solution.network.numArcCaps):
                flow += self.solution.arcFlows[(e, arc)]
                backFlow += self.solution.arcFlows[(backEdge, arc)]
            if flow > 0:
                self.netVis.add_edge(int(edge[0]), int(edge[1]), label=round(flow), color="black", value=flow)
            elif backFlow > 0:
                self.netVis.add_edge(int(edge[1]), int(edge[0]), label=round(backFlow), color="black", value=backFlow)
            elif flow == 0 and backFlow == 0:
                self.netVis.add_edge(int(edge[0]), int(edge[1]), color="rgba(155, 155, 155, 0.35)", value=flow)
                self.netVis.add_edge(int(edge[1]), int(edge[0]), color="rgba(155, 155, 155, 0.35)", value=backFlow)

    def drawGraphWithLabels(self, leadingText="") -> None:
        """Displays the Solution using PyVis and a set of hardcoded options"""
        displayName = leadingText + self.solution.name + ".html"
        # print("Drawing " + displayName + "...")
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
                                "color": "rgba(0, 0, 200, 1)",
                                "strokeWidth": 2,
                                "strokeColor": "rgba(0, 200, 0, 1)"
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
                                "color": "rgba(235, 190, 0, 1)",
                                "strokeWidth": 3,
                                "strokeColor": "rgba(255, 0, 0, 1)"
                            },
                            "arrowStrikethrough": false,
                            "arrows": {
                                "to": {
                                    "scaleFactor": 1
                                }
                            },
                            "scaling": {
                                "min": 2,
                                "max": 10
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

    def drawUnlabeledGraph(self, leadingText="") -> None:
        """Displays the Solution using PyVis and a set of hardcoded options"""
        displayName = leadingText + self.solution.name + ".html"
        # print("Drawing " + displayName + "...")
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
                                    "scaleFactor": 1
                                }
                            },
                            "scaling": {
                                "min": 2,
                                "max": 10
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

    def drawBidirectionalGraphWithSmoothedLabeledEdges(self) -> None:
        """Displays the Network using PyVis and a set of hardcoded options"""
        displayName = self.solution.name + ".html"
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
                                        "color": "rgba(0, 0, 200, 1)",
                                        "strokeWidth": 2,
                                        "strokeColor": "rgba(0, 200, 0, 1)"
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
                                        "color": "rgba(235, 190, 0, 1)",
                                        "strokeWidth": 3,
                                        "strokeColor": "rgba(255, 0, 0, 1)"
                                    },
                                    "arrowStrikethrough": false,
                                    "arrows": {
                                        "to": {
                                            "scaleFactor": 1
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
