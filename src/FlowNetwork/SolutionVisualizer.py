import math

from pyvis.network import Network as netVis

from src.FlowNetwork.FlowNetworkSolution import FlowNetworkSolution


class SolutionVisualizer:
    """Class that allows visualizations of a flow network solution using PyVis"""

    def __init__(self, solution: FlowNetworkSolution):
        """Constructor of a SolutionVisualizer instance with PyVis dependency"""
        self.solution: FlowNetworkSolution = solution  # The input solution to be visualized
        self.netVis: netVis = netVis(directed=True)  # PyVis Network visualizer object to render the visualization
        self.positionScalar: int = 8  # Scales the relative position of the nodes within the graph
        self.populateGraph()  # Calls the populated graph method on instantiation

    def populateGraph(self) -> None:
        """Populates a PyVis instance with the solution data"""
        # Add source nodes
        for s in range(self.solution.graph.numSources):
            srcID = self.solution.graph.sourcesArray[s]
            srcObj = self.solution.graph.nodesDict[srcID]
            self.netVis.add_node(srcObj.nodeID, label=srcObj.nodeID, color="blue",
                                 value=self.solution.sourceFlows[s],
                                 x=int(srcObj.xPos * self.positionScalar),
                                 y=int(srcObj.yPos * self.positionScalar))
        # Add sink nodes
        for t in range(self.solution.graph.numSinks):
            sinkID = self.solution.graph.sinksArray[t]
            sinkObj = self.solution.graph.nodesDict[sinkID]
            self.netVis.add_node(sinkObj.nodeID, label=sinkObj.nodeID, color="red",
                                 value=self.solution.sinkFlows[t],
                                 x=int(sinkObj.xPos * self.positionScalar),
                                 y=int(sinkObj.yPos * self.positionScalar))
        # Add intermediate nodes
        for n in range(self.solution.graph.numInterNodes):
            nodeID = self.solution.graph.interNodesArray[n]
            nodeObj = self.solution.graph.nodesDict[nodeID]
            flow = 0
            for edge in nodeObj.incomingEdges:
                edgeIndex = self.solution.graph.edgesDict[edge]
                for arc in range(self.solution.graph.numArcsPerEdge):
                    # Try/Except/Else block as CPLEX sometimes fails to write flow decision variables to the solution object
                    try:
                        flow += self.solution.arcFlows[(edgeIndex, arc)]
                    except KeyError:
                        print("ERROR: Key error on solution.arcFlows[" + str((edgeIndex, arc)) +
                              "] (edge = " + str(edge) + "! Assuming CPLEX decided zero flow...")
                    else:
                        flow += self.solution.arcFlows[(edgeIndex, arc)]
            if flow > 0:
                self.netVis.add_node(nodeObj.nodeID, label=nodeObj.nodeID, color="rgba(0, 0, 0, 1)", value=flow,
                                     x=int(nodeObj.xPos * self.positionScalar),
                                     y=int(nodeObj.yPos * self.positionScalar))
            else:
                self.netVis.add_node(nodeObj.nodeID, label=nodeObj.nodeID, color="rgba(155, 155, 155, 0.30)",
                                     value=flow,
                                     x=int(nodeObj.xPos * self.positionScalar),
                                     y=int(nodeObj.yPos * self.positionScalar))
        # Add edges
        for edgeIndex in range(self.solution.graph.numEdges):
            edge = self.solution.graph.edgesArray[edgeIndex]
            backEdgeIndex = self.solution.graph.edgesDict[(edge[1], edge[0])]
            flow = 0
            backFlow = 0
            numArcsOpened = 0
            numBackArcsOpened = 0
            for arc in range(self.solution.graph.numArcsPerEdge):
                # Try/Except block as CPLEX sometimes fails to write flow decision variables to the solution object
                # This assumes that values CPLEX does not write should zero flow
                try:
                    flow += self.solution.arcFlows[(edgeIndex, arc)]
                    backFlow += self.solution.arcFlows[(backEdgeIndex, arc)]
                    if self.solution.arcFlows[(edgeIndex, arc)] > 0:
                        numArcsOpened += 1
                    if self.solution.arcFlows[(backEdgeIndex, arc)] > 0:
                        numBackArcsOpened += 1
                except KeyError:
                    print("ERROR: Key error on solution.arcFlows[" + str((edgeIndex, arc)) +
                          "] (edge = " + str(edge) + "! Assuming CPLEX decided zero flow...")
            # Print a warning if there are opposing flows
            if flow > 0 and backFlow > 0:
                print("WARNING: Opposing flows check thrown on edge index [" + str(edgeIndex) + "] and back-edge [" +
                      str(backEdgeIndex) + "]! Ignoring and rendering the forward flow only...")
            if flow > 0:
                rgbTuple = self.makeScaledRGBTupleFromArcsOpened(numArcsOpened)
                rgbString = "rgba(" + str(rgbTuple[0]) + ", " + str(rgbTuple[1]) + ", " + str(rgbTuple[2]) + ", 1)"
                self.netVis.add_edge(int(edge[0]), int(edge[1]), label=round(flow), color=rgbString, value=flow)
            elif backFlow > 0:
                rgbTuple = self.makeScaledRGBTupleFromArcsOpened(numBackArcsOpened)
                rgbString = "rgba(" + str(rgbTuple[0]) + ", " + str(rgbTuple[1]) + ", " + str(rgbTuple[2]) + ", 1)"
                self.netVis.add_edge(int(edge[1]), int(edge[0]), label=round(backFlow), color=rgbString, value=backFlow)
            elif flow == 0 and backFlow == 0:
                self.netVis.add_edge(int(edge[0]), int(edge[1]), color="rgba(155, 155, 155, 0.35)", value=flow)
                self.netVis.add_edge(int(edge[1]), int(edge[0]), color="rgba(155, 155, 155, 0.35)", value=backFlow)

    def makeScaledRGBTupleFromArcsOpened(self, arcsOpened: int) -> tuple:
        """Returns an RGB tuple from arcs opened on an edge, where
        (0,0,0) = 1 arc, (0,0,255) = 2 arcs; (255,0,0) = max arcs"""
        # CURRENT IMPLEMENTATION RAMPS FROM BLUE (FEW ARCS) TO RED (MANY ARCS) WITH BLACK AS ONE ARC
        if arcsOpened <= 1:
            return 0, 0, 0
        else:
            maxArcs = self.solution.graph.numArcsPerEdge
            redVal = int(255 * (math.log(arcsOpened)) / (math.log(maxArcs)))
            blueVal = min((int(255 / ((math.log(arcsOpened)) / (math.log(maxArcs)))) - 255), 255)
            return redVal, 0, blueVal

    def drawUnlabeledSolution(self, leadingText="") -> None:
        """Displays the solution using PyVis and a set of hardcoded options"""
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
                                "min": 3,
                                "max": 9
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
                                    "scaleFactor": 0.20
                                }
                            },
                            "scaling": {
                                "min": 3,
                                "max": 15
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

    def drawLabeledSolution(self, leadingText="") -> None:
        """Displays the solution using PyVis and a set of hardcoded options"""
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
                                "min": 3,
                                "max": 9
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
                                    "scaleFactor": 0.20
                                }
                            },
                            "scaling": {
                                "min": 3,
                                "max": 15
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
