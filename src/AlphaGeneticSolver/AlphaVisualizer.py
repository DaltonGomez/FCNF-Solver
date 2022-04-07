import networkx as nx
from pyvis.network import Network as netVis


class AlphaVisualizer:
    """Class that allows visualizations of an alpha-relaxed FCFN"""

    def __init__(self, alphaIndividual, graphType="fullGraph"):
        """Constructor of a AlphaVisualizer instance with NetworkX and PyVis dependencies
        NOTE: individual must be of type AlphaIndividual (Not type hinted to prevent circular import)"""
        self.individual = alphaIndividual
        self.nx = nx.DiGraph()
        if graphType == "fullGraph":
            self.populateFullGraph()
        elif graphType == "solutionOnly":
            self.populateSolutionGraphOnly()
        elif graphType == "withLabels":
            self.populateGraphWithEdgesLabelsOnly()

    def populateFullGraph(self) -> None:
        """Populates a NetworkX instance with the full graph data of the AlphaIndividual"""
        # Create sets of opened and unopened nodes and edges
        allNodesSet = set(self.individual.FCFN.nodesDict)
        openedNodesSet = set(self.individual.openedNodesDict)
        unopenedNodesSet = allNodesSet.difference(openedNodesSet)
        allEdgesSet = set(self.individual.FCFN.edgesDict)
        openedEdgesSet = set(self.individual.openedEdgesDict)
        unopenedEdgesSet = allEdgesSet.difference(openedEdgesSet)
        # Add nodes to NX instance
        for node in self.individual.FCFN.nodesDict.keys():
            if node[0] == "s":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="blue", label=str(round(nodeValues[1])))
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="blue")
            elif node[0] == "t":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="red", label=str(round(nodeValues[1])))
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="red")
            elif node[0] == "n":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="black", label=str(round(nodeValues[1])))
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="grey")
        # Add edges to NX instance
        for edge in self.individual.FCFN.edgesDict.keys():
            edgeObj = self.individual.FCFN.edgesDict[edge]
            if edge in openedEdgesSet:
                edgeValues = self.individual.openedEdgesDict[edge]
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeValues[0], color="black",
                                 label=str(round(edgeValues[1])))
            elif edge in unopenedEdgesSet:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=0.0, color="grey")

    def populateSolutionGraphOnly(self) -> None:
        """Populates a NetworkX instance with only the solution data of the AlphaIndividual"""
        for node in self.individual.openedNodesDict:
            nodeValues = self.individual.openedNodesDict[node]
            if node[0] == "s":
                self.nx.add_node(node, value=nodeValues[0], color="blue", label=str(round(nodeValues[1])))
            elif node[0] == "t":
                self.nx.add_node(node, value=nodeValues[0], color="red", label=str(round(nodeValues[1])))
            elif node[0] == "n":
                self.nx.add_node(node, value=nodeValues[0], color="black", label=str(round(nodeValues[1])))
        for edge in self.individual.openedEdgesDict:
            edgeValues = self.individual.openedEdgesDict[edge]
            edgeObj = self.individual.FCFN.edgesDict[edge]
            self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeValues[0], color="black",
                             label=str(round(edgeValues[1])))

    def drawGraph(self, name: str) -> None:
        """Displays the FCNF using PyVis and a set of hardcoded options"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + "_Target=" + str(
            self.individual.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        # NOTE: The order in which nodes and edges are added to the NX instance affects the consistency of the layout
        visual.set_options("""
                            var options = {
                                "autoResize": true,
                                "width": "1000px",
                                "height": "800px",
                                "layout": { 
                                    "randomSeed":""" + str(self.individual.visSeed) + "," +
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

    def drawGraphUiOptions(self, name: str) -> None:
        """Displays the FCNF using PyVis and provides a UI for customizing options, which can be copied as JSON"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + "_Target=" + str(
            self.individual.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        # NOTE: The order in which nodes and edges are added to the NX instance affects the consistency of the layout
        visual.set_options("""
                                    var options = {
                                        "autoResize": true,
                                        "width": "1000px",
                                        "height": "800px",
                                        "layout": { 
                                            "randomSeed":""" + str(self.individual.visSeed) + "," +
                           """
                           "improvedLayout": true
                           },
                           "configure": {
                               "enabled": true
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

    def populateGraphWithEdgesLabelsOnly(self) -> None:
        """Populates a NetworkX instance with the full graph data of the AlphaIndividual"""
        # Create sets of opened and unopened nodes and edges
        allNodesSet = set(self.individual.FCFN.nodesDict)
        openedNodesSet = set(self.individual.openedNodesDict)
        unopenedNodesSet = allNodesSet.difference(openedNodesSet)
        allEdgesSet = set(self.individual.FCFN.edgesDict)
        openedEdgesSet = set(self.individual.openedEdgesDict)
        unopenedEdgesSet = allEdgesSet.difference(openedEdgesSet)
        # Add nodes to NX instance
        for node in self.individual.FCFN.nodesDict.keys():
            if node[0] == "s":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="blue")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="blue")
            elif node[0] == "t":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="red")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="red")
            elif node[0] == "n":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="black")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="grey")
        # Add edges to NX instance
        for edge in self.individual.FCFN.edgesDict.keys():
            edgeObj = self.individual.FCFN.edgesDict[edge]
            if edge in openedEdgesSet:
                edgeValues = self.individual.openedEdgesDict[edge]
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeValues[0], color="black",
                                 label=edge + ":\n(F= " + str(round(edgeValues[0])) + ",\nC= " + str(
                                     int(edgeValues[1])) + ")")
            elif edge in unopenedEdgesSet:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=0.0, color="grey", label=edge)

    def populateLabeledGraph(self) -> None:
        """Populates a NetworkX instance with the full graph data of the AlphaIndividual"""
        # Create sets of opened and unopened nodes and edges
        allNodesSet = set(self.individual.FCFN.nodesDict)
        openedNodesSet = set(self.individual.openedNodesDict)
        unopenedNodesSet = allNodesSet.difference(openedNodesSet)
        allEdgesSet = set(self.individual.FCFN.edgesDict)
        openedEdgesSet = set(self.individual.openedEdgesDict)
        unopenedEdgesSet = allEdgesSet.difference(openedEdgesSet)
        # Add nodes to NX instance
        for node in self.individual.FCFN.nodesDict.keys():
            if node[0] == "s":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="blue",
                                     label=node + ": (" + str(round(nodeValues[0])) + ", " + str(
                                         int(nodeValues[1])) + ")")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="blue", label=node)
            elif node[0] == "t":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="red",
                                     label=node + ": (" + str(round(nodeValues[0])) + ", " + str(
                                         int(nodeValues[1])) + ")")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="red", label=node)
            elif node[0] == "n":
                if node in openedNodesSet:
                    nodeValues = self.individual.openedNodesDict[node]
                    self.nx.add_node(node, value=nodeValues[0], color="black",
                                     label=node + ": (" + str(round(nodeValues[0])) + ", " + str(
                                         int(nodeValues[1])) + ")")
                elif node in unopenedNodesSet:
                    self.nx.add_node(node, value=0, color="grey", label=node)
        # Add edges to NX instance
        for edge in self.individual.FCFN.edgesDict.keys():
            edgeObj = self.individual.FCFN.edgesDict[edge]
            if edge in openedEdgesSet:
                edgeValues = self.individual.openedEdgesDict[edge]
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=edgeValues[0], color="black",
                                 label=edge + ": (" + str(round(edgeValues[0])) + ", " + str(int(edgeValues[1])) + ")")
            elif edge in unopenedEdgesSet:
                self.nx.add_edge(edgeObj.fromNode, edgeObj.toNode, value=0.0, color="grey", label=edge)

    def drawGraphWithLabels(self, name: str) -> None:
        """Displays the FCNF using PyVis and a set of hardcoded options"""
        displayName = name + "_Cost=" + str(round(self.individual.trueCost)) + "_Target=" + str(
            self.individual.minTargetFlow) + ".html"
        print("Drawing " + displayName + "...")
        visual = netVis("800px", "1000px", directed=True)
        visual.from_nx(self.nx)
        # Sets visualization options using a JSON format (see vis.js documentation)
        # NOTE: The order in which nodes and edges are added to the NX instance affects the consistency of the layout
        visual.set_options("""
                            var options = {
                                "autoResize": true,
                                "width": "1000px",
                                "height": "800px",
                                "layout": { 
                                    "randomSeed":""" + str(self.individual.visSeed) + "," +
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
                                   "color": "rgba(255,0,0,1)",
                                   "strokeWidth": 3,
                                   "strokeColor": "rgba(255,255,0,1)"
                               },
                               "scaling": {
                                   "min": 10,
                                   "max": 60,
                                   "label": {
                                        "enabled": false
                                        }
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
                                   "size": 75,
                                   "color": "rgba(255,0,0,1)",
                                   "strokeWidth": 3,
                                   "strokeColor": "rgba(255,255,0,1)"
                               },
                               "arrowStrikethrough": false,
                               "arrows": {
                                   "to": {
                                       "scaleFactor": 2
                                   }
                               },
                               "scaling": {
                                   "min": 1,
                                   "max": 20,
                                   "label": {
                                        "enabled": false
                                    }
                               },
                               "smooth": {
                                   "enabled": false
                               },
                               "shadow": {
                                   "enabled": false,
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
