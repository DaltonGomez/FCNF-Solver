from typing import Dict, Tuple

from Graph.Arc import Arc
from Graph.CandidateGraph import CandidateGraph
from Graph.Node import Node
from TransportationReduction.TransportationArc import TransportationArc
from TransportationReduction.TransportationDestination import TransportationDestination
from TransportationReduction.TransportationOrigin import TransportationOrigin


class TransportationProblem:
    """Class that defines a Transportation Problem object, instantiated by reducing a Candidate Graph"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, inputGraph: CandidateGraph, minTargetFlow: float):
        """Constructor of a Transportation Problem instance"""
        # Input attributes
        self.inputGraph: CandidateGraph = inputGraph
        self.minTargetFlow: float = minTargetFlow
        # Fixed-charge transportation problem reduction attributes
        self.origins: Dict[Tuple[int, int, float], TransportationOrigin] = {}
        self.destinations: Dict[int, TransportationDestination] = {}
        self.transportArcs: Dict[Tuple[Tuple[int, int, float], int], TransportationArc] = {}
        self.performReduction()
        self.numOrigins = len(self.origins)
        self.numDestination = len(self.destinations)
        self.numTransportArcs = len(self.transportArcs)

    # ===============================================
    # ============== REDUCTION METHODS ==============
    # ===============================================
    def performReduction(self) -> None:
        """Performs the reduction to the Fixed-Charge Transportation Problem on the input candidate graph instance"""
        # NOTE: Reduction only works on src/sink capacitated input graphs
        # =====================================================
        # === DESTINATION NODES (i.e. Nodes in the input graph)
        # =====================================================
        # Build destination nodes from the nodes in the input graph
        for nodeID in self.inputGraph.nodesDict.keys():
            nodeObj = self.inputGraph.nodesDict[nodeID]
            demand = 0.0
            for numOutgoingEdges in range(len(nodeObj.outgoingEdges)):
                for arcCap in self.inputGraph.possibleArcCapsArray:
                    demand += arcCap
            thisDestination = TransportationDestination(nodeObj, -demand)
            self.destinations[nodeID] = thisDestination
        # Build super-source node and reduce to a destination node
        supersourceID = -1
        supersourceNode = Node(supersourceID, 1.0, 1.0)
        for sourceID in self.inputGraph.sourcesArray:
            supersourceNode.addOutgoingEdge((-1, sourceID))
        totalSourceCapacity = 0.0
        for srcCap in self.inputGraph.sourceCapsArray:
            totalSourceCapacity += srcCap
        supersourceDemand = self.minTargetFlow - totalSourceCapacity
        supersourceDestination = TransportationDestination(supersourceNode, supersourceDemand)
        self.destinations[supersourceID] = supersourceDestination
        # Build super-sink node and reduce to a destination node
        supersinkID = -2
        supersinkNode = Node(supersinkID, 10.0, 10.0)
        for sinkID in self.inputGraph.sinksArray:
            supersinkNode.addIncomingEdge((sinkID, -2))
        supersinkDemand = -self.minTargetFlow
        supersinkDestination = TransportationDestination(supersinkNode, supersinkDemand)
        self.destinations[supersinkID] = supersinkDestination
        # ===============================================
        # === ORIGIN NODES (i.e. Arcs in the input graph)
        # ===============================================
        # Build origin nodes from the arcs in the input graph
        for arcID in self.inputGraph.arcsDict.keys():
            arcObj = self.inputGraph.arcsDict[arcID]
            thisOrigin = TransportationOrigin(arcObj)
            self.origins[arcID] = thisOrigin
        # Build origin nodes from the supersource --> source "arcs" in the input graphs
        for sourceIndex in range(len(self.inputGraph.sourcesArray)):
            sourceID = self.inputGraph.sourcesArray[sourceIndex]
            sourceCap = self.inputGraph.sourceCapsArray[sourceIndex]
            if self.inputGraph.isSourceSinkCharged:
                sourceVariableCost = self.inputGraph.sourceVariableCostsArray[sourceIndex]
            else:
                sourceVariableCost = 0.0
            superToSourceArc = Arc(-10, (-1, sourceID), sourceCap, 0.0, 0.0, sourceVariableCost)
            superToSourceOriginID = (-1, sourceID, sourceCap)
            superToSourceOrigin = TransportationOrigin(superToSourceArc)
            self.origins[superToSourceOriginID] = superToSourceOrigin
        # Build origin nodes from the sink --> supersink "arcs" in the input graphs
        for sinkIndex in range(len(self.inputGraph.sinksArray)):
            sinkID = self.inputGraph.sinksArray[sinkIndex]
            sinkCap = self.inputGraph.sinkCapsArray[sinkIndex]
            if self.inputGraph.isSourceSinkCharged:
                sinkVariableCost = self.inputGraph.sinkVariableCostsArray[sinkIndex]
            else:
                sinkVariableCost = 0.0
            sinkToSuperArc = Arc(-20, (sinkID, -2), sinkCap, 0.0, 0.0, sinkVariableCost)
            sinkToSuperOriginID = (sinkID, -2, sinkCap)
            sinkToSuperOrigin = TransportationOrigin(sinkToSuperArc)
            self.origins[sinkToSuperOriginID] = sinkToSuperOrigin
        # =======================
        # === TRANSPORTATION ARCS
        # =======================
        # Build transportation arcs
        for originID in self.origins.keys():
            originObj = self.origins[originID]
            # From node arc (with zero cost)
            fromNodeTransportArcID = (originID, originID[0])
            fromNodeTransportArcObj = TransportationArc(fromNodeTransportArcID, 0.0, 0.0)
            self.transportArcs[fromNodeTransportArcID] = fromNodeTransportArcObj
            self.destinations[originID[0]].addIncomingTransportArc(fromNodeTransportArcID)
            # To node arc (with cost equal to input arc)
            toNodeTransportArcID = (originID, originID[1])
            toNodeTransportArcObj = TransportationArc(toNodeTransportArcID, originObj.originalArc.fixedCost, originObj.originalArc.variableCost)
            self.transportArcs[toNodeTransportArcID] = toNodeTransportArcObj
            self.destinations[originID[1]].addIncomingTransportArc(toNodeTransportArcID)
