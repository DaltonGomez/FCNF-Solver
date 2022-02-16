import os

from src.FixedChargeNetwork.Edge import Edge
from src.FixedChargeNetwork.ExactSolver import ExactSolver
from src.FixedChargeNetwork.Node import Node
from src.FixedChargeNetwork.Visualizer import Visualizer


class FixedChargeFlowNetwork:
    """Class that defines a Fixed Charge Flow Network"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self):
        """Constructor of a FCNF instance"""
        # Input Network Attributes & Topology
        self.name = ""
        self.numNodes = 0
        self.numSources = 0
        self.numSinks = 0
        self.numIntermediateNodes = 0
        self.nodesDict = {}
        self.numEdges = 0
        self.edgesDict = {}

        # Solution Data
        self.solver = None
        self.isSolved = False
        self.minTargetFlow = 0
        self.totalCost = 0
        self.totalFlow = 0

        # Visualization Data
        self.visualizer = None
        self.visSeed = 1

    def addNode(self, nodeType: str, idNum: int, variableCost: int, capacity: int) -> None:
        """Adds a new node to a FCFN instance- CALL ONLY FROM GraphGeneration Class"""
        nodeName = nodeType + str(idNum)
        thisNode = Node(nodeName, variableCost, capacity)
        self.nodesDict[nodeName] = thisNode

    def addEdge(self, idNum: int, fromNode: str, toNode: str, fixedCost: int, variableCost: int, capacity: int) -> None:
        """Adds a new edge to a FCFN instance- CALL ONLY FROM GraphGeneration Class"""
        edgeName = "e" + str(idNum)
        thisEdge = Edge(edgeName, fromNode, toNode, fixedCost, variableCost, capacity)
        self.edgesDict[edgeName] = thisEdge

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def executeSolver(self, minTargetFlow: int) -> None:
        """Solves the FCFN exactly with a MILP model in CPLEX"""
        if self.solver is None:
            self.solver = ExactSolver(self, minTargetFlow)  # FYI- ExactSolver constructor does not have FCFN type hint
            self.solver.buildModel()
            self.solver.solveModel()
            self.solver.writeSolution()
            self.solver.printSolverOverview()
        elif self.solver.isRun is True and self.isSolved is False:
            print("No feasible solution exists for the network and target!")
        elif self.solver.isRun is True and self.isSolved is True:
            print("Model is already solved- Call print solution to view solution!")

    # ===================================================
    # ============== VISUALIZATION METHODS ==============
    # ===================================================
    def visualizeNetwork(self, catName="") -> None:
        """Draws the Fixed Charge Flow Network instance using the PyVis package and a NetworkX conversion"""
        if self.visualizer is None:
            self.visualizer = Visualizer(self)
            self.visualizer.drawGraph(self.name + catName)

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def loadFCFN(self, network: str) -> None:
        """Loads a FCFN from a text file encoding"""
        # Path management
        currDir = os.getcwd()
        networkFile = network + ".txt"
        catPath = os.path.join(currDir, "networks", networkFile)
        print("Loading " + networkFile + " from: " + catPath)
        # Open file, parse lines in data stream, and close file
        inputFile = open(catPath, "r")
        lines = inputFile.readlines()
        inputFile.close()
        # Assign name
        lines.pop(0)
        self.name = lines[0].split()
        self.name.pop(0)
        self.name = self.name.pop(0)
        lines.pop(0)
        # Assign seed
        self.visSeed = lines[0].split()
        self.visSeed.pop(0)
        self.visSeed = int(self.visSeed.pop(0))
        lines.pop(0)
        # Build network
        for line in lines:
            data = line.split()
            # Ignore comments
            if data[0][0] == "#":
                continue
            # Construct source node objects and add to dictionary and network
            elif data[0][0] == "s":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.numSources += 1
            # Construct sink node objects and add to dictionary and network
            elif data[0][0] == "t":
                thisNode = Node(data[0], int(data[1]), int(data[2]))
                self.nodesDict[data[0]] = thisNode
                self.numSinks += 1
            # Construct transshipment node objects and add to dictionary and network
            elif data[0][0] == "n":
                thisNode = Node(data[0], 0, 0)
                self.nodesDict[data[0]] = thisNode
                self.numIntermediateNodes += 1
            # Construct edge objects and add to dictionary and network
            elif data[0][0] == "e":
                thisEdge = Edge(data[0], data[1], data[2], int(data[3]), int(data[4]), int(data[5]))
                self.edgesDict[data[0]] = thisEdge
                self.nodesDict[data[1]].outgoingEdges.append(data[0])
                self.nodesDict[data[2]].incomingEdges.append(data[0])
        # Assign network size
        self.numNodes = len(self.nodesDict)
        self.numEdges = len(self.edgesDict)

    def saveSolutionToDisc(self) -> None:
        """Saves all the data of a Fixed Charge Flow Network solution"""
        # TODO - Implement
        pass

    # ===========================================
    # ============== PRINT METHODS ==============
    # ===========================================
    def printAllNodeData(self) -> None:
        """Prints all the data for each node in the network"""
        for node in self.nodesDict:
            thisNode = self.nodesDict[node]
            thisNode.printNodeData()

    def printAllEdgeData(self) -> None:
        """Prints all the data for each edge in the network"""
        for edge in self.edgesDict:
            thisEdge = self.edgesDict[edge]
            thisEdge.printEdgeData()

    def printFullModel(self) -> None:
        """Prints the solution data of the MILP solver to the console"""
        if self.solver is not None:
            self.solver.printModel()
        else:
            print("Solver must be initialized and executed before printing the model!")

    def printFullSolution(self) -> None:
        """Prints the solution data of the MILP solver to the console"""
        if self.solver is not None:
            self.solver.printSolution()
        else:
            print("Solver must be initialized and executed before printing the solution!")
