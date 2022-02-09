import os

from src.FixedChargeNetwork.Edge import Edge
from src.FixedChargeNetwork.ExactSolver import ExactSolver
from src.FixedChargeNetwork.Node import Node
from src.FixedChargeNetwork.Visualizer import Visualizer


class FixedChargeFlowNetwork:
    """Class that defines a Fixed Charge Flow Network with parallel edges allowed"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self):
        """Constructor of a FCNF instance"""
        # Input Network Attributes & Topology
        self.name = ""
        self.edgeCaps = []
        self.numEdgeCaps = 0
        self.edgeFixedCosts = []
        self.edgeVariableCosts = []
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

    # ============================================
    # ============== SOLVER METHODS ==============
    # ============================================
    def executeSolver(self, minTargetFlow: int):
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
    def visualizeNetwork(self, catName=""):
        """Draws the Fixed Charge Flow Network instance using the PyVis package and a NetworkX conversion"""
        if self.visualizer is None:
            self.visualizer = Visualizer(self)
            self.visualizer.drawGraph(self.name + catName)

    # =================================================
    # ============== DATA IN/OUT METHODS ==============
    # =================================================
    def loadFCFNfromDisc(self, network: str):
        """Loads a FCFN from a text file encoding"""
        # Path management
        currDir = os.getcwd()
        networkFile = network + ".txt"
        catPath = os.path.join(currDir, "networks", networkFile)
        print("Loading " + networkFile + " from: " + catPath)
        # Open file, parse lines in data stream, and close file
        inputFile = open(catPath, 'r')
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
        # Assign potential edge capacities
        edgeCapStrings = lines[0].split()
        edgeCapStrings.pop(0)
        edgeCapMap = map(int, edgeCapStrings)
        self.edgeCaps = list(edgeCapMap)
        lines.pop(0)
        # Assign potential edge fixed costs
        edgeFCStrings = lines[0].split()
        edgeFCStrings.pop(0)
        edgeFCMap = map(int, edgeFCStrings)
        self.edgeFixedCosts = list(edgeFCMap)
        lines.pop(0)
        # Assign potential edge variable costs
        edgeVCStrings = lines[0].split()
        edgeVCStrings.pop(0)
        edgeVCMap = map(int, edgeVCStrings)
        self.edgeVariableCosts = list(edgeVCMap)
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
                thisEdge = Edge(data[0], data[1], data[2])
                self.edgesDict[data[0]] = thisEdge
                self.nodesDict[data[1]].outgoingEdges.append(data[0])
                self.nodesDict[data[2]].incomingEdges.append(data[0])
        # Assign network size
        self.numNodes = len(self.nodesDict)
        self.numEdges = len(self.edgesDict)
        self.numEdgeCaps = len(self.edgeCaps)

    def generateRandomFCFN(self):
        """Generates a random Fixed Charge Flow Network using NetworkX"""
        # TODO - Implement
        pass

    def saveFCFNtoDisc(self):
        """Saves an unsolved version of a NetworkX-generated FCFN as a .txt file within the project directory"""
        # TODO - Implement
        pass

    def saveSolutionToDisc(self):
        """Saves all the data of a Fixed Charge Flow Network solution"""
        # TODO - Implement
        pass

    # ===========================================
    # ============== PRINT METHODS ==============
    # ===========================================
    def printAllNodeData(self):
        """Prints all the data for each node in the network"""
        for node in self.nodesDict:
            thisNode = self.nodesDict[node]
            thisNode.printNodeData()

    def printAllEdgeData(self):
        """Prints all the data for each edge in the network"""
        for edge in self.edgesDict:
            thisEdge = self.edgesDict[edge]
            thisEdge.printEdgeData()

    def printFullModel(self):
        """Prints the solution data of the MILP solver to the console"""
        if self.solver is not None:
            self.solver.printModel()
        else:
            print("Solver must be initialized and executed before printing the model!")

    def printFullSolution(self):
        """Prints the solution data of the MILP solver to the console"""
        if self.solver is not None:
            self.solver.printSolution()
        else:
            print("Solver must be initialized and executed before printing the solution!")
