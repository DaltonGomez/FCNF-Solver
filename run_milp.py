from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Graph.CandidateGraph import CandidateGraph
from src.Graph.GraphVisualizer import GraphVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_milp.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_milp.py
"""

if __name__ == "__main__":
    inputGraph = "medium_6.p"
    graphInstance = CandidateGraph()
    graphInstance = graphInstance.loadCandidateGraph(inputGraph)
    minTargetFlow = graphInstance.totalPossibleDemand
    graphVis = GraphVisualizer(graphInstance)
    graphVis.drawUnlabeledGraph()
    milpSolver = MILPsolverCPLEX(graphInstance, minTargetFlow, isOneArcPerEdge=True, logOutput=True)
    milpSolver.findSolution(printDetails=False)
    milpSoln = milpSolver.writeSolution()
    milpVis = SolutionVisualizer(milpSoln)
    milpVis.drawLabeledSolution()
