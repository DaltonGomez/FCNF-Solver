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
    # TODO - Update to produce graphs/output runtime data
    inputGraph = "large_3.p"
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
    milpSoln.saveSolution()
    print("\n\nMILP formulation solved optimally! Solution written to disc...")
    print("Total CPLEX runtime: " + str(milpSolver.getCplexRuntime()) + " seconds")
