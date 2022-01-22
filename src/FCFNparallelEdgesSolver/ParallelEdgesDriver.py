from src.FCFNparallelEdgesSolver.FCFNparallelEdges import FCFNparallelEdges
from src.FCFNparallelEdgesSolver.MILPpeSolver import MILPpeSolver
from src.FCFNparallelEdgesSolver.VisualizePE import VisualizePE

"""PY FILE USED TO RUN THE PARALLEL-EDGES-FCNF-SOLVER"""

# Test of the FCFN/Node/Edge Classes
FCFNpe = FCFNparallelEdges()
FCFNpe.loadFCFN("smallParallelEdges")
FCFNpe.printAllNodeData()
FCFNpe.printAllEdgeData()

# Test of the MILPsolver Class
solver = MILPpeSolver(FCFNpe, 100)
solver.buildModel()
solver.printMILPmodel()
solver.solveModel()
solver.writeSolution()

# Test of the Visualize Class
visual = VisualizePE(FCFNpe)
visual.drawGraphHardcodeOptions(FCFNpe.name)
