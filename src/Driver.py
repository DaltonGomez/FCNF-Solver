from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# Test of the FCFN
FCFNinstance = FixedChargeFlowNetwork()
FCFNinstance.loadFCFNfromDisc("small")

# Test of alpha individual
# alpha = AlphaIndividual(0, FCFNinstance)
# alpha.executeAlphaSolver(75)
# alpha.visualizeAlphaNetwork()

# Test of MILP
FCFNinstance.executeSolver(75)
FCFNinstance.visualizeNetwork()
