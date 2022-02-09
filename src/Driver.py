from src.FixedChargeNetwork.FixedChargeFlowNetwork import FixedChargeFlowNetwork

"""DRIVER PY FILE"""

# Test of the FCFN
FCFNinstance = FixedChargeFlowNetwork()
FCFNinstance.loadFCFNfromDisc("small")
FCFNinstance.executeSolver(50)
FCFNinstance.visualizeNetwork()
