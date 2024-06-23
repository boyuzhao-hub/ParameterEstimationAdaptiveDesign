import numpy as np

from tools import stInit, multi2st, stIntegration
from scenario import scenarioBrake,scenarioAcc,scenarioTurnLeft
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

deltaT = 0.01
position0 = [0, 0, 0]
steeringAngle0 = 0
velocity0 = 15
slipAngle = 0
state0_st = stInit(position0, steeringAngle0, velocity0, slipAngle)
mu = 1.0489
g = 9.81
p = parameters_vehicle2()

# Ignore Steering velocity and long velocity due to the same input
# trueState are the data from state k, while calState are calculated by the data from state k-1
# para = [m, I, lr, lf, h, C_sf, C_sr]
def error_tl(para):
    # Scenario 1: Turning left
    u = [0.2, 0]
    statesAll = scenarioTurnLeft(u, state0=state0_st, p=p)
    statesLib = multi2st(statesAll)

    simStateLib_minimize=np.empty([100,7])
    simState_minimize = state0_st
    simStateLib_minimize[0]=simState_minimize
    for i in range(1, statesLib.shape[0]):
        calStates=stIntegration(simState_minimize,deltaT,u,para)
        simStateLib_minimize[i] = calStates
        simState_minimize = calStates
    e=statesLib-simStateLib_minimize
    eSum = np.linalg.norm(e, ord=2)
    return eSum


def error_br(para):
    u = [0.15, -0.5*g]
    statesAll = scenarioBrake(u, state0=state0_st, p=p)
    statesLib = multi2st(statesAll)

    simStateLib_minimize=np.empty([100,7])
    simState_minimize = state0_st
    simStateLib_minimize[0]=simState_minimize
    for i in range(1, statesLib.shape[0]):
        calStates=stIntegration(simState_minimize,deltaT,u,para)
        simStateLib_minimize[i] = calStates
        simState_minimize = calStates
    e=statesLib-simStateLib_minimize
    eSum = np.linalg.norm(e, ord=2)
    return eSum

def error_acc(para):
    u = [0.15, 0.5*g]
    statesAll = scenarioAcc(u, state0=state0_st, p=p)
    statesLib = multi2st(statesAll)

    simStateLib_minimize = np.empty([100, 7])
    simState_minimize = state0_st
    simStateLib_minimize[0] = simState_minimize
    for i in range(1, statesLib.shape[0]):
        calStates = stIntegration(simState_minimize, deltaT, u, para)
        simStateLib_minimize[i] = calStates
        simState_minimize = calStates
    e = statesLib - simStateLib_minimize
    eSum = np.linalg.norm(e, ord=2)
    return eSum

