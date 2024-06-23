from vehiclemodels.init_mb import init_mb
from tools import mbTrueData
# Define a scenario, which is turning left.
# And plot the figure for the track of the vehicle in this scenario.
# Input: system input steering angle velocity, longitudinal acceleration, the initial state and vehicle parameter.
# Output: The measurement states of the vehicle in this scenario.
def scenarioTurnLeft(u,state0,p):
    print("Scenario: Turn Left")
    # Run multibody model to collect data
    vDelta=u[0]
    aLong=u[1]
    state0_mb = init_mb(state0, p)
    trueStates = mbTrueData(state0_mb, vDelta, aLong,p)
    return trueStates

def scenarioBrake(u,state0,p):
    print("Scenario: Brake")
    # Run multibody model to collect data
    vDelta=u[0]
    aLong=u[1]
    state0_mb = init_mb(state0, p)
    trueStates = mbTrueData(state0_mb, vDelta, aLong,p)
    return trueStates

def scenarioAcc(u,state0,p):
    print("Scenario: Accelerate")
    # Run multibody model to collect data
    vDelta=u[0]
    aLong=u[1]
    state0_mb = init_mb(state0, p)
    trueStates = mbTrueData(state0_mb, vDelta, aLong,p)
    return trueStates
