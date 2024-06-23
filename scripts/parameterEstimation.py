import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2

from error import error_br, error_tl, error_acc
from tools import stInit, stdInit,multi2st, stIntegration
from scenario import scenarioAcc,scenarioBrake,scenarioTurnLeft
import adaptiveDesign

# Set the constraints of the vehicle parameters when used to estimate the parameters
# x = [m, I, lr, lf, h, C_sf, C_sr]
cons = [{'type': 'ineq','fun':lambda x:x[0]},
        {'type': 'ineq','fun':lambda x:x[1]},
        {'type': 'ineq','fun':lambda x:x[2]},
        {'type': 'ineq','fun':lambda x:x[3]},
        {'type': 'ineq','fun':lambda x:x[4]},
        {'type': 'ineq','fun':lambda x:x[5]},
        {'type': 'ineq','fun':lambda x:x[6]},
        {'type': 'ineq','fun':lambda x:x[2]+x[3]-3}]
# Set the constraints for adaptive design
# cons_adaptive = [{'type': 'ineq','fun':lambda x:x[0]},
#                  {'type': 'eq','fun':lambda x:x[1]-x[2]}]

# Set the initial values of the estimated parameters.
C_Sf = 20
C_Sr = 20
lf = 1
lr = 1.5
h = 0.5
m = 1000
I = 1500

def turnLeftEstimated(p,state0,u):
    statesAll=scenarioTurnLeft(u,state0=state0,p=p)
    statesLib=multi2st(statesAll)
    opt_nel = optimize.minimize(fun=error_tl, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]),method='COBYLA',constraints=cons)
    opt_slsqp = optimize.minimize(fun=error_tl, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]), method='SLSQP',constraints=cons)

    simStateLib_nel = np.empty([100, 7])
    simStateLib_slsqp=np.empty([100,7])
    simState_nel = state0
    simState_slsqp = state0
    simStateLib_nel[0] = simState_nel
    simStateLib_slsqp[0]=simState_slsqp
    # Run with Euler Integration in single track model
    for i in range(1, 100):
        simStateNew_nel = stIntegration(simState_nel, 0.01, u, opt_nel.x)
        simStateNew_slsqp = stIntegration(simState_slsqp, 0.01, u, opt_slsqp.x)
        simStateLib_nel[i] = simStateNew_nel
        simStateLib_slsqp[i] = simStateNew_slsqp
        simState_nel = simStateNew_nel
        simState_slsqp = simStateNew_slsqp

    # Position comparison (x, y)
    plt.plot(simStateLib_nel[:, 0], simStateLib_nel[:, 1])
    plt.plot(simStateLib_slsqp[:, 0], simStateLib_slsqp[:, 1])
    plt.plot(statesLib[:, 0], statesLib[:, 1])
    ax=plt.gca()
    ax.set_aspect(2) #Adjust the coordinate axes proportionally
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario turn left: Position Comparison')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.show()
    # Slip Angle Comparison (beta)
    t = np.arange(0, 1, 0.01)
    plt.plot(t,simStateLib_nel[:,6])
    plt.plot(t, simStateLib_slsqp[:, 6])
    plt.plot(t, statesLib[:, 6])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario turn left: Slip Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Slip Angle (rad)')
    plt.show()
    # Yaw Comparison
    plt.plot(t,simStateLib_nel[:,4])
    plt.plot(t, simStateLib_slsqp[:, 4])
    plt.plot(t, statesLib[:, 4])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario turn left: Yaw Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (rad)')
    plt.show()
    # Velocity Comparison
    plt.plot(t,simStateLib_nel[:,3])
    plt.plot(t, simStateLib_slsqp[:, 3])
    plt.plot(t, statesLib[:, 3])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario turn left: Velocity Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.show()
    # Steering Angle Comparison
    plt.plot(t,simStateLib_nel[:,2])
    plt.plot(t, simStateLib_slsqp[:, 2])
    plt.plot(t, statesLib[:, 2])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario turn left: Steering Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.show()
    print('Parameter estimated in turning left scenario: ',opt_nel.x)
    print('Parameter estimated in turning left scenario: ',opt_slsqp.x)
    return simStateLib_slsqp

def brakeEstimated(p,state0,u):
    statesAll=scenarioBrake(u,state0=state0,p=p)
    statesLib=multi2st(statesAll)
    opt_nel = optimize.minimize(fun=error_br, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]),method='COBYLA',constraints=cons)
    opt_slsqp = optimize.minimize(fun=error_br, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]), method='SLSQP',constraints=cons)

    simStateLib_nel = np.empty([100, 7])
    simStateLib_slsqp=np.empty([100,7])
    simState_nel = state0
    simState_slsqp = state0
    simStateLib_nel[0] = simState_nel
    simStateLib_slsqp[0]=simState_slsqp
    # Run with Euler Integration in single track model
    for i in range(1, 100):
        simStateNew_nel = stIntegration(simState_nel, 0.01, u, opt_nel.x)
        simStateNew_slsqp = stIntegration(simState_slsqp, 0.01, u, opt_slsqp.x)
        simStateLib_nel[i] = simStateNew_nel
        simStateLib_slsqp[i] = simStateNew_slsqp
        simState_nel = simStateNew_nel
        simState_slsqp = simStateNew_slsqp

    # Position comparison (x, y)
    plt.plot(simStateLib_nel[:, 0], simStateLib_nel[:, 1])
    plt.plot(simStateLib_slsqp[:, 0], simStateLib_slsqp[:, 1])
    plt.plot(statesLib[:, 0], statesLib[:, 1])
    ax=plt.gca()
    ax.set_aspect(2)
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario brake into the corner: Position Comparison')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.show()
    # Slip Angle Comparison (beta)
    t = np.arange(0, 1, 0.01)
    plt.plot(t,simStateLib_nel[:,6])
    plt.plot(t, simStateLib_slsqp[:, 6])
    plt.plot(t, statesLib[:, 6])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario brake into the corner: Slip Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Slip Angle (rad)')
    plt.show()
    # Yaw Comparison
    plt.plot(t,simStateLib_nel[:,4])
    plt.plot(t, simStateLib_slsqp[:, 4])
    plt.plot(t, statesLib[:, 4])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario brake into the corner: Yaw Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (rad)')
    plt.tight_layout()
    plt.show()
    # Velocity Comparison
    plt.plot(t,simStateLib_nel[:,3])
    plt.plot(t, simStateLib_slsqp[:, 3])
    plt.plot(t, statesLib[:, 3])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario brake into the corner: Velocity Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.show()
    # Steering Angle Comparison
    plt.plot(t,simStateLib_nel[:,2])
    plt.plot(t, simStateLib_slsqp[:, 2])
    plt.plot(t, statesLib[:, 2])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario brake into the corner: Steering Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.show()
    print('Parameter estimated in braking scenario: ',opt_nel.x)
    print('Parameter estimated in braking scenario: ',opt_slsqp.x)
    return opt_slsqp.x

def accEstimated(p,state0,u):
    statesAll=scenarioAcc(u,state0=state0,p=p)
    statesLib=multi2st(statesAll)
    opt_nel = optimize.minimize(fun=error_acc, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]),method='COBYLA',constraints=cons)
    opt_slsqp = optimize.minimize(fun=error_acc, x0=np.array([m, I, lr, lf, h, C_Sf, C_Sr]), method='SLSQP',constraints=cons)
    simStateLib_nel = np.empty([100, 7])
    simStateLib_slsqp=np.empty([100,7])
    simState_nel = state0
    simState_slsqp = state0
    simStateLib_nel[0] = simState_nel
    simStateLib_slsqp[0]=simState_slsqp

    # Run with Euler Integration in single track model
    for i in range(1, 100):
        simStateNew_nel = stIntegration(simState_nel, 0.01, u, opt_nel.x)
        simStateNew_slsqp = stIntegration(simState_slsqp, 0.01, u, opt_slsqp.x)
        simStateLib_nel[i] = simStateNew_nel
        simStateLib_slsqp[i] = simStateNew_slsqp
        simState_nel = simStateNew_nel
        simState_slsqp = simStateNew_slsqp

    # Position comparison (x, y)
    plt.plot(simStateLib_nel[:, 0], simStateLib_nel[:, 1])
    plt.plot(simStateLib_slsqp[:, 0], simStateLib_slsqp[:, 1])
    plt.plot(statesLib[:, 0], statesLib[:, 1])
    ax=plt.gca()
    ax.set_aspect(2)
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario accelerate out of the corner: Position Comparison')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.show()
    # Slip Angle Comparison (beta)
    t = np.arange(0, 1, 0.01)
    plt.plot(t,simStateLib_nel[:,6])
    plt.plot(t, simStateLib_slsqp[:, 6])
    plt.plot(t, statesLib[:, 6])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario accelerate out of the corner: Slip Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Slip Angle (rad)')
    plt.show()
    # Yaw Comparison
    plt.plot(t,simStateLib_nel[:,4])
    plt.plot(t, simStateLib_slsqp[:, 4])
    plt.plot(t, statesLib[:, 4])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario accelerate out of the corner: Yaw Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (rad)')
    plt.show()
    # Velocity Comparison
    plt.plot(t,simStateLib_nel[:,3])
    plt.plot(t, simStateLib_slsqp[:, 3])
    plt.plot(t, statesLib[:, 3])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario accelerate out of the corner: Velocity Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.show()
    # Steering Angle Comparison
    plt.plot(t,simStateLib_nel[:,2])
    plt.plot(t, simStateLib_slsqp[:, 2])
    plt.plot(t, statesLib[:, 2])
    plt.legend(['ST_COBYLA','ST_SLSQP', 'Multibody'])
    plt.title('Scenario accelerate out of the corner: Steering Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Angle (rad)')
    plt.show()
    print('Parameter estimated in accelerating scenario: ', opt_nel.x)
    print('Parameter estimated in accelerating scenario: ', opt_slsqp.x)
    return opt_slsqp.x

def adaptiveEstimated_tl(p,state0,u,origin):
    statesAll=scenarioTurnLeft(u,state0=state0,p=p)
    statesLib=multi2st(statesAll)
    opt_adaptive_m = optimize.minimize(fun=adaptiveDesign.adaptive_m_tl, x0=[1000.03652,24.9541496,17.0535105], method='COBYLA')
    # opt_adaptive_m = optimize.minimize(fun=adaptiveDesign.adaptive_m_tl, x0=[1000.03652,24.9541496,17.0535105], method='COBYLA',constraints=cons_adaptive) #Do adaptive design with constraints
    simStateLib_slsqp=np.empty([100,7])
    simState_slsqp = state0
    simStateLib_slsqp[0]=simState_slsqp
    # Run with Euler Integration in single track model
    for i in range(1, 100):
        simStateNew_slsqp = stIntegration(simState_slsqp, 0.01, u, [opt_adaptive_m.x[0],1499.97571,1.26647953,1.73352047,1.78063348,opt_adaptive_m.x[1],opt_adaptive_m.x[2]])
        simStateLib_slsqp[i] = simStateNew_slsqp
        simState_slsqp = simStateNew_slsqp

    print("the estimated parameters are ",opt_adaptive_m.x)

    # Position comparison (x, y)
    plt.plot(simStateLib_slsqp[:, 0], simStateLib_slsqp[:, 1])
    plt.plot(statesLib[:, 0], statesLib[:, 1])
    plt.plot(origin[:, 0], origin[:, 1])
    ax=plt.gca()
    ax.set_aspect(2)
    plt.legend(['ST_adaptive', 'Multibody','ST_SLSQP'])
    plt.title('Scenario turn left Adaptive Design: Position Comparison')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.show()
    # Slip Angle Comparison (beta)
    t = np.arange(0, 1, 0.01)
    plt.plot(t, simStateLib_slsqp[:, 6])
    plt.plot(t, statesLib[:, 6])
    plt.plot(t, origin[:, 6])
    plt.legend(['ST_adaptive', 'Multibody','ST_SLSQP'])
    plt.title('Scenario turn left Adaptive Design: Slip Angle Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Slip Angle (rad)')
    plt.show()
    # Yaw Comparison
    plt.plot(t, simStateLib_slsqp[:, 4])
    plt.plot(t, statesLib[:, 4])
    plt.plot(t, origin[:, 4])
    plt.legend(['ST_adaptive', 'Multibody','ST_SLSQP'])
    plt.title('Scenario turn left Adaptive Design: Yaw Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw Angle (rad)')
    plt.show()


if __name__=='__main__':

    # Initialize vehicle state
    p = parameters_vehicle2()
    deltaT=0.01
    position0=[0,0,0]
    steeringAngle0=0
    velocity0=15
    slipAngle=0
    g=9.81
    state0=stInit(position0,steeringAngle0,velocity0,slipAngle)

    # Scenario 1: Turning left
    steeringAngleVel_tl=0.2
    acc_tl=0
    u_tl=[steeringAngleVel_tl,acc_tl]
    estimatedPara_tl=turnLeftEstimated(p,state0,u_tl)

    # Scenario 2: Braking into the turner
    # steeringAngleVel_br = 0.15
    # # steeringAngleVel_br = 0  ##Purely linear motion
    # acc_br = -0.5 * g
    # u_br = [steeringAngleVel_br,acc_br]
    # estimatedPara_br=brakeEstimated(p,state0,u_br)

    # Scenario 3: Accelerating
    # steeringAngleVel_ac = 0.15
    # acc_ac = 0.5 * g
    # u_ac = [steeringAngleVel_ac,acc_ac]
    # estimatedPara_acc = accEstimated(p, state0,u_ac)

    # Adaptive Design
    # Scenario turning left
    # adaptiveEstimated_tl(p,state0,u_tl,estimatedPara_tl)
