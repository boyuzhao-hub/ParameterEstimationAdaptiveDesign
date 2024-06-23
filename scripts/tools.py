import numpy as np
import math
from scipy.integrate import odeint
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

def func_MB(x, t, u, p):
    f = vehicle_dynamics_mb(x, u, p)
    return f

# Initialize the single track model, generate state 0 of the vehicle.
# Input: vehicle position, steering angle, velocity, slip angle
# Output: vehicle initial state
def stInit(position,steeringAngle,velocity,slipAngle):
    x0=position[0]
    y0=position[1]
    delta0=steeringAngle
    v0=velocity
    psi0=position[2]
    psiDot0=0
    beta=slipAngle
    return [x0,y0,delta0,v0,psi0,psiDot0,beta]

def stdInit(position,steeringAngle,velocity,slipAngle):
    x0=position[0]
    y0=position[1]
    delta0=steeringAngle
    v0=velocity
    psi0=position[2]
    psiDot0=0
    beta=slipAngle

    return [x0,y0,delta0,v0,psi0,psiDot0,beta]


# Euler Integration to single track model. Generate the next state of the vehicle
# Input: state, deltaT, system input, vehicle parameters.
# Output: The next state of the vehicle under the parameters and inputs.
def stIntegration(state,deltaT,u,p):
    mu=1.0489
    g=9.81
    C_Sf = p[5]
    C_Sr = p[6]
    lf = p[3]
    lr = p[2]
    h = p[4]
    m = p[0]
    I = p[1]

    psiDD=-mu * m / (state[3] * I * (lr + lf)) * (
                         lf ** 2 * C_Sf * (g * lr - u[1] * h) + lr ** 2 * C_Sr * (g * lf + u[1] * h)) * state[5] \
             + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h)) * state[6] \
             + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - state[1] * h) * state[2]
    betaD=(mu / (state[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1) * \
          state[5] \
             - mu / (state[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * state[6] \
             + mu / (state[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * state[2]

    x_k1=state[0]+deltaT*state[3]*math.cos(state[6]+state[4])
    y_k1=state[1]+deltaT*state[3]*math.sin(state[6]+state[4])
    delta_k1=state[2]+deltaT*u[0]
    v_k1=state[3]+deltaT*u[1]
    psi_k1=state[4]+deltaT*state[5]
    psiDot_k1=state[5]+deltaT*psiDD
    beta_k1=state[6]+deltaT*betaD
    return [x_k1,y_k1,delta_k1,v_k1,psi_k1,psiDot_k1,beta_k1]

# Convert the variables of the state in multi-body into single track states correspondingly.
# Input: all the states of multi-body model
# Output: the states with variables of multi-body which can correspond to single track.
# beta=arctan(v_y/v_x)
# v=sqrt(v_x^2+v_y^2)
def multi2st(state):
    beta=np.empty([state.shape[0]])
    vel=np.empty([state.shape[0]])
    for i in range(state.shape[0]):
        beta[i] = math.atan(state[i, 10] / state[i, 3])
        vel[i]=math.sqrt(math.pow(state[i,3],2)+math.pow(state[i,10],2))
    newStates=np.empty([state.shape[0],7])
    newStates[:,0:3]=state[:,0:3]
    newStates[:,3]=vel
    newStates[:,4:6]=state[:,4:6]
    newStates[:, 6] = beta
    # newStates[:,6]=np.nan_to_num(beta)
    return newStates

# Convert the variables of the state in multi-body into kinematic single track states correspondingly.
# Input: all the states of multi-body model
# Output: the states with variables of multi-body which can correspond to kinematic single track.
def multi2kst(state):
    vel=np.empty([state.shape[0]])
    for i in range(state.shape[0]):
        vel[i]=math.sqrt(math.pow(state[i,4],2)+math.pow(state[i,11],2))
    newStates=np.empty([state.shape[0],7])
    newStates[:,0:3]=state[:,0:3]
    newStates[:,3]=vel
    newStates[:,4]=state[:,4]
    return newStates

# Generate all the states of the vehicle in multi-body model, which are as the measurements.
# Input: state, system input steering angle velocity, longitudinal acceleration.
# Output: All the states in multi-body model.
def mbTrueData(state,vDelta,aLong,p):
    tStart = 0  # start time
    tFinal = 1  # start time
    t = np.arange(tStart, tFinal, 0.01)
    u = [vDelta, aLong]
    trueLib = odeint(func_MB, state, t, args=(u, p))
    return trueLib
