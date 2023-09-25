#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from math import pi
from qiskit import *
from qiskit import QuantumCircuit,execute,Aer,IBMQ
from qiskit.compiler import transpile,assemble
from matplotlib import style
from qiskit.circuit.library import QFT
from qiskit.tools.monitor import job_monitor
from itertools import repeat
from qiskit import IBMQ, Aer, transpile, assemble,execute
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.compiler import transpile,assemble
import operator
import time 
import random
from numpy.random import seed
from numpy.random import randint 
import numpy.linalg as linalg
import copy


# In[3]:


#qc stand for quatum computer but it's actually just a neat way for psi or quantum state
#it's the key variable in any system' it's the block upon which this system is applied
def Kinetic_Energy(qc, hopping_angle, num_qubits):
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(-np.pi/2.0, i)
        qc.cx(i + 1, i)
        qc.ry(-hopping_angle, i)
        qc.cx(i + 1, i)
        qc.ry(hopping_angle, i)
        qc.rz(np.pi/2.0, i)
        qc.cx(i, i + 1)

    
# psi = QuantumCircuit(3)
# Kinetic_Energy(psi, 0.002, 3)
# psi.draw(output="mpl")


# In[4]:


def Potential_Energy(qc, potential_angle, num_qubits): 
    for i in range(num_qubits):
        if i == int ((num_qubits-1)/2):
            continue
        r = (i-((num_qubits-1)/2))
        qc.p(potential_angle*r**2,i)

    
# # sanity check
# psi=QuantumCircuit(3)
# Potential_Energy(psi, -0.001, 3)
# psi.draw(output="mpl")


# In[5]:


##=================== Complete SHO N times ===================##

# 'time_steps' defines the number of time steps 'dt' to be taken 
# so that the system evolves with total time T = time_steps * dt 

def Uintary_Time_Evolution_SHO(qc, hopping_angle, potential_angle, time_steps, num_qubits):
    for i in range(time_steps):
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits)
        Potential_Energy(qc, potential_angle, num_qubits)
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits)


# # Sanity check. This is a deep circuit
# psi = QuantumCircuit(3)
# Uintary_Time_Evolution_SHO(psi, 0.002, -0.001, 10, 3)
# psi.draw(output="mpl")


# In[8]:


## ================== QFT Function ============== ##

## defining the rotations done by QFT
def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for i in range(n):
        circuit.cp(pi/2**(n-i), i, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

## doing swaps
def swap_registers(circuit, n): #swapping the qubits so that we get the correct QFT application
    for i in range(n//2):
        circuit.swap(i, n-i-1)
    return circuit

#######################
# The Whole Circuit
#######################
def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

# # Let's see how it looks:
# qc = QuantumCircuit(3)
# qft(qc,3)
# qc.draw()


# In[9]:


## ================== Inverse QFT Function ================= ##

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates
# psi0.measure(range(n),range(n))
# # #let's see
# qc = QuantumCircuit(3)
# inverse_qft(qc, 3)
# qc.decompose().draw()


# In[14]:


## ================== Phase Estimation Function ============== ##
# place-holding the variables
def U_SHO_QPE(psi, #wavefunction
          hopping_angle, #h*dt.  hopping
          potential_angle, #-l*dt.  PE 
          time_steps, #number of steps
          num_qubits, #number of quibts in the wavefunction
          m_cr #number of counting registers
          ):
    
    #create a circuit of n+m qubits and m classical registers (for measurment)
    qc = QuantumCircuit(num_qubits+m_cr, m_cr)   
    
    #embidding psi into the circuit from the n qubit to the n+m qubit
    qc.append(psi, list(range(num_qubits, num_qubits+m_cr))) 
    
    #custom gate? Tepically this's to convert the circuit into a gate using .to_gate()
    vector = QuantumCircuit(m_cr) #a circuit of m qubits
    Uintary_Time_Evolution_SHO(vector, hopping_angle, potential_angle, time_steps, num_qubits=m_cr) 
    custom = vector.to_gate(label='U_SHO_Gate').control(1) #converting the U_SHO circuit into a gate
    #but this gate is in control mode with the 1st qubit; the gate only apllies if the control bit is 1. 
    
    repeat = 1
    for i in range(num_qubits):
        qc.h(i)
        for j in range(repeat):
            #appending the custom gate from the m qubit to the n+m qubit
            qc.append(custom, [i]+list(range(num_qubits, num_qubits+m_cr))) 
        repeat = 2 * repeat                          
      #custom gate acts on psi which's been emidded onto from the m qubit to the n+m qubit
    
    # qft(qc, n)   we don't need qft itself we only need qft inverse.
    inverse_qft(qc, num_qubits)
    
    return qc


# In[15]:


#let's see
lat = 1/197.33    # lattice spacing
m_lat = 940*lat   # particle mass
h = 0.5/m_lat     # hopping parameter
b_lat = 5.5*lat   #binding energy
dt = 0.01/b_lat   # time step
l = (2*h**2-b_lat**2)/b_lat  # potential coupling
hopping_angle = h*dt
potential_angle = -l*dt
time_steps = 10
num_qubits= 3
m_cr = 3

psi = QuantumCircuit(num_qubits)
result = U_SHO_QPE(psi, hopping_angle, potential_angle, time_steps, num_qubits, m_cr)
    
result.draw(output='mpl')


# In[ ]:




