# Quantum Computing and Quantum Machine learning

Quantum computing has made significant strides in theoretical and experimental research, while Quantum Machine Learning (QML) is still a growing research field. The potential of QML to speed up training and bring significant changes to machine learning algorithms is widely discussed, but the limitations of the number of qubits and quantum error present in current quantum systems prevent us from running QML circuits on real quantum hardware. While classical machine learning algorithms are progressing well, QML's growth is still uncertain. However, we should not solely aim to make classical algorithms run faster in a quantum setting but rather seek to develop QML algorithms that provide new insights into quantum computing or classical machine learning algorithms. 

It is essential to focus on whether a QML algorithm can perform more efficiently than its classical counterpart instead of aiming solely for better performance. Current quantum computing hardware belongs to the family of Noisy Intermediate Scale Quantum (NISQ), which faces significant challenges of noise and is therefore unable to simulate quantum circuits of large depth or with complicated gate expressions. For any QML algorithm to be of practical use in the near future, it must be simple enough to run on hardware that does not require much error correction but complex enough to be able to learn something meaningful. Finally, the encoding function and the choice of its associated gate operations determine the expressive power of a quantum circuit for classical data. Therefore, it is crucial to search for more candidates of encoding functions that can potentially boost the performance of a quantum machine learning algorithm.

# Familiar Quantum Algorithms

I have familiar with algorithms like Quantum Kernels and Quantum neural networks. I am also well versed with the Quantum Convolutional Neural Networks (QCNN) from my previous year project in GSoC 2022. I am also familiar with Variational Quantum Eigensolver (VQE) which is an algorithm used for finding the ground state energy of a system using its Hamiltonian. The VQE algorithm uses both quantum and classical computations and is based on the variational principle of quantum mechanics.

## VQE

The variational principle states that irrespective of whether we know or don't know the exact ground state, an approximation can be found given a Hamiltonian $H$ and ground state $|\psi_g \rangle$ i.e. $|\tilde{ \psi}_g\rangle \approx |\psi_g \rangle$. We begin with a parameterized state known as an ansatz $|\psi(\boldsymbol{\theta})\rangle$, where $\boldsymbol{\theta} \equiv (\theta_1, \theta_2,\dots)$ are the parameters, in order to determine the approximate ground state $|\tilde{ \psi}_g\rangle$. The ansatz should roughly approximate the ground state $|\psi(\boldsymbol{\theta})\rangle \approx |\psi_g\rangle$ for some combinations of parameter values, but we need a method to identify those parameter combinations. Using both quantum and conventional computers, this is accomplished. 

Numerous quantum and classical computing cycles are used to optimise the parameters and move the ansatz closer to the approximate ground state. For a given set of parameter values, the quantum computer calculates the energy expectation value of the Hamiltonian $H$ acting on the parameterized ansatz.

$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle$$

This is the cost function. The measurement information from the quantum computer is used by a classical computer to determine how the parameter values should be changed to further reduce the energy $E(\boldsymbol{\theta})$. The parameter space is searched and the approximate ground state is reached by the classical and quantum computers as they loop through numerous iterations.

# Familiar Quantum Software

I primarily use Qiskit and PennyLane as my go-to quantum libraries, and have also worked with Cirq. My aim is in project is to use PennyLane with [Jraph](https://github.com/deepmind/jraph) library.  Jraph is a libraray developed by deepmind for graph neural networks. PennyLane recently released [Catalyst](https://github.com/PennyLaneAI/catalyst) which enables just-in-time (JIT) compilation of hybrid quantum-classical programs and I plan to use it.