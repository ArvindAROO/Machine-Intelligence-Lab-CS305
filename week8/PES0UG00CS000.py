import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emission probabilites
    """
    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.states_dict = {i: self.states[i] for i in range(self.N)}
        self.emissions_dict = {self.emissions[i]: i for i in range(self.M)}
 
    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emissions dict)
        Returns:
            nu: Probability of the hidden state at time t given an observation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        nu = np.zeros((len(seq), self.N))
        psi = np.zeros((len(seq), self.N), dtype=int)
        for k in range(self.N):
            nu[0, k] = self.pi[k] * self.B[k, self.emissions_dict[seq[0]]]
            psi[0, k] = 0
        for i in range(1, len(seq)):
            for k in range(self.N):
                nu_max = -np.inf
                psi_max = -np.inf
                for s_prev in range(self.N):
                    temp = nu[i - 1, s_prev] * self.A[s_prev, k] * self.B[k, self.emissions_dict[seq[i]]]
                    if temp > nu_max:
                        nu_max = temp
                        psi_max = s_prev
                nu[i, k] = nu_max
                psi[i, k] = psi_max
        nu_max = -np.inf
        psi_max = -np.inf
        for k in range(self.N):
            temp = nu[len(seq) - 1, k]
            if temp > nu_max:
                nu_max = temp
                psi_max = k
        hiddenSequence = [psi_max]
        for i in range(len(seq) - 1, 0, -1):
            hiddenSequence.append(psi[i, hiddenSequence[-1]])

        # print(self.states_dict)
        return [self.states_dict[i] for i in hiddenSequence[::-1]]
    