import numpy as np
class Energy:
    """
    The Energy class performs computations related to constructing matrices, 
    removing specific states, and calculating ground-state energies for a given 
    range of coupling constants (g). It includes tools for diagonalizing matrices 
    and comparing the full and reduced configurations.

    Attributes:
        g (list[float]): A list of coupling constants for which calculations will be performed.
        FCI (numpy.ndarray): Array storing the ground-state energies of the full matrices.
        CI (numpy.ndarray): Array storing the ground-state energies of the reduced matrices.
        eig (numpy.ndarray): Array of eigenvalues for the full matrices.
    """

    def __init__(self, g):
        """
        Initializes the Energy class with a list of coupling constants.

        Args:
            g (list[float]): List of coupling constants.
        """
        self.g = g

    def construct_matrix(self, g):
        """
        Constructs a symmetric matrix with diagonal and off-diagonal elements 
        determined by the coupling constant g.

        Args:
            g (float): Coupling constant.

        Returns:
            numpy.ndarray: The constructed matrix.
        """
        diagonal = np.array([2 - g, 4 - g, 6 - g, 6 - g, 8 - g, 10 - g])
        size = len(diagonal)
        matrix = np.zeros((size, size))

        # Fill the diagonal
        np.fill_diagonal(matrix, diagonal)

        # Fill off-diagonal elements (-g/2) except for the opposite diagonal
        for i in range(size):
            for j in range(size):
                if i != j and i + j != size - 1:
                    matrix[i, j] = -g / 2
        return matrix

    def remove_state(self, matrix, index=5):
        """
        Removes a specified state (row and column) from a matrix.

        Args:
            matrix (numpy.ndarray): The input matrix.
            index (int): The index of the state to remove (default is 5).

        Returns:
            numpy.ndarray: The reduced matrix with the specified state removed.
        """
        reduced_matrix = np.delete(matrix, index, axis=0)  # Remove row
        reduced_matrix = np.delete(reduced_matrix, index, axis=1)  # Remove column
        return reduced_matrix

    def find_ground_state_energy(self, matrix):
        """
        Calculates the ground-state energy of a matrix by finding its smallest eigenvalue.

        Args:
            matrix (numpy.ndarray): The input matrix.

        Returns:
            float: The ground-state energy (smallest eigenvalue).
        """
        eigenvalues, _ = np.linalg.eigh(matrix)
        return np.min(eigenvalues)

    def run(self):
        """
        Executes the energy calculations for all values of g, including:
        - Constructing the full matrix and calculating its ground-state energy.
        - Removing the specified state from the full matrix and recalculating the ground-state energy.
        - Storing eigenvalues for the full matrices.

        Updates the following attributes:
            FCI (numpy.ndarray): Full configuration ground-state energies.
            CI (numpy.ndarray): Reduced configuration ground-state energies.
            eig (numpy.ndarray): Eigenvalues of full matrices for each g.
        """
        g_values = self.g
        original_energies = []
        reduced_energies = []
        eigenvalues = []

        for g in g_values:
            # Construct the full matrix and compute ground-state energy
            full_matrix = self.construct_matrix(g)
            original_energies.append(self.find_ground_state_energy(full_matrix))
            eigenvalues.append(np.linalg.eigh(full_matrix)[0])

            # Remove the 6th state and compute ground-state energy for the reduced matrix
            reduced_matrix = self.remove_state(full_matrix)
            reduced_energies.append(self.find_ground_state_energy(reduced_matrix))

        self.FCI = np.array(original_energies)
        self.CI = np.array(reduced_energies)
        self.eig = np.array(eigenvalues)

    def Hartree_Fock(self, g):
        """
        Calculates a simplified Hartree-Fock energy approximation for a given coupling constant.

        Args:
            g (float): Coupling constant.

        Returns:
            float: Hartree-Fock energy approximation.
        """
        return 2 - g
