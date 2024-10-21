import numpy as np
import matplotlib.pyplot as plt
from tools import Setup, SetupMatrix

class HF_Solver:
    """
    The HF_Solver class implements the Hartree-Fock method to solve the ground-state energy
    for atoms using a density matrix approach. It computes the energy convergence through 
    iterations until convergence is achieved within a specified tolerance.

    Attributes:
        setup (Setup): Contains the Fermi level and state configuration.
        matrix (SetupMatrix): Contains matrix elements for one-body and two-body interactions.
        Z (int): Atomic number of the atom.
        parts (int): Number of particles.
        states (int): Number of states to consider.
        max_iterations (int): Maximum number of Hartree-Fock iterations.
        tolerance (float): Convergence tolerance for the Hartree-Fock algorithm.
        memory (numpy.ndarray): Stores eigenvalues for convergence checking.
        energyConvergence (list): Stores energy values over iterations to track convergence.
        SPS (list): List of single-particle states (index, spin) generated from setup.
    """

    def __init__(self, setup, matrix, Z, parts, states, max_iterations, tolerance):
        """
        Initializes the HF_Solver object with the setup, matrix elements, and Hartree-Fock parameters.

        Args:
            setup (Setup): Object managing the Fermi level and state indices.
            matrix (SetupMatrix): Object managing the one-body and two-body interaction matrices.
            Z (int): Atomic number of the atom.
            parts (int): Number of particles.
            states (int): Number of states.
            max_iterations (int): Maximum number of iterations for Hartree-Fock calculations.
            tolerance (float): Tolerance value for convergence.
        """
        self.setup = setup
        self.matrix = matrix
        self.Z = Z
        self.parts = parts
        self.states = states
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.memory = None  # Stores the energy eigenvalues for stopping condition
        self.energyConvergence = []  # Tracks the energy convergence during iterations

        # Generate single-particle states (SPS) from setup, with spins +0.5 and -0.5
        temp = []
        for i in range(setup.max_index):
            temp.append((i, 0.5))
            temp.append((i, -0.5))
        self.SPS = temp  # List of single-particle states

    def make_density_matrix(self, C):
        """
        Construct the density matrix from the coefficient matrix C.

        Args:
            C (numpy.ndarray): Coefficient matrix of the Hartree-Fock orbitals.

        Returns:
            numpy.ndarray: The density matrix ρ.
        """
        rho_Mat = np.zeros_like(C)

        for n in range(self.states):
            for m in range(self.states):
                temp = 0
                for l in range(self.parts):
                    temp += C[l, n] * C[l, m]
                rho_Mat[n, m] = temp

        return rho_Mat

    def gs(self, rho):
        """
        Compute the ground-state energy using the current density matrix ρ.

        Args:
            rho (numpy.ndarray): Current density matrix.
        """
        ii = 0
        temp = 0

        # One-body energy contribution
        for i in self.SPS:
            temp += rho[ii, ii] * self.matrix.one_body(i)
            ii += 1

        # Two-body energy contribution (using density matrix terms)
        ii = 0  # Reset the counter
        for i in self.SPS:
            jj = 0
            for j in self.SPS:
                kk = 0
                for k in self.SPS:
                    ll = 0
                    for l in self.SPS:
                        temp += 0.5 * rho[ii, kk] * rho[jj, ll] * self.matrix.two_body(i, j, k, l)
                        ll += 1
                    kk += 1
                jj += 1
            ii += 1

        self.E = temp  # Store the calculated energy

    def fill_matrix(self, rho):
        """
        Build the Hartree-Fock matrix from the current density matrix.

        Args:
            rho (numpy.ndarray): Current density matrix.

        Returns:
            numpy.ndarray: The Hartree-Fock matrix.
        """
        Mat = np.zeros_like(rho)

        ii = 0
        # One-body matrix elements
        for i in self.SPS:
            Mat[ii, ii] = self.matrix.one_body(i)
            ii += 1

        # Two-body matrix elements (interaction terms)
        ii = 0
        for i in self.SPS:
            jj = 0
            for j in self.SPS:
                temp = 0
                kk = 0
                for k in self.SPS:
                    ll = 0
                    for l in self.SPS:
                        temp += rho[kk, ll] * self.matrix.two_body(i, k, j, l)
                        ll += 1
                    kk += 1
                Mat[ii, jj] += temp
                jj += 1
            ii += 1

        return Mat

    def HartreeFock(self):
        """
        Perform the Hartree-Fock iterations to solve for the ground state energy.
        Tracks the energy convergence and stops when convergence criteria are met.
        """
        C = np.eye(self.states, self.states)  # Initialize coefficient matrix as identity
        rho = self.make_density_matrix(C)  # Initial density matrix

        for i in range(self.max_iterations):
            HF = self.fill_matrix(rho)  # Construct Hartree-Fock matrix from density matrix

            E, C = self.eigenvalues(HF)  # Solve for eigenvalues and eigenvectors (coefficients)
            
            if i == 0:
                print(E)

            stop = self.early_stop(E)  # Check for convergence

            if stop:
                break

            rho = self.make_density_matrix(C.T)  # Update density matrix with transposed(!!) coefficients

            self.gs(rho)  # Compute ground-state energy
            self.energyConvergence.append(self.E)  # Track energy convergence
        if stop:
            print(f'Convergence reached after {i + 1} iterations!')
        else:
            print('No convergence reached :(, stopping...')

    def eigenvalues(self, mat):
        """
        Compute the eigenvalues and eigenvectors of a matrix.

        Args:
            mat (numpy.ndarray): Input matrix.

        Returns:
            tuple: Eigenvalues and eigenvectors.
        """
        E, C = np.linalg.eigh(mat) # eigh, not eig!!
        return E, C

    def early_stop(self, E):
        """
        Check if the energy has converged based on the current and previous eigenvalues.

        Args:
            E (numpy.ndarray): Current eigenvalues.

        Returns:
            bool: True if convergence is achieved, False otherwise.
        """
        if self.memory is None:
            self.memory = E
            return False
        else:
            if np.linalg.norm(np.abs(E - self.memory)) / self.states < self.tolerance:
                return True
            else:
                self.memory = E
                return False

def solve_atom(Z, parts, fermi_idx=0, max_idx=3, states=3):
    """
    Solve for the ground state energy of an atom using the Hartree-Fock method.

    Args:
        Z (int): Atomic number of the atom.
        parts (int): Number of particles.
        fermi_idx (int, optional): Fermi level index. Defaults to 0.
        max_idx (int, optional): Maximum index for the states. Defaults to 3.
        states (int, optional): Number of states to consider. Defaults to 3.

    Returns:
        tuple: Minimum eigenvalue (ground state energy) and the energy convergence history.
    """
    setup = Setup(fermi_idx, max_idx)
    matrix = SetupMatrix(Z, states, setup)
    solver = HF_Solver(setup, matrix, Z, parts, 6, max_iterations=1, tolerance=1e-15)
    solver.HartreeFock()
    return np.min(solver.E), solver.energyConvergence

# SOLVE FOR HELIUM AND BERYLLIUM
atoms = [
    {"name": "Helium", "Z": 2, "P": 2, "idx": 0},
    {"name": "Beryllium", "Z": 4, "P": 4, "idx": 1}
]

for atom in atoms:
    eigenvalue, history = solve_atom(atom['Z'], atom['P'], atom['idx'])
    print(f"================ {atom['name']} ================")
    print(f"{atom['name']} ground state energy: {eigenvalue:.2f} a.u.")
    print()
    plt.title('Energy convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Energy [a.u.]')
    plt.plot(history)
    plt.show()
