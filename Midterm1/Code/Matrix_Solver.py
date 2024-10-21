import numpy as np
from tools import Setup, SetupMatrix

class MatrixSolver:
    """
    Class for solving matrix-based quantum mechanical problems, such as finding the ground state energy 
    of an atomic system using a configuration interaction approach. The class leverages one-body and two-body
    matrix elements to compute ground state energies and overlaps.

    Attributes:
        setup (Setup): An instance of the Setup class, used to define the Fermi level and manage occupied states.
        matrix (SetupMatrix): An instance of SetupMatrix that provides access to one-body and two-body matrix elements.
        N (int): Size of the matrix for configuration interaction calculations (default is 5).
    """

    def __init__(self, setup, matrix, N=5):
        """
        Initialize the MatrixSolver with the setup and matrix objects.

        Args:
            setup (Setup): Setup object containing the Fermi level and state information.
            matrix (SetupMatrix): Matrix object providing access to one-body and two-body matrix elements.
            N (int, optional): Size of the matrix for configuration interaction calculations. Default is 5.
        """
        self.setup = setup
        self.matrix = matrix
        self.N = N

    def gs(self):
        """
        Compute the ground state energy of the system.

        This function calculates the total energy by summing the one-body matrix elements
        for all states below the Fermi level and half the two-body matrix elements for pairs of these states.

        Returns:
            float: Total ground state energy.
        """
        total_energy = 0
        for i in self.setup.below():
            total_energy += self.matrix.one_body(i)
            for j in self.setup.below():
                total_energy += 0.5 * self.matrix.two_body(i, j, i, j)
        return total_energy

    def overlap(self, *indices):
        """
        Calculate the overlap between various states based on the provided indices.

        Args:
            *indices: Variable length argument list that can contain 2 or 4 indices.
                If 2 indices are provided (i, alpha), calculate the overlap for a specific state.
                If 4 indices are provided (i, alpha, j, beta), calculate the interaction for pairs of states.

        Returns:
            float: The overlap value based on the input indices.
        
        Raises:
            ValueError: If an invalid number of indices is provided.
        """
        M = self.matrix
        below_states = self.setup.below()

        if len(indices) == 2:
            i, alpha = indices
            return sum(M.two_body(i, j, alpha, j) for j in below_states)

        elif len(indices) == 4:
            i, alpha, j, beta = indices
            temp1 = M.two_body(alpha, j, i, beta)

            temp2 = sum(M.two_body(alpha, k, beta, k) * (i == j) - M.two_body(j, k, i, k) * (alpha == beta) 
                        for k in below_states)

            if (i == j) and (alpha == beta):
                temp3 = (M.one_body(alpha) - M.one_body(i))
            else:
                temp3 = 0

            temp4 = 0
            if (i == j) and (alpha == beta):
                temp4 = sum(M.one_body(k) + sum(0.5 * M.two_body(k, l, k, l) for l in below_states) 
                            for k in below_states)

            return temp1 + temp2 + temp3 + temp4

        else:
            raise ValueError("Invalid number of indices for overlap calculation")

    def fill_final_matrix(self, holes, parts):
        """
        Fill the final matrix for the configuration interaction problem.

        The matrix is filled with the ground state energy, overlaps, and interaction terms between
        particles and holes.

        Args:
            holes (list): List of tuples representing hole states.
            parts (list): List of tuples representing particle states.
        """
        H_Mat = np.zeros((self.N, self.N))
        H_Mat[0, 0] = self.gs()
    
        for n in range(len(H_Mat) - 1):
            H_Mat[n+1, 0] = self.overlap(parts[n], holes[n])  # horizontal
            H_Mat[0, n+1] = H_Mat[n+1, 0]  # vertical
    
            for m in range(n, len(H_Mat) - 1):
                H_Mat[n+1, m+1] = self.overlap(parts[m], holes[m], parts[n], holes[n])  # fill remaining
                H_Mat[m+1, n+1] = H_Mat[n+1, m+1]
    
        self.H_Mat = H_Mat
        np.set_printoptions(precision=4)
        print(self.H_Mat)

    def eigenvalues(self):
        """
        Calculate the eigenvalues of the Hamiltonian matrix and return the lowest eigenvalue.

        Returns:
            float: The lowest eigenvalue, which corresponds to the ground state energy.
        """
        E, _ = np.linalg.eig(self.H_Mat)
        return np.min(E)

def solve_atom(Z, holes, parts, fermi_idx=0, max_idx=3, states=3):
    """
    Solve for the ground state energy of an atom using the MatrixSolver.

    Args:
        Z (int): Atomic number of the atom.
        holes (list): List of hole state tuples.
        parts (list): List of particle state tuples.
        fermi_idx (int, optional): Index of the Fermi level. Default is 0.
        max_idx (int, optional): Maximum index for states. Default is 3.
        states (int, optional): Number of states to consider. Default is 3.

    Returns:
        float: The ground state energy of the atom.
    """
    setup = Setup(fermi_idx, max_idx)
    matrix = SetupMatrix(Z, states, setup)
    solver = MatrixSolver(setup, matrix)
    solver.fill_final_matrix(holes, parts)
    return solver.eigenvalues()

# SOLVE FOR HELIUM AND BERYLLIUM
atoms = [
    {"name": "Helium",
     "Z": 2, 
     "holes": [(1, 0.5), (1, -0.5), (2, 0.5), (2, -0.5)],
     "parts": [(0, 0.5), (0, -0.5), (0, 0.5), (0, -0.5)],
     'idx': 0},
    
    {"name": "Beryllium",
     "Z": 4,
     "holes": [(2, 0.5), (2, -0.5), (2, 0.5), (2, -0.5)],
     "parts": [(0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5)],
     "idx": 1}
]

for atom in atoms:
    print(f"================ {atom['name']} ================")
    eigenvalue = solve_atom(atom['Z'], atom['holes'], atom['parts'], atom['idx'])
    print(f"{atom['name']} ground state energy: {eigenvalue:.2f} a.u.")
    print()
