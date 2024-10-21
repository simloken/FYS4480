import numpy as np 

class Setup:
    """
    The Setup class defines the Fermi level and manages the state information, 
    including states below and above the Fermi level, based on a specified index.

    Attributes:
        fermi_index (int): Index indicating the Fermi level.
        max_index (int): Maximum state index to consider for the calculations.
    """

    def __init__(self, fermi_index, max_index):  
        """
        Initialize the Setup object with the Fermi level and maximum index.

        Args:
            fermi_index (int): Index representing the Fermi level.
            max_index (int): Maximum index for state levels.
        """
        self.set_fermi_level(fermi_index, max_index)
        
    def set_fermi_level(self, fermi_index, max_index):
        """
        Set the Fermi level and the maximum index for the states.

        Args:
            fermi_index (int): Index representing the Fermi level.
            max_index (int): Maximum index for state levels.
        """
        self.fermi_index = fermi_index
        self.max_index = max_index

    def below(self):
        """
        Generator that yields states below the Fermi level, including spin information.
        
        Yields:
            tuple: State index and spin value for each state below the Fermi level.
        """
        idx, spin = 0, 0.5
        while idx <= self.fermi_index:
            yield idx, spin
            
            if spin > 0:
                spin = -0.5
            else:
                spin = 0.5
                idx += 1 

    def above(self):
        """
        Generator that yields states above the Fermi level, including spin information.
        
        Yields:
            tuple: State index and spin value for each state above the Fermi level.
        """
        idx, spin = self.fermi_index + 1, 0.5
        while idx < self.max_index:
            yield idx, spin
            
            if spin > 0:
                spin = -0.5
            else:
                spin = 0.5
                idx += 1 
    

class SetupMatrix:
    """
    The SetupMatrix class manages the matrix elements required for quantum calculations,
    specifically one-body and two-body matrix elements, using the provided atomic number 
    and state configuration. It reads values from a file to fill the interaction matrix.

    Attributes:
        Z (int): Atomic number of the atom.
        states (int): Number of states to consider for the matrix elements.
        mat (numpy.ndarray): 4D matrix storing the interaction values for the system.
        values (dict): Dictionary that stores matrix element values read from the file.
    """

    def __init__(self, Z, states, setup):
        """
        Initialize the SetupMatrix object and fill the matrix with values.

        Args:
            Z (int): Atomic number of the atom.
            states (int): Number of states to consider.
            setup (Setup): Setup object containing the Fermi level and state information.
        """
        self.Z = Z
        self.states = states
        self.mat = np.zeros(shape=(states, states, states, states))  # 4D interaction matrix
        self.read_values_and_fill_matrix()
        
    def read_values_and_fill_matrix(self):
        """
        Read matrix element values from a file and fill the 4D interaction matrix.
        """
        values = {}
        with open('../Data/Values.txt', 'r') as f:
            for line in f:
                key, value = line.split('=')
                key = key.strip()
                value = value.replace('Z', 'self.Z')  # Replace Z with the atomic number
                value = value.replace('Sqrt', 'np.sqrt')  # Replace 'Sqrt' with numpy sqrt function
                value = value.replace('[', '('); value = value.replace(']', ')')  # Convert brackets to parentheses
                value = eval(value.strip())  # Evaluate the string to a numerical value
                values[key] = value
                
                # Extract matrix indices from the key
                bra, ket = key.split('V')[0][1:3], key.split('V')[1][1:3]
                i = int(bra[0])-1; j = int(bra[1])-1
                k = int(ket[0])-1; l = int(ket[1])-1
                
                # Fill the matrix with the calculated value
                self.mat[i, j, k, l] = value

            self.values = values  # Store values in case they are needed later
    
    def one_body(self, i):
        """
        Calculate the one-body matrix element for a given state.

        Args:
            i (tuple): Tuple representing a state (index, spin).

        Returns:
            float: One-body energy value for the given state.
        """
        return -(self.Z**2) / (2 * (i[0] + 1)**2)
    
    def two_body(self, i, j, k, l):
        """
        Calculate the two-body matrix element for a pair of states, considering spin conservation.

        Args:
            i (tuple): First state (index, spin).
            j (tuple): Second state (index, spin).
            k (tuple): Third state (index, spin).
            l (tuple): Fourth state (index, spin).

        Returns:
            float: Two-body interaction energy between the states.
        """
        spin1 = (i[1] == k[1]) * (j[1] == l[1])  # Spin alignment condition 1
        spin2 = (i[1] == l[1]) * (j[1] == k[1])  # Spin alignment condition 2

        # Return the difference between the two spin-aligned matrix elements
        return spin1 * self.mat[i[0], j[0], k[0], l[0]] - spin2 * self.mat[i[0], j[0], l[0], k[0]]
