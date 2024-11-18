class RayleighSchrodinger:
    """
    The RayleighSchrodinger class implements perturbation theory calculations for energy corrections 
    to the ground-state energy of a quantum system. It calculates corrections up to the fourth order 
    using nested summations and customizable formulas for the terms.

    Attributes:
        g (float): Coupling constant for the interaction.
        p (list[int]): List of particle indices.
        h (list[int]): List of hole indices.
    """

    def __init__(self, p, h, g):
        """
        Initializes the RayleighSchrodinger object with the number of particles, holes, and the coupling constant.

        Args:
            p (int): Number of particles.
            h (int): Number of holes.
            g (float): Coupling constant for the interaction.
        """
        self.g = g
        self.p = list(range(1, p + 1))  # Particle indices.
        self.h = list(range(len(self.p) + 1, len(self.p) + h + 1))  # Hole indices.

    def _sum_over(self, iterables, formula):
        """
        Helper function to calculate a sum over multiple iterables using a formula.

        Args:
            iterables (list[iterable]): List of iterables for nested summation.
            formula (function): Lambda function representing the formula to evaluate.

        Returns:
            float: Sum of the evaluated formula over all combinations of inputs.
        """
        from itertools import product
        return sum(formula(*args) for args in product(*iterables))

    def E0(self):
        """
        Computes the zeroth-order energy correction.

        Returns:
            float: Zeroth-order energy.
        """
        return 2

    def E1(self):
        """
        Computes the first-order energy correction.

        Returns:
            float: First-order energy.
        """
        return -self.g + self.E0()

    def E2(self):
        """
        Computes the second-order energy correction.

        Returns:
            float: Second-order energy.
        """
        term1 = self._sum_over([self.p, self.h], lambda p, h: self.g**2 / (p - h))
        return term1 / 8 + self.E1()

    def E3(self):
        """
        Computes the third-order energy correction.

        Returns:
            float: Third-order energy.
        """
        term1 = self._sum_over(
            [self.h, self.p, self.p],
            lambda h, p, pp: self.g**3 / ((h - p) * (h - pp))
        ) / -32

        term2 = self._sum_over(
            [self.h, self.h, self.p],
            lambda h, hh, p: self.g**3 / ((h - p) * (hh - p))
        ) / -32

        term3 = self._sum_over(
            [self.h, self.p],
            lambda h, p: self.g**3 / ((h - p)**2)
        ) / 8  # Divided by 8 because diagrams 8 and 9 are duplicates.

        return term1 + term2 + term3 + self.E2()

    def E4(self):
        """
        Computes the fourth-order energy correction.

        Returns:
            float: Fourth-order energy.
        """
        term1 = self._sum_over(
            [self.h, self.h, self.p, self.p],
            lambda h, hh, p, pp: self.g**4 * (
                1 / ((h - p) * (hh - p) * (hh - pp)) +
                1 / ((h - p) * (h - pp) * (hh - pp)) +
                1 / ((h - p) * (h + hh - p - pp) * (h - pp)) +
                1 / ((h - p) * (h + hh - p - pp) * (hh - p))
            )
        )

        term2 = self._sum_over(
            [self.h, self.p, self.p, self.p],
            lambda h, p, pp, ppp: self.g**4 / ((h - p) * (h - pp) * (h - ppp))
        )

        term3 = self._sum_over(
            [self.p, self.h, self.h, self.h],
            lambda p, h, hh, hhh: self.g**4 / ((h - p) * (hh - p) * (hhh - p))
        )

        denominator = 128**6  # Common denominator for all terms.
        return (term1 + term2 + term3) / denominator + self.E3()
