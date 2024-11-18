from energy import Energy
from RSPT import RayleighSchrodinger
import numpy as np
import matplotlib.pyplot as plt


def exercise2():
    g = np.linspace(-1, 1, 100)
    obj = Energy(g)
    obj.run()
    
    plt.figure(figsize=(10, 6))

    plt.plot(g, obj.FCI, label='FCI', color='blue')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energy as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))

    j = 0
    for i in obj.eig.T:
        plt.plot(g, i, label=f'$e_{j}$', alpha=0.5)
        j+= 1

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energy as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

            
def exercise3():
    
    
    g = np.linspace(-1, 1, 100)
    obj = Energy(g)
    obj.run()
    
    plt.figure(figsize=(10, 6))

    plt.plot(g, obj.FCI, label='FCI', color='blue')
    plt.plot(g, obj.CI, label='CI (No $\Phi_5$)', color='orange')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI - obj.CI, label='Energy Difference', color='green')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Difference as a Function of $g$ (FCI v CI)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def exercise5():
    
    g = np.linspace(-1, 1, 100)
    obj = Energy(g)
    obj.run()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI, label='FCI', color='blue')
    plt.plot(g, obj.CI, label='CI (No $\Phi_5$)', color='orange')
    plt.plot(g, obj.Hartree_Fock(g), label='HF', color='red')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI - np.array(obj.Hartree_Fock(g)), label='Energy Difference', color='green')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Difference as a Function of $g$ (FCI v HF)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    obj2 = RayleighSchrodinger(2, 2, g)    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI, label='FCI', color='blue')
    plt.plot(g, obj2.E3(), label='$E^{(3)}$', color='red')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI - obj2.E3(), label='Energy Difference', color='green')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Difference as a Function of $g$ (FCI v $E^{(3)}$)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    


def exercise7():
    g = np.linspace(-1, 1, 100)
    
    obj = Energy(g)
    obj2 = RayleighSchrodinger(2, 2, g)
    obj.run()
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(g, obj.FCI, label='FCI', color='blue')
    plt.plot(g, obj2.E4(), label='$E^{(4)}$', color='red')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI - obj2.E4(), label='Energy Difference', color='green')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Difference as a Function of $g$ (FCI v $E^{(4)}$)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, abs(abs(obj.FCI - obj2.E3()) - abs(obj.FCI -  obj2.E4())), label='Difference between absolute differences', color='green')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Difference between absolute differences ($E^{(3)}$ v $E^{(4)}$)')
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI, label='FCI', color='black')
    plt.plot(g, obj.CI, label='CI (No $\Phi_5$)', linestyle='-.', alpha=0.6)
    plt.plot(g, obj.Hartree_Fock(g), label='HF', linestyle='-.', alpha=0.6)
    
    plt.plot(g, obj2.E1(), label='$E^{(1)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E2(), label='$E^{(2)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E3(), label='$E^{(3)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E4(), label='$E^{(4)}$', linestyle='-.', alpha=0.6)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$ (all)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(g, obj.FCI, label='FCI', color='black')
    plt.plot(g, obj.CI, label='CI (No $\Phi_5$)', linestyle='-.', alpha=0.6)
    plt.plot(g, obj.Hartree_Fock(g), label='HF', linestyle='-.', alpha=0.6)
    
    plt.plot(g, obj2.E1(), label='$E^{(1)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E2(), label='$E^{(2)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E3(), label='$E^{(3)}$', linestyle='-.', alpha=0.6)
    plt.plot(g, obj2.E4(), label='$E^{(4)}$', linestyle='-.', alpha=0.6)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Ground State Energies as a function of $g$ (all)")
    plt.xlabel("$g$")
    plt.ylabel("Energy")
    plt.xlim([-1/3, 1/3])
    plt.ylim([1.5, 2.5])
    plt.legend()
    plt.grid(True)
    plt.show()

    
    
if __name__ == "__main__":
    exercise2()
    
    exercise3()
    
    exercise5()
    
    
    
    exercise7()