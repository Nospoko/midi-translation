import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(start=0, stop=8, num=99)
    y = (np.exp(x) - 1) / 1000
    plt.plot(y)
    plt.show()
    print(f"  {len(x) + 1}:")
    for val in y:
        print(f"  - {val:6.7f}")
