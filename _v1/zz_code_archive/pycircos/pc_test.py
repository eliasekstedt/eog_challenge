




import pycircos
import numpy as np

# Sample data setup
np.random.seed(12345)
gcircle = pycircos.Gcircle()
loci = [pycircos.gene("chr1", i, i + np.random.randint(3, 15)*int(1e6), np.random.choice(["+","-"])) for i in range(0, int(249e6), int(5e6))]
gcircle.add_loci(loci, "outer")

# Adjust the links here with width and alpha
for i in range(0, len(loci), 2):
    start = loci[i]
    end = loci[i+1]
    gcircle.add_link(start, end, width=2, alpha=0.6)

# Display the plot
gcircle.save("pycircos_example.png")