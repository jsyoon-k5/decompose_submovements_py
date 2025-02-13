# Submovement Decomposer

This repository contains a Python implementation of a submovement decomposition algorithm, migrated from the original MATLAB implementation: https://github.com/sgowda/decompose_submovements. 
The submovement decomposition technique is widely used in motion analysis to break down complex trajectories into elementary movements, providing insights into the underlying motor control processes.

This Python implementation uses scipy.optimize.minimize for the optimization process. Since there is no exact equivalent of MATLAB's fmincon in Python, decomposition performance may vary from the original work.
Depending on the dataset and the settings, these differences could manifest in the optimization results.

Gowda, S., Overduin, S. A., Chen, M., Chang, Y. H., Tomlin, C. J., & Carmena, J. M. (2015). Accelerating submovement decomposition with search-space reduction heuristics. IEEE Transactions on Biomedical Engineering, 62(10), 2508-2515.
