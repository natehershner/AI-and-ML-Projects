Optdigits Analysis and Feature Selection



## Overview

The assignment is divided into three main problems:

1. **Forward Feature Selection**  
   - **Objective:** Iteratively select features that improve classification performance using a Random Forest classifier.
   - **Outcome:**  
     - A list of selected feature indices.  
     - The number of features selected.  
     - A plot showing the cross-validation accuracy as features are added.

2. **Principal Component Analysis (PCA)**  
   - **Objective:** Investigate the trade-off between dimensionality reduction and reconstruction error on the optdigits dataset.
   - **Outcome:**  
     - A graph of reconstruction error (Mean Squared Error) versus the number of principal components (ranging from 1 to 64).
     - A 3D scatter plot visualizing the data projected onto the first three principal components.

3. **Eigendigits Visualization**  
   - **Objective:** Compute and visualize the mean digit and the first eight principal components (eigendigits) as 8x8 grayscale images.
   - **Outcome:**  
     - An image of the mean digit.
     - Images of the first eight principal components (eigendigits) that capture the main variance directions in the dataset.
