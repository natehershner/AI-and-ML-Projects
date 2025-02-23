# Parametric Gaussian Classifier

## Assignment Overview

### Problem Statement

The goal is to build a classifier that distinguishes between:
- **Class C1**: Points falling within the first (blue) rectangle.
- **Class C2**: Points falling within the second (black) rectangle.
- **No Class (0)**: Points outside both rectangles.

Alternatively, you can think of the problem as detecting if a point is inside one of the two rectangles (each representing a different class), with all other points not belonging to any class.

### Assignment Tasks

1. **Parametric Classifier Training (90 pts total; 30 pts per problem)**
   - **Model Building**: Train a classifier by assuming a multivariate Gaussian distribution for each class. The appropriate model complexity must be selected as discussed in the course (Sections 5.5 and 5.6 of the textbook).
   - **Custom Functions**:
     - **`gtrain(x)`**: For a given class, this function calculates the mean vector and covariance matrix of the data points and computes a rejection threshold (the minimum probability density observed on the training data).
     - **`grec(classpdf1, classpdf2, x)`**: This function uses the estimated parameters for the two classes to classify new data points. A point is assigned a class label based on which class yields a higher probability density, with a rejection option if the density is below the precomputed threshold.

2. **Impact of Training Sample Size**
   - Analyze how the number of training data points affects the accuracy of the classifier.
   - Experiment by varying the training sample size (e.g., from 50 to 1000 points) and plotting the classification error as a function of sample size.

3. **Visualization and Analysis**
   - Visualize the classifier’s performance on an independent test set.
   - Plot the decision regions and overlay the true class boundaries.

---

## Code Structure

- **Data Generation**:
  - Generates 2D points uniformly distributed over the interval [0,8]×[0,8].
  - Points are labeled as follows:
    - **1 (C1)**: Inside the blue rectangle.
    - **2 (C2)**: Inside the black rectangle.
    - **0**: Outside both rectangles.
  - The code includes a visualization of the training data along with the true rectangular boundaries.

- **Classifier Training (`gtrain`)**:
  - Estimates the mean and covariance for a given class using the provided data.
  - Computes probability densities with `scipy.stats.multivariate_normal.pdf` and determines a rejection threshold based on the minimum observed density.

- **Classification (`grec`)**:
  - Computes class-conditional densities for each new data point.
  - Assigns labels based on the higher density, and applies a rejection mechanism if the density does not meet the threshold.

- **Testing and Evaluation**:
  - An independent test set is generated.
  - The classifier's performance is evaluated by computing the classification error.
  - Test data predictions are visualized to confirm that the decision regions align with the expected class boundaries.
  - Decision regions are plotted using a grid over the input space.

- **Experiment on Training Sample Size**:
  - The classifier is trained on different sample sizes.
  - A fixed test set is used to evaluate performance.
  - A plot shows how the classification error changes with increasing training sample size, providing insight into the effect of data volume on classifier accuracy.

---

## Dependencies

- **Python 3.x**
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting graphs and data visualization.
- **SciPy**: Specifically, the `multivariate_normal` function from `scipy.stats` is used for probability density calculations.
