# Bayesian Network Structure and Parameter Learning

## Overview

This project implements **Bayesian Network Structure and Parameter Learning** using the `bnlearn` library. It constructs a Bayesian Network (BN) with predefined **Conditional Probability Distributions (CPDs)** and then performs **structure learning** and **parameter learning** based on sampled data. 

The script:
- Defines a **Bayesian Network** using manually specified CPDs.
- Generates **samples** from the Bayesian Network.
- Learns **network structure** and **parameters** from the sampled data.
- Iterates over different **sample sizes** to analyze learning performance.
- Saves and visualizes the learned **Directed Acyclic Graphs (DAGs)**.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
