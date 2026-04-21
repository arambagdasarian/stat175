# AMLWorld Synthetic Graphs — Evaluating Performance of Anonymized Data

## 1. Introduction and Motivation

The application of machine learning to anti–money laundering (AML) has introduced a fundamental challenge: the most valuable datasets for model development are also the most restricted. Financial transaction data is highly sensitive, and regulatory constraints limit its availability for external analysis, benchmarking, and model development.

Synthetic data generation offers a principled approach to addressing this constraint. Instead of sharing real transaction data, a financial institution may release a synthetic graph that approximates the statistical and structural properties of the original dataset. The central question, however, is not whether the synthetic data “looks similar,” but whether it preserves the information necessary for downstream tasks.

This project investigates the following problem:

Can a model trained on a synthetic transaction graph achieve comparable performance when evaluated on the original real transaction graph?

To answer this, we construct an experimental framework centered on anonymized graphs. Specifically, we compare model performance when trained on real data versus synthetic data, holding the evaluation dataset fixed. This provides a task-driven measure of synthetic data quality.


## 2. Data and Graph Representation

The experiments are conducted using the AMLWorld HI-Small dataset, which models financial activity as a directed transaction graph.

- Nodes represent accounts  
- Edges represent transactions between accounts  

Each transaction includes attributes such as amount, timestamp, and payment format, along with labels indicating whether it is associated with laundering activity.

The dataset is converted into a graph representation using PyTorch Geometric. This involves:

- Constructing a directed edge index from sender to receiver  
- Defining node features (e.g., bank identity, entity type)  
- Defining edge features (e.g., log-transformed amount, relative time, payment type)  

The resulting graph captures both relational structure and transactional attributes, enabling the application of graph-based learning methods.


## 3. Fraud Detection Model

### 3.1 GraphSAGE for Node Representation

Fraud detection in transaction networks is inherently relational, as suspicious behavior often arises from patterns of interaction rather than isolated events. To capture these dependencies, we employ GraphSAGE, a graph neural network that learns node representations through neighborhood aggregation.

At each layer, a node updates its representation by combining its own features with an aggregate of its neighbors’ features. Stacking multiple layers allows the model to incorporate information from progressively larger neighborhoods. A linear classifier is then applied to these embeddings to predict whether an account is associated with fraudulent activity.


### 3.2 Edge Classification

In addition to node-level inference, the model performs edge classification. For each transaction, a feature vector is constructed by concatenating:

- The embedding of the source node  
- The embedding of the destination node  
- The edge feature vector  

A multilayer perceptron is then used to classify whether the transaction is fraudulent. This setup allows the model to use both the surrounding graph structure and the transaction’s own attributes.


## 4. Synthetic Graph Generation

The project evaluates two approaches to synthetic graph generation.

### 4.1 Degree-Preserving Baseline

The baseline generator preserves the in-degree and out-degree sequences of the original graph. Edges are reassigned randomly while maintaining these degree constraints. Node and edge features are sampled independently from empirical distributions.

This method keeps basic graph statistics intact but does not attempt to reproduce more complex patterns like clusters or repeated transaction motifs.


### 4.2 GraphMaker

GraphMaker is a diffusion-based model for generating graphs with attributes. The idea is to learn how to reconstruct a graph after it has been gradually corrupted with noise.

During training, the model observes increasingly noisy versions of the original graph and learns how to reverse that process. At generation time, it starts from noise and iteratively builds a graph.

In practice, GraphMaker can first generate node attributes and then generate edges based on those attributes. This is useful for financial data, where the type of account often influences how it interacts with others.

Compared to the baseline, GraphMaker is designed to capture more complex structure in the data rather than just matching simple statistics.


## 5. Experimental Framework

The pipeline follows a consistent structure:

1. Build a graph from AMLWorld data  
2. Train a GraphSAGE-based model on the real graph  
3. Generate a synthetic graph  
4. Train the same model on the synthetic graph  
5. Evaluate both models on the real test set  

The main comparison is between:

- Training on real data and testing on real data  
- Training on synthetic data and testing on real data  

This gives a direct sense of how useful the synthetic data is for the actual task.

Because fraud is extremely rare, a few practical adjustments are needed:

- The loss function is weighted to account for imbalance  
- Extremely large class weights are capped to avoid unstable training  
- Training schedules are slightly adjusted to ensure convergence  
- Dropout is used for regularization  


## 6. Results

Experiments are conducted on an 80,000-transaction subset of AMLWorld.

### Node Classification (Account Fraud)

| Training Regime       | Degree-Preserving | GraphMaker |
|----------------------|------------------|------------|
| Real → Real          | 0.971            | 0.845      |
| Synthetic → Real     | 0.013            | 0.268      |
| Δ (Utility Gap)      | 0.958            | 0.577      |


### Edge Classification (Transaction Fraud)

| Training Regime       | Degree-Preserving | GraphMaker |
|----------------------|------------------|------------|
| Real → Real          | 0.969            | 0.972      |
| Synthetic → Real     | 0.851            | 0.904      |
| Δ (Utility Gap)      | 0.118            | 0.068      |


## 7. Discussion

On both tasks, the GraphMaker graph performs better on the baseline degree-preservation graph. These results show that anonymizing graph through the GraphMaker method could be good to use for detecting transaction fraud. Detecting account fraud seems to be harder with anonymized graph and need to work more on improving performance there. 

A useful way to read these results is to compare how the two tasks beha


## 8. Code Structure and Reproducibility
Tried to organize the codebase in a fairly modular way:

- data/amlworld.py handles loading the CSV files and turning them into a graph  
- generators/ contains the different synthetic graph methods  
- models/ contains the GraphSAGE model and training utilities  
- pipeline/transfer_experiment.py runs the full experiment end to end  
- evaluation/ includes similarity checks and supporting metrics  


To reproduce results, you can run the experiment script with the desired generator, which outputs metrics as JSON files. There are also scripts to visualize results and compare runs.

Hopefully, the structure is design so you can easily swap out components (e.g. trying a different model without rewriting the entire pipeline)