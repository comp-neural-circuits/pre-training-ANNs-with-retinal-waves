# Code for the preprint: Pre-training artificial neural networks with spontaneous retinal activity improves motion prediction in natural scenes

In this repository, we provide the code used to generate the results we present in our preprint:

[May, Lilly, and Julijana Gjorgjieva. "Pre-training artificial neural networks with spontaneous retinal activity improves motion prediction in natural scenes." bioRxiv (2024): 2024-06.](https://www.biorxiv.org/content/10.1101/2024.06.15.599143v1).

We provide scripts for generating the data, training the artificial neural networks (ANNs), and evaluating the performance and characteristics of the trained ANNs.

## Data generation
We provide the code to generate the following datasets:
- A dataset of natural scenes with prominent motion, specifically a virtual maze simulation. This dataset was generated using the 3D animation software Blender and the Python API of Blender.
- A dataset of spontaneous retinal activity. This dataset was generated based on the model introduced by [Teh et al. (2023)](https://www.science.org/doi/full/10.1126/sciadv.adf4240).

All generated datasets are available via [Zenodo](https://zenodo.org/records/10317798).

## ANN models
We provide the code to train and evaluate the ANN models designed for the task of Next Frame Prediction.
We implemented convolutional recurrent neural networks, however, we focused on building highly modular models, for instance by supporting different types of recurrent layers, such as LSTM, GRU, and vanilla RNN.