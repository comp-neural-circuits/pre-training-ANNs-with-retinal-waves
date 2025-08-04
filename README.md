# Pre-training artificial neural networks with spontaneous retinal activity improves motion prediction in natural scenes

In this repository, you'll find the code used to generate the results presented in our publication:

[May, Lilly, Alice Dauphin, and Julijana Gjorgjieva. "Pre-training artificial neural networks with spontaneous retinal activity improves motion prediction in natural scenes." PLOS Computational Biology 21.3 (2025): e1012830.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012830).

We provide scripts for generating the data, training the artificial neural networks (ANNs), and evaluating the performance and characteristics of the trained ANNs.

## 📂 Datasets
This repository includes code to generate the following datasets, located in the `dataset` folder:

- **Virtual corridor simulation**: A dataset of a natural scene with prominent optic flow, created using the 3D animation software [Blender](https://www.blender.org) and its Python API. This dataset mimics the experience of a mouse navigating through a corridor with visual cues on the walls.
- **Retinal waves**: A dataset of spontaneous retinal activity, also known as retinal waves, which propagate across the retina with a directional bias. The dataset was generated based on the retinal wave model introduced by [Teh et al. (2023)](https://www.science.org/doi/full/10.1126/sciadv.adf4240).

📦 **Accessing the Datasets**:
The pre-generated datasets are available via [Zenodo](https://doi.org/10.5281/zenodo.10317797).

## ⚙️ ANN models
We provide code to train (`model/train_cli.py`) and evaluate (`evaluation` folder) ANN models designed for the task of Next-Frame Prediction.

Our implementation is based on convolutional recurrent neural networks, with a strong focus on modularity. For example, our models support interchangeable recurrent layers, including:
- Vanilla RNN
- GRU
- LSTM

This flexible architecture makes it easy to experiment with different configurations and evaluate their performance.

## 💬 Got questions or ran into issues?
Open an issue in this repo or contact [Lilly](https://lilly-may.github.io/) - happy to help!

<p align="center">
  <img width="500" alt="Mouse in a virtual corridor" src="https://github.com/user-attachments/assets/7ce5640b-0a6e-46b3-9116-3cebf73741d0" />
</p>
