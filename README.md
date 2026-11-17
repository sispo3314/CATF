# Disagreement-Conditioned Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition
<img width="2417" height="1536" alt="overall" src="https://github.com/user-attachments/assets/d27c4ada-9ed8-4d2b-885c-9bd7591ad50d" />

This repository implements the methodology proposed in the paper "Disagreement-Conditioned Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition"

## Paper Overview

**Abstract**: Wearable-sensor-based human activity recognition (HAR) supports activity tracking and fitness feedback on consumer devices such as smartwatches and smartphones. Time-domain and filterbank-oriented representations provide complementary information for wearable-sensor-based human activity recognition (HAR), but their consistency can vary across individual samples and activity conditions. Existing dual-branch HAR approaches commonly combine heterogeneous representations through fixed aggregation or generic learned fusion. This paper proposes a compact disagreement-conditioned time–filterbank reconciliation framework that uses sample-dependent discrepancy to regulate cross-branch interaction. A Perceiver-style encoder captures temporal dependencies, while a learnable multi-scale filterbank extracts complementary motion characteristics. A bounded sample-wise reconciliation coefficient is computed from feature similarity, relative feature distance, branch predictive entropy, and top-1 prediction agreement. The coefficient jointly regulates feature reconciliation and cross-branch agreement learning, but is not intended to identify the correct branch or provide a calibrated reliability estimate. Experiments on five public HAR datasets show strong recognition performance under subject-independent evaluation. The proposed method achieves the highest Macro-F1 among the controlled fusion alternatives across the five datasets, although the aggregate gains are dataset-dependent and modest when conventional fusion already performs strongly. Conflict-stratified analyses, coefficient interventions, controlled sensing perturbations, and edge-reference measurements further characterize the functional contribution and performance–cost trade-off of the proposed mechanism.

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **MHEALTH** dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **WISDM** dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php
- **PAMAP2** dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Codebase Overview
- `model.py`: Implementation of the proposed disagreement-conditioned time–filterbank reconciliation framework, including a Perceiver-style time-domain encoder, a learnable multi-scale filterbank encoder, auxiliary branch classifiers, a disagreement-descriptor and coefficient estimator, a disagreement-conditioned reconciliation module, and the complete training objective.
- `Comparison/`: Implementations of the representative HAR baselines used for the controlled comparison in the paper.
  - `DeepConvLSTM.py`: DeepConvLSTM baseline with stacked 1D convolutional layers followed by a two-layer LSTM.
  - `CNN-attention.py`: CNN–attention baseline with 1D convolutional blocks followed by multi-head self-attention and temporal average pooling.
  - `TransformerEncoder.py`: Transformer encoder baseline with learnable positional embeddings and a pre-norm Transformer encoder.
  - `Mamba.py`: Mamba-like selective state-space model (SSM) baseline with stacked selective SSM blocks.

## Citing this Repository
If you use this code in your research, please cite:
```
@article{CATF,
  title   = {Disagreement-Conditioned Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition},
  author  = {Jimin Kim and Myung-Kyu Yi},
  journal = {},
  volume  = {},
  number  = {},
  pages   = {},
  year    = {},
  publisher = {}
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please contact:
  - Jimin Kim: sispo3314@gmail.com

