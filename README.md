# Conflict-Aware Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition
<img width="1697" height="887" alt="image" src="https://github.com/user-attachments/assets/2bb1b90b-5da3-4d8c-808a-2bdfba6a6f96" />
This repository implements the methodology proposed in the paper "Conflict-Aware Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition"

## Paper Overview

**Abstract**: Time-domain and filterbank-oriented representations provide complementary information for wearable-sensor-based HAR, but their relative usefulness can vary across activities and individual samples. Existing dual-branch HAR approaches typically combine heterogeneous representations through fixed aggregation or generic learned fusion, without explicitly modeling sample-wise agreement or disagreement between the representation branches. This paper proposes a lightweight conflict-aware time–filterbank reconciliation framework that explicitly models sample-wise disagreement between temporal and filterbank-oriented representations. A compact Perceiver-style encoder captures long-range temporal dependencies, while a learnable multi-scale filterbank extracts complementary multi-scale motion characteristics. Rather than introducing another generic fusion operator, the proposed method estimates a bounded conflict coefficient from detached feature similarity, relative feature distance, predictive uncertainty, and class agreement. This coefficient jointly regulates feature reconciliation and cross-branch agreement learning, allowing the interaction between the two representations to adapt to their sample-specific consistency. Experiments on five public HAR datasets show that the proposed framework consistently outperforms independently trained single-branch models and conventional concatenation, mean-fusion, and learned-gating baselines, achieving Macro-F1 scores from 0.9653 to 0.9845. Component ablations and branch-conflict analyses support the role of disagreement-aware reconciliation, while controlled test-time perturbations show that the complete framework generally degrades less than mean fusion. With approximately 0.105M parameters in the UCI-HAR configuration and sub-15-ms inference on a Raspberry Pi 4B, the proposed framework provides a practical trade-off between recognition reliability and deployment cost.

## Dataset
This repository does not include datasets. Please download them from the official sources below and configure the dataset path accordingly.
- **UCI-HAR** dataset is available at https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **MotionSense** dataset is available at https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
- **MHEALTH** dataset is available at https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **WISDM** dataset is available at https://www.cis.fordham.edu/wisdm/dataset.php
- **PAMAP2** dataset is available at https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Codebase Overview


## Citing this Repository
If you use this code in your research, please cite:
```
@article{CATF,
  title   = {Conflict-Aware Reconciliation of Time-Domain and Filterbank Representations for Wearable Human Activity Recognition},
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

