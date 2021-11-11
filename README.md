Interpreting Models Interpreting Brain Dynamics 
=====================
This repository provides code to replicate the findings in:
**Interpreting Models Interpreting Brain Dynamics** by<br/>
*M. M. Rahman, U. Mahmood, N. Lewis, H. Gazula, A. Fedorov, Z. Fu, V. D. Calhoun & S. M. Plis*.

![main_diagram](https://user-images.githubusercontent.com/45178290/141242644-6934f195-67ea-4656-bb02-2cdf6ba078bc.png)

# <img src="https://github.com/mmrahman21/model_introspection/tree/master/doc/figures/main_diagram.png" width="700" height="500">

[comment]: # This is a comment

## Overview
Brain dynamics are highly complex and yet hold the key to understanding brain function and dysfunction. The dynamics captured by resting-state functional magnetic resonance imaging data are noisy, high-dimensional, and not readily interpretable. The typical approach of reducing this data to low-dimensional features and focusing on the most predictive features comes with strong assumptions and can miss essential aspects of the underlying dynamics. In contrast, introspection of discriminatively trained deep learning models may uncover disorder-relevant elements of the signal at the level of individual time points and spatial locations. Yet, the difficulty of reliable training on high-dimensional low sample size datasets and the unclear relevance of the resulting predictive markers prevent the widespread use of deep learning in functional neuroimaging. In this work, we introduce a deep learning framework to learn from high-dimensional dynamical data while maintaining stable, ecologically valid interpretations. Results successfully demonstrate that the proposed framework enables learning the dynamics of resting-state fMRI directly from small data and capturing compact, stable interpretations of features predictive of function and dysfunction. 


## Data
+ Function Biomedical Informatics Research Network (FBIRN)
+ Open Access Series of Imaging Studies (OASIS)
+ Autism Brain Imaging Data Exchange (ABIDE) 
+ Human Connectome Project (HCP) (For pre-training only)


#### Experiments
In the  scripts folder, all of the scripts required to build and evaluate **Standard Machine Learning (SML) ** models, pretrain **whole MILC**, generate **post-hoc explanations** evaluate explanations using **RAR** method are provided:

- *run_sml.py*: is used to build and evaluate standard machine learning models from the raw data (ICA time-courses).

- *run_milc_pretraining.py*: is used to pretrain **whole MILC** model leveraging only healthy (control) subjects from HCP.

- *run_downstream_model*: is used to train and evaluate downstream models for different disorders.

- *run_downstream_sal_compute.py*: is used to compute saliency maps per (model-disorder) pair.

- *run_rar_fresh.py*: is used to build new **SML** models using **RAR** and **SVM** with top (5% - 30%) features guided by computed **Saliency**.

- *run_random_rar.py*: is used to build **SML** models using random (5% - 30%)feature selection and **SVM** models.

#### How to run:

```
python run_sml.py dataset_id
```
**dataset_id** options:
- FBIRN: 0
- OASIS: 2
- ABIDE: 3

