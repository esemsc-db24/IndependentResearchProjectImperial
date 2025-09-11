**Predicting Individual Physiological Responses to Pollution Using Transformer-Based Time-Series Models**

This repository consists of all the different steps useful to get from raw unprocessed data to final model predictions. 

**Datapreprocessing.ipynb**: In the data_engineering folder. This is the notebook used on the Imperial INHALE portal to load, preprocess and save the clean set of data

**modules**: Includes all the different modules use throughout the independent research project. 
- data_loader_24h_.py was useful to load the data in batches. 
- model_.pyincludes the hybrid model. With both the Transformer model and the Variational one
- train_24h_.py is used to train the model and manage the flow between encoder, variational latent space (with generator and discriminator) and decoder
- utils_24h_.py was used to store all the different functions that will be utilised in the main notebook. Mostly to either extract latent vectors, make predictions, or scan individuals risk to pollutions. 
- visuals.py instead was useful to plot models prediction either by minute or hour and to see they perform vs ground truth.

Below you will see the abstract, project aim and how to run main_notebook_rollout

**ABSTRACT**:
Air pollution remains a major global health and environmental concern, contributing to an estimated seven million deaths annually because of the combined effects of outdoor and household exposure (WHO, 2025). While pollution levels are projected to decline, the ongoing impacts of climate change continue to pose serious risks. Simultaneously, advancements in wearable sensor technologies allow for the systematic collection of high-resolution physiological data over long periods of time (Roos & Slavich, 2023).

**PROJECT AIM**:
Develop an identity map linking varying levels of air pollution to individual physiological responses. Such a framework will enable the prediction of health responses to pollution exposure, facilitating early warnings and personalised health recommendations. To achieve this, we propose a two-model approach: an initial general model to capture general population temporal trends, and a personalised one specialised on individual characteristics. Together, these models will enhance the precision of forecasting and contribute to more effective, data-driven health interventions when reacting to a polluted environment. 

**HOW TO RUN**:

1. Clone the repository with

```bash
git clone https://github.com/ese-ada-lovelace-2024/irp-db24.git
```

2. Open the terminal and run

```bash
conda env create -f env_irp.yml
conda activate irpdavideenv
```

3. Open Main_notebook_quick_run_saved_model.ipynb and run all 

