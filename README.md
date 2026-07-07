# MasterThesis - identifying spatiotemporal patterns of European heatwaves using deep learning 

This project investigates heatwave propagation using spatiotemporal deep learning models. 
The goal is to predict and analyze the evolution of heatwaves using both binary labels 
(`is_heatwave`) and continuous Complex Network (CN) coefficients (e.g., clustering coefficient `CC`) based on the dataset from 1990-2020.

The pipeline includes:
- data preprocessing and alignment
- statistical analysis 
- monthly CN distributions of the coefficients 
- feature engineering using network coefficients (feature ablation) - ablation_results.csv 
- sequence generation for temporal modeling
- ConvLSTM-based training
- SpatioTemporal Transformer training 
- evaluation for both classification and regression tasks
- trajectory tracking analysis 

# FILES

- requirements.txt and requirements_snellius.txt


- final_experiments/ all the files use as a directory IN/OUT Snellius - FileZilla explorer, all the files in this folder where used for the final experiment runs 

    - all_results/
     summary of results for each model (Transformer or if not stated it is ConvLSTM), for HW, CC, DC, BC prediction and multiple days ahead (if not stated it is +1 day ahead)

    - cn_analysis_outputs/ cn distribution stats for each case study year, monthly average of CN variables (map distribution)

    - notebooks/cn_analysis.ipynb - notebook to generate cn distribution stats for each variable 
    - notebooks/statistical_test.ipynb - to see the statistical significance of results

    - inference_only.py - to generate inference
    - run_batch_experiments*.py - all the files to run experiments for chosen regions and coefficients (CC, DC, BC, HW), the files are separate for transformer, and for multiple days ahead (_3d (+ 3 days) file works for 7 days too once the variable is changed)
    - trajectory_analysis_cc.py - used for generating CC trajectory from the CC predictions (only +1 day ahead since those were usable)

- dataset_creation/
    - merge-datasets-ERA5-CN.ipynb - merges CN and ERA5 dataset together

- NUTS_regions - downloaded files for the prediction maps 

- exploration_pathway_detection/ - exploratory phase, eventually it was not implemented, because the values were not significant, open for further exploration 

- exploration/data-exploration.ipynb
  + initial data exploration, see distribution of the coefficients, data cleaning, fill missing values, inspect min/max values
  + align the is_heatwave dataset with other CN coefficients (CC, BC, DC, ID, OD)
  + decide on the data split and make yearly sequences for the model ConvLSTM
    
  
- exploration/bc_vs_hw.ipynb
  + compare Betweenness Coefficients vs is_heatwave to see if BC is a relevant predictor
  
- exploration/model-coefficients.ipynb
  This notebook contains an earlier attempt at building the pipeline.
  - Initial exploration of data and model setup
  - Testing different preprocessing strategies
  - Prototyping ConvLSTM training
  - Debugging normalization and alignment issues

  Differences from final version:
  - Less modular structure
  - Limited support for multiple targets
  - No unified evaluation pipeline
  - Incomplete handling of regression targets (e.g., scaling issues with ID/OD)

  This notebook is kept for reference to document the development process and design decisions.
  
- initial_experiment_pipelines/run-model-pipeline.ipynb
This notebook contains the end-to-end pipeline used for initial experiments.

  + **Data preparation**
  - Loading E-OBS and network coefficient datasets with spatial and temporal alignment 
  - Region selection (Iberia / Mediterranean Basin)

  + **Feature selection**
  - Flexible selection of input coefficients (`CC`, `BC`, `DC`, `ID`, `OD`, `is_heatwave`)
  - Support for ablation experiments 

  + **Target definition**
  - Binary classification: `is_heatwave`
  - Regression: `CC`, `OD`, etc.
  - Temporal shift (`t → t+1`) for prediction

  + **Sequence generation**
  - Creation of sliding windows per year
  - Input shape: `(N, seq_len, channels, lat, lon)`

  + **Normalization**
  - Per-channel normalization for input features
  - Target normalization for regression tasks

  + **Model**
  - ConvLSTM architecture
  - Configurable for classification and regression

  + **Training**
  - BCEWithLogitsLoss for binary task
  - MSE / SmoothL1 for regression
  - Class imbalance handling via `pos_weight`

  + **Evaluation**
  - Classification metrics: F1, IoU, Precision, Recall, PR-AUC, ROC-AUC
  - Regression metrics: RMSE, MAE, R², Pearson correlation

  + **Ablation experiments**
  - Systematic comparison of different input combinations
  - Results saved to CSV

  + **Visualization**
  - Heatmaps of predictions vs ground truth
  - Error maps
  - Spatial comparison of `is_heatwave` and network coefficients


multi_head_training/
    - train_multihead_.py - exploratory multihead output head - not used in the final experimentation

    - train_threehead_.py - exploratory threehead output head - not used in the final experimentation 

    - train_onehead_.py - onehead output head - used in the final experimentation 
