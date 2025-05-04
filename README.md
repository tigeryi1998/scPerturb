# scPerturb

This is the Github repo of the single cell Perturbation scPerturb for the NeurIPS 2023 Kaggle Competition. https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview

You should have a folder called "data" in the main directory. You should download the data files in the "data" folder, especially the de_train.parquet main data file. The data is very big and it is available to download at https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data

The report folder contains the paper pdf. The presentation folder contains the presentation pptx. 

In order to run everything please follow the scripts. Alternatively you can also use miniconda to install the virtual enviroments. 

```bash
python3 -m venv myenv # create a new environment 
source myenv/bin/activate # activate hte myenv
pip3 install -r requirements.txt # install the dependency
```

The main script is called the main.py. Currently it supports 2 models to train: MLP multi layer preceptron, and Transformer. You need to change the main function between the MLP and Transformer to build the different models. Then you can run 

```bash
python3 main.py # run the main script to build everything
```

The main function will first load and process the data in process.py

Then it will build the dataloader as the model input. 

The model architecture is defined in model.py 

It will then call the train.py to train the model. 

After training the model, it will call submit_mlp or submit_transformer to make inference and generate predictions. 

