# scPerturb

This repository contains code for the NeurIPS 2023 Kaggle competition on single-cell perturbations: [Link to Kaggle Competition](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview).

## Setup Instructions

Ensure you have a `data` folder in the main directory with the required dataset. Then, set up the environment by following these steps:

1. Create a virtual environment:

    ```bash
    python3 -m venv myenv
    ```

2. Activate the virtual environment:

    ```bash
    source myenv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip3 install -r requirements.txt
    ```

4. (Optional) To deactivate the environment, use:

    ```bash
    deactivate
    ```

## Running the Code

The main script is `main.py`, which supports training both an MLP (Multi-layer Perceptron) and a Transformer model. To switch between them, open `main.py` and modify the `main` function to select the desired model architecture.

### To run the script:

```bash
python3 main.py  # Run the main script to build and train the model
```

### Workflow:
1. Loads and processes data using `process.py`
2. Builds the data loader for model input `dataset.py`
3. Defines model architecture in `model.py`
4. Trains the model with `train.py`
5. Generates predictions using `submit_mlp.py` or `submit_transformer.py`
6. Everything is orchestrated by `main.py`and parameters `config.py`

## Notes
- The dataset is large; ensure sufficient disk space.
- If needed, use Kaggle's API to download data:

    ```bash
    kaggle competitions download -c open-problems-single-cell-perturbations
    ```

## Folder Structure

- **data/**: Contains the large dataset. The primary data file is `de_train.parquet`. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data).
- **report/**: Contains the project paper in PDF format.
- **presentation/**: Contains the project presentation in PPTX format.

```
scPerturb
├── data/
│   ├── de_train.parquet
│   ├── id_map.csv
│   └── sample_submission.csv
├── report/
├── presentation/
├── pictures/
├── requirements.txt  
├── config.py 
├── utils.py              
├── process.py
├── dataset.py
├── model.py
├── train.py
├── submit_mlp.py
├── submit_transformer.py
└── main.py  
```