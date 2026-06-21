# hands on project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Hands on project to show data science capabiities.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── app                <- FastAPI application for model serving
│   └── main.py        <- Main application entrypoint
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── latex_beamer       <- LaTeX Beamer presentation files
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         hop and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── src                <- Additional source code for training and prediction
│   ├── predict        <- Code to run model inference with trained models
│   ├── train          <- Code to train models
│   └── utils          <- Utility functions
│
├── tests              <- Test cases for the project
│
└── hop   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes hop a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Setup Environment

1. Create a virtual environment:
   ```bash
   make create_environment
   ```
2. Activate the environment:
   ```bash
   source env/bin/activate
   ```
3. Install the dependencies:
   ```bash
   make requirements
   ```

## Running the API

You can start the FastAPI server to serve the models using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

Once the API is running, you can access the automatic interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). 

The following main endpoints are available:
- `GET /experiments/best_runs`: Retrieves the best performing runs from MLflow.
- `POST /train`: Upload a CSV file to trigger the training pipeline.
- `POST /predict`: Upload a CSV file and specify MLflow experiment parameters to run predictions.

## Experiment Tracking

This project uses MLflow to track experiments and Optuna for hyperparameter tuning. To view the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then navigate to `http://127.0.0.1:5000` in your browser.

## Running Tests

To run the pytest suite, execute:
```bash
make test
```

## Useful Commands

- `make lint`: Run flake8, isort, and black checks.
- `make format`: Auto-format code using isort and black.
- `make clean`: Remove compiled python files and caches.

--------

