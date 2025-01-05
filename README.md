
# Youtube Comment Sentiment Analyzer

A tool for real-time sentiment analysis of YouTube comments, capable of predicting positive, neutral, and negative sentiments. This project features a Chrome plugin for seamless integration, a high-accuracy machine learning model, and a FastAPI backend deployed on AWS. With interactive visualizations and robust MLOps practices, it delivers a reliable and scalable solution for understanding comment sentiment.


Project Organization
------------
    ├── api                <- FastApi For model inference
    ├── chrome-plugin      <- Frontend of Chrome plugin 
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├──  data_ingestion.py
    │   │   └──  data_preprocessing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── feature_engineering.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model_training.py
    │   │   ├── model_evaluation.py
    │   │   └── model_registry.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Authors

- [@Anmol25](https://www.github.com/Anmol25)

