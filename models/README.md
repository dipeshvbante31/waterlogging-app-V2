# models/

Created by `python scripts/train_model.py`. A model is installed only if training succeeded:

| file | purpose |
|---|---|
| `waterlogging_model.joblib` | fitted scikit-learn pipeline (preprocessing + classifier) |
| `model_metadata.json` | feature schema, target definition, coverage area, data lineage, SHA-256 of the joblib file, versions |
| `model_metrics.json` | evaluation metrics from the real run (read by the Model Performance page) |
| `archive/<version>/` | a copy of every trained version, kept automatically |

The web app refuses to load a model whose checksum, feature schema or scikit-learn version does not match its metadata and then
uses the labelled fallback instead. Only ever place files here that you trained yourself or fully trust: joblib files are
Python pickles and loading an untrusted one can execute code.
