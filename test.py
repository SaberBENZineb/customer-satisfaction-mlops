import mlflow

# Search for registered models
deployed_models = mlflow.search_registered_models()
print(deployed_models)