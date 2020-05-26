from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self, feature_count, features, ground_truth):
        self.feature_count = feature_count
        self.model_params = {
            "hidden_layer_sizes": (feature_count, 100, 2),
            "activation": 'relu',
            'solver': 'adam',
            "max_iter": 10,
            "verbose": True
        }
        self.model = MLPClassifier(self.model_params)
        self.features = features
        self.ground_truth = ground_truth

    def run(self):
        self.model.fit(self.features, self.ground_truth)

    def predict(self, features):
        return self.model.predict(features)
