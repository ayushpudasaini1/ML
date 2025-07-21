import joblib

model = joblib.load('model/iris_rf.pkl')

def predict(features):
    return model.predict(features)