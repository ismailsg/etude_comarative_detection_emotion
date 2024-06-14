import joblib

# Load the saved pipeline
pipeline_file = open("emotion_classifier_pipe_lr_03_june_2021.pkl", "rb")
loaded_pipeline = joblib.load(pipeline_file)
pipeline_file.close()

# Now you can use the loaded pipeline for predictions
ex1 = "stop talking to me "
prediction = loaded_pipeline.predict([ex1])
prediction_prob = loaded_pipeline.predict_proba([ex1])
classes = loaded_pipeline.classes_

print("Predicted emotion:", prediction[0])
print("Prediction probabilities:", prediction_prob[0])
print("Available classes:", classes)
