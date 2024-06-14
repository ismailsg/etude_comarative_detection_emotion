import cv2
import os
import shutil
from deepface import DeepFace
from fer import FER
from rmn import RMN
#pip install tensorflow==2.12
from feat import Detector

import matplotlib.pyplot as plt
main_folder = 'emotions/'

def make_face_detection(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw rectangles around the detected faces
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Crop the region of interest (ROI) containing the face
        face_roi = image[y:y+h, x:x+w]

        return face_roi

def detect_emotion_with_deepface(image_path):
    frame = cv2.imread(image_path)
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    if result:
        emotion = result[0]['emotion']
        dominant_emotion = max(emotion, key=emotion.get)
        print("deep",dominant_emotion)

        return dominant_emotion
    else:
        return "No Face Detected"

def detect_emotion_with_fer(image_path):
    frame = cv2.imread(image_path)
    detector = FER()
    result = detector.detect_emotions(frame)

    if result:
        dominant_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
        print("fer",dominant_emotion)
        return dominant_emotion
    else:
        return "No Face Detected"
    

def detect_emotion_with_feat(image_path):
    
    detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

    single_face_prediction = detector.detect_image(image_path)

    emotions_dict = single_face_prediction.emotions.to_dict()
# Extraire la clé (émotion) avec la valeur la plus grande
    max_emotion = max(emotions_dict, key=lambda x: list(emotions_dict[x].values())[0])
    if max_emotion == "anger": return "angry"
    if max_emotion == "happiness": return "happy"
    if max_emotion == "sadness": return "sad"
    print("feat: ",max_emotion)
    return max_emotion  

def detect_emotion_with_rmn(image_path):
    frame = cv2.imread(image_path)
    m = RMN()
    results = m.detect_emotion_for_single_frame(frame)
    
    if results:
        results = results[0]['emo_label']
        print("RMN",results)
        return results
    else:
        return "No Face Detected"

def main():
    data_dir = "/home/ismail/Documents/test2"
    emotions = ["anger","happy", "sad", "surprise", "disgust", "fear", "neutral"]
    #emotions = ["sad"]

    results = {emotion: {method: [] for method in ['deepface', 'fer', 'feat', 'rmn']} for emotion in emotions}
    #results = {emotion: {method: [] for method in ['feat']} for emotion in emotions}

    # Parcourir chaque dossier d'émotion
    for emotion in emotions:
        folder_path = os.path.join(data_dir, emotion)
        
        # Vérifier si le dossier existe
        if os.path.exists(folder_path):
            # Parcourir chaque image dans le dossier
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                print(img_path)
                
                # Vérifier si l'image existe
                if os.path.isfile(img_path):
                    # Faire des prédictions avec chaque méthode de détection d'émotion
                    results[emotion]['deepface'].append(detect_emotion_with_deepface(img_path))
                    results[emotion]['fer'].append(detect_emotion_with_fer(img_path))
                    results[emotion]['feat'].append(detect_emotion_with_feat(img_path))
                    results[emotion]['rmn'].append(detect_emotion_with_rmn(img_path))
                else:
                    print(f"Warning: Image {img_path} not found.") 


    # Calculer l'accuracy pour chaque méthode de détection d'émotion
    #accuracies = {method: {} for method in ['deepface', 'fer', 'feat', 'rmn']}
    accuracies = {method: {} for method in ['feat']}
    #for method in ['feat']:
    for method in ['deepface', 'fer', 'feat', 'rmn']:
        for emotion in emotions:
            correct_predictions = sum([1 for true_emotion, predicted_emotion in zip([emotion]*len(results[emotion][method]), results[emotion][method]) if true_emotion == predicted_emotion])
            accuracy = correct_predictions / len(results[emotion][method]) * 100
            accuracies[method][emotion] = accuracy

    # Afficher les accuracies
    for method, accuracy in accuracies.items():
        print(f"\nAccuracy for {method}:")
        for emotion, acc in accuracy.items():
            print(f"{emotion}: {acc:.2f}%")    

    

if __name__ == "__main__":
    main()
