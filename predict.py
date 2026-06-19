import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Sentiment Analysis System Ready!")

while True:
    text = input("Enter text (or type 'exit'): ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    print("Sentiment:", prediction[0])