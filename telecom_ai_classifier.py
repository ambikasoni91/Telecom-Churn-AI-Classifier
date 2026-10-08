import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. डेटा तैयार करना
X = np.array([
    [-30, 5],   
    [-45, 12],  
    [-85, 50],  
    [-90, 65],  
    [-40, 8]    
])

Y = np.array([1, 1, 0, 0, 1])

# 2. AI मॉडल लोड और ट्रेन करना
classifier = LogisticRegression()
classifier.fit(X, Y)

# 3. नए सिग्नल पर भविष्यवाणी (Prediction)
new_signal = np.array([[-75, 40]])
prediction = classifier.predict(new_signal)
probability = classifier.predict_proba(new_signal)

# 4. आउटपुट दिखाना (सरल तरीके से बिना किसी ब्रैकेट गड़बड़ के)
print("--- AI Signal Classifier Result ---")
if prediction[0] == 1:
    print("AI ka faisla: Yeh ek GOOD Signal hai!")
else:
    print("AI ka faisla: Yeh ek BAD Signal hai!")

print("Bad hone ki sambhavna:", round(probability[0][0] * 100, 2), "%")
print("Good hone ki sambhavna:", round(probability[0][1] * 100, 2), "%")