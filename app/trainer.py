from keras import layers
from keras import models
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
import json

# Model definition - convolutional network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Loading CSVs and adding user data to training set
traindf=pd.read_csv("train.csv")
user_df=pd.read_csv("user_data.csv")
testdf=pd.read_csv("test.csv")
traindf = pd.concat([traindf, user_df])

#Initialize training features
all_features = traindf.drop("label",axis=1) #copy of the traindf without label "feature"
Targeted_feature = traindf["label"]

train_images, test_images, train_labels, test_labels = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)

# data reshaping for a convolutional neural net
train_images = train_images.to_numpy()
train_labels = train_labels.to_numpy()
test_images = test_images.to_numpy()
test_labels = test_labels.to_numpy()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


t1 = time.time()
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
dt = time.time()-t1
obj = {
    "Trained At": datetime.now().strftime("%X %x"),
    "Time to Train": str(int(dt))+" seconds",
    "User Generated Rows": user_df.shape[0],
    "Rows from MNIST": train_images.shape[0] - user_df.shape[0],
    "Total Training Rows": train_images.shape[0],
    "Testing Rows": test_images.shape[0],
    "Model Accuracy": str(round(history.history["accuracy"][-1], 2)*100)+"%",
}
m =  open("./modelstats.txt", "w")
m.write(json.dumps(obj))
m.close()

#save model 
fname = "./digit_model.sav"
pickle.dump(model, open(fname, "wb"))