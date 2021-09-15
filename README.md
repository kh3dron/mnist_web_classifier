#MNIST Web Classifier

A web page for interacting with a Neural Network trained off of the MNIST digit data set. 

I had the idea for this project after completing the Kaggle competition for the "Digit Recognizer" challenge. I got my model up to 96% accuracy off of the testing data. I then wanted a way to deploy my model somehow, so I wrote this simple web page that lets you "write" a digit and feed it to the classifier. 

Here's the steps I went through in creating this project:

1. Create an HTML page that allows users to draw a number (using html and JavaScript canvas)
2. Read from the Canvas into the format of the digit dataset: break the canvas into 28x28 pixels, computing the average "darkness" of each pixel, and storing that in an array
3. Export the Neural Network from my Kaggle notebook to load into other python programs without the need to retrain the model (done with pickle)
4. Configure a Python Flask server to host the static HTML from Steps 1 and 2
5. Wrap the Neural Network from Step 3 in an API, to be accessible in the Flask server
6. Combine steps 2 and 5 with some JavaScript buttons for a click-to-classify button, that returns the model's prediction of the drawn data on the webpage. 

#todo 
7. Host entire flask site on AWS (Beanstalk)
8. Better web design 


The predictions from the site are far less than 96% accurate. I think this is due to handwriting from  a cursour or mouse is very different than handwriting scanned from a written page. Predictions off of my mouse-writing are around 70% accurate - your milage may vary. 