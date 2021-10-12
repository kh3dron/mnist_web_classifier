# MNIST Web Classifier

A web page for interacting with a Neural Network trained off of the MNIST digit data set.

I had the idea for this project after completing the Kaggle competition for the "Digit Recognizer" challenge. I got my model up to 96% accuracy off of the testing data. I then wanted a way to deploy my model somehow, so I wrote this simple web page that lets you "write" a digit and feed it to the classifier. 

Technologies used in this application:

- Python AI/ML tools to train the classifier. Training done using Pandas, matplotlib, and scikit-learn. 
- Javascript: to create the canvas page, and format the data from the handdrawn canvas into the pixel formatting that the NN can handle. 
- Flask, a python web server framework that let me connect the model to the canvas webpage. This let me put the neural network prediction methon into an API call. 
- Docker: to containerize all the dependencies for the web server and the neural network.
- AWS Elastic Contianer Registry: to host this container.  
- AWS Elastic Container Service: to deploy the dockerized application to a site, which you can check out here: http://34.213.195.239:5000/

The predictions from the site are far less than 96% accurate. I think this is due to handwriting from  a cursour or mouse is very different than handwriting scanned from a written page. Predictions off of my mouse-writing are around 70% accurate - your milage may vary. 



---

TODO:

- add DNS to the container
- https://stackoverflow.com/questions/53767231/how-can-you-launch-ecs-fargate-containers-having-a-public-dns
- https://aws.amazon.com/blogs/compute/microservice-delivery-with-amazon-ecs-and-application-load-balancers/


- here's what we need:
- https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/using-domain-names-with-elb.html
- https://aws.amazon.com/getting-started/hands-on/deploy-docker-containers/