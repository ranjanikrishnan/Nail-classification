
Nail classification
==============================

Convolutional neural network based model to classify manufactured nail images as good or bad (bent).
As the data set is small, transfer learning is a highly recommended method for the task. A pretrained model is used 
as the building block for the classification task. For this task, a pretrained vgg-16 model with a customized top 
is implemented using keras.
A simple CNN model is also implemented to establish a baseline. 


Prerequisites 
==============================

The requirements.txt file contains all the necessary packages for the whole task

Steps for Running 
==============================
- Clone the repository:
```
git clone https://github.com/ranjanikrishnan/Nail-classification
```
- Copy the nail images to /data folder.

- Run the model:
  * This will train the model and show prediction results.
```
python src/model.py
```
- Run the server:
```
python src/server.py
```
- For classifying an image:
   Open a new terminal, copy url to the image and run the following for vgg16 model prediction.
```
 curl http://localhost:5000/predict?image_url=<url-to-nail-image>
```
- For predicting using baseline model:
```
 curl http://localhost:5000/baseline/predict?image_url=<url-to-nail-image>
```


Run using docker:
==============================
- Build the Dockerfile
```
docker build -t nail-classifier .
```
-  Run the docker container
```
docker run -it nail-classifier
```
- For classifying an image:
   Open a new terminal, copy url to the image and run the following for vgg16 model prediction.
```
 curl http://localhost:5000/predict?image_url=<url-to-nail-image>
```
- For predicting using baseline model:
```
 curl http://localhost:5000/baseline/predict?image_url=<url-to-nail-image>
```





Questions to keep in mind
==============================
* How well does your model perform? How does it compare to the simplest baseline model
 you can think of?
  - The simple baseline model got a validation accuracy of 66.7% after training for 30 epochs whereas using 
  the VGG-16 model for only 12 epochs gave a validation accuracy of 83.3 %.
 
* How many images are required to build an accurate model?
  - To build an accurate model at least 1k-5k images would be needed. Data augmentation steps like horizontal/vertical
  flipping, rotation, zoom could prove to be useful.

* Where do you see the main challenge in building a model like the one we asked here?
  - Choosing the right hyperparameters like learning rate, optimization algorithms in the network architecture, so as 
  to avoid overfitting problems.  

* What would you do if you had more time to improve the model?
  - The existing model could've been improved by adding a cropping routine (using PIL or openCV) to reduce further 
    unnecessary background. In essence, it would capture only the target entity and crop around it. This pre-processing 
    step could significantly increase the performance. 

* What problems might occur if this solution would be deployed to a factory that requires
automatic nails quality assurance?
  - The procesing speed could be one main problem when deployed in a factory. Also, proper light needs to be available
  in the image for the classification task to perform.   




