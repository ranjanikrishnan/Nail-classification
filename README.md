
Nail classification
==============================

Convolutional neural network based model to classify manufactured nail images as good or bad (bent).
As the data set is pretty small, it is better to use a pretrained model as the building block for the classification task. For this task, a pretrained vgg-16 model with a customized top is implemented using keras.
-also a simple cnn


Prerequisites 
==============================

requirements.txt file contains all the necessary packages for the whole task

Steps for Running 
==============================
1. Clone the repository:
```
git clone https://github.com/ranjanikrishnan/Nail-classification
```
2. Run the model:
  * This will train the model and show prediction results.
```
python src/model.py
```
3. Run the server:
```
python src/server.py
```
4. For classifying an image:
   Open a new terminal, attach path to the image and run the following.
```
curl -X POST -F image=@<path_to_image> 'http://localhost:5000/predict' 
```

Run using docker:
==============================
1. Build the Dockerfile
```
docker build -t deevio .
```
2. Run the docker container
```
docker run -it deevio
```
3. 





Questions to keep in mind
==============================
* How well does your model perform? How does it compare to the simplest baseline model you can think of?
* How many images are required to build an accurate model?
* Where do you see the main challenge in building a model like the one we asked here?
* What would you do if you had more time to improve the model?
* What problems might occur if this solution would be deployed to a factory that requires
automatic nails quality assurance?




