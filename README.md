# Missing-Person-Detection

Building a product, which will take a photo of a person as an input, and given a set of CCTV footages, to check whether the person was identified in the footage, and return the frames and timestamp of that person, if present. The other use case is identification of a wanted person in the CCTV footage.

The main python script to run is flask_app.py, which has an explicit call to all the functions fron the Intelligent_Vision.py


PS: The following project was carried out in a group of 3, as a part of the Academic Capstone Project and was completed in 6 weeks, simultaneously along with the regular academics.

Steps Involved:

1. Face Detection: Used multi-task cascaded Convolutional Network
2. Feature Extraction(Vector Embeddings): Used pre-trained VGG Face (Transfer Learning)
3. Dimensionality Reduction: Explored the possibility of reducing the embedded vectors using PCA, t-SNE, and the combination of both
4. Clustering: Performed X-means clustering (similar to k-means, here 'k' is dynamically estimated by comparing the BICs.
5. Image Matching: Achieved matching by Spatial Euclidean Distance calculation between the embedded vectors of the face in the probe photo                    and of all the faces captured from the video.
6. Model Deployment: Model was deployed on Flask, the API returns the top 3 photos matching, along with the timestamp they were found in the video.



