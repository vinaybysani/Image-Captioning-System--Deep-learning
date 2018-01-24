## Abstract:
Automatic Image captioning is one of the most recent problem that caught interest of computer vision community and Natural Language Processing communities alike. This is major part of scene understanding in computer vision. Not one only do we have to recognize objects in the image, but we have to find relationships in natural language. This project addresses the problem by using a deep neural network model. The model would make use of Convolution neural networks to read image data and Recurrent neural networks for learning sentences/captions for image. There are various open datasets like Flickr8K, Flickr30K, MSCOCO that can be used to train the model.
	
## Why it is interesting: 
This is a generation obsessed with social networks, images and selfies. As per a recent InfoTrends forecast, in 2017 alone, 1.2 Trillion photographs digital will be taken. We intend to create a deep model to caption images. We would train on Flickr8k dataset of 8000 labelled images, and when we have a new photo, the model should generate a natural language description of the photograph. Majority of photographs in internet are not captioned. Auto generating them using deep learning would find its application in sentence based image search.

## Prior work:
It’s amazing how far image recognition has come far in the last few years. One of the most notable mention is the ImageNet project, where they crowdsourced millions of labelled images and trained models for the last decade to recognize objects in the image. Since 2010, the annual ImageNet Large Scale Visual Recognition Challenge (ILSCRC) holds a competition each year, to compete for highest accuracy on various visual recognition tasks. Now the deep CNN networks have more accuracy than humans in recognition. However Captioning images could be much challenging task, since it involves object recognition and finding relationships among them. This has been impossible until recently, owing to huge improvement in computational power. Even though there are multiple researchers working on the same problem, there are two teams that stood out with their algorithms. One from Google, and the other from Stanford University. Google released a paper “Show and Tell: A Neural Image Caption Generator” [1] in 2014. Their model is trained to maximize the likelihood of the target description sentence, given the image. The model is trained on various datasets like Flickr30K, SBU, MSCOCO and has achieved human level performance in generating captions. When google first released a paper in 2014, the system used the “Inception V1” image classification model which achieved 89.6% accuracy. The latest release in 2016 used “Inception V3” model, which achieves 93.9% accuracy. Before Google, image captioning was possible using DistBelief sotware framework. Then Google released TensorFlow implementation, which makes use of GPU power and compared to earlier implementations, the training time is reduced by a factor of 4. The other team that achieved well in solving the problem is from Stanford University - Fei-Fei Li and Andrej Karpathy. Their paper “Deep Visual-Semantic Alignments for Generating Image Descriptions” [2] which released in 2015, leverages images and descriptions to learn about multi-modal correspondences between language and visual data. They’ve used RNN and CNN to achieve the task. Their implementation is batched. It makes use of Torch library, which runs on GPU and supports CNN finetuning, which increased training speed by huge factor

![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/1.png)
## Dataset:
There are three popular open datasets that can be used to train deep network model. These are often referred as benchmark collection of image caption datasets to train and evaluate a model’s accuracy. These datasets have complex day to day scenes with various objects. MSCOCO is the largest dataset of all with 328k images in its corpus.
1.	Flickr8K
2.	Flickr30K
3.	MSCOCO – The Microsoft Common Objects in Context
MSOCO dataset is widely used currently. This corpus has 91 common object categories, 82 of which have 5000+ labelled instances. There are few other datasets that caught scientific community before MSOCO, for ex: ImageNet, was one dataset that caught attention of everybody in scientific community. MSCOCO has less categories than ImageNet, however more instances in each category, giving us more freedom to train the network well. This helps in having detailed model of objects and precise 2D localization. Due to limited computational resources, we are using Flickr8K during initial phases of model development and scale it to MSCOCO. Flickr8K has 8000 images from flickr. Flickr8k is distributed by University of Illinois at Urbana–Champaign(UIUC). Training, validation and testing split would be (6000, 1000, 1000). Each image in dataset has 5 reference captions. 
Preprocessing:
We start preprocessing Flickr8k by converting texts to lowercase and any non-alphanumeric are discarded. All words that doesn’t repeat more than some threshold number, would be ignored while building a vocabulary. Building vocabulary is achieved using NLTK toolkit for lemmatization, stop words and TFIDF. It has been observed that vocabulary size to about three thousand words. Dropout can be used to avoid any overfitting. All images are resized to one common size (224*224) before training.










## Network/Model:
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/2.png)
 To solve the problem of image captioning, we need two models.
1.	Image model
2.	Language model
One of the reason why the problem is difficult, is because we need two models and seamless integration between them. The network can be viewed as a combination of encoder and decoder. Encoder would be a convolutional neural network(CNN). Image is processed by CNN layer and features are extracted. End of CNN layer is connected to a Long short-term memory(LSTM) networks, a special kind of Recurrent Neural Network(RNN). LSTM’s are capable of learning long term dependencies. Our application doesn’t make use of spark and hence does not have any sort of data analytic representation. Most of the core work is focused on building efficient deep neural network.
The model is built using Keras, a deep learning library in python. Keras is a high-level library that is above Theano and Tensorflow. The API is very simple and makes use of Tensorflow backend. Keras is minimalistic in code, and is growing fast in the community. Keras is well documented and can run on GPU.

![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/3.png)
We designed the above model and trained it. The first subsection of layers is the image model which takes image as input and feeds to CNN layer. The CNN gets trained on images and features are extracted. The second subsection of layers is language model(LSTM). Here the input is captions from dataset. Since the words dictionary could be huge, using one hot encoding could be extremely expensive. The embedding model outputs a vector of dimension (1*256). LSTM network is trained to predict next word, given previous word and image feature vector.
#### 1.	Image model
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/4.png)
 
We used transfer learning to use one popular model for CNN implementation. Transfer learning is a major topic in machine learning that involves storing knowledge from one model and applying to another problem. The reason we use pretrained network is because, CNN models are difficult to train from scratch and it could be very computationally expensive that it takes several hours on GPU. In the scientific community, it is very common to use pretrained model on larger dataset and then using the model as a feature extractor. 

After rigorous research, we plan to use a pretrained CNN model called “VGG16” or “OxfordNet”. This model has won the 2014 ImageNet challenge - ImageNet Large Scale Visual Recognition Competition (ILSVRC) in localization and classification tracks. The model is found to be easily generalizable to other datasets, with very good results. The original VGG model is written on Caffe, however we use Keras implementation [3]. Details about the architecture can be found in the paper – “Very Deep Convolutional Networks for Large-Scale Image Recognition” [4]. The model architecture is portrayed in the below figure. The pooling layers are used as down samplers, and the initial pooling layers detect details like edges, lines etc. As the network comes to end, the pooling layers detect abstract features and shapes. The input would be images of fixed size (224*224), with 3 channels. The last SoftMax layer gives the probability distribution for object recognition. We removed the SoftMax layer, and the updated model generates a weight vector of dimension 4096. 
 
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/5.png)
We now use the dense vector to change the output to fixed embedding size which is 128 in our case. Then we use the repeat vector to repeat it as one vector for each word since we will be merging this output with Language model output.

#### 2.	Language model
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/6.png)
 
The output of image model acts as input to language model. To understand the captions under the images, we use a Recurrent Neural Networks(RNN) to solve the problem. Vanilla neural networks work in several scenarios, but the API is very limited. RNN’s output is determined not just by the input, but with series of inputs.  In practice, we use Long Short-term memory (LSTM), which is a variation of RNN. LSTM works better and has powerful update equation and backpropagation. LSTM is a language model and decoder trained on feature vectors.
LSTM’s had phenomenal influence and success in different problems like language modelling, speech recognition, translation etc. The power of RNN/LSTM’s has been well documented by Andrej Karpathy in this blog post [5]. Consider the problem of guessing the next word. The prediction depends on last few words, to understand the context and RNN’s does that with memory. Before the LSTM’s were proposed by Hochreiter & Schmidhuber [6], long term dependencies were a problem. Having a memory of information isn’t a problem anymore.
 
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/7.png)
LSTM has three inputs – the input from current time step, output from previous time step and memory of previous unit. LSTM’s have ability to add/remove information via gates, which consists of sigmoid neural nets and pointwise multiplication operations. LSTMs’ are a major reason why image captioning model is successful. LSTM picks part of image and maps to the appropriate word in the caption.
Here, we create an embedding layer to get a vector representation for each word in the caption. Then we input the output vector to LSTM for the model to learn the neighboring words for each word. We then convert the LSTM output to fixed dimension using dense layer. In this case we will use TimeDistributed because it’s a 3D tensor.
Now, we combine the outputs from both Language Model and Image model, and input the vector to LSTM. LSTM learns the different captions for that image in training phase. We convert the LSTM output to the size of vocabulary size using the dense layer and activate the model using activation method. 
In testing phase, LSTM predict the captions for the image. LSTM predicts next word for the given image with the partial caption available at that stage.   

## Evaluation and Results:
We intend to use “BLEU metric”, which is widely used by machine translation experts. BLEU is a precision based and compares system output to human translation. BLEU breaks captions to words and we compare these with human translations. It’s like a bag of words similarity. There is an inherent problem with this metric system, given the fact that different people use different vocabulary to represent a similar meaning, sometimes with no overlap. This is also due to the fact, that language model has limited vocabulary and training set. There are other evaluations strategies like METEOR metric. This metric was designed to fix shortcomings of BLEU. It addresses issues in BLUE like stemming, synonym and paraphrase matching etc. METEOR has extended support other than English language. We can also make use of brute force human evaluation by taking random output samples from model and evaluate. When the model is evaluated on the test dataset, the BLEU score is observed as shown below. Upon running the model for 50 epochs, we’ve attained BLEU score ~ 0.51 which is pretty good given the limited training dataset. We could improve the score when tried with more epochs, however running 25 epochs on a general-purpose laptop with NVIDIA GTX 960M GPU has taken 30 hours to train the model. This is not a model’s limitation, but the nature of problem is such that image and language models are computationally expensive.
 

![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/8.png)
## Limitations and Conclusion: 
There are few limitations of this model. This doesn’t do detailed object recognition. Originality is a problem too. The model tries to frame sentences based on words found in training set alone. The captions are generated only based on the captions it gets trained on and the features that are extracted from the images. Even MSCOCO dataset isn’t sufficient to generalize the solution to any new image unseen by the network. In MSCOCO dataset, there are a lot of giraffe’s and most of the giraffes are close to trees in their natural habitat. If we input an image with a tree alone, the model outputs “A giraffe standing next to tree”, since the model has associated giraffe and tree to be often together – based on the dataset it is provided with.
  
![alt text](https://github.com/vinaybysani/Image-Captioning-System--Deep-learning/blob/master/Images/9.jpg)
The rise of general artificial intelligence relies on understanding and integrating cross domains to create a map of the world as humans do. The image captioning generator is a great framework to combine language and image models for a single task. The visual features are mapped to semantics and this is close to how animals and humans perceive the world 

## References:
1.	https://arxiv.org/abs/1411.4555
2.	http://cs.stanford.edu/people/karpathy/cvpr2015.pdf
3.	https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 - VGG16
4.	https://arxiv.org/abs/1409.1556
5.	http://karpathy.github.io/2015/05/21/rnn-effectiveness/
6.	http://www.bioinf.jku.at/publications/older/2604.pdf
7.	https://arxiv.org/abs/1405.0312
8.	http://bplumme2.web.engr.illinois.edu/Flickr30kEntities/
9.	http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html 
10.	http://www.image-net.org/
11.	https://en.wikipedia.org/wiki/BLEU
12.	https://blogs.technet.microsoft.com/machinelearning/2014/11/18/rapid-progress-in-automatic-image-captioning/
13.	https://arxiv.org/pdf/1509.04942.pdf
14.	http://www.nltk.org/ 
15.	http://colah.github.io/posts/2015-08-Understanding-LSTMs/
16.	http://cs.stanford.edu/people/karpathy/deepimagesent/


## Need to install the following
h5py, Keras, numpy==1.11.2, tensorflow==1.1.0, pandas==0.16.2, Pillow
