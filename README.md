# tfhub

TF-Hub is a platform to share machine learning expertise packaged in reusable resources, notably pre-trained modules. This tutorial is organized into two main parts.

# Introduction: Training a text classifier with TF-Hub

We will use a TF-Hub text embedding module to train a simple sentiment classifier with a reasonable baseline accuracy. We will then analyze the predictions to make sure our model is reasonable and propose improvements to increase the accuracy.

# Advanced: Transfer learning analysis

In this section, we will use various TF-Hub modules to compare their effect on the accuracy of the estimator and demonstrate advantages and pitfalls of transfer learning.

# Data

We will try to solve the Large Movie Review Dataset v1.0 task from Mass et al. The dataset consists of IMDB movie reviews labeled by positivity from 1 to 10. The task is to label the reviews as negative or positive.

# Feature columns

TF-Hub provides a feature column that applies a module on the given text feature and passes further the outputs of the module. In this tutorial we will be using the nnlm-en-dim128 module. For the purpose of this tutorial, the most important facts are:

  * The module takes a batch of sentences in a 1-D tensor of strings as input.
  * The module is responsible for preprocessing of sentences (e.g. removal of punctuation and splitting on spaces).
  * The module works with any input (e.g. nnlm-en-dim128 hashes words not present in vocabulary into ~20.000 buckets).
# Further improvements

* Regression on sentiment: we used a classifier to assign each example into a polarity class. But we actually have another categorical feature at our disposal - sentiment. Here classes actually represent a scale and the underlying value (positive/negative) could be well mapped into a continuous range. We could make use of this property by computing a regression (DNN Regressor) instead of a classification (DNN Classifier).
* Larger module: for the purposes of this tutorial we used a small module to restrict the memory use. There are modules with larger vocabularies and larger embedding space that could give additional accuracy points.
* Parameter tuning: we can improve the accuracy by tuning the meta-parameters like the learning rate or the number of steps, especially if we use a different module. A validation set is very important if we want to get any reasonable results, because it is very easy to set-up a model that learns to predict the training data without generalizing well to the test set.
More complex model: we used a module that computes a sentence embedding by embedding each individual word and then combining them with average. One could also use a sequential module (e.g. Universal Sentence Encoder module) to better capture the nature of sentences. Or an ensemble of two or more TF-Hub modules.
* Regularization: to prevent overfitting, we could try to use an optimizer that does some sort of regularization, for example Proximal Adagrad Optimizer.

# Advanced: Transfer learning analysis

Transfer learning makes it possible to save training resources and to achieve good model generalization even when training on a small dataset. In this part, we will demonstrate this by training with two different TF-Hub modules:

  * nnlm-en-dim128 - pretrained text embedding module,
  * random-nnlm-en-dim128 - text embedding module that has same vocabulary and network as nnlm-en-dim128, but the weights were just randomly initialized and never trained on real data.

And by training in two modes:

  * training only the classifier (i.e. freezing the module), and
  * training the classifier together with the module.
