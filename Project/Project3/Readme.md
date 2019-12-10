# Text classification using semi-supervised techniques

With the massive volume of data available online such as news feed, email, medical records; automati-
cally classifying text documents is a problem. Such text classification usually requires large amount of
labeled data to train any model which affects the accuracy. Hence, to improve the accuracy, unlabeled
documents can be used to augment the labeled ones.  We focus on semi-supervised algorithms to
classify news articles into 20 newsgroups.

## Dataset

The data set used in this project is the Twenty Newsgroups, which contains approx., 1000 text articles
posted to each of 20 online newsgroups, for a total of 18846 articles.  The “label” of each article
is the category (newsgroups) to which each of the article belongs to. We imported the dataset from scikit-learn.

## Requirements

* python 2.7 
* python modules:
  - scikit-learn
  - numpy
  - matplotlib
  - nltk
  - pickle
  - scipy etc
  
## Code

The code is pretty straight forward and well documented. The preprocessing of the documents and the implementation of classifiers have been done from scratch and then the results have been compared to inbuilt sklearn's classifiers. The code has been arranged in form of IPython Notebooks, each notebook corresponds to a particular "classifier" or "technique" used for classifying the dataset.
To implement semi-supervised techniques, implementation of the algorithms which were not built-in, modules were also used.

## Running the test

All codes are in code/ folder. `

* s3vm.ipynb, self_learning.ipynb, Graphbased.py and EM.ipynb contain our implemntations of the semi-supervised algorithms on this dataset
* frameworks.ipynb, methods.ipynb and qns3vm.ipynb, Semi_EM_NB.py are the modules required for implmentations of the above algorithms.
* Tu run graphbased.py, use
    ```
     $python Graphbased.py
    ```
    
## Results
Below, we see the validation and testing accuracy of supervised and semi-supervised techniques.

![alt text][https://github.com/anushamanur/CSC591_ML/tree/master/Project/Project3/images/Results.png]

<div align="center">
    <img src="/https://github.com/anushamanur/CSC591_ML/tree/master/Project/Project3/images/Results.png" width="100"  height="100" /> 
</div>

<div align="center">
    <img src="https://github.com/anushamanur/CSC591_ML/tree/master/Project/Project3/images/graph_variation.png" width="400px"</img> 
</div>


## Reference

* s3vm and self learning were implmented using [Semi-supervised learning frameworks for Python](https://github.com/tmadl/semisup-learn)
* EM with NB was implemented using [semi-supervised learning with EM](https://github.com/jerry-shijieli/Text_Classification_Using_EM_And_Semisupervied_Learning)
