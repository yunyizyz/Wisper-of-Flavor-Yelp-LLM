# Wisper of Flavor - Yelp Text Analytics Project

Dependencies:

- Python 3.6 or higher
- TensorFlow 2.x
- PyTorch
- Transformers library
- scikit-learn
- tqdm

Install dependencies using pip install name. Replace "name" with the actual name of the package in dependencies list above.

### Dataset:
- df_labeled_test.csv: this is our itermediate dataset. The whole 1000000 data with predicted labels using BERT(comes from file BERT). Please download the data from this link: https://drive.google.com/file/d/10PcVCoBCnqxw4uqgUA45UfAvxuzkakqd/view?usp=sharing
- labeled_review_id.xlsx: manually labeled 1979 records of reviews. It is our sample for training and testing.
- yelp_academic_dataset_business.json: raw data downloaded from yelp official website, the link is here: https://www.yelp.com/dataset. (please download the file and use the data inside, the data is too large, we can't upload it to github :(
- yelp_academic_dataset_review.json: raw data downloaded from yelp official website, the link is here: https://www.yelp.com/dataset.


### Word2Vec_TFIDF_traditionalML_RNN
This folder contains code for using TF-IDF and Word2Vec to deal with text data. The subsequent steps involve applying traditional machine learning models ( Decision Tree, Naive Bayes, Linear SGD Classifier, Logistic Regression, Random Forest, and SVC) and applying RNN (Recurrent Netural Network) and evaluating models' performance.

### Multilabel classification using BERT
This folder contains code for training a text multilabel classification model using BERT (Bidirectional Encoder Representations from Transformers). The model is designed to classify Yelp reviews into five categories: 'food_quality', 'environment', 'service', 'convenience', 'cost_effectiveness'.

### EDA_cluster_time_series
This folder used the result predicted from BERT. 





