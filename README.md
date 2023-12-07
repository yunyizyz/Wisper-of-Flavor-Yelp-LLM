# Wisper of Flavor - Yelp Text Analytics Project

Dependencies:

- Python 3.6 or higher
- TensorFlow 2.x
- PyTorch
- Transformers library
- scikit-learn
- tqdm

Install dependencies using pip install name. Replace "name" with the actual name of the package in dependencies list above.

### Datasets:
- yelp_academic_dataset_business.json: raw data downloaded from yelp official website. Please download the data from this link:: https://www.yelp.com/dataset. 
- yelp_academic_dataset_review.json: raw data downloaded from yelp official website. Please download the data from this link: https://www.yelp.com/dataset.
- df_labeled_test.csv: The whole 1,000,000 data with predicted labels using BERT. Please download the data from this link: https://drive.google.com/file/d/10PcVCoBCnqxw4uqgUA45UfAvxuzkakqd/view?usp=sharing
- labeled_review_id.xlsx: manually labeled records of reviews. This dataset is the sample data for training and testing.

### Word2Vec_TFIDF_traditionalML_RNN
This folder contains code for using TF-IDF and Word2Vec to deal with text data. The subsequent steps involve applying traditional machine learning models ( Decision Tree, Naive Bayes, Linear SGD Classifier, Logistic Regression, Random Forest, and SVC), RNN (Recurrent Netural Network) model and model evaluation.

### Multilabel classification using BERT
This folder contains code for training a text multilabel classification model using BERT (Bidirectional Encoder Representations from Transformers). The model is designed to classify Yelp reviews into five categories: 'food_quality', 'environment', 'service', 'convenience', 'cost_effectiveness'.

### EDA_cluster_time_series
This folder used the result predicted from BERT. The codes involve time series analysis on review text and numerical factors, data visualization on how the pandemic affected restaurants’ performances and customer demand in five dimensions.



