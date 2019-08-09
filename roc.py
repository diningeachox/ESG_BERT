import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
import bert_multi_label as bml
from bert_multi_label import BertLayer
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import csv
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score
from matplotlib import pyplot


np.set_printoptions(suppress=True)
# Initialize session
sess = tf.Session()
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 128
# Instantiate variables
bml.initialize_vars(sess)
#Load custom layers such as BertLayer
model = tf.keras.models.load_model("least_loss_model_d05_lr=3_ft=0.h5",
custom_objects={'BertLayer': bml.BertLayer, 'precision_m': bml.precision_m,
'recall_m': bml.recall_m})
#model = load_model("bert_model.h5", custom_objects={'BertLayer': bml.BertLayer})

print("BERT model succesfully loaded.")

#Take datasets to be the tweets csv
train_df, test_df = bml.load_datasets_csv("modified_tweets.csv")

test_text = test_df["tweet_content"].tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df.iloc[:, 6:-1].values.tolist()
#Convert str to float
test_label = [[float(entry) for entry in row] for row in test_label]

# Instantiate tokenizer
tokenizer = bml.create_tokenizer_from_hub_module(bert_path)

examples = bml.convert_text_to_examples(test_text, test_label)
# Convert to features
(
    test_input_ids,
    test_input_masks,
    test_segment_ids,
    test_labels,
) = bml.convert_examples_to_features(
    tokenizer, examples, max_seq_length=max_seq_length
)

pred = model.predict([test_input_ids, test_input_masks, test_segment_ids])

#FPR-TPR curves
x = np.arange(-0.5,0.51,0.01) #x ranges from 0 to 1 in increments of 0.05
pre = [K.eval(bml.precision_m(test_labels, pred, val)) for val in x]
rec = [K.eval(bml.recall_m(test_labels, pred, val)) for val in x]
fpr = [K.eval(bml.fpr_m(test_labels, pred, val)) for val in x]
F1 = [2 * (x * y) / (x + y + K.epsilon()) for x, y in zip(pre, rec)]
one_minus_fpr = [1 - K.eval(bml.fpr_m(test_labels, pred, val)) for val in x]

'''
# plot ROC curve
pyplot.plot(fpr, rec, marker='.', label="ROC curve")
#x = y line
pyplot.plot([0,1], [0,1], linestyle='--')
# Add X and y Label
pyplot.xlabel('FPR')
pyplot.ylabel('TPR')
'''

#Plot TPR vs 1 - FPR
pyplot.plot(x, tpr, marker='.', label="TPR")
#pyplot.plot(x, one_minus_fpr, marker='.', label="1 - FPR")

# Add X and y Label
pyplot.xlabel('Cutoff')
pyplot.ylabel('Score')

# Add a grid
pyplot.grid(alpha=.4,linestyle='--')

# Add a Legend
pyplot.legend()

pyplot.show()
