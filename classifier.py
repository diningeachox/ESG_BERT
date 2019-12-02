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
from keras import backend

os.environ['TF_KERAS'] = '1' #environment variable for RAdam
from keras_radam import RAdam

np.set_printoptions(suppress=True)
# Initialize session
sess = tf.Session()
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 128
# Instantiate variables
bml.initialize_vars(sess)
#Load custom layers such as BertLayer
model = tf.keras.models.load_model("2stage_model_v2.h5",
custom_objects={'BertLayer': bml.BertLayer, 'precision_m': bml.precision_m,
'recall_m': bml.recall_m, 'RAdam': RAdam})
#model = load_model("bert_model.h5", custom_objects={'BertLayer': bml.BertLayer})

print("BERT model succesfully loaded.")
#Take datasets to be the tweets csv
train_df, test_df = bml.load_datasets_csv("train_data.csv", "test_data.csv")

test_text = test_df["tweet_content"].tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df.iloc[:, 7:-1].values.tolist()

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


#Write sample predictions to predictions.csv
with open('predictions.csv', mode='w') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
    #Column names
    file_writer.writerow(["Corporate Behaviour - Business Ethics",
    "Corporate Behaviour - Anti-Competitive Practices",
    "Corporate Behaviour - Corruption & Instability",
    "Privacy & Data Security",
    "Human Capital - Discrimination (Added by RiskLab Team)",
    "Pollution & Waste - Toxic Emissions & Waste",
    "Human Capital - Health & Demographic Risk",
    "Human Capital - Supply Chain Labour Standards or Labour Management",
    "Climate Change - Carbon Emissions",
    "Product Liability - Product Quality & Safety",
    "Positive",
    "Negative",
    "Tied to Specific Company(Y/N)"])
    for i in range(len(test_label)):
        #Make 25 sample prediction using the trained model from test data
        pred = model.predict([test_input_ids[i:i+1], test_input_masks[i:i+1], test_segment_ids[i:i+1]])
        #Round to 8 decimal places
        file_writer.writerow(list(np.around(np.array(pred[0]),8)))
        file_writer.writerow(test_labels[i]) #Actual labels
