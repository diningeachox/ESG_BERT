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
import inspect

#SKLearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score

from matplotlib import pyplot
from fpdf import FPDF

os.environ['TF_KERAS'] = '1' #environment variable for RAdam
from keras_radam import RAdam

#np.set_printoptions(suppress=True)
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

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dropout):
        print(layer.name)
        layer.rate = 0 #No dropout for test data
model.compile(loss="binary_crossentropy", optimizer=RAdam(), metrics=["accuracy"])
model.summary()

print("BERT model succesfully loaded.")

#Take datasets to be the tweets csv
train_df, test_df = bml.load_datasets_csv("train_data.csv","test_data.csv")

test_text = test_df["tweet_content"].tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df.iloc[:, 7:-1].values.tolist()
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


for i in range(0, 13):
    #Calculate ROC curves
    fpr, tpr, _ = metrics.roc_curve(test_labels[:, i], pred[:, i])
    roc_auc = auc(fpr, tpr)
    #Calculate PR curves
    precision, recall, _ = precision_recall_curve(test_labels[:, i], pred[:, i])
    pr_auc = auc(recall, precision)

    # Smooth out the plot
    step_kwargs = ({'step': 'post'}
                   if 'step' in inspect.signature(pyplot.fill_between).parameters
                   else {})
    pyplot.step(recall, precision, color='b', alpha=0.2,
             where='post')
    pyplot.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.ylim([0.0, 1.05])
    pyplot.xlim([0.0, 1.0])
    pyplot.title('2-class Precision-Recall Curve: AP={0:0.2f}'.format(
              pr_auc))

    #Save the graph to a pgn file
    pyplot.savefig("new_curves/pr_class{}.png".format(i+1))
    print("Saved pr curve for class {}\n".format(i+1))
    pyplot.clf()


    # plot ROC curve
    # Smooth out the plot
    step_kwargs = ({'step': 'post'}
                   if 'step' in inspect.signature(pyplot.fill_between).parameters
                   else {})
    pyplot.step(fpr, tpr, color='b', alpha=0.2,
             where='post')
    pyplot.fill_between(fpr, tpr, alpha=0.2, color='b', **step_kwargs)

    pyplot.xlabel('FPR')
    pyplot.ylabel('TPR')
    pyplot.ylim([0.0, 1.05])
    pyplot.xlim([0.0, 1.0])
    pyplot.title('2-class Receiver Operating Characteristic Curve: AP={0:0.2f}'.format(
              roc_auc))

    #Save the graph to a pgn file
    pyplot.savefig("new_curves/roc_class{}.png".format(i+1))
    print("Saved roc curve for class {}\n".format(i+1))
    pyplot.clf()

#Save the pgn files to a single PDF
path = "new_curves\\"
# r=root, d=directories, f = files
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            files.append(os.path.join(r, file))
pdf = FPDF()
for image in files:
    pdf.add_page()
    pdf.image(image, x=0, y=0)
pdf.output("new_curves\\charts.pdf", "F")
