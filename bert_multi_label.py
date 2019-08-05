import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm
from tensorflow.keras import backend as K
import csv
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# Initialize session
sess = tf.Session()

classes = 14 #14 classes in labelling

# Load tweets from the csv file
def load_datasets_csv(file_path):
    # Read csv file and store it as a pandas Dataframe
    df = pd.read_csv(file_path)
    #Replace the NaN's by 0
    df.replace(np.nan, '0', inplace=True)

    # 80% training data, 20% testing data
    total_rows = len(df.index)
    train_rows = np.floor(0.8 * total_rows).astype(int)
    test_rows = total_rows - train_rows

    #Shuffle the dataframe to randomize the samples
    df.sample(frac=1).reset_index(drop=True)
    #Divide train/test into two separate Dataframes
    train_df = df.head(train_rows)
    test_df = df.tail(test_rows)

    return train_df, test_df


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) list. The labels of the example stored in a list. \
      This should be specified for train and dev examples, but not for test
      examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        # Make sure that the labels tensor gets reshaped to have 14 columns
        np.array(labels).reshape(-1, classes),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, multi_label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=multi_label)
        )
    return InputExamples


class BertLayer(tf.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    #Save  custom data so that the BertLayer can be loaded
    def get_config(self):
        config = {"n_fine_tune_layers": self.n_fine_tune_layers,
        "trainable": self.trainable,
        "pooling": self.pooling,
        "bert_path": self.bert_path
        }
        #base_config = super(BertLayer, self).get_config()
        #return dict(list(base_config.items()) + list(config.items()))
        return config

#Threshold for outputs. We will use t = 0.2
def pred_thresh(y, t):
    return np.round(y - t + 0.5)

#Recall score: true positives / (true positives + true negatives)
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

#Recall score: true positives / (true positives + false positives)
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

#F-beta score for evaluating multi-label predictions, usually beta = 1 is used
def F_beta(beta, y_true, y_pred):
    #Convert output to actual predictions with threshold 0.2

    y_pred_rounded = pred_thresh_tensor(y_pred, 0.2)
    precision = precision_m(y_true, y_pred_rounded)
    recall = recall_m(y_true, y_pred_rounded)
    return (1 + beta ** 2) * ((precision * recall)/(beta ** 2 * precision + recall + K.epsilon()))

def F1(y_true, y_pred):
    return F_beta(1, y_true, y_pred)

# Build model
def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    #Added dropout layers to deal with early overfitting
    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    bert_output = tf.keras.layers.Dropout(0.5)(bert_output)
    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
    dense = tf.keras.layers.Dropout(0.5)(dense)
    pred = tf.keras.layers.Dense(classes, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

def main():
    # Params for bert model and tokenization
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    #Shorten max_seq_length to reduce memory
    max_seq_length = 128

    #Take datasets to be the tweets csv
    train_df, test_df = load_datasets_csv("modified_tweets.csv")

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df["tweet_content"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    #Extract the 14 columns containing the ESG categories
    train_label = train_df.iloc[:, 6:-1].values.tolist()

    test_text = test_df["tweet_content"].tolist()
    test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df.iloc[:, 6:-1].values.tolist()

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path)

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    # Convert to features
    (
        train_input_ids,
        train_input_masks,
        train_segment_ids,
        train_labels,
    ) = convert_examples_to_features(
        tokenizer, train_examples, max_seq_length=max_seq_length
    )
    (
        test_input_ids,
        test_input_masks,
        test_segment_ids,
        test_labels,
    ) = convert_examples_to_features(
        tokenizer, test_examples, max_seq_length=max_seq_length
    )

    print(test_input_ids)

    model = build_model(max_seq_length)

    # Instantiate variables
    initialize_vars(sess)

    #Early stopping that waits 20 epochs after first increase in val_loss in order to avoid local minima
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    #Automatically save the best model (according to accuracy)
    #mc = ModelCheckpoint('best_acc_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    #Save the model with least loss
    mc = ModelCheckpoint('least_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #Save model with best F1 score
    #mc = ModelCheckpoint('best_F1_model.h5', monitor='val_F1', mode='max', verbose=1, save_best_only=True)

    history = model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_labels,
        validation_data=(
            [test_input_ids, test_input_masks, test_segment_ids],
            test_labels,
        ),
        epochs=100,
        batch_size=32,
        callbacks=[es, mc],
        verbose=1
    )

    # evaluate the model
    _, train_acc = model.evaluate([train_input_ids, train_input_masks, train_segment_ids], train_labels, verbose=0)
    _, test_acc = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    #Write sample predictions to predictions.csv
    with open('predictions.csv', mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #Column names
        file_writer.writerow(["Actual/Predicted",
        "Corporate Behaviour - Business Ethics",
        "Corporate Behaviour - Business Ethics",
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
        "Neutral",
        "Tied to Specific Company(Y/N)"])
        for i in range(50):
            #Make 25 sample prediction using the trained model from test data
            pred = model.predict([test_input_ids[i:i+1], test_input_masks[i:i+1], test_segment_ids[i:i+1]])
            pred = map(lambda x: np.round(x + 0.3), pred)
            file_writer.writerow(pred) #Predicted labels
            file_writer.writerow(test_labels[i]) #Actual labels
            file_writer.writerow("--------------------------------------------")

if __name__ == "__main__":
    main()
