# ESG_BERT

Corporations all around the world have become more interested in ventures which follow ESG categories.
ESG stands for Environmental, Social, and Governance. Examples include climate change, inclusivity & diversity, and corruption. These are categories designed to gauge an organization's non-financial behavior. Studies have shown that companies who display characteristics of ESG categories also tend to be more profitable, and that companies which violate these categories also present more financial risk.

This project uses the BERT model in NLP to classify tweets related to S&P 500 companies, in order to detect any red flags regarding their ESG categories. 

Data collection:
  
We collected 6617 tweets (including retweets) which mentioned or discussed S&P 500 companies over a period of one month from June 8, 2019 to July 8, 2019. Each tweet was given a positive or negative label for each ESG category it is in violation of.
For example, the tweet "copyright suit against microsoft is precluded by earlier patent suit | the recorder https://t.co/ltpe5zrq3v" is given a positive label in the copyright category (which falls under the governmence (G) part of ESG).
This method of labelling allows the model to learn to identify "red flags" in each ESG category. 

Our data set covers 10 ESG topics:

▪ Governance: Business Ethics, Anti-Competitive Practices, Corruption, & Instability

▪ Social: Discrimination, Health & Demographic Risk, Supply Chain Labour Standards or Labour Management, Privacy & Data Security, Product Liability

• The Discrimination category was added by the RiskLab team due to the prevalence of such issues being expressed on social media.

▪ Environmental: Climate Change and Carbon Emissions, Toxic Emissions & Waste

Model:
We employed the Bidirectional Encoder Representations from Transformers (BERT) model, which is state-of-the-art in natural language processing (NLP). A big advantage of this model is that it is very good at learning atomized parts of a word such as prefixes and suffixes. This makes it a particularly good fit for analyzing tweets, which often contain abbreviations and shortened words due to the 144 character constraint. 

Following is the full list of hyperparameters we used to train our classifier:

▪ We use an output size of 768 and a max sequence length of 128 for the BERT encoder layers, a batch size of 32, and a dropout rate of 0.1.

▪ We add one hidden layer, of dimension 256, right after the initial (CLS token) BERT embedding, with dropout of 0.5.

▪ We use the binary cross-entropy loss for each category, as the ESG issues may not be mutually exclusive.

▪ We utilize a two-stage pre-training approach, with a patience of 5 epochs. In the first phase, we train the hidden and output layer; in the second, the full network (with all 10 BERT encoder layers) is fine tuned.

▪ We utilize the RADAM (Liu et al. 2020) optimizer with learning rates of Embedded Image during the first stage of training, and Embedded Image during the second stage of training (where BERT encoder layers are fine tuned); the RADAM optimizer simplifies the learning rate hyperparameter search and removes the need for running “warm-up” epochs. We used no warm-up for training our model.

Identifying clusters in feature space:
Already with a relatively small dataset of ~6000 samples, we can identify certain clusters in the data corresponding to certain categories.
<p>
<img src="https://github.com/diningeachox/ESG_BERT/blob/master/F4.large.jpg" width=500> The data breach cluster</img>
</p>

See our paper here: https://jesg.pm-research.com/content/early/2021/01/09/jesg.2021.1.010
