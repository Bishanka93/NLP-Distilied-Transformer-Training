# Text Classification and Causal Language Modeling Using Transformer Models

## 📌 Overview

This experiment focuses on implementing Transformer-based models for text classification and text generation using the Hugging Face Transformers library and PyTorch.

The experiment is divided into two major parts:

- **Part A – Text Classification:** Fine-tuning the pretrained DistilBERT model on the IMDb movie review dataset to classify reviews as positive or negative.
- **Part B – Causal Language Modeling:** Fine-tuning the pretrained DistilGPT2 model on the ELI5 dataset to generate contextually relevant text based on a given prompt.

The experiment explores important NLP concepts such as tokenization, self-attention, pretrained language models, sequence classification, causal language modeling, and text generation strategies.

---

## 🎯 Aim

To implement Transformer-based models for text classification and text generation, and understand the role of tokenization, self-attention, and pretrained language models in Natural Language Processing (NLP).

---

## 📚 Learning Outcomes

After completing this experiment, the following learning outcomes are achieved:

- Understand tokenization and Transformer-based language representations.
- Understand the self-attention mechanism in Transformer architectures.
- Learn how pretrained language models process textual data.
- Fine-tune DistilBERT for sentiment classification.
- Perform binary sentiment classification using the IMDb dataset.
- Fine-tune DistilGPT2 for causal language modeling.
- Generate text using pretrained Transformer models.
- Evaluate classification performance using accuracy.
- Evaluate language generation performance using perplexity.
- Understand top-k, top-p, and sampling techniques for text generation.

---

## 🧠 Theory

### Transformer-Based Language Models

Transformer-based language models are deep learning models that use the Transformer architecture to understand and generate human language.

Unlike traditional sequence models such as Recurrent Neural Networks (RNNs), Transformers use a self-attention mechanism that allows them to capture relationships between different words or tokens across an entire input sequence.

The Transformer architecture forms the foundation of modern NLP applications, including sentiment analysis, text classification, question answering, and text generation.

### 1. Tokenization

Tokenization is the process of converting input text into smaller units called tokens.

A tokenizer converts these tokens into numerical representations that can be processed by a Transformer model.

Common tokenizer outputs include:
- Input IDs
- Attention masks
- Token type IDs (when supported by the model)

### 2. Self-Attention Mechanism

Self-attention allows each token to focus on other relevant tokens in the input sequence.

It helps the model understand contextual relationships between words and learn meaningful language representations.

The scaled dot-product attention mechanism is represented as:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = Dimension of the key vectors

### 3. DistilBERT for Text Classification

DistilBERT is a smaller and faster version of BERT that retains much of its language understanding capability.

It uses a Transformer encoder architecture and can be fine-tuned for classification tasks.

In this experiment, DistilBERT is trained on the IMDb movie review dataset to classify movie reviews into two categories:

- Positive (1)
- Negative (0)

The model processes tokenized text and uses a classification head to predict the sentiment.

### 4. DistilGPT2 for Causal Language Modeling

DistilGPT2 is a distilled version of GPT-2 designed for autoregressive text generation.

It uses a Transformer decoder architecture and predicts the next token based on previously generated tokens.

The model learns the probability of the next token in a sequence:

\[
P(x_1,x_2,\ldots,x_n)=\prod_{t=1}^{n}P(x_t|x_1,\ldots,x_{t-1})
\]

During generation, the model repeatedly predicts and appends the next token until the generation process stops.

### 5. Text Generation Techniques

Different generation strategies can control the diversity and randomness of generated text.

**Greedy Search:**
- Selects the token with the highest probability at each generation step.
- Produces relatively predictable output.

**Sampling:**
- Selects the next token randomly according to the predicted probability distribution.
- Introduces diversity into generated text.

**Top-k Sampling:**
- Restricts the selection of the next token to the k most probable tokens.
- Helps control randomness.

**Top-p Sampling (Nucleus Sampling):**
- Selects tokens from the smallest set whose cumulative probability reaches the specified threshold p.
- Allows more flexible and dynamic text generation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained Transformer models |
| Hugging Face Datasets | Dataset loading and preprocessing |
| DistilBERT | Text classification |
| DistilGPT2 | Causal language modeling |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Evaluate | Model evaluation |
| Jupyter Notebook / Google Colab | Development environment |

---

# PART A: TEXT CLASSIFICATION USING DISTILBERT

## 📂 Dataset – IMDb Movie Reviews

**Dataset Name:** IMDb Movie Reviews Dataset

The IMDb dataset contains movie reviews labeled according to their sentiment.

It is a binary classification dataset with two categories:

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

The dataset is used to fine-tune DistilBERT for sentiment classification.

### ⚙️ Methodology / Procedure

#### Step 1: Import Required Libraries
- Import PyTorch, Transformers, Datasets, and evaluation libraries.
- Configure the development environment.

#### Step 2: Dataset Loading and Description
- Load the IMDb dataset using Hugging Face Datasets.
- Explore the training and testing samples.
- Understand the positive and negative sentiment labels.

#### Step 3: Load Tokenizer
- Load the pretrained DistilBERT tokenizer.
- Convert movie reviews into numerical token representations.

#### Step 4: Data Preprocessing
- Apply truncation to limit sequence length.
- Apply padding to make sequences uniform.
- Generate attention masks.

#### Step 5: Load Pretrained Transformer Model
- Load DistilBERT using `AutoModelForSequenceClassification`.
- Configure the model for two output classes.

#### Step 6: Configure and Train the Model
- Define training arguments.
- Configure batch size, learning rate, and number of epochs.
- Fine-tune DistilBERT using the IMDb training dataset.

#### Step 7: Model Evaluation
- Evaluate the trained model using the test dataset.
- Calculate classification accuracy.
- Analyze the prediction performance.

#### Step 8: Text Prediction
- Provide new movie reviews as input.
- Tokenize the input reviews.
- Predict whether the sentiment is positive or negative.

### 📊 Part A – Results and Observations

The fine-tuned DistilBERT model is evaluated based on its ability to correctly classify movie reviews.

The following aspects are analyzed:

1. Classification accuracy on the test dataset.
2. Prediction results for positive movie reviews.
3. Prediction results for negative movie reviews.
4. Model performance on new and unseen text samples.

**Evaluation Metric: Accuracy**

\[
Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}
\]

---

# PART B: CAUSAL LANGUAGE MODELING USING DISTILGPT2

## 📂 Dataset – ELI5

**Dataset Name:** ELI5 (Explain Like I'm Five)

ELI5 is a question-answering dataset containing questions and corresponding answers written in natural language.

In this experiment, a suitable subset of the ELI5 dataset is used to fine-tune DistilGPT2 for causal language modeling.

The answer text is used to train the model to predict subsequent tokens based on previously observed tokens.

### ⚙️ Methodology / Procedure

#### Step 1: Dataset Loading and Description
- Load the ELI5 dataset using Hugging Face Datasets.
- Select a suitable subset for experimentation.
- Explore the question-answer data.

#### Step 2: Data Preprocessing
- Extract the answer text from the dataset.
- Load the pretrained DistilGPT2 tokenizer.
- Tokenize the answer text into numerical representations.

#### Step 3: Group Text into Blocks
- Concatenate tokenized sequences.
- Divide the concatenated sequences into fixed-length blocks.
- Prepare the blocks for causal language modeling.

#### Step 4: Create Data Collator
- Use `DataCollatorForLanguageModeling`.
- Configure `mlm=False` for causal language modeling.
- Prepare batches for model training.

#### Step 5: Load Pretrained Causal Language Model
- Load DistilGPT2 using `AutoModelForCausalLM`.
- Initialize the pretrained model for fine-tuning.

#### Step 6: Configure and Fine-tune the Model
- Define training arguments.
- Configure learning rate, batch size, and training epochs.
- Fine-tune DistilGPT2 using the Trainer API.

#### Step 7: Model Evaluation
- Evaluate the fine-tuned model.
- Calculate evaluation loss.
- Calculate perplexity to analyze language modeling performance.

**Perplexity Formula:**

\[
Perplexity=e^{Evaluation\ Loss}
\]

Lower perplexity generally indicates that the model predicts the evaluation text more effectively.

#### Step 8: Text Generation
- Provide a text prompt to the fine-tuned model.
- Generate new text using the model.
- Observe how the model predicts subsequent tokens.

#### Step 9: Analyze Generated Text
- Compare outputs generated using different sampling strategies.
- Experiment with top-k sampling.
- Experiment with top-p sampling.
- Observe the effects of sampling and randomness on generated text.

### 📊 Part B – Results and Observations

The fine-tuned DistilGPT2 model is evaluated based on its ability to generate contextually relevant text.

The following aspects are analyzed:

1. Evaluation loss and perplexity.
2. Generated text based on different prompts.
3. Effect of top-k sampling on generated text.
4. Effect of top-p sampling on generated text.
5. Differences between greedy decoding and sampling.
6. Coherence, relevance, and diversity of generated responses.

---

## 📈 Results and Overall Analysis

This experiment demonstrates two important applications of Transformer-based language models.

| Task | Model | Dataset | Evaluation |
|---|---|---|---|
| Text Classification | DistilBERT | IMDb | Accuracy |
| Causal Language Modeling | DistilGPT2 | ELI5 | Evaluation Loss and Perplexity |
| Text Generation | DistilGPT2 | ELI5 | Qualitative Analysis |

The experiment demonstrates how pretrained Transformer models can be fine-tuned for different NLP tasks.

DistilBERT learns contextual representations for sentiment classification, while DistilGPT2 learns to predict subsequent tokens and generate natural language text.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Bishanka93/NLP-Distilied-Transformer-Training.git
```

Navigate to the project directory:

```bash
cd Text-Classification-and-Causal-Language-Modeling
```

Install the required dependencies:

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn numpy matplotlib
```

Run the Jupyter Notebook:

```bash
jupyter notebook experiment.ipynb
```

---

## 🔍 Conclusion

In this experiment, Transformer-based models were implemented for text classification and causal language modeling using the Hugging Face Transformers library and PyTorch.

In Part A, DistilBERT was fine-tuned on the IMDb movie review dataset to perform binary sentiment classification. The model learned contextual language representations and predicted whether movie reviews expressed positive or negative sentiment.

In Part B, DistilGPT2 was fine-tuned on a subset of the ELI5 dataset for causal language modeling. The model learned to predict subsequent tokens and generate text based on input prompts. Different generation techniques, including sampling, top-k, and top-p, were explored to understand their effects on generated text.

Overall, this experiment provided practical understanding of Transformer architectures, tokenization, self-attention, pretrained language models, text classification, and natural language generation.

---

## 📖 References

1. [Hugging Face – Text Classification Documentation](https://huggingface.co/docs/transformers/tasks/sequence_classification)
2. [Hugging Face – Causal Language Modeling Documentation](https://huggingface.co/docs/transformers/tasks/language_modeling)
3. [Hugging Face – Transformers Documentation](https://huggingface.co/docs/transformers)
4. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
5. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Proceedings of NAACL-HLT.
6. Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).

---

## 👨‍💻 Author

**Name:** Bishanka Sarma

**Topic:** Text Classification and Causal Language Modeling

**Models Used:** DistilBERT and DistilGPT2

**Framework:** Hugging Face Transformers & PyTorch

---

⭐ If you found this project useful, consider giving the repository a star!
