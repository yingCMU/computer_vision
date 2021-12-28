# RNN
[pythorch - RNN source code](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN)

asume each word in the sequence is a one-hot encoding vector across the word dictionary

why not a standara network?
- inputs, outputs can be different lengths in different examples
- doesn't share features learned across different positions of text. e.g if Harry Porter is a name, no matter where it is in the position of text, it should always be learned as a name

in RNN, each input word is processed one by one

problem with RNN: it does not use informatin earlier in the sequence, not later in the sequence.

## Vanishing Gradients/ Exploding Gradient
exploding: meaning Expotential large gradients; seeing NaN/not a number. use gradient clipping to solve.
Vanishing gradient is harder to solve.

## Different Types of RNN

input lenth Tx and output lenth Ty can be very different.
![Tux, the Linux mascot](./images/rnn_types.png)
- sentiment clasfficiation: many to one
- music generation: one to many, e.g gener to music note sequence
- machine translation: many to many, but different length
    ![Tux, the Linux mascot](./images/many2many.png)

## Bidirectional RNN
The disadvantage of the bidirectional RNN is that, you do need the entire sequence of data before you can make predictions anywhere. for example, if you're building a speech recognition system then BRNN will let you take into account the entire speech other friends. But if you use this straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it, and make a speech recognition prediction

![Tux, the Linux mascot](./images/brnn.png)

`
## Language modeling

Given a sentence, what is the probability of that sentence.
speech recognition:
  - p(The apple and paire salad) =?
  - p(The apple and pear salad) =?

### How to train language model?


## Use case
- [MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION](https://arxiv.org/pdf/1412.7755.pdf)
- [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623.pdf)

## How
RNN output vector’s contents are influenced not only by the input you just fed in, but also on the entire history of inputs you’ve fed in in the past.


## Embedding
lower dimention representation. Very powerful because it can examine very large text corpuses, maybe found on the internet,  **unlabeled text**. It also carry out transfer learning, where you take info learned from huge amounts of unlabeled text and transfer that knowledge to a task, such as named entity rocognition, for which you have a small labeled training set.
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

Steps to do transfer learning and word embeddings:
1. learn word embeddings from large text corpus (1-100B words). Or download pre-trained embedding online
2. transfer embedding to new tasks with smaller training set (say 100k words)
3. optional): contine to finetune the word embedding with new data , if yor task has pretty big data set

Note word embeddings tend to make the biggest difference when the task has relatively small train set. It has been less useful for language modeling, machine translation, especially if you are accessing a task with a lot of data dedicated to the task.

Similarly in CNN, we train network and ghet low-dimention representaion for different faces (e.g 128 dimensions). You can compare these encodings in order to figure out if these two pictures are of the same face.

[Linguistic Regularities in Continuous Space Word Representations](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)

### Analogies using word embedding

to solve this e_man - e_womain = e_king - e_? => find word wi where argmax sim(e2, e_king-e_man+e_woman)

most common used similarity is cosine similartiy. The nominator is basically dot product;
you can also use square distance (u-v)^2

### Embedding Matrix(table)
When you implement an algorithm to learn a word embedding, what you end up learning is an embedding matrix.
    ![Tux, the Linux mascot](./images/embedding_matrix.png)

### Learning Word Embeddings

**1. N Gram**<br>
(N-gram) Large language models in machine translation<br>

**2. Nueral network based language models** <br>
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). (**This paper is most well written to understand everything**)

posing of machine lenaring problem to learn embedding: input the context like the last 4 words and predict some target word

The embedding matrix can be thought of either a look up table or a matrix of weight to be tuned. For a trining example, with one-hot encoding as a known input, multiple by weights (embedding matrix):
parameters of the mapping C are simply the feature vectors themselves, represented by a jVj m
matrix C whose row i is the feature vector C(i) for word i. The function g may be implemented by a
feed-forward or recurrent neural network or another parametrized function, with parameters w. The
overall parameter set is q = (C;w).

Building a neural language model is  a way to learn a set of embeeings. A feedforward neural network with a linear projection layer and a non-linear hidden layer was used to
learn jointly the word vector representation and a statistical language model.
    ![Tux, the Linux mascot](./images/neural_language_model.png)

Here we learn embedding matrix E: 300 x 10k dimension

Other learning context:
    ![Tux, the Linux mascot](./images/other_context.png)

**3. word2vec** <br>
(skip gram): [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) <br>
- [NEURAL NETWORK BASED LANGUAGE MODELS FOR HIGHLY INFLECTIVE LANGUAGES](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4960686)

**The problem**: softmax step is very expensive to calculate because needing to sum over your entire vocabulary size into the denominator of the softmax<br>
**Speedup Softmax**  : using 10K classes to calculate softmax is really slow. Use hierachy softmax classifier<br>
**how to sample context**: use heuristics to sample more uncommon word<br>
**Measuring Similarity**: [Linguistic Regularities in Continuous Space Word Representations](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf)
expectation that not only will similar words tend to be close to each other, but that
words can have multiple degrees of similarity

**4. Negative Sampling** <br>
[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)<br>

It generates positive/negative learning examples (supervised learning): given the context and target word, label 1 if they are generated from the same sequence, label 0 if they are random put together. It also avoids the large softmax problem. It runs a ginat softmax to each binary classificaiton problem (K+1)
k is the number of negative samples for each context word:
- large dataset: 2-5
- small dataset: 5-20

An inherent limitation of word representations is their indifference to word order
and their inability to represent idiomatic phrases. For example, the meanings of
“Canada” and “Air” cannot be easily combined to obtain “Air Canada”. Motivated
by this example, we present a simple method for finding phrases in text, and show
that learning good vector representations for millions of phrases is possible

**how to create random negative samples?**
- sample based on word frequency in the dataset: but this end up with a lot of words like the, of, an, etc
- sample uniformely 1/|V|, but that is also very unpresentative of the distribution of engilish words
- reported in the paper, something in between works better. Frequency^3/4

**5.GloVe Word Vectors** <br>
[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

## Application using word embedding
### Sentiment Classification
naive solution embedding lookup -> sum or average pooling among all words -> softmax prediction rating.
Consider sentence 'completely  lacking in good taste, good service, and good ambience'. This include3 multiple good, but it is reverse meaning. So simply sum or average  and ignore word order, you end up having a lot of representation of good in the final feature vector;, you classifier will probably think this is a good review.
Using RNN:
    ![Tux, the Linux mascot](./images/rnn_sentiment.png)
### Debiasing word embeddings
[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)



# Sequence to Seuqnce Architectures
## Sequence to Sequence (encoder/decoder)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

### Image Captioning
Similar papers at same time:
- [Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)](https://arxiv.org/abs/1412.6632)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/abs/1412.2306)

### Picking the most likely sentence (Machine Translationo)
in machine translation, given an input Frensch sentence, you don't want to output a random English translation, you want to ouput the best and the most likely Englihs translation.. The same is true for speech recognition where given an input audio clip, you don't want to output a random text transcript of that audio, you wanto output the most likely text transcript
![Tux, the Linux mascot](./images/most_likely.png)

greedy search won't work: for each output word y, if pick the current word of highest probability one by one, the whole sequence is not optimal.
Instead what is optimal is to find each y where the:
argmax p(y_1, y_2, ... , y_n | x)

### Beam Search Algorithm
B is bean width. Let's say B=3 for step2. Hard wiring the previous word, to evaluate the next word probability given the previous word.
 Every step it instantiates B copies of the network to evaluate B x |V| possibilies.
![Tux, the Linux mascot](./images/beam_search.png)

# Attention
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) <br>


encoder read and memorize the whole sentence, and decoder output translated sentence. Encoder/Decoder works poorly on long sentences (Bleu Score drop at length 20)
![Tux, the Linux mascot](./images/attention.png)

intuitively, how much attention to pay for each previous timestep should depend the most on what is your own hidden state from the previou time step. You don't have the current state activation yet. It it natural that attention weight, and calculated attention value should depend on thse two quantities, feature at t_prime, and previous hidden state. But we don't know what the functionis. So jus train a very small neural network to learn whatever this funciton should be.
![Tux, the Linux mascot](./images/computing_attention.png)

The drawback is it does take quadratic time or quadratic cost to run. T_x is input length and T_y is output sequence length, then total number of these attention parameters is T_x times T_y.  Although in machine translation applications neither input nor output sentences is that long maybe quadratic cost is acceptable.

Attention is also implemented in image caption (CV), attention is paid to a region of picture. See paper [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)

# Sequene to Sequence on Audio Data
## Speech Recognition
audio clip -> transcript <br>
you can use attention model with audio as input and transcript as output.
The input timestamps can be much bigger than output timestamps. Audio input is high frequency , 100hz e.g, but the ouput is only a few words.
### CTC cost for speech recognition.
[Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

## Trigger word detection
![Tux, the Linux mascot](./images/trigger.png)

# Transformer
## why?
with increased complexity for sequential model, each unit becomes a bottlneck: each unit was like a bottleneck to the flow of information. Because to compute the output of this final unit, for example, you first have to compute the outputs of all of the units that come before
## what is transformer?
[attention is all you need](https://arxiv.org/abs/1706.03762)
RNN process one output at a time; CNN instead can take in a lot of pixels and process in parallel.
## Self Attention
![Tux, the Linux mascot](./images/sa_i.png)
use q,k, v to learn what is the value of word in context to the word being calculated. This makes learning attention between current word and any word in context to be parallel. weights for Q, K, V is parameters being learned.
![Tux, the Linux mascot](./images/self_attention.png)
## Multi-Head Attention
![Tux, the Linux mascot](./images/m_a.png)
## Transformer Architecture
![Tux, the Linux mascot](./images/transformer_arch.png)
with positional encoding * Masked multi-head attention:
![Tux, the Linux mascot](./images/attention_position.png)