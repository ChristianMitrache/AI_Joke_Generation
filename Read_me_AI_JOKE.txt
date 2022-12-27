# AI Joke Generation

Humour is a complex subject that is not easily grasped by many humans. This makes the task of teaching an AI model to learn the subtle intricacies of humour all the more challenging. The aim of this project is to train a state-of-the-art transformer based model in order to achieve humour generation that is both funny and human-like.

## Selected Examples of Model Output:

Human: "How many professors does it take to change a lightbulb?" 
Model: "None, they just sit in the dark and blame the students"

Human: "What kind of coffee does an executioner drink?"
Model: "Decapresso"

Human: "Why did the pilot hit the alps?"
Model: "He was tired of all the altitude."

## Previous Work

Efforts to computationally generate humour go back to the 1990s. In the recent past, deep
learning models (in particular models based on the transformer architecture) have pushed
the state of the art in all major NLP tasks. Previous work in humour generation has used sequence-to-sequence
modesl with LSTM units for both the encoder and decoder to generate jokes. Some previous GITHUB repos fine-tuned GPT2 models to  This project
aims to improve on this by fine-tuning RoBERTaSHARE; an encoder-decoder architecture
which uses robustly pre-trained BERT architectures for both the encoder (Trained as a
Masked Language Model) and decoder (Trained as a Causal Language model).

## Data-Set 

The dataset titled "One Million Reddit Jokes" by HuggingFace was used. Features that were extracted from the data-set consisted of build-ups, punchlines and scores (up-votes). Only jokes that had fewer then 50 words in the buildup and fewer then 20 words in the punchline were kept. Restricting the jokes to these sequence lengths provids 2 major benefits. First, the shorter sequence lengths drastically improved the speed and memory requirements of gradient updates enabling the training of larger models on google collab GPUs. Secondly, shorter less complex sequences should make it easier for the model to learn relationships between the buildups and punchlines. After the data pre-processing, the data-set contained approximately 350,000 jokes.

Preliminary analysis on the remaining jokes revealed that the remaining jokes contain many semi-duplicates, (jokes which have very similar buildups or punchlines). It was decided to keep these jokes in the training data as this has the effect of biasing the model to more popular punchline formats. Jokes with many semi-duplicates usually have larger scores, so keeping semi-duplicates in training can be viewed as over-sampling jokes with higher scores.


## Architecture

The architecture that was used consists of a transformer based encoder-decoder model were the encoder consists of a pre-trained RoBERTa model and the decoder consists of a similar model with added randomly initialized cross attention layers.

### Encoder Architecture

The Encoder architecture consists solely of a pre-trained RoBERTa architecture which takes in sentences and outputs a rich sentence embedding which can be used for any task. The main architecture of a RoBERTa model consists of 12 encoder blocks. Each encoder block consists of a self attention layer followed by 2 fully connected layers which form a bottleneck. At the end of the 12 encoder blocks, we are left with a sentence embedding consisting of a sequence of vector representations for the original sentence. Because RoBERTa is trained as a masked language model, the context from the entire sentence is captured in each vector representation (regardless of position) and a pooling can be done by only feeding in the first vector representation to the decoder.

### Decoder Architecture

The decoder architecture consists of 12 decoder blocks which have similar architectures as the 12 encoder blocks in the previous section. The only difference between the 2 blocks is that in a decoder block, there is a randomly intialized cross attention layer between the self attention layer and the two fully connected layers. 

The output from the 12 decoder blocks is then passed into a fully connected which is trained to output a sequence of log distributions for each word in the sentence (conditioned on the buildup given to the model). In probabilistic terms this can be viewed as outputting the log distributions: $log(Pr_1(V_1,...,V_M|g_1,...g_k)),log(Pr_2(V_1,...,V_M|g_1,...g_k))...$ where $g_1,...,g_k$ are the words in the buildup given by the user and $V_1,...,V_M$ are the words in the vocabulary.

### Weight Sharing Among Decoder and Encoder Blocks

The RoBERTa model used in the encoder uses pre-trained weights to make fine-tuning on jokes computationally feasible with a google-collab GPU. In addition to this, common layers between the encoder and decoder blocks also share the same pre-trained weights during training. The only layers that were trained from scratch are the randomly initialized cross attention layers in the decoder blocks.

## Training

A weighted loss that gives higher weight to better jokes was used. To achieve this, a log transformation was applied to the scores with the result being multiplied by the loss associated to that joke. Let $N_B$ be the number of batches in an epoch and let $B$ denote batch size. For joke $t$, denote the sequence of tokens in the buildup as $g_{1,t},...g_{p,t}$ and the sequence of tokens in the punchline as $S_{1,t},...,S_{l,t}$ . The weighted training loss is defined by: 

$$L = -\sum_{i=1}^{N_B} \sum_{j \in B_i} \sum_{k = 1}^{l_j} w_{i,j}\times log(Pr(S_{k,j}|g_{1,j},...,g_{p,j}))$$

Where $w_{i,j}$ is given by: $w_{i,j} = log(score_{i,j}+1)+1$  

## Joke Generation

In order to generate jokes from the model, a sequence of words needs to be sampled sequentially the model. To see how this is done, consider the punchline: 

"Why did the chicken cross the road?"

Passing this in to the model yields a sequence of log distributions. We only sample from the first log distribution: $log(Pr_1(V_1,...,V_M|'why','did',...'?'))$. 

We get the word "To". 

Passing "Why did the chicken cross the road? To" into the model and sampling from the first log distribution yields the word: "get".

We can then continue this process until we get an "<EOS>" (End of Sequence token) which signals to the model to stop the process. After sequentially passing in the previous outputs into the model and sampling we are given the punchline: "To get to the other side.".
  
There are tricks that I used to making the joke generation smoother and more human-sounding, however I do not get into this here. If you are interested, check out this notebook: https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/02_how_to_generate.ipynb.