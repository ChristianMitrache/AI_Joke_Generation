# AI Joke Generation Project

I train a Seq2Seq model to randomly generate funny punchlines to joke buildups supplied by the user. This project currently has a front end for model inference.



![image](https://user-images.githubusercontent.com/102553420/209865848-47ed819f-4c43-4a02-bbf4-e7f9d21375ce.png)

Currently, the website is not hosted anywhere but this will change soon! For now, follow the instructions below to run the program.

## To run this program:

1) clone the github repo onto your local machine 
2) Install the necessary requirements into a virtual environment by running the command: 'pip install -r Requirements.txt'
3) type 'python -m streamlit run StreamLitCode.py' into your terminal

# The Model:

## Data-Set 

The dataset titled "One Million Reddit Jokes" by HuggingFace was used. Features that were extracted from the data-set consisted of build-ups, punchlines and scores (up-votes). Only jokes that had fewer then 50 words in the buildup and fewer then 20 words in the punchline were kept. Restricting the jokes to these sequence lengths provided 2 major benefits. First, the shorter sequence lengths drastically improved the speed and memory requirements of gradient updates enabling the training of larger models on google collab GPUs. Secondly, shorter less complex sequences should make it easier for the model to learn relationships between the buildups and punchlines. After the data pre-processing, the data-set contained approximately 350,000 jokes.

Preliminary analysis on the remaining jokes revealed that the remaining jokes contain many semi-duplicates, (jokes which have very similar buildups or punchlines). It was decided to keep these jokes in the training data as this has the effect of biasing the model to more popular punchline formats. Jokes with many semi-duplicates usually have larger scores, so keeping semi-duplicates in training can be loosely viewed as over-sampling jokes with higher scores during training.


## Architecture

The architecture that was used consists of a transformer based encoder-decoder model.

### Encoder Architecture

The Encoder architecture is given by a pre-trained RoBERTa architecture which takes in sentences and outputs a rich sentence embedding which can be used for any task. The RoBERTa model consists of 12 encoder blocks. Each encoder block consists of a self attention layer followed by 2 fully connected layers which form a bottleneck. At the end of the 12 encoder blocks, we are left with a sentence embedding consisting of a sequence of vector representations for the original sentence. Because RoBERTa is trained as a masked language model, the context from the entire sentence is captured in each vector representation (regardless of position) and a pooling can be done by only feeding in the first vector representation to the decoder.

### Decoder Architecture

The decoder architecture consists of 12 decoder blocks which have similar architectures as the 12 encoder blocks in the previous section. The only difference between the 2 blocks is that in a decoder block, there is a randomly intialized cross attention layer between the self attention layer and the two fully connected layers. 

The output from the 12 decoder blocks is then passed into a fully connected which is trained to output a sequence of log distributions for each word in the sentence (conditioned on the buildup given to the model). In probabilistic terms this can be viewed as outputting the log distributions: $log(Pr_1(V_1,...,V_M|g_1,...g_k)),log(Pr_2(V_1,...,V_M|g_1,...g_k))...$ where $g_1,...,g_k$ are the words in the buildup given by the user and $V_1,...,V_M$ are the words in the vocabulary.

### Weight Sharing Among Decoder and Encoder Blocks

The RoBERTa model used in the encoder uses pre-trained weights to make fine-tuning on jokes computationally feasible with a google-collab GPU. In addition to this, common layers between the encoder and decoder blocks also share the same pre-trained weights during training. The only layers that were trained from scratch are the randomly initialized cross attention layers in the decoder blocks.

## Training

A weighted loss that gives slightly higher weight to better jokes was used. To achieve this, a log transformation was applied to the reddit scores with the result being multiplied by the loss associated to that joke. Let $N$ denote batch size. For joke $t$, denote the sequence of tokens in the buildup as $B_{1,t},...B_{p,t}$ and the sequence of tokens in the punchline as $S_{1,t},...,S_{l,t}$ . The weighted training loss for a batch is then defined by: 

$$L = -\sum_{j \in N} \sum_{k = 1}^{l_j} w_{i,j}\times log(Pr(S_{k,j}|B_{1,j},...,B_{p,j},S_{1,j}...,S_{k-1,j}))$$

Where $w_{i,j}$ is given by: $w_{i,j} = log(score_{i,j}+1)+1$  
