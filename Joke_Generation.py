from transformers import EncoderDecoderModel, RobertaTokenizerFast
import pandas as pd
import streamlit as st

class TrainedJokeGenerator:
    """
    Class variables
    model: variable that stores an instance of trained joke generator.
    encoder_tokenizer: variable that stores tokenizer for encoding model input
    decoder_tokenizer: variable that stores tokenizer for decoding model output
    weighted_loss: variable that indicates whether loss was weighted by reddit scores.
    """

    def __init__(self, model_directory,streamlit_dict = False,bad_words_directory = None):
        """
        Goes through model directory and initializes model+ tokenizers for the model.
        :param model_directory: directory where the model and strings for model are stored
        """
        self.model = EncoderDecoderModel.from_pretrained(model_directory)
        # Moving model to GPU
        self.model.to("cuda")
        # Setting Model to evaluation state
        self.model.eval()
        # Initializing Tokenizers
        self.encoder_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.decoder_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        if bad_words_directory:
            vulgar_words_data = pd.read_csv(bad_words_directory)
            vulgar_list = vulgar_words_data["2g1c"].tolist()
            # Encoding bad-words
            vulgar_token_id_list = []
            for vulgar_word in vulgar_list:
                vulgar_token_id_list.append(self.encoder_tokenizer.encode(vulgar_word))  # ,add_prefix_space = True))
        else:
            vulgar_token_id_list = None
        self.bad_word_list = vulgar_token_id_list

    def format_jokes(self,buildup, joke_list):
        """
        formats the jokes from the generate function.
        :param joke_list:
        :return: neat string representations of buildups and jokes
        """
        return "TODO"

    def add_extra_bad_words(self,bad_word_list):
        """
        Allows user to append more bad words onto bad_word list
        :param bad_word_list:
        :return:
        """
        print("TODO")

    def generate(self, input_text, top_k=None, top_p=None, num_sequences=4, no_repeat_ngram_size=3,
                 remove_vulgar=True, repetition_penalty=1,temperature = 1):
        """
        This function specifies how to sample from the conditional word log probabilities to create jokes.
        :param input_text: the buildup that the user specifies.
        :param top_k: Samples from only top-k most likely next words from vocab.
        :param top_p: Samples from only the top-p % of words from vocab.
        :param num_sequences: number of punchlines to randomly generate.
        :param no_repeat_ngram_size: prevent the exact same sequence of characters from appearing twice. default is 4.
        :param remove_vulgar: boolean to specify if you want to remove vulgar words
        (these will need to be specified in bad_words_directory)
        :param repetition_penalty: repetition penalty: default is 1: (no penalty)
        :return: list of generated punchlines.
        """
        # Dealing with vulgar words:

        inputs = self.encoder_tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        # outputs = model.generate(input_ids, attention_mask=attention_mask)
        outputs = self.model.generate(input_ids, do_sample=True, max_length=50,
                                      top_k=top_k, top_p=top_p, num_return_sequences=num_sequences,
                                      attention_mask=attention_mask, no_repeat_ngram_size=no_repeat_ngram_size,
                                      bad_words_ids=self.bad_word_list,
                                      repetition_penalty=repetition_penalty,temperature=temperature)
        return self.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)
