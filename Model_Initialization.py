from transformers import EncoderDecoderModel
from transformers import RobertaTokenizerFast

def load_roberta_encoder_decoder(directory= None,dropout_p = 0.1,tie_params = True):
    """
    Loads a pre-trained Roberta-2-Roberta model from a directory in google drive.
    :param dropout_p: dropout for cross attention layers (default: 0.1)
    :param directory: directory to be loaded.
    :return: tuple of model with share
    """
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    if directory:
        roberta_shared = EncoderDecoderModel.from_pretrained(directory)
    else:
        roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base",
                                                                             tie_encoder_decoder= tie_params)
    # set special tokens
    roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id
    roberta_shared.config.eos_token_id = tokenizer.eos_token_id
    roberta_shared.config.pad_token_id = tokenizer.pad_token_id
    roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size
    # Setting Dropout
    roberta_shared.config.decoder.hidden_dropout_prob = dropout_p
    roberta_shared.config.encoder.hidden_dropout_prob = dropout_p

    return roberta_shared