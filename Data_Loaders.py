import pyarrow as pa
from datasets import Dataset
from transformers import RobertaTokenizerFast
from Clean_Data_Functions import get_training_eval_log_scores_data

def tokenize_input_output(batch,tokenizer,encoder_max_length,decoder_max_length):
    """
    :param batch: dictionary containing buildups and punchlines.
    :param tokenizer: tokenizer being used.
    :param encoder_max_length: max length of encoder of seq-2-seq model
    :param decoder_max_length: max length of decoder of seq-2-seq model
    :return: inputs,outputs -- which are the tokenized buildups and punchlines respectively
    """
    inputs =  tokenizer(batch["buildup"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["punchline"], padding="max_length", truncation=True, max_length=decoder_max_length)
    return inputs,outputs


def load_and_tokenize_data(directory ="/content/drive/MyDrive/Final_Project_940/Clean_Joke_Data.csv",
              num_words_buildup = 50, num_words_punch = 20, remove_duplicate_punchlines = True,
              discrete_scores = False,batch_size = 64):
    # Computing log scores according to formula in documentation
    train_data, eval_data = get_training_eval_log_scores_data(num_words_build=num_words_buildup,
                                                              num_words_punch=num_words_punch,
                                                              directory=directory,
                                                              is_discrete=discrete_scores)
    # Removing Punchlines
    if remove_duplicate_punchlines:
        train_data = train_data.drop_duplicates(subset=['buildup'])
        train_data = train_data.sort_values(['score'], ascending=False).drop_duplicates(subset=["buildup"],
                                                                                        keep="first")
        train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Moving train data from pandas to huggingface dataset class
    train_data = Dataset(pa.Table.from_pandas(train_data))
    eval_data = Dataset(pa.Table.from_pandas(eval_data))

    # Initializing tokenizer for pre-trained model and specifying encoder-decoder max lengths
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    encoder_max_length = num_words_buildup
    decoder_max_length = num_words_punch

    def tokenize_data_to_model_inputs(batch):
        """
        tokenize the raw text in a batch.
        :param batch: dictionary containing buildups and punchlines.
        :return: same batch object but with tokenized buildups and punchlines
        """
        inputs, outputs = tokenize_input_output(batch,tokenizer,encoder_max_length,decoder_max_length)
        # batch["score"] = batch["score"]
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        return batch

    train_data = train_data.map(
        tokenize_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["buildup", "punchline"]
    )
    eval_data = eval_data.map(
        tokenize_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["buildup", "punchline"]
    )

    train_data.set_format(
        type="torch", columns=["score", "input_ids", "attention_mask", "labels"],
    )

    eval_data.set_format(
        type="torch", columns=["score", "input_ids", "attention_mask", "labels"],
    )
    return train_data, eval_data, tokenizer