import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_short_jokes(jokes_dataframe, num_words_build, num_words_punch):
    """
  This function returns all short jokes with short buildups (<num_words_build) and short punchlines (<num_words_punch)
  """
    return jokes_dataframe[(jokes_dataframe['punchline'].str.split().str.len() < num_words_punch) &
                           (jokes_dataframe['buildup'].str.split().str.len() < num_words_build)]


def get_training_eval_log_scores_data_continuous(num_words_build, num_words_punch,
                                                 directory="/content/drive/MyDrive/Final_Project_940/Clean_Joke_Data.csv"):
    total_data = pd.read_csv(directory)
    total_data = get_short_jokes(total_data, num_words_build, num_words_punch)
    # Seperating training data and evaluation data.

    # Making the scores ints.
    total_data = total_data[total_data["score"].str.isnumeric() == True]
    total_data["score"] = pd.to_numeric(total_data["score"], downcast="float")
    total_data.reset_index(inplace=True, drop=True)

    total_data["score"] = np.log10(total_data['score'].values + 1) + 1
    # Normalizing the above:

    # Plotting the log scores
    _ = plt.hist(total_data['score'].values, bins=900)  # arguments are passed to np.histogram
    plt.title("Histogram of log scores")
    plt.show()

    train_data = total_data[:-2000]
    eval_data = total_data[-2000:]

    return train_data, eval_data


def get_training_eval_log_scores_data_discrete(num_words_build,
                                               num_words_punch, directory="/content/drive/MyDrive/Final_Project_940/Clean_Joke_Data.csv"):
    total_data = pd.read_csv(directory)
    total_data = get_short_jokes(total_data, num_words_build, num_words_punch)
    # Seperating training data and evaluation data.

    # Making the scores ints.
    total_data = total_data[total_data["score"].str.isnumeric() == True]
    total_data["score"] = pd.to_numeric(total_data["score"])
    total_data.reset_index(inplace=True, drop=True)

    # Logging the scores in the train_data
    for i in range(0, len(total_data)):
        if total_data["score"][i] >= 1:
            total_data['score'][i] = min(np.log10(total_data['score'][i] + 1) + 1, 5)
    total_data["score"] = total_data["score"] + 1

    # Plotting the log scores
    _ = plt.hist(total_data['score'].values, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of log scores")
    plt.show()

    # Printing the categories:
    print("Categories:")
    print("------------------")
    print("1 ---> 0 likes")
    print("2 ---> 1-8 likes")
    print("3 ---> 9-98 likes")
    print("4 ---> 99- 998 likes")
    print("5 ---> 999-9998 likes")
    print("6 ---> 9999+ likes")

    train_data = total_data[:-2000]
    eval_data = total_data[-2000:]

    return train_data, eval_data


def get_training_eval_log_scores_data(num_words_build, num_words_punch,
                                      directory="/content/drive/MyDrive/Final_Project_940/Clean_Joke_Data.csv",
                                      is_discrete=False):
    if is_discrete:
        return get_training_eval_log_scores_data_discrete(num_words_build, num_words_punch, directory)
    else:
        return get_training_eval_log_scores_data_continuous(num_words_build, num_words_punch, directory)
