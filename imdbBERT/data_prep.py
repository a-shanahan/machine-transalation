import glob
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def get_text_list_from_files(files: list) -> list:
    """
    Read in data from files
    :param files: File list
    :return: Text list
    """
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name: str) -> pd.DataFrame:
    """
    Extract data from downloaded files
    :param folder_name: Directory name
    :return: Dataframe containing text
    """
    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


def cln_text(text: str) -> bytes:
    """
    Remove punctuation and html tags from text
    :param text: Text to clean
    :return: Cleansed text
    """
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+-/:;<=>?@\^_`{|}~.,"), "").numpy()


def save_tf_data(tf_data: tf.data.Dataset, dir_name: str) -> None:
    path = os.path.join('data', dir_name)
    tf.data.Dataset.save(tf_data, path)


# Read in datasets
train_df = get_data_from_text_files("train")
test_df = get_data_from_text_files("test")

# Combine and cleanse
all_data = train_df.append(test_df)
all_data['review_cln'] = all_data['review'].map(cln_text)

# Split into train/test
X_train, X_test = train_test_split(all_data['review_cln'], test_size=0.33, random_state=42)
train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test)

try:
    os.mkdir('data')
except FileExistsError:
    print('Directory already exisits')

# Save data as tf records
save_tf_data(train_dataset, 'train_data')
save_tf_data(test_dataset, 'test_data')
