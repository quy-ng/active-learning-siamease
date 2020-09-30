import copy
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import nlpaug.augmenter.char as nac

__all__ = ['augment']

from ultils import synonym_dict


def augment(data_frame: pd.DataFrame, columns: tuple = ("name", "address"), rep_columns: tuple = ("address",)):
    df_generated = generate_new_rows(data_frame[list(columns)], synonym_dict, list(rep_columns), low=2, high=3)
    df_generated = pre_processing_pipeline(df_generated, rep_columns[0])
    df_generated = pd.concat([data_frame[list(columns)], df_generated], ignore_index=True)

    ind = helper_find_index(list(columns), list(rep_columns))

    new_rows = []
    # Augment by fault of human keyboard
    for row in df_generated.itertuples():
        # Augment data base on keyboard mistakes and deletion
        new_row = list(row)[1:]
        aug = nac.KeyboardAug()
        augmented_texts = aug.augment(new_row[ind], n=2)
        for text in augmented_texts:
            tmp_row = copy.copy(new_row)
            tmp_row[ind] = text
            new_rows.append(tuple(tmp_row))
        new_row = list(row)[1:]
        aug = nac.RandomCharAug(action="delete")
        augmented_texts = aug.augment(new_row[1], n=2)
        for text in augmented_texts:
            tmp_row = copy.copy(new_row)
            tmp_row[ind] = text
            new_rows.append(tuple(tmp_row))

    augmented = pd.DataFrame(new_rows, columns=columns)

    return df_generated


def helper_find_index(list_names: list, name: str) -> int:
    """
    list_names = [a, e, b, c]
    name = b; output -> 2
    name = a; output -> 0
    """
    for i, v in enumerate(list_names):
        if v == name:
            break
    return i


def generate_new_rows(targeted_df, synonym_tabl, rep_columns, low=3, high=5):
    """
    Generate rows by using synonym and acronym with columns name
    :param targeted_df: dataframe
    :param synonym_tabl: dictionary of synonym and acronym
    :param rep_columns: column that use want to duplicate
    :param low: min duplications for each row
    :param high: max duplications for each row
    :return: new dataframe
    """
    df_cols = targeted_df.columns.to_list()
    new_rows = []
    for row in targeted_df.itertuples():
        # Random duplicating a rows from low times to high times
        for i in range(np.random.randint(low, high, 1)[0]):
            new_row = list(row)[1:]  # 1: to drop index in dataframe
            # Generate new row for selected column(s)
            for column in rep_columns:
                ind = helper_find_index(df_cols, column)
                result = generate_row(new_row[ind], synonym_tabl, syn_type=column)
                if result["code"] == 1:
                    new_row[ind] = result["content"]
            # Collect all new rows and store them into new_rows
            new_rows.append(new_row)
    df_result = pd.DataFrame(pd.DataFrame(new_rows, columns=targeted_df.columns))
    return df_result


def generate_row(target, synonym_dict, syn_type):
    """
    Generate new row by using synonym and acronym
    :param target: content
    :param synonym_dict: dictionary of synonym and acronym
    :param syn_type: type of synonym and acronym
    :return: {'code': 1 for successfull and 0 for failed, {content}: new row if code = 1
    """
    result = {"code": 0}
    try:
        target_lowered = target.lower()  # get lowered target
    except:
        # out if target_lowered is Nan
        return result
    # each cluster of synonym (name)
    for i in range(len(synonym_dict[syn_type])):
        synonym = synonym_dict[syn_type][i]
        # iterrate words in a synonyms cluster
        for j in range(len(synonym)):
            syn = synonym[j]
            target_lowered = target.lower()  # get lowered target
            match = re.search("{}".format(syn), target_lowered)
            if match is None:
                continue

            syn_len = len(syn)  # Get length of replaceable word
            target_sta = match.start()
            target_end = target_sta + syn_len
            if target_end < len(target) and (
                    target[target_end].isdigit() or target[target_end].isalpha()
            ):
                # if word neither is the last word of sentence nor a sub of a word, a number
                continue

            # gamble or not base (75%)
            if np.random.randint(0, 4, 1) == 0:
                break

            # position of synonym's replacement
            syn_rep_pos = j
            while syn_rep_pos == j:
                syn_rep_pos = np.random.randint(0, len(synonym), 1)[0]
            syn_rep = synonym[syn_rep_pos]

            # Generate new target
            target = target[:target_sta] + " " + syn_rep + " " + target[target_end:]
            target = target.lower()
            # Remove all regex special characters
            target = re.sub(r"\[\.\]", ".", target)
            target = re.sub(r"\[([a\-z.\^09]*)\]", "", target)
            target = re.sub(r"\{[0-9]+\}", "", target)
            break
    result = {"code": 1, "content": target}
    return result


def pre_processing_pipeline(df, rep_col):
    temp_df = df.copy()
    address_list = df[rep_col].values.copy()
    # All the cases are handled with a certain probability to prevent bias
    for index, target in tqdm(enumerate(address_list)):
        # ---- System Error - missing some thing
        random = np.random.randint(0, 100, 1)
        target_as_list = target.split(",")  # Split sentence as a list
        if random < 16 and len(target_as_list) > 2:
            pos_to_drop = list(set(np.random.randint(2, len(target_as_list), 3)))
            for pos in sorted(pos_to_drop, reverse=True):
                # Drop position that has been chosen
                target_as_list.pop(pos)
            target = ",".join(target_as_list)

        # ---- Zip-code case ~ 13%
        # Zipcode error
        zipcode = ["000000", "111111", "123456", "0", "1"]
        random = np.random.randint(0, 100, 1)
        if random <= 13:
            match = re.search(r"[0-9]{4,6}[\.]*$", target)
            if match is not None:
                match = match.start()
                target = (
                        target[:match] + zipcode[np.random.randint(0, len(zipcode), 1)[0]]
                )

        # ---- Misunderstanding
        random = np.random.randint(0, 100, 1)
        if random < 10:
            mis_understanding = [
                [
                    ["avenue", "[^a-zA-Z]ave[.]{1}", "[^a-zA-Z]ave[^a-zA-Z.]"],
                    ["boulevard", "[^a-zA-Z]blvd[.]{1}", "blvd[^.]"],
                    ["drive", "[^a-zA-Z]dr[^a-zA-Z.]", "[^a-zA-Z]dr[.]{1}"],
                ],
                [
                    ["lane", "[^a-zA-Z]ln[.]{1}", "[^a-zA-Z0-9]ln[^a-zA-Z.]"],
                    ["road", "rd[.]{1}", "[^a-zA-Z0-9]rd[^a-zA-Z.]"],
                ],
                [["[^a-zA-Z]zone[^a-zA-Z]"], ["[^a-zA-Z]area[^a-zA-Z]"]],
                [
                    ["[^a-zA-Z]suite[^a-zA-Z]", "[^a-zA-Z]ste[^a-zA-Z]"],
                    ["[^a-zA-Z]plot[^a-zA-Z]"],
                ],
            ]
            for each_type in mis_understanding:
                change_start = None
                change_end = None
                for c_id in range(0, len(each_type)):
                    # Loop through each class
                    c = each_type[c_id]
                    for element in c:
                        # Find if in string exists an element in each class to replace it
                        # by a random element in the other class of the same type
                        match = re.search("{}".format(element), target)
                        if match is not None:
                            change_start = match.start()
                            change_end = match.end()
                            break
                    if change_start is not None:
                        # Stop finding if it has founded that exist at least one element
                        # in the string and change it
                        change_class = c_id
                        while change_class == c_id:
                            change_class = np.random.randint(0, len(each_type), 1)[0]
                        new_class = np.random.choice(each_type[change_class])
                        target = (
                                target[:change_start]
                                + " "
                                + new_class
                                + " "
                                + target[change_end:]
                        )
                        target = target.lower()
                        # Remove all regex special characters
                        target = re.sub(r"\[\.\]", "", target)
                        target = re.sub(r"\[([a\-z.\^09]*)\]", "", target)
                        target = re.sub(r"\{[0-9]+\}", "", target)
                        break

        address_list[index] = target
    temp_df[rep_col] = address_list
    return temp_df
