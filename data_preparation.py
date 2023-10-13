import os, json
import pandas as pd


def main():
    list = []
    directory = 'sample_training_data'
    count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename)) as f:
                data = json.load(f)
                if (len(data) > 0):
                    list.append(data)
                    count += 1

        if (count == 2):
            break

    list_no_dups = remove_duplicates(list)
    pretty_print(list_no_dups)


def remove_duplicates(list):
    # concatenate all nested items into single list
    single_list = []
    for e1 in list:
        for e2 in e1:
            single_list.append(e2)

    list_no_duplicates = []
    seen = set()

    for e in single_list:
        joined_string = ''.join(e['sentence_tokens'])
        print(joined_string, "\n")
        if (joined_string not in seen):
            list_no_duplicates.append(e)

        seen.add(joined_string)

    # print("-   Seen Count: {}".format(len(single_list)))
    # print("- Original Count: {}".format(list))
    # print("-   Unique Count: {}\n".format(len(list_no_duplicates)))

    return list_no_duplicates

    
def pretty_print(list):
    for e in list:
        print(e, "\n")

main()