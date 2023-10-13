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
                # print("data: ", data, "")
                if (len(data) > 0):
                    list.append(data)
                    count += 1

        if (count == 2):
            break
    no_dups = remove_duplicates(list)


def remove_duplicates(list):
    # concatenate all nested items into single list
    list2 = []
    for e1 in list:
        for e2 in e1:
            list2.append(e2)
            print(e2)


            

def pretty_print(list):
    for e1 in list:
        for e2 in e1:
            print(e2)
            print()
    
    

main()