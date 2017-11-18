import pandas as pd


def get_words_of_interest(filepath_to_arff):
    f = open(filepath_to_arff)

    f.readline()

    read_arr = f.readline().split()

    words_of_interest = []

    print read_arr

    #print read_arr[0]

    while read_arr[0] == "@ATTRIBUTE":

        words_of_interest.append(read_arr[1])

        read_arr = f.readline().split()


    f.close()


    return words_of_interest
    
def read_csv_dataset(csv_filepath):
    
    try:
        freq_data = pd.read_csv(csv_filepath)
        print "Dataset loaded!"
        return freq_data
    except:
        print "Dataset could not be loaded."
        return None

    
    