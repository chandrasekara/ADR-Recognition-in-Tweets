def count_hashtags(filepath):
    
    #Returns a pandas series corresponding to the number of hashtags used in each tweet
    
    try:
        raw_tweets_data = open(filepath, "r")
    except:
        print "Raw tweet file could not be opened!"
        exit()

    nextline = raw_tweets_data.readline()

    num_hashtags = []

    while nextline:
        line_arr = nextline.split()

        line_num_hashtags = 0

        for token in line_arr:
            token_arr = list(token)
            if token_arr[0] == "#":
                line_num_hashtags += 1

        num_hashtags.append(line_num_hashtags)

        nextline = raw_tweets_data.readline()

    raw_tweets_data.close()

    return num_hashtags