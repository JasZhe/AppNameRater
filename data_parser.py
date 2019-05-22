import csv
import pickle
import math

RAW_DATA_FILE = 'data.pickle'
GOOGLE_APP_DATA = 'data/googleplaystore.csv'

def load_data():
    data = []
    with open(GOOGLE_APP_DATA, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    with open(RAW_DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    return data

def get_app_info():
    with open(RAW_DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    app_title = []
    app_rating = []
    for app_info in data[1:]:
        cur_rating = float(app_info[2])
        if not math.isnan(cur_rating):
            cur_rating = int(cur_rating * 1)
            if cur_rating <= 5 and cur_rating >= 1:
                app_title.append(app_info[1].replace('_', '') + " " + app_info[0]) # add category into this at the beginning
                app_rating.append(cur_rating - 1)

    return app_title, app_rating