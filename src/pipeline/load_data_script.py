
#######################################
# Script to extract data from gzip file, process it and save it as csv
#######################################

import json
from collections import Counter
from pre_processing_functs import process_data, find_relevant_ratings, process_books, load_data
from genres_fetching_functs import find_genres_parallel
import pandas as pd

interactions_DIR = 'data/raw/goodreads_interactions_dedup.json.gz' 
books_DIR = 'data/raw/goodreads_books.json.gz'
works_DIR = 'data/raw/goodreads_book_works.json.gz'

######################################
# pick only the most read 1k books and most active 1k users
#######################################
subset_size = 1000
out_users, out_books = process_data(interactions_DIR, stop=1e8, step = 1e6)

# merge counters for each batch
out_users_cum = Counter()
for x in out_users:
    out_users_cum = out_users_cum+x

out_books_cum = Counter()
for x in out_books:
    out_books_cum = out_books_cum+x
    
out_users_cum = Counter()
for x in out_users:
    out_users_cum = out_users_cum+x

out_books_cum = Counter()
for x in out_books:
    out_books_cum = out_books_cum+x

# use counters to find the most common users and books
top_users = dict(out_users_cum.most_common(subset_size))
top_books = dict(out_books_cum.most_common(subset_size))

top_users_id = list(top_users.keys())
top_books_id = list(top_books.keys())

# extract ratings for the selected users and books
selected_ratings, selected_uids, selected_bids = find_relevant_ratings(interactions_DIR, top_users_id, top_books_id, start=0, stop=1e8, step=1e6)

# extract book information for the selected books
out, out_book_id, out_description, out_work_id = process_books(books_DIR, top_books_id, start=0, stop=1e8, step=1e6)

# create mapping from work_id to title and best_book_id
work_data = load_data(works_DIR, start=0, end=1e8)
work_id_title = {x['work_id']:x['original_title'] for x in work_data}
work_id_best_book_id = {x['work_id']:x['best_book_id'] for x in work_data}

# extract genres for the selected books
genres = find_genres_parallel(out, max_workers=20)

dt = pd.DataFrame({'rating':selected_ratings, 'user_id':selected_uids, 'book_id':selected_bids})
other_dt = pd.DataFrame({'book_id':out_book_id, 'work_id':out_work_id, 'genres':genres})
merged_dataset = pd.merge(dt, other_dt, how='left', on='book_id')

merged_dataset['book_id'] = merged_dataset.apply(
    lambda row: work_id_best_book_id.get(row['work_id'], row['book_id']), axis = 1)

# select only a subset of the genres to binarise
merged_dataset['romance'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Romance', 'romance']))>0 else 0)
merged_dataset['history'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['History', 'history']))>0 else 0)
merged_dataset['biography'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Biography', 'biography', 'Autobiography']))>0 else 0)
merged_dataset['fantasy'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Fantasy', 'fantasy']))>0 else 0)
merged_dataset['fiction'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Fiction', 'fiction']))>0 else 0)
merged_dataset['mistery'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Mystery', 'mystery']))>0 else 0)
merged_dataset['classic'] = merged_dataset['genres'].apply(lambda x: 1 if len(set(x).intersection(['Classic', 'classic', 'Classics', 'classics']))>0 else 0)


# save everything in data/processed
merged_dataset.to_csv('data/processed/dataset_clean.csv')

with open("data/processed/user_counts.json", "w") as f:
    json.dump(dict(out_users_cum), f)
    
with open("data/processed/book_counts.json", "w") as f:
    json.dump(dict(out_books_cum), f)

with open("data/processed/selected_users_ids.json", "w") as f:
    json.dump(selected_uids, f)

with open("data/processed/selected_books_ids.json", "w") as f:
    json.dump(selected_bids, f)

with open("data/processed/ratings.json", "w") as f:
    json.dump(selected_ratings, f)

with open('data/processed/book_info.json', 'w') as f:
    json.dump([out, out_book_id, out_description, out_work_id], f)

# should not raise exception if all books are processed correctly
# raises exception if only a subset of the dataset is used
if merged_dataset.isna().any().any():
    raise Exception("Merged dataset contains NaN values. Please check the data processing steps.")

# save everything in data/processed
with open("data/processed/user_counts.json", "w") as f:
    json.dump(dict(out_users_cum), f)
    
with open("data/processed/book_counts.json", "w") as f:
    json.dump(dict(out_books_cum), f)

with open("data/processed/selected_users_ids.json", "w") as f:
    json.dump(selected_uids, f)

with open("data/processed/selected_books_ids.json", "w") as f:
    json.dump(selected_bids, f)

with open("data/processed/ratings.json", "w") as f:
    json.dump(selected_ratings, f)

with open('data/processed/book_info.json', 'w') as f:
    json.dump([out, out_book_id, out_description, out_work_id], f)
    

    
