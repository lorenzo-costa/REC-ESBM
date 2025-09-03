########################
# Functions used for loading .gz file 
############################

import gzip
import json
from collections import Counter
from tabnanny import verbose



def load_data(file_name, start, end):
    """Loads a gzip file as JSON object.

    Parameters
    ----------
    file_name : str
        Path to the gzip-compressed file
    start : int
        The starting line number (0-based, inclusive) from which to begin loading data.
    end : int
        The maximum number of lines to process (0-based, inclusive).

    Returns
    -------
    list
        A list of JSON objects loaded from the specified range of lines.
    """
    
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            count += 1
            if count < start:
                continue
            if count > end:
                break
            
            d = json.loads(l)
            data.append(d)
    return data

def process_data(dir, stop, step, verbose=False):
    """Loads a gzip file and return user-book pairs where book is read

    Parameters
    ----------
    dir : str
        Directory containing the gzip file
    stop : int
        Maximum number of lines to process
    step : int
        Number of lines to process in each batch
    verbose : bool, optional
        If True, print progress messages, by default False

    Returns
    -------
    tuple
        A tuple containing the user IDs and book IDs.
    """
    
    temp = [0]
    start = 0
    if stop < step:
        end = stop
    else:
        end = step
    
    out_users = []
    out_books = []
    count = 0
    while count < stop:
        temp = load_data(dir, start=start, end = end)
        if len(temp) == 0: #stops if file is empty
            break
        start = end+1
        end = start + step
        user_id = []
        book_id = []
        
        for x in temp:
            if x['is_read'] is True:
                user_id.append(x['user_id'])
                book_id.append(x['book_id'])
                
        out_users.append(Counter(user_id))
        out_books.append(Counter(book_id))
        count += len(temp)
        del temp
        del user_id
        del book_id
        
        if verbose is True:
            print(count)
        
    return out_users, out_books

def find_relevant_ratings(dir, to_take_u, to_take_b, start=0, stop=100, step = 1, verbose=False):
    """Find relevant ratings for user-book pairs.

    Parameters
    ----------
    dir : str
        Directory containing the gzip file
    to_take_u : list
        List of user IDs to consider
    to_take_b : list
        List of book IDs to consider
    start : int, optional
        Starting line number (0-based, inclusive), by default 0
    stop : int, optional
        Stopping line number (0-based, inclusive), by default 100
    step : int, optional
        Number of lines to process in each batch, by default 1
    verbose : bool, optional
        If True, print progress messages, by default False

    Returns
    -------
    tuple
        A tuple containing the relevant ratings, user IDs, and book IDs.
    """
    
    temp = [0]
    start = 0
    count = 0
    end = step
    out_ratings = []
    out_users = []
    out_books = []
    while count < stop:
        temp = load_data(dir, start=start, end = end)
        
        if len(temp) == 0:
            break
        start = end+1
        end = start + step
        
        for x in temp:
            if x['is_read'] is True:
                if x['user_id'] in to_take_u and x['book_id'] in to_take_b:
                    out_ratings.append(x['rating'])
                    out_users.append(x['user_id'])
                    out_books.append(x['book_id'])
    
        count += len(temp)
        del temp
        
        if verbose is True:
            print(count)
        
    return out_ratings, out_users, out_books

def process_books(dir, to_take, start, stop, step, verbose=False):
    temp = [0]
    start = 0
    count = 0
    end = step
    out = []
    out_book_id = []
    out_description = []
    out_work_id = []
    while count < stop:
        temp = load_data(dir, start=start, end = end)
        
        if len(temp) == 0:
            break
        start = end+1
        end = start + step
        
        for x in temp:
            if x['book_id'] in to_take:
                out.append(x)
                out_book_id.append(x['book_id'])
                out_description.append(x['description'])
                out_work_id.append(x['work_id'])
    
        count += len(temp)
        del temp
        
        if verbose is True:
            print(count)
        
    return out, out_book_id, out_description, out_work_id

def find_relevant_ratings(dir, to_take_u, to_take_b, start=0, stop=100, step = 1, verbose=False):
    temp = [0]
    start = 0
    count = 0
    end = step
    out_ratings = []
    out_users = []
    out_books = []
    while count < stop:
        temp = load_data(dir, start=start, end = end)
        
        if len(temp) == 0:
            break
        start = end+1
        end = start + step
        
        for x in temp:
            if x['is_read'] is True:
                if x['user_id'] in to_take_u and x['book_id'] in to_take_b:
                    out_ratings.append(x['rating'])
                    out_users.append(x['user_id'])
                    out_books.append(x['book_id'])
    
        count += len(temp)
        del temp
        
        if verbose is True:
            print(count)
        
    return out_ratings, out_users, out_books

def process_books(dir, to_take, start, stop, step, verbose=False):
    temp = [0]
    start = 0
    count = 0
    end = step
    out = []
    out_book_id = []
    out_description = []
    out_work_id = []
    while count < stop:
        temp = load_data(dir, start=start, end = end)
        if len(temp) == 0:
            break
        start = end+1
        end = start + step
        
        for x in temp:
            if x['book_id'] in to_take:
                out.append(x)
                out_book_id.append(x['book_id'])
                out_description.append(x['description'])
                out_work_id.append(x['work_id'])
    
        count += len(temp)
        del temp

        if verbose is True:
            print(count)

    return out, out_book_id, out_description, out_work_id



