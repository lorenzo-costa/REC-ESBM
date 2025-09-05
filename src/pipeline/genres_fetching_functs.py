import requests
from bs4 import BeautifulSoup
import concurrent.futures

def fetch_and_parse_genres(book):
    """Fetches and parses genres for a single book's URL from Goodreads.

    Parameters
    ----------
    book : dict
        A dictionary containing book information, including the 'link' key.

    Returns
    -------
    list
        A list of genres associated with the book.
    """
       
    url = book.get('link')
    if not url:
        print("Warning: Book entry missing 'link'")
        return []

    headers = {'User-Agent': 'Mozilla/5.0'}
    genres_for_book = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        genre_elements = soup.select('a[href*="/genres/"]')
        
        if not genre_elements:
            genre_spans = soup.find_all('span', class_='BookPageMetadataSection__genreButton')
            genres_for_book = [span.get_text(strip=True) for span in genre_spans]
        else:
            genres_for_book = [elem.get_text(strip=True) for elem in genre_elements if '/genres/' in elem.get('href', '')]
            genres_for_book = list(dict.fromkeys(genres_for_book))

        if not genres_for_book:
            print(f"Warning: No genres found for {book.get('title'), url}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

    return genres_for_book

def find_genres_parallel(book_list, max_workers=10):
    """Parallelises execution of fetch_and_parse_genres to fetch data from Goodreads.

    Parameters
    ----------
    book_list : list
        A list of dictionaries containing book information.
    max_workers : int, optional
        The maximum number of worker threads to use, by default 10
    Returns
    -------
    list
        A list of lists, where each inner list contains genres for a book.
    """
    
    all_genres = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map applies the function to each item in book_list concurrently and returns 
        # results in the order the tasks were submitted
        results = executor.map(fetch_and_parse_genres, book_list)
        all_genres = list(results)

    return all_genres