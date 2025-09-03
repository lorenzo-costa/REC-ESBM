import requests
from bs4 import BeautifulSoup
import concurrent.futures

def fetch_and_parse_genres(book):
    """Fetches and parses genres for a single book's URL."""
    
    url = book.get('link')
    if not url:
        print("Warning: Book entry missing 'link'")
        return [] # Return empty list if no link

    headers = {'User-Agent': 'Mozilla/5.0'} # Be a good citizen
    genres_for_book = []
    try:
        # Consider adding a timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.content, "html.parser")
        # Updated selector based on potential Goodreads changes (check current site structure)
        # Common patterns: data-testid="genres" or specific class names
        genre_elements = soup.select('a[href*="/genres/"]') # Example selector, ADJUST AS NEEDED
        
        # If the above doesn't work, revert to original or inspect the page source:
        if not genre_elements:
             genre_spans = soup.find_all('span', class_='BookPageMetadataSection__genreButton') # Original selector
             genres_for_book = [span.get_text(strip=True) for span in genre_spans]
        else:
             # Process the elements found by the selector
             # This might need adjustment depending on the exact HTML structure
             genres_for_book = [elem.get_text(strip=True) for elem in genre_elements if '/genres/' in elem.get('href', '')]
             # Simple deduplication if needed
             genres_for_book = list(dict.fromkeys(genres_for_book))


        if not genres_for_book:
            print(f"Warning: No genres found for {book.get('title'), url}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        # Decide how to handle errors: return empty list, None, or raise exception
        return []
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return [] # Return empty list on parsing error

    return genres_for_book

# --- Parallel execution function ---
def find_genres_parallel(book_list, max_workers=10):
    """Finds genres for a list of books in parallel using threads."""
    all_genres = []
    # Use ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map applies the function to each item in book_list concurrently
        # It returns results in the order the tasks were submitted
        results = executor.map(fetch_and_parse_genres, book_list)
        all_genres = list(results) # Convert the iterator to a list

    return all_genres