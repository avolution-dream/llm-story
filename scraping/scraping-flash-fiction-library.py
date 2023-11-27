"""
Scraping stories from https://flashfictionlibrary.com.
"""

import os
import requests
from bs4 import BeautifulSoup


# List of categories
category_list = [
    'fantasy', 
    'uncategorized', 
    'horror', 
    'romance', 
    'scifi', 
    'science-fiction'
]


# Base URL
base_url = 'https://flashfictionlibrary.com/category/'


def get_story_links(category_url):
    """
    Get the story link from the home url.
    """
    
    # Get the response
    try:
        response = requests.get(category_url)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {category_url}: {e}")
        return []

    # Parse the html file
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')

    # Get the links from the header of the article section
    story_links = []
    for article in articles:
        header = article.find('header', class_='entry-header')
        if header:
            a_tag = header.find('a')
            if a_tag and 'href' in a_tag.attrs:
                story_links.append(a_tag['href'])

    return story_links


def get_story_content(url):
    """
    Get story content from the url.
    """
    
    # Get the response
    response = requests.get(url)
    if response.status_code != 200:
        return 'Story content not found', ''
    
    # Parse the html file
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get the title
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True).split(' – ')[0] if title_tag else 'Title not found'
    
    # Find the story content
    story_div = soup.find('article')
    if story_div:
        paragraphs = story_div.find_all('p')
        story_content = '\n\n'.join(paragraph.get_text(strip=True) for paragraph in paragraphs)
    else:
        story_content = 'Story content not found'

    return title, story_content


def main():
    """
    Get stories from the base urls and download txts.
    """
    for category in category_list:
        dir_path = f'./flash-fiction-library/{category}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for page in range(1, 11):  # Iterate over pages 1 to 10
            page_url = f'{base_url}{category}/page/{page}/'
            story_links = get_story_links(page_url)

            for link in story_links:
                title, story_content = get_story_content(link)
                formatted_title = title.replace(' ', '-').lower()
                file_path = os.path.join(dir_path, f'{formatted_title}.txt')

                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f"Title: {title}\n\n{story_content}")

                print(f"Saved: {file_path}")

                
if __name__ == '__main__':
    main()
