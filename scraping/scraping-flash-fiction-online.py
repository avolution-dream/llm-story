"""
Scraping stories from https://www.flashfictiononline.com.
"""

import os
import requests
from bs4 import BeautifulSoup


# List of categories
category_list = [
    'classic-flash', 
    'fantasy', 
    'horror', 
    'humor', 
    'literary', 
    'mainstream', 
    'science-fiction'
]

# Set the base URL
base_url = 'https://www.flashfictiononline.com/article-categories/'


def get_story_links(page_url):
    """
    Get the links from the category page.
    """
    
    # Get the response
    try:
        response = requests.get(page_url)
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {page_url}: {e}")
        return []
    
    # Find the article division
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')

    # Get the links from the division
    story_links = []
    for article in articles:
        figure = article.find('figure', class_='post-image')
        if figure:
            a_tag = figure.find('a')
            if a_tag and 'href' in a_tag.attrs:
                story_links.append(a_tag['href'])

    return story_links


def get_html(url):
    """
    Get the html texts.
    """
    response = requests.get(url)
    return response.text if response.status_code == 200 else ''


def get_story_details(html_content):
    """
    Get the story text in the html contents.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title_tag = soup.find('span', class_='main-head')
    title = title_tag.get_text(strip=True) if title_tag else 'Title not found'

    story_div = soup.find('div', class_='module module-post-content tb_iy83113')
    if story_div:
        paragraphs = story_div.find_all('p')
        story_content = '\n\n'.join(paragraph.get_text(strip=True) for paragraph in paragraphs)
        story_content = story_content.split("Share this")[0].strip()
    else:
        story_content = 'Story content not found'

    return title, story_content


def main():
    """
    Get stories from the base urls and download txts.
    """
    for category in category_list:
        
        # Create directory structure if it doesn't exist
        dir_path = f'./flash-fiction-online/{category}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Iterate over pages
        for page in range(1, 10):
            page_url = f'{base_url}{category}/page/{page}/'
            story_links = get_story_links(page_url)

            for link in story_links:
                html_content = get_html(link)
                title, story_content = get_story_details(html_content)

                formatted_title = title.replace(' ', '-').lower()
                file_path = os.path.join(dir_path, f'{formatted_title}.txt')

                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f"Title: {title}\n\n{story_content}")

                print(f"Saved: {file_path}")

                
if __name__ == '__main__':
    main()