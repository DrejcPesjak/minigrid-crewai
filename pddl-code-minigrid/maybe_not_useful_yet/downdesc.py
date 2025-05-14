import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

# Base URLs for MiniGrid and BabyAI environments
BASE_URLS = {
    'MiniGrid': 'https://minigrid.farama.org/environments/minigrid/',
    'BabyAI': 'https://minigrid.farama.org/environments/babyai/'
}

def get_environment_links(base_url):
    """Fetch environment links from the base URL."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    env_links = []

    # Find all links within the environment grid
    for a_tag in soup.select('div.env-grid a[href]'):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        env_links.append(full_url)

    return list(set(env_links))  # Remove duplicates

def parse_environment_page(url):
    """Parse an individual environment page to extract details."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract environment name
    name_tag = soup.find('h1')
    name = name_tag.text.strip() if name_tag else 'Unknown'
    if name[-1] == '#':
        name = name[:-1]

    # Extract description
    description = ''
    desc_section = soup.find('section', id='description')
    if desc_section:
        desc_paragraph = desc_section.find('p')
        if desc_paragraph:
            description = desc_paragraph.text.strip()

    # Extract registered configurations
    configs = []
    reg_section = soup.find('section', id='registered-configurations')
    if reg_section:
        for li in reg_section.find_all('li'):
            code_tag = li.find('code')
            if code_tag:
                configs.append(code_tag.text.strip())

    return {
        'sub_link': url.split('/')[-1],  # Extract sub-link from URL
        'name': name,
        'description': description,
        'configs': configs
    }

def main():
    all_envs = {}

    for category, base_url in BASE_URLS.items():
        print(f'Processing {category} environments...')
        env_links = get_environment_links(base_url)
        env_data = []

        for link in env_links:
            print(f'Parsing {link}')
            data = parse_environment_page(link)
            env_data.append(data)

        all_envs[category] = env_data

    # Save the data to a JSON file
    with open('minigrid_babyai_envs.json', 'w') as f:
        json.dump(all_envs, f, indent=2)

if __name__ == '__main__':
    main()
