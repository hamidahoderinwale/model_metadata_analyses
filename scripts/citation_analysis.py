import re
import requests
import xml.etree.ElementTree as ET

def extract_arxiv_id(bibtex_str):
    """
    Extract arXiv ID from a BibTeX entry string.
    Looks for patterns like 'arXiv:2505.14683' or 'arXiv preprint arXiv:2505.14683'.
    """
    match = re.search(r'arXiv[: ]+(\d{4}\.\d{4,5})(v\d+)?', bibtex_str, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def fetch_arxiv_metadata(arxiv_id):
    """
    Query arXiv API to get metadata XML for a given arXiv ID.
    """
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch arXiv metadata: HTTP {response.status_code}")

def parse_categories_from_xml(xml_str):
    """
    Parse arXiv categories (subject classifications) from API XML response.
    Returns a list of category shorthand strings like ['cs.DL', 'cs.CV'].
    """
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml_str)
    categories = []
    # Each entry corresponds to a paper
    for entry in root.findall('atom:entry', ns):
        for category in entry.findall('atom:category', ns):
            term = category.attrib.get('term')
            if term:
                categories.append(term)
    return categories

def get_arxiv_subject_classifications_from_bibtex(bibtex_str):
    """
    Given a BibTeX entry string, extract arXiv ID and fetch subject classifications.
    """
    arxiv_id = extract_arxiv_id(bibtex_str)
    if not arxiv_id:
        raise ValueError("No arXiv ID found in BibTeX entry.")
    xml_metadata = fetch_arxiv_metadata(arxiv_id)
    categories = parse_categories_from_xml(xml_metadata)
    return categories

# Example usage:
bibtex_entry = """
@article{deng2025bagel,
title = {Emerging Properties in Unified Multimodal Pretraining},
author = {Deng, Chaorui and Zhu, Deyao and Li, Kunchang and Gou, Chenhui and Li, Feng and Wang, Zeyu and Zhong, Shu and Yu, Weihao and Nie, Xiaonan and Song, Ziang and Shi, Guang and Fan, Haoqi},
journal = {arXiv preprint arXiv:2505.14683},
year = {2025}
}
"""

if __name__ == "__main__":
    try:
        subjects = get_arxiv_subject_classifications_from_bibtex(bibtex_entry)
        print("ArXiv subject classifications:", subjects)
    except Exception as e:
        print("Error:", e)
