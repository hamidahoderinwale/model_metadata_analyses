import re
import requests
from bs4 import BeautifulSoup
import bibtexparser
from urllib.parse import urlparse, urljoin
import time

class SubjectClassificationExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_bibtex(self, bibtex_string):
        """Parse BibTeX string and extract relevant information"""
        try:
            bib_db = bibtexparser.load(bibtex_string)
            entries = bib_db.entries
            if not entries:
                return None
            
            entry = entries[0]  # Take first entry
            return {
                'title': entry.get('title', ''),
                'url': entry.get('url', ''),
                'eprint': entry.get('eprint', ''),
                'archivePrefix': entry.get('archivePrefix', ''),
                'doi': entry.get('doi', ''),
                'entry_type': entry.get('ENTRYTYPE', ''),
                'raw_entry': entry
            }
        except Exception as e:
            print(f"Error parsing BibTeX: {e}")
            return None
    
    def extract_arxiv_id(self, paper_info):
        """Extract arXiv ID from various sources"""
        # Direct eprint field
        if paper_info.get('eprint'):
            return paper_info['eprint']
        
        # From URL
        url = paper_info.get('url', '')
        arxiv_patterns = [
            r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv:([0-9]{4}\.[0-9]{4,5})'
        ]
        
        for pattern in arxiv_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_arxiv_subject_classification(self, arxiv_id):
        """Get subject classification from arXiv"""
        try:
            # arXiv API endpoint
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            
            response = self.session.get(api_url)
            response.raise_for_status()
            
            # Parse XML response
            soup = BeautifulSoup(response.content, 'xml')
            entry = soup.find('entry')
            
            if not entry:
                return None
            
            # Extract categories
            categories = []
            for category in entry.find_all('category'):
                categories.append(category.get('term'))
            
            # Extract abstract for additional context
            abstract = entry.find('summary')
            abstract_text = abstract.text.strip() if abstract else ""
            
            return {
                'primary_category': categories[0] if categories else None,
                'all_categories': categories,
                'subject_classifications': self.decode_arxiv_categories(categories),
                'abstract': abstract_text[:500] + "..." if len(abstract_text) > 500 else abstract_text
            }
            
        except Exception as e:
            print(f"Error fetching arXiv data: {e}")
            return None
    
    def decode_arxiv_categories(self, categories):
        """Decode arXiv category codes to human-readable subjects"""
        category_mapping = {
            # Physics - Astrophysics
            'astro-ph': 'Astrophysics',
            'astro-ph.GA': 'Astrophysics of Galaxies',
            'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
            'astro-ph.EP': 'Earth and Planetary Astrophysics',
            'astro-ph.HE': 'High Energy Astrophysical Phenomena',
            'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
            'astro-ph.SR': 'Solar and Stellar Astrophysics',
            
            # Physics - Condensed Matter
            'cond-mat': 'Condensed Matter Physics',
            'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
            'cond-mat.mtrl-sci': 'Materials Science',
            'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
            'cond-mat.other': 'Other Condensed Matter',
            'cond-mat.quant-gas': 'Quantum Gases',
            'cond-mat.soft': 'Soft Condensed Matter',
            'cond-mat.stat-mech': 'Statistical Mechanics',
            'cond-mat.str-el': 'Strongly Correlated Electrons',
            'cond-mat.supr-con': 'Superconductivity',
            
            # Physics - General Relativity and Quantum Cosmology
            'gr-qc': 'General Relativity and Quantum Cosmology',
            
            # Physics - High Energy Physics
            'hep-ex': 'High Energy Physics - Experiment',
            'hep-lat': 'High Energy Physics - Lattice',
            'hep-ph': 'High Energy Physics - Phenomenology',
            'hep-th': 'High Energy Physics - Theory',
            
            # Physics - Mathematical Physics
            'math-ph': 'Mathematical Physics',
            
            # Physics - Nonlinear Sciences
            'nlin': 'Nonlinear Sciences',
            'nlin.AO': 'Adaptation and Self-Organizing Systems',
            'nlin.CG': 'Cellular Automata and Lattice Gases',
            'nlin.CD': 'Chaotic Dynamics',
            'nlin.SI': 'Exactly Solvable and Integrable Systems',
            'nlin.PS': 'Pattern Formation and Solitons',
            
            # Physics - Nuclear
            'nucl-ex': 'Nuclear Experiment',
            'nucl-th': 'Nuclear Theory',
            
            # Physics - General
            'physics': 'Physics',
            'physics.acc-ph': 'Accelerator Physics',
            'physics.app-ph': 'Applied Physics',
            'physics.ao-ph': 'Atmospheric and Oceanic Physics',
            'physics.atom-ph': 'Atomic Physics',
            'physics.atm-clus': 'Atomic and Molecular Clusters',
            'physics.bio-ph': 'Biological Physics',
            'physics.chem-ph': 'Chemical Physics',
            'physics.class-ph': 'Classical Physics',
            'physics.comp-ph': 'Computational Physics',
            'physics.data-an': 'Data Analysis, Statistics and Probability',
            'physics.flu-dyn': 'Fluid Dynamics',
            'physics.gen-ph': 'General Physics',
            'physics.geo-ph': 'Geophysics',
            'physics.hist-ph': 'History and Philosophy of Physics',
            'physics.ins-det': 'Instrumentation and Detectors',
            'physics.med-ph': 'Medical Physics',
            'physics.optics': 'Optics',
            'physics.soc-ph': 'Physics and Society',
            'physics.ed-ph': 'Physics Education',
            'physics.plasm-ph': 'Plasma Physics',
            'physics.pop-ph': 'Popular Physics',
            'physics.space-ph': 'Space Physics',
            
            # Physics - Quantum Physics
            'quant-ph': 'Quantum Physics',
            
            # Mathematics
            'math': 'Mathematics',
            'math.AG': 'Algebraic Geometry',
            'math.AT': 'Algebraic Topology',
            'math.AP': 'Analysis of PDEs',
            'math.CT': 'Category Theory',
            'math.CA': 'Classical Analysis and ODEs',
            'math.CO': 'Combinatorics',
            'math.AC': 'Commutative Algebra',
            'math.CV': 'Complex Variables',
            'math.DG': 'Differential Geometry',
            'math.DS': 'Dynamical Systems',
            'math.FA': 'Functional Analysis',
            'math.GM': 'General Mathematics',
            'math.GN': 'General Topology',
            'math.GT': 'Geometric Topology',
            'math.GR': 'Group Theory',
            'math.HO': 'History and Overview',
            'math.IT': 'Information Theory',
            'math.KT': 'K-Theory and Homology',
            'math.LO': 'Logic',
            'math.MP': 'Mathematical Physics',
            'math.MG': 'Metric Geometry',
            'math.NT': 'Number Theory',
            'math.NA': 'Numerical Analysis',
            'math.OA': 'Operator Algebras',
            'math.OC': 'Optimization and Control',
            'math.PR': 'Probability',
            'math.QA': 'Quantum Algebra',
            'math.RT': 'Representation Theory',
            'math.RA': 'Rings and Algebras',
            'math.SP': 'Spectral Theory',
            'math.ST': 'Statistics Theory',
            'math.SG': 'Symplectic Geometry',
            
            # Computer Science
            'cs': 'Computer Science',
            'cs.AI': 'Artificial Intelligence',
            'cs.CL': 'Computation and Language',
            'cs.CC': 'Computational Complexity',
            'cs.CE': 'Computational Engineering, Finance, and Science',
            'cs.CG': 'Computational Geometry',
            'cs.GT': 'Computer Science and Game Theory',
            'cs.CV': 'Computer Vision and Pattern Recognition',
            'cs.CY': 'Computers and Society',
            'cs.CR': 'Cryptography and Security',
            'cs.DS': 'Data Structures and Algorithms',
            'cs.DB': 'Databases',
            'cs.DL': 'Digital Libraries',
            'cs.DM': 'Discrete Mathematics',
            'cs.DC': 'Distributed, Parallel, and Cluster Computing',
            'cs.ET': 'Emerging Technologies',
            'cs.FL': 'Formal Languages and Automata Theory',
            'cs.GL': 'General Literature',
            'cs.GR': 'Graphics',
            'cs.AR': 'Hardware Architecture',
            'cs.HC': 'Human-Computer Interaction',
            'cs.IR': 'Information Retrieval',
            'cs.IT': 'Information Theory',
            'cs.LO': 'Logic in Computer Science',
            'cs.LG': 'Machine Learning',
            'cs.MS': 'Mathematical Software',
            'cs.MA': 'Multiagent Systems',
            'cs.MM': 'Multimedia',
            'cs.NI': 'Networking and Internet Architecture',
            'cs.NE': 'Neural and Evolutionary Computing',
            'cs.NA': 'Numerical Analysis',
            'cs.OS': 'Operating Systems',
            'cs.OH': 'Other Computer Science',
            'cs.PF': 'Performance',
            'cs.PL': 'Programming Languages',
            'cs.RO': 'Robotics',
            'cs.SI': 'Social and Information Networks',
            'cs.SE': 'Software Engineering',
            'cs.SD': 'Sound',
            'cs.SC': 'Symbolic Computation',
            'cs.SY': 'Systems and Control',
            
            # Quantitative Biology
            'q-bio': 'Quantitative Biology',
            'q-bio.BM': 'Biomolecules',
            'q-bio.CB': 'Cell Behavior',
            'q-bio.GN': 'Genomics',
            'q-bio.MN': 'Molecular Networks',
            'q-bio.NC': 'Neurons and Cognition',
            'q-bio.OT': 'Other Quantitative Biology',
            'q-bio.PE': 'Populations and Evolution',
            'q-bio.QM': 'Quantitative Methods',
            'q-bio.SC': 'Subcellular Processes',
            'q-bio.TO': 'Tissues and Organs',
            
            # Quantitative Finance
            'q-fin': 'Quantitative Finance',
            'q-fin.CP': 'Computational Finance',
            'q-fin.EC': 'Economics',
            'q-fin.GN': 'General Finance',
            'q-fin.MF': 'Mathematical Finance',
            'q-fin.PM': 'Portfolio Management',
            'q-fin.PR': 'Pricing of Securities',
            'q-fin.RM': 'Risk Management',
            'q-fin.ST': 'Statistical Finance',
            'q-fin.TR': 'Trading and Market Microstructure',
            
            # Statistics
            'stat': 'Statistics',
            'stat.AP': 'Applications',
            'stat.CO': 'Computation',
            'stat.ML': 'Machine Learning',
            'stat.ME': 'Methodology',
            'stat.OT': 'Other Statistics',
            'stat.TH': 'Statistics Theory',
            
            # Electrical Engineering and Systems Science
            'eess': 'Electrical Engineering and Systems Science',
            'eess.AS': 'Audio and Speech Processing',
            'eess.IV': 'Image and Video Processing',
            'eess.SP': 'Signal Processing',
            'eess.SY': 'Systems and Control',
            
            # Economics
            'econ': 'Economics',
            'econ.EM': 'Econometrics',
            'econ.GN': 'General Economics',
            'econ.TH': 'Theoretical Economics'
        }
        
        decoded = []
        for cat in categories:
            if cat in category_mapping:
                decoded.append(f"{cat}: {category_mapping[cat]}")
            else:
                decoded.append(f"{cat}: Unknown category")
        
        return decoded
    
    def get_doi_subject_classification(self, doi):
        """Attempt to get subject classification from DOI (basic implementation)"""
        try:
            # Try to resolve DOI
            doi_url = f"https://doi.org/{doi}"
            response = self.session.get(doi_url, allow_redirects=True)
            
            # This is a basic implementation - different publishers have different formats
            # You might need to customize this based on the specific publishers you're dealing with
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for common subject/keyword indicators
            subjects = []
            
            # Check for keywords meta tags
            keywords_meta = soup.find('meta', {'name': 'keywords'})
            if keywords_meta:
                subjects.extend(keywords_meta.get('content', '').split(','))
            
            # Check for subject classifications in various formats
            subject_tags = soup.find_all(['span', 'div', 'p'], class_=re.compile(r'subject|keyword|classification', re.I))
            for tag in subject_tags:
                if tag.text.strip():
                    subjects.append(tag.text.strip())
            
            return {
                'subjects': list(set(subjects)),
                'url': response.url
            }
            
        except Exception as e:
            print(f"Error fetching DOI data: {e}")
            return None
    
    def process_bibtex_citation(self, bibtex_string):
        """Main function to process BibTeX citation and extract subject classification"""
        print("Parsing BibTeX citation...")
        paper_info = self.parse_bibtex(bibtex_string)
        
        if not paper_info:
            return {"error": "Failed to parse BibTeX citation"}
        
        print(f"Paper title: {paper_info['title']}")
        
        results = {
            'title': paper_info['title'],
            'classifications': {}
        }
        
        # Try arXiv first
        arxiv_id = self.extract_arxiv_id(paper_info)
        if arxiv_id:
            print(f"Found arXiv ID: {arxiv_id}")
            print("Fetching arXiv classification...")
            arxiv_data = self.get_arxiv_subject_classification(arxiv_id)
            if arxiv_data:
                results['classifications']['arxiv'] = arxiv_data
                print(f"Primary category: {arxiv_data['primary_category']}")
                print(f"All categories: {', '.join(arxiv_data['all_categories'])}")
        
        # Try DOI if available
        doi = paper_info.get('doi')
        if doi:
            print(f"Found DOI: {doi}")
            print("Fetching DOI-based classification...")
            time.sleep(1)  # Be respectful to servers
            doi_data = self.get_doi_subject_classification(doi)
            if doi_data:
                results['classifications']['doi'] = doi_data
        
        # If no arXiv or DOI, try to extract from URL
        url = paper_info.get('url')
        if url and not arxiv_id and not doi:
            print(f"Trying to extract from URL: {url}")
            # This would need custom implementation based on the specific sites
            
        return results

# Example usage
def main():
    extractor = SubjectClassificationExtractor()
    
    # Example BibTeX citation (replace with your actual citation)
    sample_bibtex = """
    @article{vaswani2017attention,
      title={Attention is all you need},
      author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Łukasz and Polosukhin, Illia},
      journal={Advances in neural information processing systems},
      volume={30},
      year={2017},
      eprint={1706.03762},
      archivePrefix={arXiv}
    }
    """
    
    # Process the citation
    result = extractor.process_bibtex_citation(sample_bibtex)
    
    print("\n" + "="*50)
    print("SUBJECT CLASSIFICATION RESULTS")
    print("="*50)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Title: {result['title']}")
    print()
    
    if 'arxiv' in result['classifications']:
        arxiv_data = result['classifications']['arxiv']
        print("arXiv Classification:")
        print(f"  Primary: {arxiv_data['primary_category']}")
        print("  All categories:")
        for cat in arxiv_data['subject_classifications']:
            print(f"    - {cat}")
        print()
    
    if 'doi' in result['classifications']:
        doi_data = result['classifications']['doi']
        print("DOI-based subjects:")
        for subject in doi_data['subjects']:
            print(f"  - {subject}")
        print()

if __name__ == "__main__":
    # Make sure to install required packages:
    # pip install requests beautifulsoup4 bibtexparser lxml
    main()
