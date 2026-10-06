# Natural Language Processing — Lab Work

**Course:** AIC365 – Natural Language Processing Fundamentals  
**Semester:** 6th Semester

## Repository Structure & Branches

This repository is organized into branches for each lab session. The `main` branch contains base files, manuals, and assignments, while the actual lab code is located in their respective branches.

| Branch | Content |
|--------|---------|
| [`main`](https://github.com/atharkakahail/NLP-Lab-Work/tree/main) | Base repository files, Lab Manual PDF, Assignments |
| [`lab-1`](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-1) | Lab 01 — Python Essentials for NLP |
| [`lab-2`](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-2) | Lab 02 — Data Acquisition |
| [`lab-3`](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-3) | Lab 03 — Regex, Tokenization & Sentence Segmentation |

---

### [Lab 01 — Python Essentials for NLP](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-1)
Covers core Python data structures and idioms for text manipulation:
- **Task 01:** Strings as Text Storage
- **Task 02:** Lists for Token Management
- **Task 03:** Dictionaries for Word Mapping
- **Task 04:** Sets for Unique Vocabulary
- **Task 05:** Tuples for Fixed Linguistic Data
- **Task 06:** String Cleaning Methods
- **Task 07:** String Validation Methods
- **Task 08:** Functions and Lambda for NLP Pipelines
- **Task 09:** map(), filter(), sorted() and Counter

---

### [Lab 02 — Data Acquisition](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-2)
Covers practical skills for extracting text from real-world formats:
- **Lab Task 1:** PDF Text Extraction (PyPDF2)
- **Lab Task 2:** DOCX Paragraph Extraction (python-docx)
- **Lab Task 3:** JSON API Fetching (requests)
- **Lab Task 4:** HTML Parsing (BeautifulSoup)
- **Lab Task 5:** Web Scraping Structured Data

---

### [Lab 03 — Regex, Tokenization & Sentence Segmentation](https://github.com/atharkakahail/NLP-Lab-Work/tree/lab-3)
Pull structured information out of raw, unstructured text and correctly break text into words and sentences:
- **Activity 1:** Regex Extraction (Emails, Order numbers, Phone numbers)
- **Activity 2:** spaCy Token Attributes (Currency amounts and numeric quantities)
- **Activity 3:** NLTK vs spaCy Tokenization
- **Activity 4:** NLTK vs spaCy Sentence Segmentation

---

## How to Run

1. Clone the repository and switch to the desired lab branch:
	```powershell
	git clone https://github.com/atharkakahail/NLP-Lab-Work.git
	cd NLP-Lab-Work
	git checkout lab-1  # Or lab-2, lab-3
	```
2. Install [Jupyter Notebook](https://jupyter.org/install) or use VS Code with the Jupyter extension.
3. Ensure you have the required dependencies for the respective lab (e.g., `requests`, `beautifulsoup4`, `PyPDF2`, `python-docx`, `nltk`, `spacy`).
4. Open the `.ipynb` file and run the cells.
