from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import arxiv, torch, requests, os, hashlib
from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup
import gradio as gr
import tempfile
import shutil

# Load models and tokenizers once
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
device = "cuda" if torch.cuda.is_available() else "cpu"
hypo_model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-sci-simplify").to(device)
hypo_tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-sci-simplify", legacy=False)

def grobid_extract(pdf_path, output_dir="./grobid_out"):
    os.makedirs(output_dir, exist_ok=True)
    client = GrobidClient(config_path="config.json")
    client.process("processFulltextDocument", pdf_path, output=output_dir, consolidate_header=True)

    tei_files = [f for f in os.listdir(output_dir) if f.endswith(".tei.xml")]
    if not tei_files:
        raise FileNotFoundError(f"No TEI XML file found in {output_dir}")

    tei_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
    tei_path = os.path.join(output_dir, tei_files[0])

    with open(tei_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_sections_from_tei(tei_xml):
    soup = BeautifulSoup(tei_xml, "xml")

    def get_text(tag):
        return " ".join([p.get_text(separator=" ") for p in soup.find_all(tag)])

    title = soup.find("titleStmt").find("title").text if soup.find("titleStmt") and soup.find("titleStmt").find("title") else "Unknown Title"
    abstract = get_text("abstract") or "No abstract available."
    fulltext = get_text("p") or "No fulltext extracted."
    return {"title": title, "abstract": abstract, "fulltext": fulltext}

def extract_pdf_with_grobid(pdf_url, filename="temp.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch PDF from ArXiv.")
        tmp_file.write(response.content)
        tmp_pdf_path = tmp_file.name
    
    tei = grobid_extract(tmp_pdf_path)
    return extract_sections_from_tei(tei)

# NLP Functions
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    output = model.generate(inputs["input_ids"], max_length=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_hypothesis(summary):
    prompt = f"""A hypothesis is a testable statement of what you believe to be true. It is a prediction of the relationship between two or more variables.
You are a scientific researcher. Based on the following summary of a research paper, generate a **novel**, **specific**, and **testable** scientific hypothesis that is inspired by the paper’s findings, but not merely a repetition.

Summary:
{summary}

Hypothesis (1-2 sentences):"""
    inputs = hypo_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(hypo_model.device)
    output = hypo_model.generate(inputs["input_ids"], max_length=150)
    return hypo_tokenizer.decode(output[0], skip_special_tokens=True)

# ArXiv Search
def search_arxiv(query, max_results=1):
    search = arxiv.Search(query=query + " AND cat:cs.AI", max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    return list(search.results())

# Main Pipeline
def pipeline(query, n_hypotheses=3):
    try:
        results = search_arxiv(query)
        if not results:
            return "No results found.", "", "", ""

        paper = results[0]
        sections = extract_pdf_with_grobid(paper.pdf_url)

        # Smart truncation
        tokens = tokenizer.tokenize(sections["abstract"] + "\n\n" + sections["fulltext"])
        truncated = tokenizer.convert_tokens_to_string(tokens[:4000])

        summary = summarize(truncated)
        hypotheses = "\n\n".join([f"- {generate_hypothesis(summary)}" for _ in range(n_hypotheses)])

        return paper.title, sections["abstract"], summary, hypotheses
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

# Gradio Interface
demo = gr.Interface(
    fn=pipeline,
    inputs=[
        gr.Textbox(placeholder="Enter an AI research topic...", label="Search Query"),
        gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Number of Hypotheses")
    ],
    outputs=[
        gr.Textbox(label="Paper Title"),
        gr.Textbox(label="Abstract"),
        gr.Textbox(label="Summarized Content"),
        gr.Textbox(label="Generated Hypotheses (Multiple)")
    ],
    title="Scientific Hypothesis Generator",
    description="Searches ArXiv, parses the paper with GROBID, summarizes it, and generates novel hypotheses."
)
demo.launch()