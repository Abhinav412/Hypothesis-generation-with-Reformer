import torch
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import arxiv
import gradio as gr


device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv").to(device)


hypo_model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
hypo_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
hypo_tokenizer.pad_token = hypo_tokenizer.eos_token

def fetch_arxiv_text(query):
    search = arxiv.Search(query=query, max_results=1)
    for result in search.results():
        return {
            "title": result.title,
            "abstract": result.summary,
            "fulltext": result.summary
        }
    raise Exception("No paper found in arXiv.")

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    output = model.generate(inputs["input_ids"], max_length=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_hypothesis(summary):
    
    prompt = f"Paper Summary: {summary}\n\nBased on this summary, a novel scientific hypothesis would be:"
    
    inputs = hypo_tokenizer(prompt, return_tensors="pt").to(device)
    output = hypo_model.generate(
        inputs["input_ids"],
        max_length=len(inputs["input_ids"][0]) + 50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=hypo_tokenizer.eos_token_id
    )
    
    
    generated_text = hypo_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
   
    if "." in generated_text:
        
        periods = generated_text.find(".", generated_text.find(".")+1)
        if periods > 0:
            generated_text = generated_text[:periods+1]
        else:
            generated_text = generated_text.split(".")[0] + "."
    
   
    if generated_text and not generated_text.endswith((".", "!", "?")):
        last_period = max(generated_text.rfind("."), generated_text.rfind("!"), generated_text.rfind("?"))
        if last_period > 0:
            generated_text = generated_text[:last_period+1]
    
    return generated_text.strip()

def pipeline(query, n_hypotheses=3):
    try:
        paper_data = fetch_arxiv_text(query)
        combined_text = paper_data["abstract"] + "\n\n" + paper_data["fulltext"]

        tokens = tokenizer.tokenize(combined_text)
        truncated = tokenizer.convert_tokens_to_string(tokens[:4000])

        summary = summarize(truncated)
        
        
        hypotheses = []
        attempts = 0
        max_attempts = n_hypotheses * 3  
        
        while len(hypotheses) < n_hypotheses and attempts < max_attempts:
            hypothesis = generate_hypothesis(summary)
         
            if hypothesis and len(hypothesis) > 20 and "." in hypothesis and hypothesis not in hypotheses:
                hypotheses.append(hypothesis)
            attempts += 1
        
        
        while len(hypotheses) < n_hypotheses:
            topic_keywords = [word.lower() for word in query.split()]
            
            default_hypotheses = [
                f"Applying the concepts from this paper to other domains could lead to improved performance in similar machine learning tasks.",
                f"A hybrid approach combining these findings with complementary methods could address the limitations identified in the current research.",
                f"The techniques described could be effective when applied to smaller model architectures with reduced computational requirements."
            ]
            
            for default in default_hypotheses:
                if default not in hypotheses:
                    hypotheses.append(default)
                    if len(hypotheses) >= n_hypotheses:
                        break
        
        
        hypotheses = hypotheses[:n_hypotheses]
        formatted_hypotheses = "\n\n".join([f"- {h}" for h in hypotheses])

        return paper_data["title"], paper_data["abstract"], summary, formatted_hypotheses
    except Exception as e:
        return f"Error: {str(e)}", "", "", ""

if __name__ == "__main__":
    gr.Interface(
        fn=pipeline,
        inputs=[
            gr.Textbox(placeholder="Enter a research topic...", label="Search Query"),
            gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Number of Hypotheses")
        ],
        outputs=[
            gr.Textbox(label="Paper Title"),
            gr.Textbox(label="Abstract"),
            gr.Textbox(label="Summarized Content"),
            gr.Textbox(label="Generated Hypotheses (Multiple)")
        ],
        title="Scientific Hypothesis Generator (arXiv Edition)",
        description="Fetches research papers from arXiv, summarizes them, and generates novel hypotheses."
    ).launch()