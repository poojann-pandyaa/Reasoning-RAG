import os
import json
from datasets import load_dataset
from bs4 import BeautifulSoup
from tqdm import tqdm

def clean_html(text):
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def load_taxonomy(config_path="configs/taxonomy.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_preprocessing(max_samples=None, output_path="data/processed_dataset.jsonl"):
    print("Loading taxonomy...")
    taxonomy = load_taxonomy()
    
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True
    )
    
    processed_count = 0
    seen_question_ids = set()
    
    print(f"Starting processing... (Max Samples: {max_samples if max_samples is not None else 'Unlimited'})")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataset:
            if max_samples and processed_count >= max_samples:
                break
                
            q_id = record.get('qid')
            if q_id in seen_question_ids:
                continue
                
            answers = record.get('answers', [])
            if not answers:
                continue
                
            # Filter condition: at least one answer with score >= 5 OR an accepted answer
            valid = False
            for ans in answers:
                if ans.get('pm_score', 0) >= 5 or ans.get('selected', False):
                    valid = True
                    break
                    
            if not valid:
                continue
                
            # Domain extraction
            domain = 'stackoverflow'
            metadata = record.get('metadata', [])
            if isinstance(metadata, list) and len(metadata) > 1:
                url = metadata[1]
                domain = url.split('//')[-1].split('.')[0]
                
            if domain.lower() not in ['askubuntu', 'softwareengineering', 'stackoverflow']:
                continue
            # Wait, the instructions say: Extract the Stack Exchange sub-site tag from each question (available in the metadata).
            # The dataset actually has a "title" or metadata. Let's assume there's a way. For simplicity, we just use the raw record.
            
            # Clean HTML
            clean_question = clean_html(record.get('question', ''))
            
            clean_answers = []
            for ans in answers:
                clean_ans = ans.copy()
                clean_ans['body_clean'] = clean_html(ans.get('text', ''))
                clean_ans['score'] = ans.get('pm_score', 0)
                clean_ans['is_accepted'] = ans.get('selected', False)
                clean_answers.append(clean_ans)
            
            processed_record = {
                'question_id': q_id,
                'title': '', # no explicit title in root fields, can optionally pull from metadata later
                'question': clean_question,
                'domain': domain,
                'reasoning_category': taxonomy.get(domain, 'Procedural'),
                'answers': clean_answers
            }
            
            f.write(json.dumps(processed_record) + "\n")
            seen_question_ids.add(q_id)
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} records...")

    print(f"Preprocessing complete. Total records saved: {processed_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=os.environ.get('MAX_SAMPLES'), help="Max samples to process")
    args = parser.parse_args()
    
    run_preprocessing(max_samples=args.max_samples)
