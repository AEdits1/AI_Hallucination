import json
import pandas as pd

def prepare_data(input_file, output_file, max_samples=500):
    data = []
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= max_samples:
                break
            try:
                item = json.loads(line)
                # Entailment pair
                data.append({
                    'premise': item['document'],
                    'hypothesis': item['right_summary'],
                    'label': 1  # Entailment
                })
                # Contradiction/Hallucination pair
                data.append({
                    'premise': item['document'],
                    'hypothesis': item['hallucinated_summary'],
                    'label': 0  # Contradiction
                })
                count += 1
            except Exception as e:
                pass

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")

if __name__ == '__main__':
    prepare_data('HaluEval/data/summarization_data.json', 'hallucination_dataset.csv', max_samples=250)
