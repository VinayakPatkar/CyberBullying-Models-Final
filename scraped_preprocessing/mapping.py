import openai
import csv
import json
from openai import OpenAI
def returnlabels():
    api_key = "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"
    csv_file_path = 'C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/data_store/english_comments.csv'
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data_to_classify = [row['text'] for row in reader]
    prompt_template = "Given the following sentence, determine if it constitutes cyberbullying or not. Provide a binary classification label (1 for cyberbullying, 0 for not cyberbullying). Get the proper context of the sentence and provide proper results. Only give me the 0 or 1.\n\n**Sentence:** {sentence}"
    instructions = [prompt_template.format(sentence=s) for s in data_to_classify]
    client = OpenAI(api_key=api_key)
    labels = []
    for instruction in instructions:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_template.format(sentence= "I have so much homework that I want to kill myself")}],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
        )
        labels.append(response.choices[0].message.content)
    return labels
