import csv
from langdetect import detect

csv_file_path = "C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/data_store/youtube_comments.csv"
filtered_csv_file_path = "C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/data_store/english_comments.csv"  


def is_english(comment):
    try:
        lang = detect(comment)
        return lang == "en"
    except:
        return False


def filter_english_comments():
    with open(csv_file_path, mode='r', encoding='utf-8') as read_file:
        csv_reader = csv.reader(read_file)
        
        try:
            header = next(csv_reader)  
        except StopIteration:
            print("CSV file is empty.")
            return

        english_comments = []
        english_comments.append(["text"])  

        for row in csv_reader:
            if len(row) > 0:
                comment_text = row[0].strip()

                if is_english(comment_text):
                    english_comments.append([comment_text])

    
    with open(filtered_csv_file_path, mode='w', newline='', encoding='utf-8') as write_file:
        csv_writer = csv.writer(write_file)
        csv_writer.writerows(english_comments)

    print("Filtered English comments written to", filtered_csv_file_path)