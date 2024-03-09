import os
import googleapiclient.discovery
import csv
import time

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyAX-gcgeqYdlhvcgX-x0o4k1qF1-y8p-gY"
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY
)

def get_video_ids(query, max_results=5, region_code="US"):
    request = youtube.search().list(
        part="id",
        q=query,
        type="video",
        maxResults=max_results,
        regionCode=region_code
    )
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    return video_ids


csv_file_path = "C:/Users/prati/Desktop/CyberMod/CyberBullying-Mods/data_store/youtube_comments.csv"


csv_directory = os.path.dirname(csv_file_path)
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

processed_video_ids = set()
iterations_before_reset = 10
current_iteration = 0
max_results = 5 
total_csv_entries = 0

while True:
    genres = ["gaming", "vlogs", "education", "travel"]

    for genre in genres:
        video_ids = get_video_ids(query=genre, max_results=max_results)
        if all(video_id in processed_video_ids for video_id in video_ids):
            max_results *= 2 

        for VIDEO_ID in video_ids:
            if VIDEO_ID in processed_video_ids:
                print(f"Video ID {VIDEO_ID} already processed. Skipping...")
                continue

            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=VIDEO_ID,
                    maxResults=100
                )
                response = request.execute()

                comments = []
                for item in response["items"]:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append([comment['textDisplay']])
                with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(['text'])  
                    csv_writer.writerows(comments)
                total_csv_entries += len(comments)

                print("Comments appended to", csv_file_path, "for VIDEO_ID:", VIDEO_ID)
                processed_video_ids.add(VIDEO_ID)

            except googleapiclient.errors.HttpError as e:
                error_message = str(e)
                if "disabled comments" in error_message:
                    print(f"Comments disabled for VIDEO_ID: {VIDEO_ID}")
                else:
                    print(f"Error for VIDEO_ID {VIDEO_ID}: {error_message}")
        if total_csv_entries >= 5000:
            print("CSV file has reached 5000 entries. Exiting the script.")
            exit()

    
    current_iteration += 1

    
    time.sleep(5)
