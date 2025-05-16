# create a class that will search for videos on a topic using the YouTube API
# the scraper will also load a transcript of the video based on the video id and return
# a dataframe with the transcript and metadata associated with the video
# finally, the scraper will save the dataframe to a csv file

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api._api import YouTubeTranscriptApi


class YTLoader:
    def __init__(self, video_ids: list[str], api_key: str) -> None:
        self.video_ids = video_ids
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.transcripts = []
        self.metadata = []
        for video_id in self.video_ids:
            self.transcripts.append(self.get_transcript(video_id))
            self.metadata.append(self.get_video_metadata(video_id))

    def get_transcript(self, video_id: str):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry["text"] for entry in transcript])
            return full_text
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {e}")
            return None  # No transcript available

    def get_video_metadata(self, video_id: str):
        try:
            request = self.youtube.videos().list(part="snippet,statistics", id=video_id)
            response = request.execute()
            item = response["items"][0]["snippet"]
            stats = response["items"][0]["statistics"]
            return {
                "title": item["title"],
                "description": item["description"],
                "channel": item["channelTitle"],
                "publishedAt": item["publishedAt"],
                "viewCount": stats.get("viewCount", "0"),
            }
        except Exception as e:
            print(f"Error getting video metadata for video {video_id}: {e}")
            return None


@dataclass
class YoutubeScraper:
    """
    Scrape youtube videos based on a topic
    """

    topic: str
    max_results: int = 10
    api_key: str | None = field(default_factory=lambda: YoutubeScraper._load_api_key())
    video_ids: list[str] = field(default_factory=list)
    videos_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @staticmethod
    def _load_api_key() -> str | None:
        load_dotenv()
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            print(
                "WARNING: YOUTUBE_API_KEY environment variable is not set or is empty"
            )
        else:
            print(f"YouTube API key found with length: {len(api_key)}")
        return api_key

    def search_videos(self) -> None:
        """Search for youtube videos based on a topic and save a list of
        video ids that can be accessed via https://youtu.be/{video_id}
        """
        if not self.api_key:
            print(
                "Warning: No YouTube API key provided. Please set the YOUTUBE_API_KEY environment variable."
            )
            self.video_ids = []
            return

        try:
            youtube = build("youtube", "v3", developerKey=self.api_key)
            request = youtube.search().list(
                q=self.topic,
                part="id,snippet",
                maxResults=self.max_results,
                type="video",
                relevanceLanguage="en",
            )
            response = request.execute()

            if "items" not in response or not response["items"]:
                print(f"Warning: No videos found for topic '{self.topic}'")
                self.video_ids = []
                return

            self.video_ids = [item["id"]["videoId"] for item in response["items"]]
            print(f"Found {len(self.video_ids)} videos for topic '{self.topic}'")
        except Exception as e:
            print(f"Error searching for videos: {e}")
            self.video_ids = []

    def _load_video_data(
        self, video_id: str, add_video_info: bool = False
    ) -> pd.DataFrame | None:
        """Protected method that loads transcript and metadata of a single video

        Args:
            video_id (str): video id that references the given video
            add_video_info (bool, optional): flag that if set to true will
            add the video info by calling the youtube api. Defaults to False.

        Returns:
            pd.DataFrame | None: dataframe containing the transcript and metadata
            of a single youtube video, or None if transcript is unavailable
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        if not self.api_key:
            print(f"Warning: No YouTube API key provided for video {video_id}.")
            return None

        loader = YTLoader(video_ids=[video_id], api_key=self.api_key)
        transcript = loader.transcripts[0]
        metadata = loader.metadata[0]
        if transcript is None:  # Skip videos with no transcript
            return None
        doc_df = pd.DataFrame([metadata])
        doc_df["video_url"] = url
        doc_df["page_content"] = transcript
        return doc_df

    def load_videos_data(self, add_video_info: bool = False) -> None:
        """Load the data of multiple videos and save it to a dataframe

        Args:
            add_video_info (bool, optional): flag that if set to true will
            add the video info by calling the youtube api. Defaults to False.
        """
        videos = []
        for video_id in tqdm(self.video_ids, desc="Loading videos"):
            try:
                video = self._load_video_data(video_id, add_video_info=add_video_info)
                time.sleep(2)
                if video is not None:  # Only add videos with valid transcripts
                    videos.append(video)
            except Exception as e:
                print(f"Error loading video {video_id}: {e}")

        # Check if we have any videos to concatenate
        if not videos:
            print("Warning: No videos were loaded successfully.")
            self.videos_df = pd.DataFrame()  # Create an empty DataFrame
        else:
            self.videos_df = pd.concat(videos)

    def save_videos_data(self, csv_path: str) -> None:
        """Save the dataframe to a csv file

        Args:
            csv_path (str): path to the csv file
        """
        # Create directory if it doesn't exist
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        self.videos_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # example usage
    scraper = YoutubeScraper(topic="Hybrid RAG")
    scraper.search_videos()
    print(scraper.video_ids)
    scraper.load_videos_data(add_video_info=False)
    scraper.videos_df.head()
    scraper.save_videos_data("./data/videos.csv")
    print(len(scraper.videos_df))
