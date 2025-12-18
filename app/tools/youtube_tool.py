from __future__ import annotations
import os
import requests
from langchain_core.tools import tool

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

@tool
def youtube_search(query: str) -> str:
    """
    YouTube Data API로 query 관련 상위 영상 1개를 찾아서 '제목 - 링크' 한 줄로 반환한다.
    (항상 한 줄 텍스트로 반환)
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        return "유튜브 검색을 위한 YOUTUBE_API_KEY가 설정되어 있지 않습니다."

    params = {
        "part": "snippet",
        "q": query,
        "key": api_key,
        "type": "video",
        "maxResults": 1,
        "safeSearch": "strict",
        "relevanceLanguage": "ko",
    }
    try:
        r = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", []) or []
        if not items:
            return "추천할 만한 유튜브 영상을 찾지 못했습니다."

        results = []
        for it in items:
            vid = (it.get("id") or {}).get("videoId")
            title = ((it.get("snippet") or {}).get("title") or "").replace("\n", " ").strip()
            if vid:
                url = f"https://www.youtube.com/watch?v={vid}"
                results.append(f"{title} - {url}")

        if not results:
            return "추천할 만한 유튜브 영상을 찾지 못했습니다."

        # 한 줄로 합쳐서 반환
        return "추천영상: " + " | ".join(results)

    except Exception:
        return "유튜브 검색 중 오류가 발생했습니다."
