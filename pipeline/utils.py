import os
import re

from google.genai import types


def extract_gcs_uris(text: str) -> list[types.Part]:
    """Finds all markdown images in the context and converts them to native GCS URI Parts for Gemini Vision."""
    parts = []
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not bucket_name:
        return parts
    bucket_name = bucket_name.replace('"', "").replace("'", "")

    image_regex = r"!\[.*?\]\((.*?)\)"
    matches = re.findall(image_regex, text)

    seen_uris = set()
    for url in matches:
        clean_path = url.split("?")[0]
        if "storage.googleapis.com" in clean_path:
            clean_path = clean_path.split(f"{bucket_name}/")[-1]
        else:
            clean_path = (
                clean_path.replace("http://localhost:8000/assets/", "")
                .replace("./", "")
                .strip("/")
            )

        gs_uri = f"gs://{bucket_name}/{clean_path}"
        if gs_uri not in seen_uris:
            seen_uris.add(gs_uri)
            parts.append(types.Part.from_uri(file_uri=gs_uri, mime_type="image/jpeg"))

    return parts
