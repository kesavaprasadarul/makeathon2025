import os
import numpy as np
import asyncio
import base64
import pathlib
import time
import re
import json
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from urllib.request import urlopen
from PIL import Image

# --- Configuration ---
API_KEY = os.getenv('OPENAI_API_KEY', 'REDACTED')
CONCURRENCY = 30
OBJECT_EXTRACTION_MODEL = 'gpt-4o-mini'
DETECTION_MODEL = 'gpt-4o'

# Regex to parse the model's reply
PATTERN = re.compile(
    r"(.*):.*Approval\s*:\s*(Yes|No)\s*,\s*Confidence\s*:\s*(100|[1-9]?\d)",
    re.I,
)

# Initialize async clients
object_client = AsyncOpenAI(api_key=API_KEY)
image_client = AsyncOpenAI(api_key=API_KEY)
sem = asyncio.Semaphore(CONCURRENCY)

# Template for the detection prompt
DETECTION_TEMPLATE = (
    """
Does this photo contain any of the following objects: {objects}? 
You must be ABSOLUTELY certain that you are actually observing these objects, 
and not some other items. Quantify between 0 and 100 how confident you are that at least one of these objects is present. 
25 means you think it might contain them, 50 means you are highly confident, 100 means absolutely certain (no false positives). 
Return exactly in this format:
"Approval: {{Yes|No}}, Confidence: {{confidence_score}}"
"""
)

async def extract_objects_from_prompt(prompt_text: str) -> list[str]:
    """
    Calls a lightweight model to parse the user's textual prompt and extract a list of object keywords.
    Strips any Markdown code fences before parsing JSON.
    """
    system_msg = {
        'role': 'system',
        'content': (
            'You are an assistant that extracts the key objects or items a user wants detected in images.'
            'The user might either specify the objects directly or their relative location.'
            'Given a user instruction, utilize both forms of description and return a JSON array of strings, each an object to look for.'
        )
    }
    user_msg = {'role': 'user', 'content': prompt_text}

    response = await object_client.chat.completions.create(
        model=OBJECT_EXTRACTION_MODEL,
        messages=[system_msg, user_msg]
    )
    text = response.choices[0].message.content.strip()

    # Strip markdown code fences
    fenced = re.match(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        objects = json.loads(text)
        if not isinstance(objects, list):
            raise ValueError('Expected a JSON list')
    except Exception:
        # Fallback: split on commas/semicolons/newlines
        objects = [obj.strip() for obj in re.split(r'[,;\n]+', text) if obj.strip()]
    print(objects)
    return objects

async def ask_image(path: Path, detection_prompt: str) -> str:
    """
    Sends one image plus the detection prompt to the model, retries on rate limits.
    """
    encoded = encode_image(path)
    backoff = 2
    while True:
        try:
            async with sem:
                chat = await image_client.chat.completions.create(
                    model=DETECTION_MODEL,
                    messages=[
                        { 'role': 'user', 'content': [
                            {'type': 'text', 'text': detection_prompt},
                            {'type': 'image_url', 'image_url': {'url': encoded, 'detail': 'low'}}
                        ]}
                    ]
                )
            return f"{path.name}: {chat.choices[0].message.content}"
        except Exception as e:
            if 'RateLimit' in str(e):
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                raise

async def main_async(user_prompt: str, top_num: int = 15) -> list[str]:
    # Extract objects
    objects = await extract_objects_from_prompt(user_prompt)
    detection_prompt = DETECTION_TEMPLATE.format(objects=', '.join(objects))
    print(detection_prompt)
    # Gather images
    uploads = Path.cwd() / 'static' / 'uploads'
    subdirs = sorted(d for d in uploads.iterdir() if d.is_dir())
    if not subdirs:
        raise RuntimeError('No folders found under uploads')
    IMAGES_DIR = subdirs[0]
    images = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp'}]
    return [img for img in images]


    # Query all images
    tasks = [ask_image(img, detection_prompt) for img in images]
    hits = []
    for reply in await asyncio.gather(*tasks):
        print(reply)
        m = PATTERN.search(reply)
        if not m:
            continue
        fname, approval, conf = m.group(1), m.group(2).lower(), int(m.group(3))
        if approval == 'yes' and conf >= 90:
            hits.append((conf, fname))

    hits.sort(reverse=True)
    return [fname for _, fname in hits[:top_num]]


def run_async_ai(prompt_text: str) -> list[str]:
    """Blocking helper so Flask routes can call async code"""
    return asyncio.run(main_async(prompt_text))


def encode_image(path: Path) -> str:
    mime = 'image/' + path.suffix.lstrip('.')
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'data:{mime};base64,{b64}'

# Example usage:
# results = run_async_ai('Detect all pallets and crates in these images')
# print(results)
