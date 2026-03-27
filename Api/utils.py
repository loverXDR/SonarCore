"""Utility functions for API"""

import os
import uuid
import httpx
import tempfile
from contextlib import asynccontextmanager
from fastapi import UploadFile, HTTPException


@asynccontextmanager
async def handle_audio_input(
    file: UploadFile = None,
    url: str = None,
    file_path: str = None,
):
    """
    Context manager to handle audio input from multiple sources.

    Ensures that exactly one input method is provided and yields a local file path.
    Cleans up any downloaded or uploaded temporary files upon exiting.

    Args:
        file (UploadFile, optional): Uploaded multipart file. Defaults to None.
        url (str, optional): Remote URL to an audio file. Defaults to None.
        file_path (str, optional): Local file path on the server. Defaults to None.

    Yields:
        str: Absolute local file path to the audio file.

    Raises:
        HTTPException: If none or multiple inputs are provided, or on download/save errors.
    """
    inputs_provided = sum([file is not None, url is not None, file_path is not None])
    if inputs_provided == 0:
        raise HTTPException(status_code=400, detail="Must provide one of: file, url, file_path")
    if inputs_provided > 1:
        raise HTTPException(status_code=400, detail="Must provide EXACTLY ONE of: file, url, file_path")

    temp_path = None
    try:
        if file_path:
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            yield file_path

        elif file:
            temp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{file.filename}")
            try:
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")
            yield temp_path

        elif url:
            temp_path = os.path.join(tempfile.gettempdir(), f"download_{uuid.uuid4().hex}.wav")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error downloading url: {str(e)}")
            yield temp_path

    finally:
        # Cleanup temporary file if we created one
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
