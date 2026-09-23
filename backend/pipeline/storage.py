"""storage.py — recording upload/playback via Neon Object Storage
(S3-compatible; the "voice-recordings" bucket declared in neon.ts)."""
import os

import boto3
from botocore.config import Config

BUCKET = "voice-recordings"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_recording(local_path: str, key: str) -> str:
    _client().upload_file(local_path, BUCKET, key, ExtraArgs={"ContentType": "audio/wav"})
    return key


def presign_recording(key: str, expires_in: int = 3600) -> str:
    return _client().generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=expires_in,
    )
