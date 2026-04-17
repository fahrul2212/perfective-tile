"""
core/sam3_client.py — DEPRECATED: Backward Compatibility Proxy
─────────────────────────────────────────────────────────────────
SAM3 client telah dipindahkan ke services/sam3_client.py (Clean Architecture).
File ini tetap ada agar import lama tidak break.

Gunakan: from services.sam3_client import sam3_client
"""
from services.sam3_client import SAM3Client, sam3_client

__all__ = ["SAM3Client", "sam3_client"]
