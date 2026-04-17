"""
utils/tile_catalog.py
─────────────────────────────────────────────────────────────────
Tile Catalog — scan folder assets/tile/ dan return daftar tile.

Mendukung:
  - Auto-scan berdasarkan file extension (.jpg, .jpeg, .png)
  - Parse nama file → tile name, size (dari pattern nama)
  - Generate thumbnail path
  - Return JSON-ready dict
"""
import os
import re
from pathlib import Path
from typing import List, Dict
from core.config import Config


TILE_DIR = Config.ROOT_DIR / "assets" / "tile"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_tile_name(filename: str) -> Dict:
    """
    Parse nama file tile → metadata.
    Format yg didukung: 'Brand-SIZExSIZE-CODE-NAME.ext'
    Contoh: 'Concord-60x60-PGC66K001-ALASKA WHITE.jpeg'
    """
    stem = Path(filename).stem
    
    # Coba parse pattern: Brand-SIZExSIZE-CODE-NAME
    size_match = re.search(r'(\d+)x(\d+)', stem)
    tile_size = f"{size_match.group(1)}x{size_match.group(2)}" if size_match else "60x60"
    
    # Bersihkan nama
    display_name = stem
    # Hapus pattern ukuran dari nama display
    display_name = re.sub(r'\d+x\d+[-_]?', '', display_name)
    # Hapus kode produk (huruf+angka panjang)
    display_name = re.sub(r'[A-Z]{2,}\d{2,}[A-Z]*\d*[-_]?', '', display_name)
    # Bersihkan separator
    display_name = display_name.replace('-', ' ').replace('_', ' ').strip()
    # Capitalize
    display_name = display_name.title() if display_name else stem.title()
    
    # Generate ID (slug dari filename)
    tile_id = stem.lower().replace(' ', '-').replace('_', '-')
    tile_id = re.sub(r'[^a-z0-9\-]', '', tile_id)
    tile_id = re.sub(r'-+', '-', tile_id).strip('-')
    
    return {
        "id": tile_id,
        "name": display_name,
        "size": tile_size,
        "filename": filename,
    }


def get_tile_catalog() -> List[Dict]:
    """
    Scan folder assets/tile/ dan return daftar tile.
    
    Returns:
        List of dict: [{ id, name, size, filename, path, thumbnail_url }]
    """
    if not TILE_DIR.exists():
        return []
    
    tiles = []
    for f in sorted(TILE_DIR.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXT and f.is_file():
            tile = _parse_tile_name(f.name)
            tile["path"] = str(f.relative_to(Config.ROOT_DIR))
            tile["thumbnail_url"] = f"/static/tile-thumbs/{f.stem}.jpg"
            tile["full_url"] = f"/outputs/tiles/{f.name}"
            tiles.append(tile)
    
    return tiles


def get_tile_path(tile_id: str) -> str:
    """
    Resolve tile_id ke absolute path file.
    
    Args:
        tile_id: ID dari catalog (slug)
        
    Returns:
        Absolute path ke file tile, atau default tile jika tidak ditemukan
    """
    catalog = get_tile_catalog()
    
    for tile in catalog:
        if tile["id"] == tile_id:
            return str(Config.ROOT_DIR / tile["path"])
    
    # Fallback: coba match partial
    for tile in catalog:
        if tile_id in tile["id"] or tile_id in tile["filename"].lower():
            return str(Config.ROOT_DIR / tile["path"])
    
    # Default: tile pertama di catalog
    if catalog:
        return str(Config.ROOT_DIR / catalog[0]["path"])
    
    # Last resort
    return str(TILE_DIR / "Concord-60x60-PGC66K001-ALASKA WHITE.jpeg")
