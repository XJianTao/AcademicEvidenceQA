"""为人工版面核对生成 PDF 页面联系页。"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def main() -> None:
    root = Path(r"E:\Project Files\VscodeProjects\rag_test\papers")
    names = [
        "2026_A fuzzy-enhanced Graph Convolutional Network with rule-basedpost-processing for Collective Sentiment Analysis.pdf",
        "2026_Efficient and privacy-preserved link prediction via condensed graphs.pdf",
    ]
    tiles: list[tuple[str, int, Image.Image]] = []
    for name in names:
        with fitz.open(root / name) as document:
            for page_number, page in enumerate(document, 1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(0.28, 0.28), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                tiles.append((name[:12], page_number, image))
    tile_width, tile_height, label_height, columns = 260, 350, 28, 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, (tile_height + label_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, page, image) in enumerate(tiles):
        x = index % columns * tile_width
        y = index // columns * (tile_height + label_height)
        sheet.paste(image.resize((tile_width, tile_height)), (x, y + label_height))
        draw.text((x + 4, y + 5), f"{name} p.{page}", fill="black")
    output = Path(r"C:\Users\XJT\AppData\Local\Temp\aeqa-m0-contact.png")
    sheet.save(output)
    print(f"Rendered {len(tiles)} pages to {output}")


if __name__ == "__main__":
    main()
