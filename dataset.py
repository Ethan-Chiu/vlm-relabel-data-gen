import io
import json
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

CONFIG_PATH = Path("config.toml")


@dataclass
class Config:
    output_dir: Path = Path("images")
    timeout: int = 10

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            output_dir=Path(data.get("output_dir", "images")),
            timeout=int(data.get("timeout", 10)),
        )


anns = [
    {
        "caption": "the tiki ball logo is displayed on this apron",
        "url": "https://image.spreadshirtmedia.com/image-server/v1/products/P1016953357T1186A359PC1026849373PA2537PT17X1Y0S28/views/1,width=300,height=300,appearanceId=359,version=1497265829/z-tiki-bar-adjustable-apron.png",
    },
    {
        "caption": "the view of an aerial pool with palm trees",
        "url": "http://uberflip.cdntwrk.com/mediaproxy?url=http%3A%2F%2Fd22ir9aoo7cbf6.cloudfront.net%2Fwp-content%2Fuploads%2Fsites%2F4%2F2018%2F01%2FAyana1.jpg&size=1&version=1517393441&sig=e3e14f09e2b062e0fd144306d56abda7&default=hubs%2Ftilebg-blogs.jpg",
    },
]

cfg = Config.load()
cfg.output_dir.mkdir(parents=True, exist_ok=True)

metadata_path = cfg.output_dir / "metadata.jsonl"

stats = {"total": len(anns), "success": 0, "failed": 0}
error_counts: dict[str, int] = defaultdict(int)

with open(metadata_path, "w") as meta_f:
    for i, ann in enumerate(anns):
        url = ann["url"]
        try:
            response = requests.get(url, timeout=cfg.timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

            suffix = Path(url.split("?")[0]).suffix or ".jpg"
            filename = f"{i:06d}{suffix}"
            out_path = cfg.output_dir / filename
            image.save(out_path)

            record = {
                "filename": filename,
                "caption": ann["caption"],
                "source_url": url,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
            meta_f.write(json.dumps(record) + "\n")

            print(f"[OK] {image.size} → {filename}  ({ann['caption'][:50]})")
            stats["success"] += 1

        except requests.exceptions.Timeout:
            error_counts["Timeout"] += 1
            stats["failed"] += 1
        except requests.exceptions.HTTPError as e:
            error_counts[f"HTTP {e.response.status_code}"] += 1
            stats["failed"] += 1
        except requests.exceptions.RequestException as e:
            error_counts[f"RequestError({type(e).__name__})"] += 1
            stats["failed"] += 1
        except UnidentifiedImageError:
            error_counts["InvalidImage"] += 1
            stats["failed"] += 1

print()
print(
    f"Results: {stats['success']}/{stats['total']} succeeded, {stats['failed']} failed"
)
print(f"Metadata: {metadata_path}")
if error_counts:
    print("Error breakdown:")
    for kind, count in sorted(error_counts.items()):
        print(f"  {kind}: {count}")
