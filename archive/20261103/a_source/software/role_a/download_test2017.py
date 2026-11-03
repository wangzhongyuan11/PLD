"""Download only surveyed official ISIC 2017 test originals inside the disk cap."""
from __future__ import annotations

import json

from software.role_a import external2017 as ext
from software.role_a import survey_full2017 as full
from software.role_a import survey_test2017 as test


def main() -> None:
    manifest = json.loads((test.ROOT / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "role-a-isic2017-test-v1" or
            manifest.get("source_masks_sha256") != ext.sha256(test.ARCHIVE.read_bytes()) or
            manifest.get("selected_count") != len(manifest.get("items", {}))):
        raise ValueError("2017 test survey or mask archive changed")
    ext.assert_budget(manifest, test.MAX_IMAGE_BYTES)
    projected = (full.tree_bytes(ext.PROJECT) + int(manifest["estimated_image_bytes"]) +
                 int(manifest["estimated_mask_bytes"]))
    if projected > test.MAX_PROJECT_BYTES:
        raise ValueError("2017 test download would exceed the 17 GB project cap")
    print(json.dumps({"selected": manifest["selected_count"],
                      "estimated_image_bytes": manifest["estimated_image_bytes"],
                      "project_cap_bytes": test.MAX_PROJECT_BYTES}), flush=True)
    result = ext.download(manifest, max_bytes=test.MAX_IMAGE_BYTES,
                          root=test.ROOT, archive=test.ARCHIVE,
                          manifest_path=test.ROOT / "manifest.json")
    actual = full.tree_bytes(ext.PROJECT)
    if actual > test.MAX_PROJECT_BYTES:
        raise RuntimeError("project exceeded the reviewed 17 GB cap")
    print(json.dumps({"verified_images": result["image_count"],
                      "image_bytes": result["image_bytes"],
                      "project_bytes": actual}))


if __name__ == "__main__":
    main()
