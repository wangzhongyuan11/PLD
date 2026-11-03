"""Execute a reviewed full-2017 survey under the whole-project disk cap."""
from __future__ import annotations

import json

from software.role_a import external2017 as ext
from software.role_a import survey_full2017 as survey


def main() -> None:
    manifest = json.loads((survey.ROOT / survey.SURVEY_NAME).read_text(encoding="utf-8"))
    if (manifest.get("source_masks_sha256") != ext.sha256(survey.MASK_ZIP.read_bytes()) or
            manifest.get("selected_count") != 949):
        raise ValueError("full 2017 survey does not match verified mask ZIP and 949-item selection")
    ext.assert_budget(manifest, survey.MANIFEST_MAX_BYTES)
    estimate = survey.estimate_disk(manifest, survey.mask_index(survey.MASK_ZIP), root=survey.PROJECT)
    survey.assert_budget(estimate)
    print(json.dumps({"selected": manifest["selected_count"],
                      "new_image_bytes": estimate["required_image_bytes"],
                      "estimated_project_bytes_after": estimate["project_bytes_after"],
                      "project_limit_bytes": survey.PROJECT_MAX_BYTES}, ensure_ascii=False), flush=True)
    result = ext.download(manifest, max_bytes=survey.MANIFEST_MAX_BYTES,
                          root=survey.ROOT, archive=survey.MASK_ZIP)
    actual_project_bytes = survey.tree_bytes(survey.PROJECT)
    if actual_project_bytes > survey.PROJECT_MAX_BYTES:
        raise RuntimeError("project exceeded the authorized 17-billion-byte disk cap")
    print(json.dumps({"downloaded_and_verified": result["image_count"],
                      "image_bytes_total": result["image_bytes"],
                      "project_bytes_actual": actual_project_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
