"""Dataset configuration for the TAI-FD segmentation project."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class SceneConfig:
    """File mapping and objective for one segmentation scene."""

    scene_id: str
    image_files: tuple[str, ...]
    mask_file: str
    objective: str


SCENES: dict[str, SceneConfig] = {
    "scene1": SceneConfig(
        scene_id="scene1",
        image_files=("Scene_1.png",),
        mask_file="GT1.png",
        objective="Segmenter chat, ciel, sol, arbres",
    ),
    "scene2": SceneConfig(
        scene_id="scene2",
        image_files=("Scene_2.png",),
        mask_file="GT2.png",
        objective="Extraire le disque lumineux",
    ),
    "scene3": SceneConfig(
        scene_id="scene3",
        image_files=("Scene_3.png",),
        mask_file="GT3.png",
        objective="Isoler les routes",
    ),
    "scene4": SceneConfig(
        scene_id="scene4",
        image_files=("Scene_4_RGB_1.png", "Scene_4_D_2.png"),
        mask_file="GT4.png",
        objective="Extraire la personne debout",
    ),
}

