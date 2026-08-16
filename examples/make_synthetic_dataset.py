"""Generate a small synthetic detection dataset to smoke-test a GPU pipeline."""

from ai_toolset.dataset import generate_synthetic

CLASSES = {
    "pokecenter": (255, 144, 30),
    "pokemart": (0, 140, 255),
    "npc": (50, 50, 220),
    "house": (43, 90, 139),
    "grass": (0, 160, 0),
}

if __name__ == "__main__":
    generate_synthetic(CLASSES, "synthetic_data")
