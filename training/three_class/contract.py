"""Single source of truth for three-class labels and audited visual families."""

CLASSES = ["verde", "maduro", "sobre_maduro"]
SOURCE_TO_CLASS = {"unripe": "verde", "ripe": "maduro", "overripe": "sobre_maduro"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Cross-label scenes that appear to show the same banana or hand over time.
# Exclude both sides instead of choosing a label.
AMBIGUOUS_CROSS_LABEL_LINKS = {
    ("VF-00005", "VF-00018"),
    ("VF-00063", "VF-00199"),
    ("VF-00189", "VF-00197"),
    ("VF-00190", "VF-00197"),
    ("VF-00191", "VF-00452"),
}

# Same-label near-identical scenes are joined for splitting to prevent leakage.
SAME_LABEL_VISUAL_LINKS = {
    ("VF-00009", "VF-00193"),
    ("VF-00092", "VF-00114"),
    ("VF-00128", "VF-00138"),
    ("VF-00171", "VF-00188"),
    ("VF-00196", "VF-00199"),
    ("VF-00200", "VF-00202"),
    ("VF-00253", "VF-00254"),
    ("VF-00033", "VF-00254"),
    ("VF-00254", "VF-00256"),
    ("VF-00076", "VF-00077"),
    ("VF-00083", "VF-00125"),
    ("VF-00197", "VF-00252"),
}
