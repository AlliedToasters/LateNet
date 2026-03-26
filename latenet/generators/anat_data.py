"""Curated human anatomy data for the anatomy generator.

All structures sourced from standard anatomy references:
- Gray's Anatomy, 42nd Edition (Standring, 2020)
- Netter's Atlas of Human Anatomy, 7th Edition (Netter, 2018)
- Terminologia Anatomica (FCAT, 1998)

Data is static and hardcoded — human anatomy doesn't change.
Only well-known structures a high school biology student would recognize are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnatomicalStructure:
    name: str
    structure_type: str  # bone, muscle, organ, nerve, vessel, gland, ligament, tendon
    body_system: str  # skeletal, muscular, cardiovascular, nervous, respiratory,
                      # digestive, endocrine, urinary, reproductive, lymphatic, integumentary
    body_region: str  # head, neck, thorax, abdomen, pelvis, upper_limb, lower_limb, back
    sub_region: str | None = None
    paired: bool = False
    related_systems: list[str] = field(default_factory=list)
    spans_regions: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Region hierarchy
# ---------------------------------------------------------------------------

REGION_HIERARCHY: dict[str, list[str]] = {
    "head": ["skull", "face", "cranium"],
    "neck": [],
    "thorax": ["chest"],
    "abdomen": [],
    "pelvis": ["perineum"],
    "upper_limb": ["shoulder", "arm", "forearm", "hand"],
    "lower_limb": ["hip", "thigh", "leg", "foot"],
    "back": ["cervical_spine", "thoracic_spine", "lumbar_spine"],
}

# Adjacent regions (for hard-tier swaps in regional containment)
ADJACENT_REGIONS: dict[str, list[str]] = {
    "head": ["neck"],
    "neck": ["head", "thorax", "back"],
    "thorax": ["neck", "abdomen", "back", "upper_limb"],
    "abdomen": ["thorax", "pelvis", "back"],
    "pelvis": ["abdomen", "lower_limb"],
    "upper_limb": ["thorax", "neck"],
    "lower_limb": ["pelvis"],
    "back": ["neck", "thorax", "abdomen"],
}

# Systems that are functionally or spatially related (for hard-tier system swaps)
RELATED_SYSTEMS: dict[str, list[str]] = {
    "cardiovascular": ["respiratory", "lymphatic"],
    "respiratory": ["cardiovascular"],
    "digestive": ["endocrine", "urinary"],
    "muscular": ["skeletal"],
    "skeletal": ["muscular"],
    "nervous": ["endocrine"],
    "endocrine": ["nervous", "digestive", "reproductive"],
    "urinary": ["reproductive", "digestive"],
    "reproductive": ["urinary", "endocrine"],
    "lymphatic": ["cardiovascular"],
    "integumentary": [],
}

ALL_SYSTEMS = [
    "skeletal", "muscular", "cardiovascular", "nervous", "respiratory",
    "digestive", "endocrine", "urinary", "reproductive", "lymphatic", "integumentary",
]

ALL_STRUCTURE_TYPES = [
    "bone", "muscle", "organ", "nerve", "vessel", "gland", "ligament", "tendon",
]

# Related structure types (for hard-tier type swaps)
RELATED_STRUCTURE_TYPES: dict[str, list[str]] = {
    "bone": ["ligament", "tendon"],
    "muscle": ["tendon", "ligament"],
    "organ": ["gland"],
    "nerve": ["vessel"],
    "vessel": ["nerve"],
    "gland": ["organ"],
    "ligament": ["tendon", "bone"],
    "tendon": ["ligament", "muscle"],
}

# Medium-relatedness structure types
MEDIUM_STRUCTURE_TYPES: dict[str, list[str]] = {
    "bone": ["muscle"],
    "muscle": ["bone"],
    "organ": ["vessel", "nerve"],
    "nerve": ["organ", "muscle"],
    "vessel": ["organ", "muscle"],
    "gland": ["nerve", "vessel"],
    "ligament": ["muscle"],
    "tendon": ["bone"],
}

# Structure type display names (for natural language)
STRUCTURE_TYPE_LABELS: dict[str, str] = {
    "bone": "bone",
    "muscle": "muscle",
    "organ": "organ",
    "nerve": "nerve",
    "vessel": "blood vessel",
    "gland": "gland",
    "ligament": "ligament",
    "tendon": "tendon",
}

# System display names
SYSTEM_LABELS: dict[str, str] = {
    "skeletal": "skeletal",
    "muscular": "muscular",
    "cardiovascular": "cardiovascular",
    "nervous": "nervous",
    "respiratory": "respiratory",
    "digestive": "digestive",
    "endocrine": "endocrine",
    "urinary": "urinary",
    "reproductive": "reproductive",
    "lymphatic": "lymphatic",
    "integumentary": "integumentary",
}

# Region display names
REGION_LABELS: dict[str, str] = {
    "head": "head",
    "neck": "neck",
    "thorax": "thorax",
    "abdomen": "abdomen",
    "pelvis": "pelvis",
    "upper_limb": "upper limb",
    "lower_limb": "lower limb",
    "back": "back",
}

# ---------------------------------------------------------------------------
# Bones (~65)
# ---------------------------------------------------------------------------

_BONES: list[AnatomicalStructure] = [
    # Head
    AnatomicalStructure("skull", "bone", "skeletal", "head", "cranium",
                        description="bony framework of the head"),
    AnatomicalStructure("mandible", "bone", "skeletal", "head", "face",
                        description="lower jaw bone"),
    AnatomicalStructure("maxilla", "bone", "skeletal", "head", "face", paired=True,
                        description="upper jaw bone"),
    AnatomicalStructure("nasal bone", "bone", "skeletal", "head", "face", paired=True,
                        description="bone forming the bridge of the nose"),
    AnatomicalStructure("zygomatic bone", "bone", "skeletal", "head", "face", paired=True,
                        description="cheekbone"),
    # Neck
    AnatomicalStructure("hyoid", "bone", "skeletal", "neck",
                        description="U-shaped bone in the neck supporting the tongue"),
    # Thorax
    AnatomicalStructure("sternum", "bone", "skeletal", "thorax", "chest",
                        description="breastbone"),
    AnatomicalStructure("ribs", "bone", "skeletal", "thorax", "chest", paired=True,
                        description="curved bones forming the rib cage"),
    # Upper limb
    AnatomicalStructure("clavicle", "bone", "skeletal", "upper_limb", "shoulder", paired=True,
                        description="collarbone"),
    AnatomicalStructure("scapula", "bone", "skeletal", "upper_limb", "shoulder", paired=True,
                        description="shoulder blade"),
    AnatomicalStructure("humerus", "bone", "skeletal", "upper_limb", "arm", paired=True,
                        description="upper arm bone"),
    AnatomicalStructure("radius", "bone", "skeletal", "upper_limb", "forearm", paired=True,
                        description="lateral forearm bone on the thumb side"),
    AnatomicalStructure("ulna", "bone", "skeletal", "upper_limb", "forearm", paired=True,
                        description="medial forearm bone on the pinky side"),
    AnatomicalStructure("carpals", "bone", "skeletal", "upper_limb", "hand", paired=True,
                        description="wrist bones"),
    AnatomicalStructure("metacarpals", "bone", "skeletal", "upper_limb", "hand", paired=True,
                        description="hand bones between wrist and fingers"),
    AnatomicalStructure("phalanges of the hand", "bone", "skeletal", "upper_limb", "hand", paired=True,
                        description="finger bones"),
    # Lower limb
    AnatomicalStructure("femur", "bone", "skeletal", "lower_limb", "thigh", paired=True,
                        description="thigh bone, the longest bone in the body"),
    AnatomicalStructure("patella", "bone", "skeletal", "lower_limb", "leg", paired=True,
                        description="kneecap"),
    AnatomicalStructure("tibia", "bone", "skeletal", "lower_limb", "leg", paired=True,
                        description="shinbone"),
    AnatomicalStructure("fibula", "bone", "skeletal", "lower_limb", "leg", paired=True,
                        description="thin bone on the outer side of the lower leg"),
    AnatomicalStructure("tarsals", "bone", "skeletal", "lower_limb", "foot", paired=True,
                        description="ankle bones"),
    AnatomicalStructure("metatarsals", "bone", "skeletal", "lower_limb", "foot", paired=True,
                        description="foot bones between ankle and toes"),
    AnatomicalStructure("phalanges of the foot", "bone", "skeletal", "lower_limb", "foot", paired=True,
                        description="toe bones"),
    AnatomicalStructure("calcaneus", "bone", "skeletal", "lower_limb", "foot", paired=True,
                        description="heel bone"),
    AnatomicalStructure("talus", "bone", "skeletal", "lower_limb", "foot", paired=True,
                        description="ankle bone connecting leg to foot"),
    # Pelvis
    AnatomicalStructure("pelvis", "bone", "skeletal", "pelvis", paired=True,
                        description="hip bone"),
    AnatomicalStructure("sacrum", "bone", "skeletal", "back", "lumbar_spine",
                        description="triangular bone at the base of the spine"),
    AnatomicalStructure("coccyx", "bone", "skeletal", "back", "lumbar_spine",
                        description="tailbone"),
    # Back / Spine
    AnatomicalStructure("cervical vertebrae", "bone", "skeletal", "back", "cervical_spine",
                        description="seven vertebrae of the neck"),
    AnatomicalStructure("thoracic vertebrae", "bone", "skeletal", "back", "thoracic_spine",
                        description="twelve vertebrae of the mid-back"),
    AnatomicalStructure("lumbar vertebrae", "bone", "skeletal", "back", "lumbar_spine",
                        description="five vertebrae of the lower back"),
]

# ---------------------------------------------------------------------------
# Muscles (~45)
# ---------------------------------------------------------------------------

_MUSCLES: list[AnatomicalStructure] = [
    # Head
    AnatomicalStructure("masseter", "muscle", "muscular", "head", "face", paired=True,
                        description="jaw muscle used for chewing"),
    AnatomicalStructure("temporalis", "muscle", "muscular", "head", "skull", paired=True,
                        description="temple muscle used for chewing"),
    # Neck
    AnatomicalStructure("sternocleidomastoid", "muscle", "muscular", "neck", paired=True,
                        description="large neck muscle for head rotation"),
    # Thorax
    AnatomicalStructure("pectoralis major", "muscle", "muscular", "thorax", "chest", paired=True,
                        description="large chest muscle"),
    AnatomicalStructure("intercostal muscles", "muscle", "muscular", "thorax", "chest", paired=True,
                        description="muscles between the ribs"),
    AnatomicalStructure("diaphragm", "muscle", "muscular", "thorax",
                        related_systems=["respiratory"],
                        description="dome-shaped muscle for breathing"),
    # Abdomen
    AnatomicalStructure("rectus abdominis", "muscle", "muscular", "abdomen", paired=True,
                        description="abdominal muscle forming the six-pack"),
    AnatomicalStructure("external oblique", "muscle", "muscular", "abdomen", paired=True,
                        description="outer abdominal muscle on the side"),
    AnatomicalStructure("internal oblique", "muscle", "muscular", "abdomen", paired=True,
                        description="inner abdominal muscle on the side"),
    AnatomicalStructure("transversus abdominis", "muscle", "muscular", "abdomen", paired=True,
                        description="deepest abdominal muscle"),
    # Back
    AnatomicalStructure("trapezius", "muscle", "muscular", "back", paired=True,
                        description="large triangular back muscle"),
    AnatomicalStructure("latissimus dorsi", "muscle", "muscular", "back", paired=True,
                        description="broadest muscle of the back"),
    AnatomicalStructure("erector spinae", "muscle", "muscular", "back", paired=True,
                        description="group of muscles running along the spine"),
    # Upper limb
    AnatomicalStructure("deltoid", "muscle", "muscular", "upper_limb", "shoulder", paired=True,
                        description="triangular shoulder muscle"),
    AnatomicalStructure("biceps", "muscle", "muscular", "upper_limb", "arm", paired=True,
                        description="muscle on the front of the upper arm"),
    AnatomicalStructure("triceps", "muscle", "muscular", "upper_limb", "arm", paired=True,
                        description="muscle on the back of the upper arm"),
    AnatomicalStructure("rotator cuff", "muscle", "muscular", "upper_limb", "shoulder", paired=True,
                        description="group of muscles stabilizing the shoulder"),
    AnatomicalStructure("forearm flexors", "muscle", "muscular", "upper_limb", "forearm", paired=True,
                        description="muscles that flex the wrist and fingers"),
    AnatomicalStructure("forearm extensors", "muscle", "muscular", "upper_limb", "forearm", paired=True,
                        description="muscles that extend the wrist and fingers"),
    # Lower limb
    AnatomicalStructure("gluteus maximus", "muscle", "muscular", "lower_limb", "hip", paired=True,
                        description="largest muscle in the body, in the buttock"),
    AnatomicalStructure("gluteus medius", "muscle", "muscular", "lower_limb", "hip", paired=True,
                        description="muscle on the outer surface of the pelvis"),
    AnatomicalStructure("quadriceps", "muscle", "muscular", "lower_limb", "thigh", paired=True,
                        description="four-part muscle on the front of the thigh"),
    AnatomicalStructure("hamstrings", "muscle", "muscular", "lower_limb", "thigh", paired=True,
                        description="muscles on the back of the thigh"),
    AnatomicalStructure("adductors", "muscle", "muscular", "lower_limb", "thigh", paired=True,
                        description="muscles on the inner thigh"),
    AnatomicalStructure("gastrocnemius", "muscle", "muscular", "lower_limb", "leg", paired=True,
                        description="large calf muscle"),
    AnatomicalStructure("soleus", "muscle", "muscular", "lower_limb", "leg", paired=True,
                        description="flat calf muscle beneath the gastrocnemius"),
    AnatomicalStructure("tibialis anterior", "muscle", "muscular", "lower_limb", "leg", paired=True,
                        description="muscle on the front of the shin"),
    AnatomicalStructure("sartorius", "muscle", "muscular", "lower_limb", "thigh", paired=True,
                        description="longest muscle in the body, crossing the thigh"),
]

# ---------------------------------------------------------------------------
# Organs (~35)
# ---------------------------------------------------------------------------

_ORGANS: list[AnatomicalStructure] = [
    # Nervous system
    AnatomicalStructure("brain", "organ", "nervous", "head", "cranium",
                        description="central organ of the nervous system"),
    AnatomicalStructure("spinal cord", "organ", "nervous", "back",
                        spans_regions=True,
                        description="long bundle of nerves running through the vertebral column"),
    # Cardiovascular
    AnatomicalStructure("heart", "organ", "cardiovascular", "thorax", "chest",
                        description="muscular organ pumping blood"),
    # Respiratory
    AnatomicalStructure("lungs", "organ", "respiratory", "thorax", "chest", paired=True,
                        description="organs for gas exchange"),
    AnatomicalStructure("trachea", "organ", "respiratory", "neck",
                        spans_regions=True,
                        description="windpipe connecting larynx to bronchi"),
    AnatomicalStructure("larynx", "organ", "respiratory", "neck",
                        description="voice box"),
    # Digestive
    AnatomicalStructure("stomach", "organ", "digestive", "abdomen",
                        description="organ for digesting food"),
    AnatomicalStructure("liver", "organ", "digestive", "abdomen",
                        description="largest internal organ, processes nutrients"),
    AnatomicalStructure("small intestine", "organ", "digestive", "abdomen",
                        description="long tube where most nutrient absorption occurs"),
    AnatomicalStructure("large intestine", "organ", "digestive", "abdomen",
                        description="absorbs water and forms stool"),
    AnatomicalStructure("gallbladder", "organ", "digestive", "abdomen",
                        description="stores bile from the liver"),
    AnatomicalStructure("pancreas", "organ", "digestive", "abdomen",
                        related_systems=["endocrine"],
                        description="produces digestive enzymes and insulin"),
    AnatomicalStructure("appendix", "organ", "digestive", "abdomen",
                        related_systems=["lymphatic"],
                        description="small pouch attached to the large intestine"),
    AnatomicalStructure("esophagus", "organ", "digestive", "thorax",
                        spans_regions=True,
                        description="tube connecting throat to stomach"),
    # Urinary
    AnatomicalStructure("kidneys", "organ", "urinary", "abdomen", paired=True,
                        description="filter blood and produce urine"),
    AnatomicalStructure("bladder", "organ", "urinary", "pelvis",
                        description="stores urine"),
    # Endocrine
    AnatomicalStructure("thyroid", "gland", "endocrine", "neck",
                        description="butterfly-shaped gland regulating metabolism"),
    AnatomicalStructure("adrenal glands", "gland", "endocrine", "abdomen", paired=True,
                        description="glands atop the kidneys producing hormones"),
    AnatomicalStructure("pituitary gland", "gland", "endocrine", "head", "skull",
                        description="master gland at the base of the brain"),
    AnatomicalStructure("pineal gland", "gland", "endocrine", "head", "cranium",
                        description="gland producing melatonin"),
    AnatomicalStructure("thymus", "gland", "lymphatic", "thorax", "chest",
                        related_systems=["endocrine"],
                        description="gland important for immune development"),
    # Lymphatic
    AnatomicalStructure("spleen", "organ", "lymphatic", "abdomen",
                        description="organ filtering blood and recycling red blood cells"),
    AnatomicalStructure("tonsils", "organ", "lymphatic", "neck", paired=True,
                        description="lymphoid tissue in the throat"),
    # Integumentary
    AnatomicalStructure("skin", "organ", "integumentary", "thorax",
                        spans_regions=True,
                        description="largest organ, covers the entire body"),
    # Sensory organs
    AnatomicalStructure("eyes", "organ", "nervous", "head", "face", paired=True,
                        description="organs of vision"),
    AnatomicalStructure("ears", "organ", "nervous", "head", "face", paired=True,
                        description="organs of hearing and balance"),
    AnatomicalStructure("tongue", "organ", "digestive", "head", "face",
                        related_systems=["muscular", "nervous"],
                        description="muscular organ for taste and swallowing"),
    # Reproductive
    AnatomicalStructure("prostate", "gland", "reproductive", "pelvis",
                        description="gland surrounding the male urethra"),
    AnatomicalStructure("uterus", "organ", "reproductive", "pelvis",
                        description="organ where fetal development occurs"),
    AnatomicalStructure("ovaries", "gland", "reproductive", "pelvis", paired=True,
                        related_systems=["endocrine"],
                        description="female gonads producing eggs and hormones"),
    AnatomicalStructure("testes", "gland", "reproductive", "pelvis", paired=True,
                        related_systems=["endocrine"],
                        description="male gonads producing sperm and testosterone"),
]

# ---------------------------------------------------------------------------
# Nerves (~15)
# ---------------------------------------------------------------------------

_NERVES: list[AnatomicalStructure] = [
    AnatomicalStructure("sciatic nerve", "nerve", "nervous", "lower_limb", "hip", paired=True,
                        description="largest nerve in the body, runs from lower back to feet"),
    AnatomicalStructure("femoral nerve", "nerve", "nervous", "lower_limb", "thigh", paired=True,
                        description="nerve supplying the front of the thigh"),
    AnatomicalStructure("radial nerve", "nerve", "nervous", "upper_limb", "arm", paired=True,
                        description="nerve supplying the back of the arm and forearm"),
    AnatomicalStructure("ulnar nerve", "nerve", "nervous", "upper_limb", "forearm", paired=True,
                        description="nerve running along the inner forearm, funny bone nerve"),
    AnatomicalStructure("median nerve", "nerve", "nervous", "upper_limb", "forearm", paired=True,
                        description="nerve supplying the palm and most fingers"),
    AnatomicalStructure("vagus nerve", "nerve", "nervous", "neck", paired=True,
                        spans_regions=True,
                        description="longest cranial nerve, extends to the abdomen"),
    AnatomicalStructure("optic nerve", "nerve", "nervous", "head", "skull", paired=True,
                        description="nerve carrying visual information from the eye to the brain"),
    AnatomicalStructure("trigeminal nerve", "nerve", "nervous", "head", "face", paired=True,
                        description="nerve responsible for face sensation and chewing"),
    AnatomicalStructure("phrenic nerve", "nerve", "nervous", "neck", paired=True,
                        related_systems=["respiratory"],
                        description="nerve controlling the diaphragm"),
    AnatomicalStructure("brachial plexus", "nerve", "nervous", "neck",
                        description="network of nerves supplying the upper limb"),
    AnatomicalStructure("tibial nerve", "nerve", "nervous", "lower_limb", "leg", paired=True,
                        description="nerve supplying the back of the leg and sole of the foot"),
    AnatomicalStructure("peroneal nerve", "nerve", "nervous", "lower_limb", "leg", paired=True,
                        description="nerve supplying the front of the leg and top of the foot"),
]

# ---------------------------------------------------------------------------
# Vessels (~15)
# ---------------------------------------------------------------------------

_VESSELS: list[AnatomicalStructure] = [
    AnatomicalStructure("aorta", "vessel", "cardiovascular", "thorax", "chest",
                        description="largest artery in the body"),
    AnatomicalStructure("superior vena cava", "vessel", "cardiovascular", "thorax", "chest",
                        description="large vein draining the upper body into the heart"),
    AnatomicalStructure("inferior vena cava", "vessel", "cardiovascular", "abdomen",
                        spans_regions=True,
                        description="large vein draining the lower body into the heart"),
    AnatomicalStructure("pulmonary artery", "vessel", "cardiovascular", "thorax", "chest",
                        related_systems=["respiratory"],
                        description="artery carrying blood from heart to lungs"),
    AnatomicalStructure("pulmonary vein", "vessel", "cardiovascular", "thorax", "chest",
                        related_systems=["respiratory"],
                        description="vein carrying oxygenated blood from lungs to heart"),
    AnatomicalStructure("carotid artery", "vessel", "cardiovascular", "neck", paired=True,
                        description="major artery supplying blood to the head"),
    AnatomicalStructure("jugular vein", "vessel", "cardiovascular", "neck", paired=True,
                        description="major vein draining blood from the head"),
    AnatomicalStructure("femoral artery", "vessel", "cardiovascular", "lower_limb", "thigh", paired=True,
                        description="main artery of the thigh"),
    AnatomicalStructure("femoral vein", "vessel", "cardiovascular", "lower_limb", "thigh", paired=True,
                        description="main vein of the thigh"),
    AnatomicalStructure("coronary arteries", "vessel", "cardiovascular", "thorax", "chest",
                        description="arteries supplying blood to the heart muscle"),
    AnatomicalStructure("hepatic portal vein", "vessel", "cardiovascular", "abdomen",
                        related_systems=["digestive"],
                        description="vein carrying blood from gut to liver"),
    AnatomicalStructure("renal artery", "vessel", "cardiovascular", "abdomen", paired=True,
                        related_systems=["urinary"],
                        description="artery supplying blood to the kidneys"),
]

# ---------------------------------------------------------------------------
# Ligaments and tendons (~12)
# ---------------------------------------------------------------------------

_CONNECTIVE: list[AnatomicalStructure] = [
    # Knee ligaments
    AnatomicalStructure("anterior cruciate ligament", "ligament", "skeletal", "lower_limb", "leg", paired=True,
                        description="ligament preventing forward sliding of the tibia"),
    AnatomicalStructure("posterior cruciate ligament", "ligament", "skeletal", "lower_limb", "leg", paired=True,
                        description="ligament preventing backward sliding of the tibia"),
    AnatomicalStructure("medial collateral ligament", "ligament", "skeletal", "lower_limb", "leg", paired=True,
                        description="ligament on the inner side of the knee"),
    AnatomicalStructure("lateral collateral ligament", "ligament", "skeletal", "lower_limb", "leg", paired=True,
                        description="ligament on the outer side of the knee"),
    # Tendons
    AnatomicalStructure("Achilles tendon", "tendon", "muscular", "lower_limb", "foot", paired=True,
                        description="tendon connecting the calf muscles to the heel"),
    AnatomicalStructure("patellar tendon", "tendon", "muscular", "lower_limb", "leg", paired=True,
                        description="tendon connecting the kneecap to the shinbone"),
    # Foot
    AnatomicalStructure("plantar fascia", "ligament", "skeletal", "lower_limb", "foot", paired=True,
                        description="thick band of tissue on the sole of the foot"),
    # Shoulder
    AnatomicalStructure("rotator cuff tendons", "tendon", "muscular", "upper_limb", "shoulder", paired=True,
                        description="tendons stabilizing the shoulder joint"),
    # Spine
    AnatomicalStructure("anterior longitudinal ligament", "ligament", "skeletal", "back",
                        description="ligament running along the front of the vertebral bodies"),
    # Wrist
    AnatomicalStructure("carpal tunnel", "ligament", "skeletal", "upper_limb", "hand", paired=True,
                        description="fibrous band forming the roof of the carpal tunnel"),
]


# ---------------------------------------------------------------------------
# Combined list
# ---------------------------------------------------------------------------

STRUCTURES: list[AnatomicalStructure] = _BONES + _MUSCLES + _ORGANS + _NERVES + _VESSELS + _CONNECTIVE
