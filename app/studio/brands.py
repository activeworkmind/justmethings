"""Brand profiles: visual theme, AI background prompt, colour grading, dietary rules and the 1-click preset.

Light direction convention used across the engine: an angle in degrees, 0 = from the right, 90 = from above,
135 = from the upper-left, 180 = from the left (counter-clockwise, as on a compass seen on screen)."""
from dataclasses import dataclass, field

GLOBAL_NEGATIVE = ("watermark, text, logo, signature, diamond icon, blurry foreground, cartoon, 3d render, "
                   "plastic, overexposed, hands, people, extra plates in center, deformed")

DIET_LABELS = {"veg": "Pure Vegetarian (green circle)", "egg": "Contains Egg (amber circle)",
               "nonveg": "Non-Vegetarian (red triangle)"}


@dataclass(frozen=True)
class BrandProfile:
    key: str
    name: str
    slug: str
    tagline: str
    vibe: str
    bg_prompt: str
    primary: str                      # hex colours used by the badge / border
    secondary: str
    accent: str
    warmth: float                     # +0.15 = +15 %
    contrast: float
    saturation: float
    vignette: float
    diets: tuple = ("veg",)
    default_diet: str = "veg"
    light_angle: float = 135.0        # where the key light comes from
    preset: tuple = field(default_factory=tuple)   # ordered (op, params) for the 1-click master preset
    style: str = "royal"              # selects the procedural background generator

    @property
    def negative_prompt(self) -> str:
        return GLOBAL_NEGATIVE


BRANDS = {
    "biryani_rules": BrandProfile(
        key="biryani_rules", name="Biryani Rules", slug="biryani-rules",
        tagline="100% PURE VEGETARIAN ROYALE  -  NO COMPROMISE ON DUM",
        vibe="Majestic, royal, dark & moody, traditional Shahi Darbar",
        bg_prompt=("Professional food photography background only, dark royal carved rosewood table, subtle warm "
                   "amber rim lighting, a few scattered green cardamom pods, star anise, and saffron strands on the "
                   "wood, dark moody royal Indian dining ambiance, shallow depth of field, 45-degree angle, "
                   "photorealistic, 8k, empty center for placing main dish."),
        primary="#14284B", secondary="#C9A227", accent="#E8A33D",
        warmth=0.15, contrast=0.20, saturation=0.12, vignette=0.25,
        diets=("veg",), default_diet="veg", light_angle=45.0, style="royal",
        preset=(("isolate", {}), ("background", {"source": "auto"}), ("framing", {"fill": 0.76}),
                ("shadows", {}), ("sunlight", {"strength": 0.25, "blinds": 0.0}), ("rim_glow", {"intensity": 0.7}),
                ("lab_pop", {}), ("steam", {"opacity": 0.35}), ("bokeh", {"strength": 0.7}),
                ("brand_grade", {})),
    ),
    "sunday_biryani": BrandProfile(
        key="sunday_biryani", name="The Sunday Biryani", slug="sunday-biryani",
        tagline="SLOW COOKED DUM PERFECTION & ROYAL FLAVOURS",
        vibe="Festive, hearty Sunday feast, celebratory, abundant family dining",
        bg_prompt=("Professional Indian feast food photography background, warm rustic teak wood table with subtle "
                   "banana leaf and brass katori bowls of raita and salan softly blurred in the background, festive "
                   "warm Sunday afternoon lighting, rich inviting tones, 45-degree angle, photorealistic, 8k, "
                   "empty center for main dish."),
        primary="#8B1A1A", secondary="#D9A21B", accent="#2F6B3A",
        warmth=0.18, contrast=0.10, saturation=0.18, vignette=0.10,
        diets=("veg", "egg", "nonveg"), default_diet="nonveg", light_angle=135.0, style="festive",
        preset=(("isolate", {}), ("background", {"source": "auto"}), ("framing", {"fill": 0.76}),
                ("shadows", {}), ("sunlight", {"strength": 0.45, "blinds": 0.0}), ("lab_pop", {}),
                ("steam", {"opacity": 0.30}), ("bokeh", {"strength": 0.55}), ("brand_grade", {})),
    ),
    "day_meal": BrandProfile(
        key="day_meal", name="The Day Meal", slug="day-meal",
        tagline="100% PURE VEGETARIAN HOMESTYLE KITCHEN",
        vibe="Comforting, fresh, everyday homestyle Indian thalis, curries and rotis",
        bg_prompt=("Clean homestyle Indian kitchen table background, light natural wood surface with a rustic woven "
                   "jute table runner, soft natural morning window sunlight coming from the left, fresh coriander "
                   "leaves and a small clay bowl in the blurred background, airy, wholesome, 45-degree angle, "
                   "photorealistic, 8k, empty center for main thali or bowl."),
        primary="#1E4D3A", secondary="#B5651D", accent="#E8DCC4",
        warmth=0.08, contrast=0.05, saturation=0.10, vignette=0.0,
        diets=("veg",), default_diet="veg", light_angle=180.0, style="homestyle",
        preset=(("isolate", {}), ("background", {"source": "auto"}), ("framing", {"fill": 0.76}),
                ("shadows", {}), ("sunlight", {"strength": 0.45, "blinds": 0.4, "angle": 180.0}),
                ("lab_pop", {}), ("bokeh", {"strength": 0.45}), ("brand_grade", {})),
    ),
}


def get_brand(key: str) -> BrandProfile:
    return BRANDS[key]
