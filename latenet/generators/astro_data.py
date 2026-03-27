"""Curated astronomical data for the astronomy generator.

All values sourced from:
- NASA Planetary Fact Sheet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/
- IAU Minor Planet Center: https://www.minorplanetcenter.net/
- SIMBAD Astronomical Database: https://simbad.u-strasbg.fr/
- IAU Constellation list: https://www.iau.org/public/themes/constellations/

Data is static and hardcoded — the solar system doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Planet:
    name: str
    order_from_sun: int
    orbital_period_years: float
    mass_kg: float
    diameter_km: float
    number_of_moons: int
    has_rings: bool
    planet_type: str  # "terrestrial", "gas_giant", "ice_giant"
    average_distance_from_sun_au: float
    is_dwarf_planet: bool = False


@dataclass(frozen=True)
class Moon:
    name: str
    parent_planet: str
    orbital_period_days: float
    diameter_km: float
    discovery_year: int | None = None


@dataclass(frozen=True)
class Star:
    name: str
    distance_ly: float
    spectral_type: str
    apparent_magnitude: float
    is_binary: bool = False


@dataclass(frozen=True)
class Constellation:
    name: str
    notable_stars: list[str] = field(default_factory=list)
    hemisphere: str = "northern"  # "northern", "southern", "both"


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------

PLANETS: list[Planet] = [
    Planet("Mercury", 1, 0.241, 3.301e23, 4879, 0, False, "terrestrial", 0.387),
    Planet("Venus", 2, 0.615, 4.867e24, 12104, 0, False, "terrestrial", 0.723),
    Planet("Earth", 3, 1.000, 5.972e24, 12756, 1, False, "terrestrial", 1.000),
    Planet("Mars", 4, 1.881, 6.417e23, 6792, 2, False, "terrestrial", 1.524),
    Planet("Jupiter", 5, 11.862, 1.898e27, 142984, 95, True, "gas_giant", 5.203),
    Planet("Saturn", 6, 29.457, 5.683e26, 120536, 146, True, "gas_giant", 9.537),
    Planet("Uranus", 7, 84.011, 8.681e25, 51118, 28, True, "ice_giant", 19.191),
    Planet("Neptune", 8, 164.790, 1.024e26, 49528, 16, True, "ice_giant", 30.069),
]

# ---------------------------------------------------------------------------
# Dwarf planets
# ---------------------------------------------------------------------------

DWARF_PLANETS: list[Planet] = [
    Planet("Ceres", 0, 4.600, 9.393e20, 946, 0, False, "dwarf_planet", 2.768, is_dwarf_planet=True),
    Planet("Pluto", 0, 247.940, 1.303e22, 2377, 5, False, "dwarf_planet", 39.482, is_dwarf_planet=True),
    Planet("Haumea", 0, 283.280, 4.006e21, 1632, 2, False, "dwarf_planet", 43.218, is_dwarf_planet=True),
    Planet("Makemake", 0, 306.200, 3.100e21, 1430, 1, False, "dwarf_planet", 45.430, is_dwarf_planet=True),
    Planet("Eris", 0, 558.040, 1.660e22, 2326, 1, False, "dwarf_planet", 67.781, is_dwarf_planet=True),
]

# ---------------------------------------------------------------------------
# Major moons
# ---------------------------------------------------------------------------

MOONS: list[Moon] = [
    # Earth
    Moon("Moon", "Earth", 27.322, 3475),
    # Mars
    Moon("Phobos", "Mars", 0.319, 22, 1877),
    Moon("Deimos", "Mars", 1.263, 12, 1877),
    # Jupiter
    Moon("Io", "Jupiter", 1.769, 3643, 1610),
    Moon("Europa", "Jupiter", 3.551, 3122, 1610),
    Moon("Ganymede", "Jupiter", 7.155, 5268, 1610),
    Moon("Callisto", "Jupiter", 16.689, 4821, 1610),
    # Saturn
    Moon("Mimas", "Saturn", 0.942, 396, 1789),
    Moon("Enceladus", "Saturn", 1.370, 504, 1789),
    Moon("Tethys", "Saturn", 1.888, 1062, 1684),
    Moon("Dione", "Saturn", 2.737, 1123, 1684),
    Moon("Rhea", "Saturn", 4.518, 1527, 1672),
    Moon("Titan", "Saturn", 15.945, 5150, 1655),
    Moon("Iapetus", "Saturn", 79.322, 1470, 1671),
    # Uranus
    Moon("Miranda", "Uranus", 1.413, 472, 1948),
    Moon("Ariel", "Uranus", 2.520, 1158, 1851),
    Moon("Umbriel", "Uranus", 4.144, 1169, 1851),
    Moon("Titania", "Uranus", 8.706, 1577, 1787),
    Moon("Oberon", "Uranus", 13.463, 1522, 1787),
    # Neptune
    Moon("Triton", "Neptune", 5.877, 2707, 1846),
    Moon("Proteus", "Neptune", 1.122, 420, 1989),
    # Pluto
    Moon("Charon", "Pluto", 6.387, 1212, 1978),
]

# ---------------------------------------------------------------------------
# Notable stars
# ---------------------------------------------------------------------------

STARS: list[Star] = [
    Star("Sun", 0.0, "G2V", -26.74),
    Star("Proxima Centauri", 4.24, "M5.5Ve", 11.13),
    Star("Alpha Centauri A", 4.37, "G2V", -0.01, is_binary=True),
    Star("Alpha Centauri B", 4.37, "K1V", 1.33, is_binary=True),
    Star("Sirius", 8.60, "A1V", -1.46, is_binary=True),
    Star("Betelgeuse", 700.0, "M1Ia", 0.42),
    Star("Rigel", 860.0, "B8Ia", 0.13, is_binary=True),
    Star("Polaris", 433.0, "F7Ib", 1.98),
    Star("Vega", 25.0, "A0V", 0.03),
    Star("Aldebaran", 65.3, "K5III", 0.85),
    Star("Antares", 550.0, "M1Iab", 1.06),
    Star("Canopus", 310.0, "A9II", -0.74),
    Star("Arcturus", 36.7, "K0III", -0.05),
    Star("Deneb", 2615.0, "A2Ia", 1.25),
    Star("Fomalhaut", 25.1, "A3V", 1.16),
]

# ---------------------------------------------------------------------------
# Constellations and star membership
# ---------------------------------------------------------------------------

CONSTELLATIONS: list[Constellation] = [
    Constellation("Orion", ["Betelgeuse", "Rigel"], "both"),
    Constellation("Ursa Major", [], "northern"),
    Constellation("Ursa Minor", ["Polaris"], "northern"),
    Constellation("Cassiopeia", [], "northern"),
    Constellation("Leo", [], "northern"),
    Constellation("Scorpius", ["Antares"], "southern"),
    Constellation("Lyra", ["Vega"], "northern"),
    Constellation("Taurus", ["Aldebaran"], "northern"),
    Constellation("Canis Major", ["Sirius"], "southern"),
    Constellation("Carina", ["Canopus"], "southern"),
    Constellation("Bootes", ["Arcturus"], "northern"),
    Constellation("Cygnus", ["Deneb"], "northern"),
    Constellation("Piscis Austrinus", ["Fomalhaut"], "southern"),
    Constellation("Centaurus", ["Alpha Centauri A", "Alpha Centauri B", "Proxima Centauri"], "southern"),
    Constellation("Gemini", [], "northern"),
    Constellation("Aquarius", [], "both"),
    Constellation("Sagittarius", [], "southern"),
    Constellation("Virgo", [], "both"),
    Constellation("Andromeda", [], "northern"),
    Constellation("Pegasus", [], "northern"),
]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

PLANET_BY_NAME: dict[str, Planet] = {p.name: p for p in PLANETS}
DWARF_BY_NAME: dict[str, Planet] = {p.name: p for p in DWARF_PLANETS}
ALL_BODIES_BY_NAME: dict[str, Planet] = {**PLANET_BY_NAME, **DWARF_BY_NAME}
MOON_BY_NAME: dict[str, Moon] = {m.name: m for m in MOONS}
STAR_BY_NAME: dict[str, Star] = {s.name: s for s in STARS}
CONSTELLATION_BY_NAME: dict[str, Constellation] = {c.name: c for c in CONSTELLATIONS}

# Moons grouped by parent planet
MOONS_BY_PARENT: dict[str, list[Moon]] = {}
for _m in MOONS:
    MOONS_BY_PARENT.setdefault(_m.parent_planet, []).append(_m)

# Star-to-constellation mapping
STAR_TO_CONSTELLATION: dict[str, str] = {}
for _c in CONSTELLATIONS:
    for _s in _c.notable_stars:
        STAR_TO_CONSTELLATION[_s] = _c.name

# Planet type groups
PLANET_TYPES = ["terrestrial", "gas_giant", "ice_giant", "dwarf_planet"]
TYPE_LABELS: dict[str, str] = {
    "terrestrial": "terrestrial planet",
    "gas_giant": "gas giant",
    "ice_giant": "ice giant",
    "dwarf_planet": "dwarf planet",
}
