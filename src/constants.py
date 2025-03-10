from enum import Enum


class Color(Enum):
    WHITE = (1, 1, 1)
    BLACK = (0, 0, 0)
    RED = (1, 0, 0)
    GREEN = (0, 1, 0)
    BLUE = (0, 0, 1)
    PALE_VIOLET = (0.342605, 0.313068, 0.496933)
    PALE_TURQUOISE = (0.239975, 0.426978, 0.533277)
    PALE_GREEN = (0.165398, 0.558341, 0.416653)
    BRIGHT_BLUE = (0.0419309, 0.154187, 0.438316)
    PALE_RED = (0.410603, 0.101933, 0.0683599)
    BEIGE = (0.496933, 0.472623, 0.331984)
    WARM_GREY = (0.502887, 0.494328, 0.456411)


class Strength(Enum):
    OFF = 0.5
    WEAK = 0.6
    MEDIUM = 0.7
    STRONG = 1.0


class Shading(Enum):
    FLAT = 'flat'
    SMOOTH = 'smooth'
    AUTO = 'auto'


class Shape(Enum):
    SPHERE = 'sphere'
    CUBE = 'cube'
    DIAMOND = 'diamond'


class Look(Enum):
    VERY_LOW_CONTRAST = 'Very Low Contrast'
    LOW_CONTRAST = 'Low Contrast'
    MEDIUM_CONTRAST = 'Medium Contrast'
    HIGH_CONTRAST = 'High Contrast'
    VERY_HIGH_CONTRAST = 'Very High Contrast'


class Shadow(Enum):
    OFF = 'off'
    VERY_HARD = 'very_hard'
    HARD = 'hard'
    MEDIUM = 'medium'
    SOFT = 'soft'
    VERY_SOFT = 'very_soft'


class Engine(Enum):
    CYCLES = 'cycles'
    EEVEE = 'eevee'


class Primitive(Enum):
    SUZANNE = 'suzanne'
    MONKEY = 'monkey'
    CUBE = 'cube'
    SPHERE = 'sphere'
    CYLINDER = 'cylinder'
    CONE = 'cone'


class Animation(Enum):
    TURN = 'turn'
    SWIVEL = 'swivel'
    TUMBLE = 'tumble'


class Light(Enum):
    OFF = 0.0
    VERY_DARK = 0.1
    DARK = 0.3
    MEDIUM = 0.5
    BRIGHT = 0.7
    VERY_BRIGHT = 1.0
