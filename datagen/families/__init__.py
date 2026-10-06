"""Registry of generators by name."""

from .grids import Bars, Calendar, ColoredGrid, Grid, Piano, Rubik
from .pips import Die, Domino, DotsInSplitRect, DotsInSquare, GlyphsOnCard, PlayingCard
from .radial import Burst, Clock, ClockNumerals, NGon, Snowflake, Star, StopSign, TickRing, TrafficLight
from .shapes import SharedShapes
from .emoji import EmojiColor, EmojiCrowd

GENERATORS = {g.name: g for g in [
    Die(), PlayingCard(), Domino(), Clock(), StopSign(), Star(), Calendar(), Piano(), Rubik(),
    TrafficLight(), Snowflake(), ClockNumerals(),
    DotsInSquare(), GlyphsOnCard(), DotsInSplitRect(), TickRing(), NGon(), Burst(), Grid(), Bars(), ColoredGrid(),
    SharedShapes(),
    EmojiColor(), EmojiCrowd(),
]}

# Exp 1 conflict families (§4.2) in a fixed order, and each one's neutral twin.
E1_FAMILIES = ["die", "playing_card", "domino", "clock", "stop_sign", "star", "calendar", "piano", "rubik"]
TWIN_OF = {"die": "twin_dots_square", "playing_card": "twin_glyph_card", "domino": "twin_split_rect",
           "clock": "twin_ticks", "stop_sign": "twin_ngon", "star": "twin_burst", "calendar": "twin_grid",
           "piano": "twin_bars", "rubik": "twin_colored_grid"}
HELD_OUT = ["traffic_light", "snowflake"]
