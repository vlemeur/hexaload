"""Plotly look applied to every hexaload figure (notebooks, reports).

Import this module once (``import hexaload.viz``) and every plotly figure gets
the hexaload template: clean white theme + Set2 qualitative palette.
deepcopy + override: nesting ``template="simple_white"`` inside a new Template
loses the custom colorway at render time.
"""

import copy

import plotly.express as px
import plotly.io as pio

_template = copy.deepcopy(pio.templates["simple_white"])
_template.layout.colorway = list(px.colors.qualitative.Set2)
pio.templates["hexaload"] = _template
pio.templates.default = "hexaload"
