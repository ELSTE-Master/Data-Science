"""Original vector illustrations for the ice-cream/sharks teaching example."""
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'figures'
PURPLE, PINK, TURQUOISE, BLUE = '#6A5ACD', '#eb078e', '#20B2AA', '#2b59c3'

def cone(x, y, scale=1):
    return f'''<g transform="translate({x} {y}) scale({scale})">
      <path d="M-30 10 L30 10 L0 105 Z" fill="#f4ddbd" stroke="{PURPLE}" stroke-width="2.5"/>
      <path d="M-21 35 L12 68 M-13 61 L6 82 M21 35 L-12 68 M13 61 L-6 82" stroke="#cba66f" stroke-width="2"/>
      <path d="M-34 6 C-45-16-25-44-5-32 C9-54 43-30 35-4 C46 16 15 24 2 14 C-10 28-36 26-34 6 Z"
       fill="#f9bedf" stroke="{PURPLE}" stroke-width="2.8"/>
      <path d="M-18-13 Q-8-23 0-15 M15-18 Q25-24 29-12" fill="none" stroke="#eb78b5" stroke-width="2"/>
    </g>'''

def shark(x, y, scale=1):
    return f'''<g transform="translate({x} {y}) scale({scale})">
      <!-- Swept dorsal and pectoral fins; asymmetric shark tail, not a fish fork. -->
      <path d="M-38-28 Q-22-49-19-86 Q-5-59 24-30 Z"
       fill="#829faf" stroke="{PURPLE}" stroke-width="2.5" stroke-linejoin="round"/>
      <path d="M-80 12 Q-90 29-100 35 L-59 17 Z"
       fill="#829faf" stroke="{PURPLE}" stroke-width="2"/>
      <path d="M-113-5 Q-136-22-166-70 Q-158-31-141-4
       Q-140 13-158 39 Q-131 29-113 7
       C-67 25 14 35 92 13 Q114 10 136-1
       Q113-14 92-22 C29-43-55-33-113-5 Z"
       fill="#a8bfcc" stroke="{PURPLE}" stroke-width="2.8" stroke-linejoin="round"/>
      <path d="M-107 5 Q-28 27 54 12 Q97 13 127 1 Q111 18 78 20 Q-10 32-107 5 Z"
       fill="#edf5f7"/>
      <path d="M24 12 Q1 42-44 64 Q-29 37-29 15"
       fill="#829faf" stroke="{PURPLE}" stroke-width="2.5" stroke-linejoin="round"/>
      <circle cx="96" cy="-11" r="3.6" fill="#283d54"/>
      <path d="M58-15 q-7 11-4 20 M47-17 q-7 12-4 21
       M36-18 q-7 12-4 21 M25-19 q-6 11-3 21 M14-19 q-5 10-3 19"
       fill="none" stroke="{PURPLE}" stroke-width="1.8"/>
      <path d="M87 7 Q105 13 123 3 Q108 18 91 12 Z" fill="#536879"/>
      <path d="M94 9 l3 5 3-4 3 4 3-5 3 3 3-5" fill="white"/>
    </g>'''

def waves(x, y, width=300):
    return f'''<g transform="translate({x} {y})" fill="none" stroke="{TURQUOISE}" stroke-width="3" stroke-linecap="round">
      <path d="M0 0 q25-14 50 0 t50 0 t50 0 t50 0 t50 0 t50 0"/>
      <path d="M15 25 q25-14 50 0 t50 0 t50 0 t50 0 t50 0" opacity=".45"/>
    </g>'''

def swimmer(x, y):
    return f'''<g transform="translate({x} {y})" fill="none" stroke="{BLUE}" stroke-width="3.5" stroke-linecap="round">
      <circle cx="0" cy="-18" r="10" fill="#e8eef8"/>
      <path d="M-30 8 L-8-2 L15 8 L40-13 M-8-2 L-20-20"/>
    </g>'''

def sun(x, y):
    rays = ''.join(f'<path d="M0-49 v-17" transform="rotate({a})"/>' for a in range(0, 360, 45))
    return f'''<g transform="translate({x} {y})" stroke="#EB811B" stroke-width="3" stroke-linecap="round">
      {rays}<circle r="34" fill="#fff0cf"/>
    </g>'''

def wrap(width, height, title, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
    <title>{title}</title><rect width="100%" height="100%" fill="white"/>
    <defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{PURPLE}"/></marker></defs>
    <g font-family="Liberation Serif, serif" fill="#33454a">{body}</g></svg>'''

association = cone(135, 160, 1.25) + shark(450, 230, .72) + waves(325, 285)
association += f'''<path d="M225 195 H320" stroke="{PINK}" stroke-width="3" stroke-dasharray="7 7"/>
  <text x="273" y="170" text-anchor="middle" fill="{PINK}" font-size="44">?</text>
  <text x="135" y="345" text-anchor="middle" font-size="27">Ice-cream sales</text>
  <text x="450" y="345" text-anchor="middle" font-size="27">Shark incidents</text>'''
(OUT / 'icecream_shark_question.svg').write_text(wrap(650, 390, 'Does an association imply a cause?', association))

cause = sun(600, 72) + cone(245, 230, 1.05)
cause += swimmer(845, 224) + swimmer(931, 235) + shark(1060, 277, .58) + waves(792, 320)
cause += f'''<text x="600" y="158" text-anchor="middle" font-size="29" fill="{PURPLE}">Warmer beach weather</text>
  <path d="M505 174 Q388 165 302 217" fill="none" stroke="{PURPLE}" stroke-width="3" marker-end="url(#arrow)"/>
  <path d="M698 174 Q787 164 845 190" fill="none" stroke="{PURPLE}" stroke-width="3" marker-end="url(#arrow)"/>
  <path d="M365 277 H740" fill="none" stroke="{PINK}" stroke-width="2.5" stroke-dasharray="7 7"/>
  <text x="553" y="260" text-anchor="middle" font-size="22" fill="#777">Association in the scatterplot</text>
  <text x="245" y="380" text-anchor="middle" font-size="29">Ice-cream sales</text>
  <text x="943" y="373" text-anchor="middle" font-size="26">More swimming</text>
  <text x="943" y="410" text-anchor="middle" font-size="24">More opportunities for shark incidents</text>'''
(OUT / 'confounding_weather.svg').write_text(wrap(1250, 440, 'A common cause can produce an association', cause))
print('Generated original ice-cream, weather, swimmer and shark SVG illustrations.')
