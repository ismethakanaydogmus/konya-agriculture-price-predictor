import json

with open('ML_TimeSeries_Training_Evaluation_Template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = []
for i, c in enumerate(nb['cells']):
    type_ = c['cell_type'].upper().ljust(8)
    first_line = "".join(c['source']).split('\n')[0][:80] if c['source'] else "EMPTY"
    lines.append(f"{i:02d}: {type_} - {first_line}")

with open('template_flow.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
