import pandas as pd
import json

all_spaces = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX',
               'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG', 'HHH', 'III', 'JJJ', 'LLL', 'MMM', 'NNN', 'OOO', 'PPP', 'QQQ', 'RRR', 'SSS', 'TTT', 'UUU', 'VVV', 'WWW',
               'AB', 'CA', 'ET', 'FT', 'GC', 'GM', 'HR', 'IL', 'MT', 'MTMT', 'NV', 'XU'
            ]

roi_width = 230
roi_height = 119
n_spaces_row = 7
x0 = 235
y0 = 913
x_offset = 320
y_offset = 119

critique_df = pd.read_csv('test_critique.csv', header=1)
poss_values = {}
for i, row in critique_df.iterrows():
    # skip critique records for CPs, scoresheet gimmicks, etc.
    if row['Letter'] not in all_spaces:
        continue
    poss_values[row['Letter']] = poss_values.get(row['Letter'], []) + [str(row['Number'])]

cm_rois = []
for i in range(len(all_spaces)):
    label = all_spaces[i]
    x = x0 + (i % n_spaces_row) * x_offset + (len(label) - 1) * 15
    y = y0 + (i // n_spaces_row) * y_offset
    allow_chars = ''.join(set().union(*[set(v) for v in poss_values.get(label, ['1234567890'])]))
    cm_rois.append({
        'id': f'cm_{label}',
        'bbox': [x, y, roi_width, roi_height],
        'poss_values': poss_values.get(label, []),
        'filter_keywords': [label],
        'ocr_config': {
            'tesseract_config': f'--psm 7 -c tessedit_char_whitelist={allow_chars}',
            'easyocr_allowlist': allow_chars
        }
    })
    print(f"{label}: [{x}, {y}, {roi_width}, {roi_height}] {allow_chars}")

# add car number roi
car_number_roi = {
    'id': 'car_number',
    'bbox': [264, 291, 183, 172],
    'filter_keywords': ['car', 'number']
}
cm_rois.append(car_number_roi)

# full config definition
config = {
    'default_filter_keywords': [],
    'default_ocr_config': {
        'preprocess': 'gaussian',
        'tesseract_config': '--psm 7 -c tessedit_char_whitelist=0123456789',
        'easyocr_allowlist': '0123456789'
    },
    'roi_locations': cm_rois
}

with open('all_cms_rois_config.json', 'w') as f:
    json.dump(config, f, indent=2)







