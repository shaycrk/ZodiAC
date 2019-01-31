#!/usr/bin/env python

import csv
import random

cms = [
    ('O', '4'),
    ('M', '16'),
    ('I', '1'),
    ('L', '7'),
    ('II', '11'),
    ('E', '3'),
    ('EE', '33'),
    ('EEE', '333'),
    ('III', '444'),
    ('LLL', '000')
]

nums = [
    '22', '2', '6', '9', '19', '10', '51', '15', '17', '12', '21', '14', '18', '20', '23', '24',
    '25', '28', '27', '29', '30', '31', '53', '54', '55', '26', '48', '50', '56', '59', '60',
    '49',' 61', '58', '57', '62', '52', '78', '42', '13', '45', '63', '64', '41', '46', '5', '76',
    '66', '89', '65'
]

lets = [chr(i) for i in range(ord('A'), ord('Z')+1)]
lets = lets + [chr(i)*2 for i in range(ord('A'), ord('Z')+1)]
lets = lets + [chr(i)*3 for i in range(ord('A'), ord('T')+1)]


for letter, number in cms:
    lets.remove(letter)

for number in nums:
    new_letter = random.choice(lets)
    lets.remove(new_letter)
    cms = cms + [(new_letter, number)]

with open('cm_letters.csv', 'w') as f:
    c = csv.writer(f)
    c.writerow(['letter', 'number'])
    c.writerows(cms)


