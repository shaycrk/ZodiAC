import json
import yaml
import argparse

# # TESTING: Load a scoresheet
# with open('new_test_output_dad.json', 'r') as f:
#     scoresheet = json.load(f)
# scoresheet['Finished'] = True
# scoresheet['Zodiac_1969'] = True
# scoresheet['Date_1886'] = True
# scoresheet['Passenger_Floyd'] = True
# scoresheet['Passenger_Lick'] = True
# scoresheet['Sunshine'] = True

# for i in range(1, 20):
#     scoresheet[f'CP_{i}'] = True
# scoresheet['CP_14'] = False

parser = argparse.ArgumentParser(description="Scoring")
parser.add_argument("-s", "--scoresheet", required=True,
                   help="Path to the scoresheet JSON file")
parser.add_argument("-o", "--output", required=True,
                   help="Path to the output YAML file")
parser.add_argument("-c", "--config", required=False, default='scoring_config.yaml',
                   help="Path to the scoring config YAML file")
args = parser.parse_args()

with open(args.scoresheet, 'r') as f:
    scoresheet = json.load(f)

with open(args.config, 'r') as f:
    scoring_config = yaml.safe_load(f)

multi = lambda x: sum([int(i) for i in x]) > 1

def check_option(option, scoresheet):
    option_type = [k for k in option.keys() if k != 'points'][0]
    option_bools = [str(scoresheet[key]) == str(value) for key, value in option[option_type].items()]
    if option_type == 'all':
        return all(option_bools)
    elif option_type == 'any':
        return any(option_bools)
    elif option_type == 'multi':
        return multi(option_bools)
    else:
        raise ValueError(f"Invalid option type: {option_type}")

scored = {}
for item, options in scoring_config.items():
    if not item.startswith('FISH -'):
        scored[item] = 0
    for option in options:
        if check_option(option, scoresheet):
            scored[item] = option['points']
            break

print(f"Total score for {args.scoresheet}: {sum(scored.values())}")

scored['RALLYE TOTAL'] = sum(scored.values())
with open(args.output, 'w') as f:
    yaml.dump(scored, f, indent=2, sort_keys=False)








