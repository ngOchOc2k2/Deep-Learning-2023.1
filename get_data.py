import json
from config import *


class TerminalColor:
    COLORS = {
        'black': '0;30',
        'red': '0;31',
        'green': '0;32',
        'yellow': '0;33',
        'blue': '0;34',
        'purple': '0;35',
        'cyan': '0;36',
        'white': '0;37'
    }

    @classmethod
    def color_text(cls, text, color='white'):
        color_code = cls.COLORS.get(color, cls.COLORS['white'])
        return f'\033[{color_code}m{text}\033[0m'
    
    

data = json.load(open('./Data Json/data_total.json', 'r'))


print(f"Max Lenth Query: {len(data)}")
while True:
    colorizer = TerminalColor()
    inputs = input('Continue: ')
    
    if inputs == 'N':
        break
    else:
        print(colorizer.color_text(f"Query: {data[int(inputs)]['query']}", color='green'))
        print(colorizer.color_text(f"Passage: {data[int(inputs)]['passage']}", color='yellow'))
        print('---'*40)