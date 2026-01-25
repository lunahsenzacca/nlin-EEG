from rich import print
from rich.pretty import Pretty
from rich.panel import Panel
from rich import box
from rich.columns import Columns
from rich.style import Style
from rich.table import Table, Column

def pdict(d: dict, max_length = 14, **kwargs):

    pretty = Pretty(d, max_length = max_length, indent_size = 2)
    panel = Panel(pretty, width = 50, **kwargs)

    print(panel)

    print('')

    return

def pdict_leg(d: dict, leg: dict, max_length = 14, **kwargs):

    table = Table('DICT',Column(header = 'LEGEND', style= Style(color='red')), title = None,
                  show_header= False, leading = 1, box = box.SIMPLE_HEAD)

    for k in d.items():

        if k[0] in leg:

            inf = leg[k[0]]

        else:

            inf = None

        table.add_row(Pretty({k[0]: k[1]}, max_length = max_length, indent_size= 2),inf)

    panel = Panel(table, width = 80, **kwargs)

    print(panel)

    print('')

    return
