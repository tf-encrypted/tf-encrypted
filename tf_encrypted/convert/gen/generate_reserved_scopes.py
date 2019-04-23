import os

from ..register import REGISTERED_SPECOPS


HD = """Reserved Scope | TF Counterpart\n---------------|---------------\n"""


def table_from_registered_specops():
    table_string = HD
    for scope, attr_dict in REGISTERED_SPECOPS.items():
        tf_name = attr_dict["tf-name"]
        hyperlink = attr_dict["hyperlink"]
        table_string += """`{scope}`|[{name}]({hyperlink})\n""".format(
            scope=scope,
            name=tf_name,
            hyperlink=hyperlink)

    return table_string


reserved_scopes_table = table_from_registered_specops()

this_file = os.path.realpath(__file__)
gen_dir = os.path.dirname(this_file)
convert_dir = os.path.dirname(gen_dir)

template_path = os.path.join(gen_dir, "readme_template.md")
readme_path = os.path.join(convert_dir, "README.md")

with open(template_path, "r") as template_file:
    template_str = template_file.read()
    with open(readme_path, "w") as target_readme:
        with_table = template_str.format(
            reserved_scopes=reserved_scopes_table)
        target_readme.write(with_table)
