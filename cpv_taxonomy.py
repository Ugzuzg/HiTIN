import json

def tree_to_taxonomy(node) -> list[str]:
    """A taxonomy file is a tab separated file with the following structure:
    Category \t Subcategory1 \t Subcategory2
    Subcategory1 \t Subcategory1.1 \t Subcategory1.2
    Subcategory2 \t Subcategory2.1 \t Subcategory2.2
    """
    if node['children'] is None or len(node['children']) == 0:
        return []

    out = []
    direct_children = [child['cpv'] for child in (node['children'] or []) if child['cpv'] != node['cpv']]
    out.append('\t'.join([node['cpv']] + direct_children) + '\n')
    for child in (node['children'] or []):
        out += tree_to_taxonomy(child)
    return out

with open('cpv/cpv_hierarchy.json', 'r') as hierarchy_file:
    with open('cpv/cpv.taxonomy', 'w') as taxonomy_file:
        hierarchy = json.load(hierarchy_file)
        hierarchy_lines = tree_to_taxonomy({ 'name': 'Root', 'cpv': 'Root', 'children': hierarchy })
        taxonomy_file.writelines(hierarchy_lines)
