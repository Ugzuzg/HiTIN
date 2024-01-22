import json

def walk_hierarchy(node, depth=0):
    if node['children'] is None or len(node['children']) == 0:
        return []

    out = []
    direct_children = [child['cpv'] for child in (node['children'] or []) if child['cpv'] != node['cpv']]
    out.append('\t'.join([node['cpv']] + direct_children) + '\n')
    for child in (node['children'] or []):
        out += walk_hierarchy(child, depth=depth+1)
    return out

with open('cpv/cpv_hierarchy.json', 'r') as hierarchy_file:
    with open('cpv/cpv.taxonomy', 'w') as taxonomy_file:
        hierarchy = json.load(hierarchy_file)
        hierarchy_lines = walk_hierarchy({ 'name': 'Root', 'cpv': 'Root', 'children': hierarchy })
        taxonomy_file.writelines(hierarchy_lines)
