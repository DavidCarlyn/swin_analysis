import os

import graphviz
import igraph

def load_data():
    data = []
    for root, dirs, files in os.walk('output'):
        for f in files:
            path = os.path.join(root, f)
            parts = f.split(".")[0].split('_')
            if parts[0] != 'depth': continue
            depth = int(parts[1])

            if depth == 0:
                name = 'root'
            else:
                name = "_".join(parts[2:])
            parent = None
            if depth == 1:
                parent = 'root'
            elif depth > 1:
                parent = "_".join(name.split("_")[:-1])
            data.append({
                'depth' : depth,
                'path' : path,
                'name' : name,
                'parent' : parent
            })
    return data

if __name__ == "__main__":
    data = load_data()
    
    g = graphviz.Digraph('Hierarchy', filename='inat_hierarchy.gv')
    edges = []
    for dp in data:
        if dp['parent'] is not None:
            edges.append([dp['parent'], dp['name']])
        #g.node(dp['name'], image=dp['path'], shape='rectangle', scale='false', fontsize='0', imagescale='true', fixedsize='true', height='1.5', width='3')
        g.node(dp['name'], image=dp['path'], shape='rectangle', fontsize='0')
    
    for edge in edges:
        g.edge(edge[0], edge[1])
    
    g.render("inat_hierarchy", format="png", view=False)
    #g.render("inat_hierarchy", format="svg", view=False)
    #g.render("inat_hierarchy", format="pdf", view=False)
        
