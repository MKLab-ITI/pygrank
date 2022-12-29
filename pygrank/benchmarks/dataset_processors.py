import os


def gml_processor(path: str):   # pragma: no cover
    groups = dict()
    pairs = list()
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.gml'):
                node_mode = False
                edge_mode = False
                with open(path+"/"+filename, "r") as file:
                    for line in file:
                        line = line.strip().split(" ")
                        if line[0] == "node":
                            node_mode = True
                        elif line[0] == "edge":
                            edge_mode = True
                        elif line[0] == "]":
                            node_mode = False
                            edge_mode = False
                        elif line[0] == "id" and node_mode:
                            nodeid = line[1]
                        elif line[0] == "value" and node_mode:
                            if line[1] not in groups:
                                groups[line[1]] = list()
                            groups[line[1]].append(nodeid)
                        elif line[0] == "source" and edge_mode:
                            src_id = line[1]
                        elif line[0] == "target" and edge_mode:
                            pairs.append((src_id, line[1]))
    if len(pairs) == 0:
        raise Exception("No file with .gml extension in "+path)

    #print("GML format parsed", len(pairs), "edges and ", len(groups), "node groups")

    with open(path+'/pairs.txt', 'w') as file:
        for p1, p2 in pairs:
            file.write(p1+'\t'+p2+'\n')

    with open(path+'/groups.txt', 'w') as file:
        for g in groups.values():
            for uid in g:
                file.write(uid+'\t')
            file.write('\n')


def amazon_processor(path: str):   # pragma: no cover
    pairs = list()
    id2title = {}
    id2group = {}
    group2ids = {}
    with open(path+'/all.txt', 'r', encoding='utf-8') as file:
        product = ""
        for line in file:
            line = line.strip()
            if line.startswith('ASIN:'):
                product = line[5:].strip()
            if line.startswith('title:'):
                id2title[product] = line[6:].strip()
            if line.startswith('group:'):
                group = line[6:].strip()
                id2group[product] = group
                group2ids[group] = group2ids.get(group, list())
                group2ids[group].append(product)
            if line.startswith('similar:'):
                for similar in [other for other in line[8:].strip().split(' ') if len(other) > 0][1:]:
                    pairs.append((product, similar))

    with open(path+'/pairs.txt', 'w') as file:
        for p1, p2 in pairs:
            file.write(p1+'\t'+p2+'\n')

    with open(path+'/groups.txt', 'w') as file:
        for g in group2ids.values():
            for uid in g:
                file.write(uid+'\t')
            file.write('\n')


def pokec_processor(path: str):   # pragma: no cover
    attributes = ["varenie",        # cooking
                  "jedlo",          # eating
                  "nakupovanie",    # shopping
                  "priatelia",      # friends
                  "anglicky",       # english
                  "sportovanie",    # sports
                  "party",          # music
                  "fotografovanie", # photography
                  "turistika",      # tourism
                  "film",           # movies
                  "malovanie",      # painting
                  "hudby",          # music
                  "kupalisko",      # swimming
                  ]
    groups = [list() for _ in range(len(attributes)+1)]
    data_path = path+'/groups.txt' if os.path.exists(path+'/groups.txt') else path+'/groups.txt.debug'
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split("\t")
            if line[3]=="1":
                groups[-1].append(line[0])
            for i, text in enumerate(attributes):
                if text in line:
                    groups[i].append(line[0])
    os.remove(data_path)

    with open(path+'/groups.txt', 'w') as file:
        for g in groups:
            for uid in g:
                file.write(uid+'\t')
            file.write('\n')