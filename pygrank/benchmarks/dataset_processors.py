def amazon_processor(path: str = "data"):   # pragma: no cover
    pairs = list()
    id2title = {}
    id2group = {}
    group2ids = {}
    with open('amazon-meta.txt', 'r', encoding='utf-8') as file:
        product = ""
        for line in file:
            line = line.strip()
            if(line.startswith('ASIN:')):
                product = line[5:].strip()
            if(line.startswith('title:')):
                id2title[product] = line[6:].strip()
            if(line.startswith('group:')):
                group = line[6:].strip()
                id2group[product] = group
                group2ids[group] = group2ids.get(group, list())
                group2ids[group].append(product)
            if(line.startswith('similar:')):
                for similar in [other for other in line[8:].strip().split(' ') if len(other)>0][1:]:
                    pairs.append((product, similar))

    with open(path+'/pairs.txt', 'w') as file:
        for p1, p2 in pairs:
            file.write(p1+'\t'+p2+'\n')

    with open(path+'/groups.txt', 'w') as file:
        for g in group2ids.values():
            for uid in g:
                file.write(uid+'\t')
            file.write('\n')
