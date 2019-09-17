import algorithms
import sklearn.metrics

"""
def display_top(algorithm, G, group, ids2names, top=None, divide_by=None):
    import operator
    print('----- Calculating scores -----')
    ranks = algorithm.rank(G, ranking_algorithms.convert_to_ranks(G, group))
    original_ranks = ranks
    if not divide_by is None:
        ranks = {v: ranks[v]/(1+divide_by[v]*0.03) for v in divide_by.keys() if v in ranks}
    sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)
    if top is None:
        minTop = min([ranks[v] for v in group if v in ranks])
        print(minTop)
        top = 100
    else:
        minTop = 0
    for item in sorted_ranks:
        key = item[0]
        if key in ids2names:
            top -= 1
            if top<0 or ranks[key]<=minTop:
                break
            if not divide_by is None:
                print("%s, %.2f, %.2f, %.2f" % (ids2names[key], original_ranks[key], divide_by[key], ranks[key]))
            else:
                print("%s, %.2f" % (ids2names[key], ranks[key]))
"""


def split_groups(groups, split, G=None):
    if split==1:
        return groups, groups
    clusters = {}
    training = {}
    for group_id, group in groups.items():
        print('----- Group',group_id,'Data -----')
        splt = int(len(group)*split)
        clusters[group_id] = group[:splt]
        training[group_id] = group[splt:]
        if not G is None:
            for cluster in training[group_id]:
                for v in cluster:
                    for u in cluster:
                        if G.has_edge(v,u):
                            G.remove_edge(v,u)
                        if G.has_edge(u, v):
                            G.remove_edge(u,v) 
        print('Training:', len(training[group_id]))
        print('Validation:', len(clusters[group_id]))
    return training, clusters


def extract_topics(algorithm, G, training, seed_size=1, group_names=None, normalize_node_distributions=False):
    print('----- Extracting Topics with',seed_size*100,'% seeds -----')
    topics = {v: {} for v in G.nodes()}
    for group_id, group in training.items():
        if group_names is None:
            print('Group',group_id)
        else:
            print(group_names[group_id])
        group = [v for v in group if v in G.nodes()]
        prior_ranks = algorithms.convert_to_ranks(G, group[:int(len(group) * seed_size)])
        ranks = algorithm.rank(G, prior_ranks)
        ranks = algorithms.normalize(ranks)
        for v in ranks.keys():
            topics[v][group_id] = ranks[v]
            
    if normalize_node_distributions:
        for v in topics.keys():
            v_sum = sum(topics[v].values())
            if v_sum!=0:
                for group_id in topics[v].keys():
                    topics[v][group_id] /= v_sum
    return topics


def _similatity(v, u):
    sim = 0
    n1 = 0
    n2 = 0
    for topic in v.keys():
        sim += v.get(topic,0)*u.get(topic,0)
        n1 += v.get(topic,0)**2
    for topic in u.keys():
        n2 += u.get(topic,0)**2
    if n1==0 or n2==0:
        return 0
    sim /= (n1*n2)**0.5
    return sim


def evaluate_linkage(topics, validation, G):  
    print(validation)
    print('----- Evaluation: Linkage -----')
    real = list()
    predicted = list()
    count = 0
    neighbors = {}
    if validation is None:
        for v in G.nodes():
            neighbors[v] = [u for u in G.neighbors(v)]
    else:
        for cluster in validation.values():
            for v in cluster:
                if v in G.nodes():
                    if not v in neighbors:
                        neighbors[v] = list()
                    for u in cluster:
                        if u!=v and u in G.nodes():
                            neighbors[v].append(u)
    print('Found neighbors')
    nodes = list(neighbors.keys())
    if len(nodes)>2000:
        from random import shuffle
        shuffle(nodes)
        nodes = nodes[:2000]
    for node1 in neighbors.keys():
        for node2 in neighbors[node1]:
            real.append(1)
            predicted.append(_similatity(topics[node1],topics[node2]))
        for node2 in nodes:
            if node1!=node2:
                if not node2 in neighbors[node1] and not G.has_edge(node1, node2):
                    real.append(0)
                    predicted.append(_similatity(topics[node1],topics[node2]))
        count += 1
        if count%1000==0:
            print(count)
    print('Produced vectors of length',len(real))
    auc = sklearn.metrics.roc_auc_score(real, predicted)
    print('AUC  : %.2f'%auc)
    return auc


def evaluate_density(topics, validation, G, unsupervised=True):
    print('----- Evaluation: Density -----')
    density = []
    for group_id, group in validation.items():
        if unsupervised:
            ranks = {v: topics[v][group_id] for v in G.nodes()}
        else:
            ranks = {v: topics[v][group_id] for v in group}
        dense = 0
        for i, j in G.edges():
            dense += ranks.get(i,0)*ranks.get(j,0)
        if dense!=0:
            dense /= sum(ranks.values())**2 - sum(ranks[i]**2 for i in ranks.keys())
        print('Group',group_id,'density:', dense)
        density.append(dense)
    print('Avg. Density: ',sum(density)/len(density))
    return sum(density)/len(density)


def _conductance(ranks, G):
    cond = 0
    condDenom = 0
    for i, j in G.edges():
        cond += ranks.get(i,0)*(1-ranks.get(j,0)) + (1-ranks.get(i,0))*ranks.get(j,0)
        condDenom += ranks.get(i,0)*ranks.get(j,0)
    if condDenom>G.number_of_edges()/2:
        condDenom = G.number_of_edges()-condDenom
    if cond!=0:
        cond /= 2*condDenom #sum(ranks.values())**2 - sum(ranks[i]**2 for i in ranks.keys())
    return cond


def evaluate_conductance(topics, validation, G, unsupervised=True):
    print('----- Evaluation: Conductance -----')
    conductance = []
    for group_id, group in validation.items():
        if unsupervised:
            ranks = {v: topics[v][group_id] for v in G.nodes()}
        else:
            ranks = {v: topics[v][group_id] for v in group}
        #/sum(topics[v][gid] for gid in validation.keys())
        cond = _conductance(ranks, G)
        print('Group',group_id,'conductance:', cond)
        conductance.append(cond)
    print('Avg. Conductance: ',sum(conductance)/len(conductance))
    return sum(conductance)/len(conductance)


def _modularity(ranks, G):
    mod = 0
    second_term = 0
    m = 2*sum(G.degree(i) for i in G.nodes())
    for i, j in G.edges():
        mod += ranks.get(i,0)*ranks.get(j, 0)
    for i in G.nodes():
        second_term += G.degree(i)*ranks.get(i, 0)
    mod -= second_term*second_term/(2*m)
    if mod!=0:
        mod /= 2*m
    return mod


def modularity(topics, validation, G):
    modularity = []
    for group_id, group in validation.items():
        ranks = {v: topics[v][group_id] for v in group}
        mod = _modularity(ranks, G)
        print('Group',group_id,'modularity:', mod)
        modularity.append(mod)
    print('Avg. Modularity: ',sum(modularity)/len(modularity))
    return sum(modularity)/len(modularity)
    

def sweep_conductance(topics, validation, G, unsupervised=True, fast=True):
    print('----- Evaluation: Sweep Conductance -----')
    conductance = []
    for group_id, group in validation.items():
        if unsupervised:
            ranks = {v: topics[v][group_id]/G.degree(v) for v in G.nodes()}
        else:
            ranks = {v: topics[v][group_id]/G.degree(v) for v in group}
        if fast:
            max_diff = 0
            max_diff_val = 0.5
            prev_rank = -1
            for v in sorted(ranks, key=ranks.get, reverse=True):
                if prev_rank>0:
                    diff = (prev_rank - ranks[v])/prev_rank
                    if diff>max_diff:
                        max_diff = diff
                        max_diff_val = ranks[v]
                prev_rank = ranks[v]
            #print(max_diff, max_diff_val, max(ranks.values()), len([v for v in ranks.keys() if ranks[v]>max_diff_val]))
            best_cond = _conductance({v: 1 if ranks[v]>max_diff_val else 0 for v in ranks.keys()}, G)
        else:
            best_cond = float('inf')
            cond = 0
            condDenom = 0
            prev = set()
            for v in sorted(ranks, key=ranks.get, reverse=True):
                for u in G.neighbors(v):
                    if u in prev:
                        cond -= 1
                        condDenom += 1
                    else:
                        cond += 1
                prev.add(v)
                if condDenom!=0 and condDenom!=G.number_of_edges():
                    if condDenom<G.number_of_edges()/2:
                        curr_cond = cond/condDenom
                    else:
                        curr_cond = cond/(G.number_of_edges()-condDenom)
                        break
                    if curr_cond<best_cond:
                        best_cond = curr_cond
                        #print(curr_cond, len(prev))
        print('Group',group_id,'sweep conductance:', best_cond)
        conductance.append(best_cond)
    print('Avg. Sweep Conductance: ',sum(conductance)/len(conductance))
    return sum(conductance)/len(conductance)
    
    
def evaluate_clusters(algorithm, G, groups, clusters):
    for cluster in clusters:
        for v in cluster:
            for u in cluster:
                if G.has_edge(v,u):
                    G.remove_edge(v,u)
                if G.has_edge(u, v):
                    G.remove_edge(u,v)
    topics = {v: {} for v in G.nodes()}
    for groupid, group in groups.items():
        print('----- Group',groupid,'-----')
        print('Size : ', len(group))
        group = [v for v in group if v in G.nodes()]
        prior_ranks = algorithms.convert_to_ranks(G, group)
        ranks = algorithm.rank(G, prior_ranks)
        ranks = algorithms.normalize(ranks)
        for v in ranks.keys():
            topics[v][groupid] = ranks[v]
    overall_similarity = list()
    count = 0
    for artists in clusters:
        similarity = 0
        n = 0
        count += 1
        for artist1 in artists:
            #if artist1 in G.nodes():
            #    print(topics[artist1])
            for artist2 in artists:
                if artist1!=artist2 and artist1 in G.nodes() and artist2 in G.nodes():
                    sim = 0
                    n1 = 0
                    n2 = 0
                    for topic in groups.keys():
                        sim += topics[artist1].get(topic,0)*topics[artist2].get(topic,0)
                        n1 += topics[artist1].get(topic,0)**2
                        n2 += topics[artist2].get(topic,0)**2
                    if n1!=0 and n2!=0:
                        sim /= (n1*n2)**0.5#n1+n2-sim
                        n += 1
                    similarity += sim
        if n!=0:
            #print(similarity/float(n))
            overall_similarity.append(similarity/float(n))
    print('Intra-sim:', sum(overall_similarity)/(len(overall_similarity)+0.00000001))
    print('Coverage :',len(overall_similarity)/float(count))


def NDCG(topics, validation):
    import math
    print('----- Evaluation: NDCG -----')
    # find which nodes are labeled
    node2group = {}
    for group_id, group in validation.items():
        for v in group:
            if v in topics:
                node2group[v] = group_id
    NDCGs = list()
    for group_id, group in validation.items():
        for v in group:
            node2group[v] = group_id
        ranks = {v: topics[v][group_id] for v in node2group.keys()}
        DCG = 0
        IDCG = 0
        i = 1
        for v in sorted(ranks, key=ranks.get, reverse=True):
            if node2group[v]==group_id:
                DCG += 1/math.log2(i+1)
            i += 1
        i = 1
        for v in node2group.keys():
            if node2group[v]==group_id:
                IDCG += 1/math.log2(i+1)
                i += 1
        NDCGs.append(DCG/IDCG)
        print('Group',group_id,'NDCG : %.2f'%(DCG/IDCG))
    print('Average NDCG: %.2f'%(sum(NDCGs)/len(NDCGs)))
    return sum(NDCGs)/len(NDCGs)


def evaluate_prediction(topics, validation, group_names = None): 
    print('----- Evaluation: Prediction -----')
    # find which nodes are labeled
    node2group = {}
    for group_id, group in validation.items():
        for v in group:
            if v in topics:
                node2group[v] = group_id
    aucs = list()
    for group_id, group in validation.items():
        for v in group:
            node2group[v] = group_id
        group_truth = [int(node2group[v]==group_id) for v in node2group.keys()] 
        estimated = [topics[v][group_id] for v in node2group.keys()]
        fpr, tpr, _ = sklearn.metrics.roc_curve(group_truth, estimated)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucs.append(auc)
        if group_names is None:
            print('Group',group_id,': %.2f'%auc)
        else:
            print(group_names[group_id],': %.2f'%auc)
    print('Average AUC: %.2f'%(sum(aucs)/len(aucs)))
    return aucs