from operator import le
import pandas as pd
import glob
import seaborn as sns
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
import copy
from networkx.algorithms.connectivity.connectivity import average_node_connectivity
from matplotlib.lines import Line2D
from itertools import chain
from scipy.stats import ranksums
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from collections import Counter


mpl.rc('font',family='Arial')

farm = "All"
results_folder = "Results"
data_list=glob.glob('Broiler/'+results_folder+'/Ecoli/Faeces/Importance - '+farm+'/*.csv')
directory = "Broiler/"+results_folder+"/Ecoli/Faeces"


chicken_sa_files=[]
for i in data_list:
    chicken_sa_files.append(i)

# Load ARG data:
args_data = pd.read_csv("CARD_ARG_drugclass.csv", header=[0])
anti_info = pd.read_csv("AMR_drug_class_to_CARD.csv", header=[0])
anti_name_info = np.array(anti_info["Abbreviation"])
    
        
class TextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,xdescent, ydescent,
                        width, height, fontsize,trans):
        h = copy.copy(orig_handle)
        h.set_position((width/2.,height/2.))
        h.set_transform(trans)
        h.set_ha("center");h.set_va("center")
        fp = orig_handle.get_font_properties().copy()
        fp.set_size(fontsize)
        # uncomment the following line, 
        # if legend symbol should have the same size as in the plot
        h.set_font_properties(fp)
        return [h]


def color_anti(anti_class):
    if anti_class == "beta lactam":
        color_name = "lightcoral"
    elif anti_class == "aminoglycoside":
        color_name = "darkorange"
    elif anti_class == "phenicol":
        color_name = "olivedrab"
    elif anti_class == "fluoroquinolone":
        color_name = "palegreen"
    elif anti_class == "glycopeptide":
        color_name = "cyan"
    elif anti_class == "tetracycline":
        color_name = "violet"
    elif anti_class == "MLSB":
        color_name = "blue"
    elif anti_class == "trimethoprim & sulfonamide" or anti_class == "trimethoprim" or anti_class == "sulfonamide":
        color_name = "saddlebrown"
    elif anti_class == "MDR":
        color_name = "indigo"
    elif anti_class == "peptide":
        color_name = "silver"
    elif anti_class == "mupirocin":
        color_name = "khaki"
    elif anti_class == "fosfomycin":
        color_name = "darkgoldenrod"
    elif anti_class == "fusidic_acid":
        color_name = "teal"
    elif anti_class == "rifamycin":
        color_name = "red"
    elif anti_class == "nucleoside":
        color_name = "tomato"
    elif anti_class == "bacteria":
        color_name = "lawngreen"
        
    return color_name

def network_chicken(g,chicken_sa_files):
    # chicken
    features_chicken = []

    for data_file in chicken_sa_files:
        anti=data_file.split('\\')[-1].split('_')[-1].split('.')[0] 
        
        if anti in ["AMC", "CAZ", "CAZ-C", "CTX-C", "FEP", "GEN"]:#"SXT"
            continue
        
        data=pd.read_csv(data_file, index_col=[0], header=[0])
        features=data.index  
        
        #feat_imp = np.array(data["Importance"])
        #idx = np.where(feat_imp > np.mean(feat_imp))[0]
        #features = features[idx]
        if len(features) > 0:
            features_chicken.append(list(features))
            id_anti = np.where(anti_name_info == anti)[0]
            anti_class = anti_info.loc[id_anti[0],"Antibiotic Class to CARD"]
            
            g.add_node(anti, color_=color_anti(anti_class))
            for f in features:
                id_arg = np.where(args_data["Source"] == f)[0]
                if len(id_arg) == 0:
                    g.add_node(f, color_=color_anti("bacteria"))
                else:
                    f_class = args_data.iloc[id_arg[0],1]
                    g.add_node(f, color_=color_anti(f_class))
                
                g.add_edge(anti,f, color="black")
    
    return g, features_chicken

def rename(g):
    node_names = g.nodes

    relabel_dict = {}
    for name in node_names:
        if name == "catII_from_Escherichia_coli_K-12":
            relabel_dict[name] = "catII"
        elif name == "Lactobacillus_reuteri_cat-TC":
            relabel_dict[name] = "cat-TC"
        elif name == "Bifidobacterium_adolescentis_rpoB_conferring_resistance_to_rifampicin":
            relabel_dict[name] = "rpoB"
        elif name == "Bifidobacteria_intrinsic_ileS_conferring_resistance_to_mupirocin":
            relabel_dict[name] = "ileS"
        elif name == "Campylobacter_coli_chloramphenicol_acetyltransferase":
            relabel_dict[name] = "C.coli_acetyltransferase"
        elif name == "Escherichia_coli_ampC1_beta-lactamase":
            relabel_dict[name] = "ampC1"
        elif name == "Vibrio_anguillarum_chloramphenicol_acetyltransferase":
            relabel_dict[name] = "cat"
        elif name == "Klebsiella_pneumoniae_acrA":
            relabel_dict[name] = "acrA"
        elif name == "Klebsiella_pneumoniae_KpnE":
            relabel_dict[name] = "kpnE"
        elif name == "Klebsiella_pneumoniae_KpnF":
            relabel_dict[name] = "kpnF"
        elif name == "Klebsiella_pneumoniae_KpnG":
            relabel_dict[name] = "kpnG"
        elif name == "Klebsiella_pneumoniae_KpnH":
            relabel_dict[name] = "kpnH"
        elif name == "Klebsiella_pneumoniae_OmpK37":
            relabel_dict[name] = "OmpK37"
        elif name == "Escherichia_coli_ampC":
            relabel_dict[name] = "ampC"
        elif name == "Escherichia_coli_emrE":
            relabel_dict[name] = "emrE"
        elif name == "E.coli_ampC1":
            relabel_dict[name] = "ampC1"
        elif name == "Enterococcus_faecalis_chloramphenicol_acetyltransferase":
            relabel_dict[name] = "E.faecalis_acetyltransferase"
        elif name == "dfrA6_from_Proteus_mirabilis":
            relabel_dict[name] = "dfrA6"
        elif name == "vga(E)_Staphylococcus_cohnii":
            relabel_dict[name] = "vga(E)"
        else:
            relabel_dict[name] = name

    g = nx.relabel_nodes(g, relabel_dict)
    return g

def relabel_args(name):
    if name == "catII_from_Escherichia_coli_K-12":
        relabel_name = "catII"
    elif name == "Lactobacillus_reuteri_cat-TC":
        relabel_name = "cat-TC"
    elif name == "Bifidobacterium_adolescentis_rpoB_conferring_resistance_to_rifampicin":
        relabel_name = "rpoB"
    elif name == "Bifidobacteria_intrinsic_ileS_conferring_resistance_to_mupirocin":
        relabel_name = "ileS"
    elif name == "Campylobacter_coli_chloramphenicol_acetyltransferase":
        relabel_name = "C.coli_acetyltransferase"
    elif name == "Escherichia_coli_ampC1_beta-lactamase":
        relabel_name = "ampC1"
    elif name == "Vibrio_anguillarum_chloramphenicol_acetyltransferase":
        relabel_name = "cat"
    elif name == "Klebsiella_pneumoniae_acrA":
        relabel_name = "acrA"
    elif name == "Klebsiella_pneumoniae_KpnE":
        relabel_name = "kpnE"
    elif name == "Klebsiella_pneumoniae_KpnF":
        relabel_name = "kpnF"
    elif name == "Klebsiella_pneumoniae_KpnG":
        relabel_name = "kpnG"
    elif name == "Klebsiella_pneumoniae_KpnH":
        relabel_name = "kpnH"
    elif name == "Klebsiella_pneumoniae_OmpK37":
        relabel_name = "OmpK37"
    elif name == "Escherichia_coli_ampC":
        relabel_name = "ampC"
    elif name == "Escherichia_coli_emrE":
        relabel_name = "emrE"
    elif name == "E.coli_ampC1":
        relabel_name = "ampC1"
    elif name == "Enterococcus_faecalis_chloramphenicol_acetyltransferase":
        relabel_name = "E.faecalis_acetyltransferase"
    elif name == "dfrA6_from_Proteus_mirabilis":
        relabel_name = "dfrA6"
    elif name == "vga(E)_Staphylococcus_cohnii":
        relabel_name = "vga(E)"
    else:
        relabel_name = name
        
    return relabel_name

def draw_network(g, axn, labels, node_max_size = 200, fontsize=6, connect=False):
    if node_max_size == 200:
        node_min_size = 50
    else:
        node_min_size = 90
        
    node_degree_dict=nx.degree(g)
    nodes_sel = [x for x in g.nodes() if node_degree_dict[x]>3]
    
    df_node = pd.DataFrame(columns=["Count","Antibiotics"])
    for n in nodes_sel:
        if n in ["AMC", "AMI", "AZM", "CAZ", "CAZ-C", "CFX", "CHL", "CTX", "CTX-C",
                 "FEP", "GEN", "KAN", "MIN", "NAL", "STR", "Sul", "SXT"]:
            continue
        
        neigh = g.neighbors(n)
        neigh_list = []
        for nn in neigh:
            neigh_list.append(nn)
        
        df_node.loc[n,"Count"] = len(neigh_list)    
        df_node.loc[n,"Antibiotics"] = ', '.join(neigh_list)
    
    df_node.to_csv(directory+'/Node_analysis_'+farm+'.csv', index_label=False)
        
    #Check distances between nodes for number of iterations
    df = pd.DataFrame(index=g.nodes(), columns=g.nodes())
    for row, data in nx.shortest_path_length(g):
        for col, dist in data.items():
            df.loc[row,col] = dist

    df = df.fillna(df.max().max())

    pos = nx.kamada_kawai_layout(g, dist=df.to_dict())
    
    color_map=nx.get_node_attributes(g, 'color_')
    color_s=[color_map.get(x) for x in g.nodes()]
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    node_size = []
    edge_colors = []
    linewidth_val = []
    alpha_val = []
    node_shape_list = []
    nodes_name = []
    for i, n in enumerate(g.nodes):
        color_n = color_map.get(n)
        nodes_name.append(n)
        if n in ["AMC", "AMI", "AZM", "CAZ", "CAZ-C", "CFX", "CHL", "CTX", "CTX-C",
                 "FEP", "GEN", "KAN", "MIN", "NAL", "STR", "Sul", "SXT"]:
            edge_colors.append(color_s[i])
            color_s[i] = "white"
            node_size.append(node_max_size)
            linewidth_val.append(3)
            alpha_val.append(1)
            node_shape_list.append("o") 
        else:
            edge_colors.append(color_n)
            node_size.append(node_min_size)
            
            if n in nodes_sel:
                alpha_val.append(1)  
                linewidth_val.append(1) 
                node_shape_list.append("o") 
            else:
                alpha_val.append(1)
                linewidth_val.append(1)
                node_shape_list.append("o") #"s"                
            
    d = dict(g.degree)
    
    print(color_s)
    
    node_shape_list = np.array(node_shape_list)
    edge_colors = np.array(edge_colors)
    node_size = np.array(node_size)
    alpha_val = np.array(alpha_val)
    linewidth_val = np.array(linewidth_val)
    color_s = np.array(color_s)
    nodes = np.array(nodes_name)
    
    id_o = np.where(node_shape_list == "o")[0]
    options_o = {"edgecolors": list(edge_colors[id_o]), "node_size": list(node_size[id_o]), "alpha": list(alpha_val[id_o]), "linewidths":list(linewidth_val[id_o])} #[v * 1000 for v in d.values()]
    id_s = np.where(node_shape_list == "s")[0]
    options_s = {"edgecolors": list(edge_colors[id_s]), "node_size": list(node_size[id_s]), "alpha": list(alpha_val[id_s]), "linewidths":list(linewidth_val[id_s])} #[v * 1000 for v in d.values()]
    
    nx.draw_networkx_nodes(g, pos, nodelist= nodes[id_o], node_shape = "o", node_color=list(color_s[id_o]), **options_o, ax=axn)
    nx.draw_networkx_nodes(g, pos, nodelist= nodes[id_s], node_shape = "*", node_color=list(color_s[id_s]), **options_s, ax=axn)
    
    nx.draw_networkx_edges(g, pos, alpha=0.3, edge_color = colors, width=1.0, ax=axn)
    nx.draw_networkx_labels(g, pos, labels, font_size=fontsize, font_color="k", ax=axn)
    axn.margins(x=0.15)

    if connect == True:
        connectivity = np.round(average_node_connectivity(g),3)
        axn.set_title("Connectivity = {}".format(connectivity), fontsize = 30)

# Plot just chicken
h=nx.Graph()
h, _ = network_chicken(h, chicken_sa_files)
h = rename(h)

color_map=nx.get_node_attributes(h, 'color_')
legend_node_names = []
legend_node_number = []
labels = {}
k = 1
for n in h.nodes:
    color_n = color_map.get(n)
    if n in ["AMC", "AMI", "AZM", "CAZ", "CAZ-C", "CFX", "CHL", "CTX", "CTX-C",
            "FEP", "GEN", "KAN", "MIN", "NAL", "STR", "Sul", "SXT"]:
        labels[n] = n
    else:    
        legend_node_number.append(str(k))
        legend_node_names.append(n)
        labels[n] = ""#str(k)
        k+=1        

legend_node_names = np.array(legend_node_names)
legend_node_number = np.array(legend_node_number)

# Networkx        
fig = plt.figure(figsize=(10, 8))

ax0 = fig.add_subplot(111)

plt.rcParams.update({'font.size': 20})
draw_network(h,ax0,labels)

color_map=nx.get_node_attributes(h, 'color_')

c_map = []
for key in color_map.keys():
    c_map.append(color_map[key])

c_map = Counter(c_map)
print(c_map)

legend_elements = []

for i in c_map.keys():
    if i == 'lightcoral':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='lightcoral', label='beta lactam',
                          color = 'w', markerfacecolor = 'lightcoral', markersize=10, alpha=1))
    elif i == 'darkorange':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='darkorange', label='aminoglycoside',
                          color = 'w', markerfacecolor = 'darkorange', markersize=10, alpha=1))
    elif i == 'olivedrab':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='olivedrab', label='amphenicol',
                          color = 'w', markerfacecolor = 'olivedrab', markersize=10, alpha=1))
    elif i == 'palegreen':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='palegreen', label='fluoroquinolone',
                          color = 'w', markerfacecolor = 'palegreen', markersize=10, alpha=1))
    elif i == 'cyan':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='cyan', label='glycopeptide',
                          color = 'w', markerfacecolor = 'cyan', markersize=10, alpha=1))
    elif i == 'violet':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='violet', label='tetracycline',
                          color = 'w', markerfacecolor = 'violet', markersize=10, alpha=1))
    elif i == 'blue':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='blue', label='MLSB',
                          color = 'w', markerfacecolor = 'blue', markersize=10, alpha=1))
    elif i == 'saddlebrown':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='saddlebrown', label='trimethoprim & sulfonamide',
                          color = 'w', markerfacecolor = 'saddlebrown', markersize=10, alpha=1))
    elif i == 'indigo':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='indigo', label='MDR',
                          color = 'w', markerfacecolor = 'indigo', markersize=10, alpha=1))
    elif i == 'silver':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='silver', label='peptide',
                          color = 'w', markerfacecolor = 'silver', markersize=10, alpha=1))
    elif i == 'khaki':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='khaki', label='mupirocin',
                          color = 'w', markerfacecolor = 'khaki', markersize=10, alpha=1))
    elif i == 'teal':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='teal', label='fusidic_acid',
                          color = 'w', markerfacecolor = 'teal', markersize=10, alpha=1))
    elif i == 'red':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='red', label='rifamycin',
                          color = 'w', markerfacecolor = 'red', markersize=10, alpha=1))
    elif i == 'darkgoldenrod':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='darkgoldenrod', label='fosfomycin',
                          color = 'w', markerfacecolor = 'darkgoldenrod', markersize=10, alpha=1))
    elif i == 'lawngreen':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='lawngreen', label='bacteria',
                          color = 'w', markerfacecolor = 'lawngreen', markersize=10, alpha=1))
    elif i == 'tomato':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='tomato', label='nucleoside',
                          color = 'w', markerfacecolor = 'tomato', markersize=10, alpha=1))


ax0.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01),
          fancybox=True, shadow=True, ncol=4, fontsize = 12,
          title="Antibiotic Class", title_fontsize=15)

plt.tight_layout()

"""
handles = []
for i in range(len(legend_node_names)):
    id_arg = np.where(args_data["Source"] == legend_node_names[i])[0]
    f_class = args_data.iloc[id_arg[0],1]
    handles.append(plt.text(1.2,0.5,str(i+1), transform=plt.gcf().transFigure,
         bbox={"boxstyle" : "circle", "color":color_anti(f_class), "alpha":0.5}))

handlermap = {type(handles[0]) : TextHandler()}
leg = plt.figlegend(handles, legend_node_names, handler_map=handlermap, bbox_to_anchor=[1.48, 0.5], loc='center right', ncol=2,
                    fancybox=True, shadow=True, fontsize = 13, title="Genes Legend", title_fontsize=16)
"""
#plt.savefig('Figure6b_C_H.png')

plt.savefig(directory+'/Figure_network_grant_'+farm+'.svg', dpi=300, bbox_inches='tight')
