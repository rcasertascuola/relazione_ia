import json

def load_models():
    with open('models.json', 'r') as f:
        return json.load(f)

def build_tree():
    models = load_models()
    model_dict = {m['modello']: m for m in models}

    # Define parent-child relationships and the reason for the change
    # Format: "Child_Modello": ("Parent_Modello", "Description of change")
    parents = {
        "Modello 1": (None, "Root Baseline ReLU"),
        "Modello 2": (None, "Root Baseline Sigmoid"),
        "Modello 3": ("Modello 1", "Epoche 100 \u2192 150"),
        "Modello 4": ("Modello 1", "Batch 16 \u2192 8"),
        "Modello 5": ("Modello 2", "Batch 16 \u2192 8"),
        "Modello 6": ("Modello 2", "Epoche 100 \u2192 150"),
        "Modello 7": ("Modello 1", "Epoche 100 \u2192 200"),
        "Modello 8": ("Modello 1", "Batch 16 \u2192 32"),
        "Modello 9": ("Modello 1", "Aggiunto hidden layer (20 relu)"),
        "Modello 10": ("Modello 1", "Aggiunto hidden layer (20 sigmoid)"),
        "Modello 11": ("Modello 9", "Epoche 100 \u2192 150 (Testo)"),
        "Modello 12": ("Modello 9", "Epoche 100 \u2192 150 (Gruppo C)"),
        "Modello 13": ("Modello 9", "Batch 16 \u2192 8"),
        "Modello 14": ("Modello 9", "Batch 16 \u2192 32"),
        "Modello 15": ("Modello 14", "Epoche 100 \u2192 200"),
        "Modello 16": ("Modello 9", "Retrain / Controllo Gruppo C"),
        "Modello 17": ("Modello 1", "Reti Profonde: 4 layer + Output ReLU"),
        "Modello 18": ("Modello 1", "Reti Profonde: 8 layer ReLU"),
        "Modello 19": ("Modello 1", "Reti Profonde: 8 layer alternati"),
        "Modello 20": ("Modello 1", "Controllo baseline in Gruppo D"),
        "Modello 21": ("Modello 1", "Dati Standardizzati (Std=S\u00ec, 25 relu)"),
        "Modello 22": ("Modello 2", "Std=S\u00ec, 3 layer sigmoid"),
        "Modello 23": ("Modello 22", "Epoche 100 \u2192 500"),
        "Modello 24": ("Modello 21", "4 layer misti (relu/sig/selu)"),
        "Modello 25": ("Modello 21", "Attivazione SELU, Epoche 500"),
        "Modello 26": ("Modello 25", "Output SELU!, Epoche 1000"),
        "Modello 27": ("Modello 21", "Attivazione tanh, Epoche 150"),
        "Modello 28": ("Modello 27", "Attivazione selu, Output TANH!"),
        "Modello 29": ("Modello 21", "2 layer (relu->selu), Output ELU!"),
        "Modello 30": ("Modello 21", "Attivazione softplus (5 neuroni)"),
        "Modello 31": ("Modello 21", "3 layer misti, Output GELU!"),
        "Modello 32": ("Modello 21", "Architettura Finale: 3 layer misti, Batch 64"),
        "Modello 33": ("Modello 32", "4 layer, Output TANH!, Epoche 300, Batch 8"),
        "Modello 34": ("Modello 21", "Architettura Finale: 2 layer, Batch 512")
    }

    tree_nodes = []

    def get_delta(child, parent):
        if not parent:
            return ""
        delta_acc = child['test_acc'] - parent['test_acc']
        return f"{delta_acc:+.2f}%"

    for mod_name, m_data in model_dict.items():
        parent_info = parents.get(mod_name, (None, ""))
        parent_name = parent_info[0]
        change_desc = parent_info[1]

        parent_data = model_dict.get(parent_name) if parent_name else None
        delta_str = get_delta(m_data, parent_data)

        node = {
            "name": m_data['modello'],
            "parent": parent_name,
            "change": change_desc,
            "delta_acc": delta_str,
            "data": m_data
        }
        tree_nodes.append(node)

    # Build hierarchical tree
    def build_hierarchy(nodes):
        node_map = {n['name']: {**n, 'children': []} for n in nodes}
        roots = []
        for n in nodes:
            if n['parent'] is None:
                roots.append(node_map[n['name']])
            else:
                if n['parent'] in node_map:
                    node_map[n['parent']]['children'].append(node_map[n['name']])
        return roots

    hierarchy = build_hierarchy(tree_nodes)

    # We might want a single root for the visualization to look like a single tree
    final_tree = {
        "name": "Modelli di Partenza",
        "change": "Root",
        "delta_acc": "",
        "children": hierarchy,
        "data": {
            "test_acc": "N/A",
            "test_loss": "N/A",
            "gruppo": "Root",
            "architettura": "N/A",
            "epoche": "-",
            "batch": "-",
            "std": "-"
        }
    }

    with open('tree_data.json', 'w') as f:
        json.dump(final_tree, f, indent=2)

if __name__ == '__main__':
    build_tree()
