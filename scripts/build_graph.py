"""
TLAG-RAG: ILO-Graph Builder
Constructs a temporal legal knowledge graph using NetworkX based on the 'Big 8' acts.
Maps: Act -> Chapter -> Section -> Version Chains -> Succeeds/Repeals
"""
import json
import pathlib
import networkx as nx
import re
import pickle
from datetime import datetime

# Paths
BASE_DIR     = pathlib.Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data_collection" / "raw_acts"
GRAPH_DIR    = BASE_DIR / "app" / "db" / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
VERSION_CHAIN_FILE = GRAPH_DIR / "legal_graph.gpickle"

# IPC to BNS Section Mapping (IPC 1860 -> BNS 2023)
IPC_BNS_MAP = {
    "302": "103",   # Murder
    "304": "105",   # Culpable homicide
    "307": "109",   # Attempt to murder
    "376": "64",    # Rape
    "420": "318",   # Cheating
    "498A": "85",   # Cruelty
    "499": "356",   # Defamation
}

class ILOGraphBuilder:
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.processed_acts = 0

    def add_legislation_node(self, act_data):
        act_id = re.sub(r'[^\w]', '_', act_data['title'][:30]).upper()
        self.G.add_node(
            act_id,
            label="Legislation",
            title=act_data['title'],
            act_number=act_data.get('act_number', ''),
            enactment_year=self._extract_year(act_data.get('enactment_date', '')),
            ministry=act_data.get('ministry', ''),
            source="IndiaCode/GitHub"
        )
        return act_id

    def add_section_nodes(self, act_id, sections):
        for sec in sections:
            # Fallback for section_id
            sec_num = sec.get('section_id', '')
            if not sec_num:
                # Try raw_db_cols
                raw = sec.get('raw_db_cols', {})
                for k in ["Section", "section", "section_number", "no"]:
                    if k in raw and raw[k]:
                        sec_num = str(raw[k])
                        break
            
            if not sec_num:
                continue

            sec_id = f"{act_id}_SEC_{re.sub(r'[^\w]', '', str(sec_num))}"
            
            # Add base Section node
            self.G.add_node(
                sec_id,
                label="Section",
                section_number=sec_num,
                section_title=sec.get('section_title', ''),
            )
            
            # Edge: Act CONTAINS Section
            self.G.add_edge(act_id, sec_id, type="CONTAINS")
            
            # Handle versions (if any explicit versions exist)
            versions = sec.get('versions', [])
            if not versions:
                # Add default version node (current)
                ver_id = f"{sec_id}_v1"
                self.G.add_node(
                    ver_id,
                    label="LegalProvision",
                    text=sec.get('rule_text', ''),
                    penalty_fine=sec.get('penalty_fine_inr'),
                    penalty_imp=sec.get('penalty_imprisonment'),
                    start_year=sec.get('start_year'),
                    end_year=sec.get('end_year'),
                    status="active"
                )
                self.G.add_edge(sec_id, ver_id, type="HAS_VERSION")
            else:
                for idx, ver in enumerate(versions):
                    ver_id = f"{sec_id}_v{idx+1}"
                    self.G.add_node(
                        ver_id,
                        label="LegalProvision",
                        text=ver.get('version_text', ''),
                        penalty_fine=ver.get('penalty_fine'),
                        penalty_imp=ver.get('penalty_imprisonment'),
                        start_year=ver.get('effective_from'),
                        end_year=ver.get('effective_until'),
                        status=ver.get('status', 'active')
                    )
                    self.G.add_edge(sec_id, ver_id, type="HAS_VERSION")
                    
                    # Temporal Edge: v2 SUPERSEDES v1
                    if idx > 0:
                        prev_ver_id = f"{sec_id}_v{idx}"
                        self.G.add_edge(ver_id, prev_ver_id, type="SUPERSEDES", date=ver.get('effective_from'))

    def apply_external_mappings(self):
        """Link IPC/CrPC/IEA to BNS/BNSS/BSA using IndLegal mappings."""
        mapping_dir = BASE_DIR / "data_collection" / "IndLegal" / "mapping"
        if not mapping_dir.exists():
            print("Mapping directory not found.")
            return

        mappings = {
            "ipc.json":  {"old_act": "PENAL CODE",      "new_title": "BHARATIYA NYAYA SANHITA, 2023", "new_id": "BNS_2023"},
            "crpc.json": {"old_act": "CRIMINAL PROCEDURE", "new_title": "BHARATIYA NAGARIK SURAKSHA SANHITA, 2023", "new_id": "BNSS_2023"},
            "iea.json":  {"old_act": "EVIDENCE ACT",    "new_title": "BHARATIYA SAKSHYA ADHINIYAM, 2023", "new_id": "BSA_2023"},
        }

        for fname, meta in mappings.items():
            path = mapping_dir / fname
            if not path.exists(): continue
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create new Legislation node
            new_act_id = meta['new_id']
            self.G.add_node(
                new_act_id,
                label="Legislation",
                title=meta['new_title'],
                enactment_year=2023,
                status="active"
            )

            # Find old act
            old_acts = [n for n, d in self.G.nodes(data=True) if d.get('label') == 'Legislation' and meta['old_act'] in d.get('title', '').upper()]
            if not old_acts: continue
            old_act_id = old_acts[0]

            for old_sec_num, new_text in data.items():
                if "REMOVED" in new_text or not new_text.strip():
                    continue

                # Parse new section number (e.g. "302 Punishment..." -> 302)
                # Some are "2(10) " - we take the first number
                m = re.match(r'(\d+)', new_text.strip())
                if not m: continue
                new_sec_num = m.group(1)

                new_sec_id = f"{new_act_id}_SEC_{new_sec_num}"
                old_sec_id = f"{old_act_id}_SEC_{old_sec_num}"

                if not self.G.has_node(new_sec_id):
                    self.G.add_node(
                        new_sec_id,
                        label="Section",
                        section_number=new_sec_num,
                        section_title="", # Could parse from text
                    )
                    self.G.add_edge(new_act_id, new_sec_id, type="CONTAINS")
                    
                    # Create provision
                    prov_id = f"{new_sec_id}_v1"
                    self.G.add_node(
                        prov_id,
                        label="LegalProvision",
                        text=new_text,
                        start_year=2024,
                        status="active"
                    )
                    self.G.add_edge(new_sec_id, prov_id, type="HAS_VERSION")

                # Add SUCCEEDS edge
                if self.G.has_node(old_sec_id):
                    self.G.add_edge(new_sec_id, old_sec_id, type="SUCCEEDS", date="2024-07-01")

    def build(self):
        json_files = list(DATA_DIR.glob("*.json"))
        print(f"Building graph from {len(json_files)} act files...")
        
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    act_data = json.load(f)
                
                act_id = self.add_legislation_node(act_data)
                self.add_section_nodes(act_id, act_data.get('sections', []))
                self.processed_acts += 1
            except Exception as e:
                print(f"Error processing {jf.name}: {e}")

        self.apply_external_mappings()
        self.save()

    def save(self):
        with open(VERSION_CHAIN_FILE, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"Graph saved to {VERSION_CHAIN_FILE}")
        print(f"Total Nodes: {self.G.number_of_nodes()}")
        print(f"Total Edges: {self.G.number_of_edges()}")

    def _extract_year(self, s):
        m = re.search(r'\b(1[89]\d\d|20[012]\d)\b', str(s))
        return int(m.group(1)) if m else None

if __name__ == "__main__":
    builder = ILOGraphBuilder()
    builder.build()
