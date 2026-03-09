"""
TLAG-RAG: Graph Manager
Provides an interface to query the ILO-Graph (NetworkX).
Handles point-in-time retrieval, multi-hop traversal, and IPC->BNS mapping.
"""
import pickle
import pathlib
import networkx as nx
import re
from loguru import logger


class GraphManager:
    def __init__(self, graph_path=None):
        if graph_path is None:
            graph_path = pathlib.Path(__file__).parent / "graph" / "legal_graph.gpickle"

        self.graph_path = pathlib.Path(graph_path)
        self.G = self._load_graph()
        self.available = self.G.number_of_nodes() > 0

    def _load_graph(self):
        if not self.graph_path.exists():
            logger.warning(f"Graph file not found: {self.graph_path} — graph features disabled")
            return nx.MultiDiGraph()
        try:
            with open(self.graph_path, 'rb') as f:
                G = pickle.load(f)
            logger.info(f"ILO-Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        except Exception as e:
            logger.error(f"Error loading graph from {self.graph_path}: {e}")
            return nx.MultiDiGraph()

    def get_legislation_by_title_fragment(self, fragment):
        """Search for legislation nodes by title."""
        results = []
        for n, d in self.G.nodes(data=True):
            if d.get('label') == 'Legislation' and fragment.upper() in d.get('title', '').upper():
                results.append({"id": n, **d})
        return results

    def get_section_node(self, act_id, section_number):
        """Find the Section node for a given act and section number."""
        # Normalize section number (remove spaces, dots)
        clean_sec = re.sub(r'[^\w]', '', str(section_number))
        sec_id_pattern = f"{act_id}_SEC_{clean_sec}"
        
        if self.G.has_node(sec_id_pattern):
            return sec_id_pattern
        
        # Fallback: scan edges from act_id
        for _, v, d in self.G.out_edges(act_id, data=True):
            if d.get('type') == 'CONTAINS':
                node_data = self.G.nodes[v]
                if str(node_data.get('section_number')) == str(section_number):
                    return v
        return None

    def get_version_at_year(self, section_node_id, year):
        """Retrieve the LegalProvision (version) valid at a specific year."""
        versions = []
        for _, v, d in self.G.out_edges(section_node_id, data=True):
            if d.get('type') == 'HAS_VERSION':
                versions.append((v, self.G.nodes[v]))
        
        # Sort or filter by year
        for ver_id, data in versions:
            start = data.get('start_year')
            end = data.get('end_year')
            
            # Simple temporal matching
            if start is not None and year < start:
                continue
            if end is not None and year > end:
                continue
            
            # If we reach here, it's a candidate
            return {"id": ver_id, **data}
        
        # If no match, return the most recent one if no end_year specified
        if versions:
            return {"id": versions[-1][0], **versions[-1][1]}
        
        return None

    def get_bns_equivalent(self, ipc_section_number):
        """Return BNS section number + text if it succeeds an IPC section."""
        # Find IPC Act node
        ipc_acts = self.get_legislation_by_title_fragment("PENAL CODE")
        if not ipc_acts:
            return None
        
        ipc_act_id = ipc_acts[0]['id']
        ipc_sec_node = self.get_section_node(ipc_act_id, ipc_section_number)
        
        if not ipc_sec_node:
            return None
            
        # Check for SUCCEEDS edge in-neighbors (BNS <- SUCCEEDS - IPC)
        # Note: In build_graph, we added G.add_edge(bns_sec_id, ipc_sec_id, type="SUCCEEDS")
        # So we look for in-edges to ipc_sec_node
        for u, v, d in self.G.in_edges(ipc_sec_node, data=True):
            if d.get('type') == 'SUCCEEDS':
                node_data = self.G.nodes[u]
                return {
                    "bns_section": node_data.get('section_number'),
                    "bns_title": node_data.get('section_title'),
                    "effective_date": d.get('date')
                }
        return None

    def get_history_chain(self, section_node_id):
        """Get the full chain of versions with their text and dates."""
        versions = []
        for _, v, d in self.G.out_edges(section_node_id, data=True):
            if d.get('type') == 'HAS_VERSION':
                versions.append({"id": v, **self.G.nodes[v]})

        versions.sort(key=lambda x: (x.get('start_year') or 0))
        return versions

    def find_successor_act(self, law_name: str):
        """Check if a law has been succeeded by a newer act (e.g. IPC -> BNS)."""
        if not self.available:
            return None
        fragment = law_name.upper()
        # Simplify for matching
        for keyword, new_info in {
            "PENAL CODE": {"new_act": "Bharatiya Nyaya Sanhita, 2023", "year": 2024},
            "CRIMINAL PROCEDURE": {"new_act": "Bharatiya Nagarik Suraksha Sanhita, 2023", "year": 2024},
            "EVIDENCE ACT": {"new_act": "Bharatiya Sakshya Adhiniyam, 2023", "year": 2024},
        }.items():
            if keyword in fragment:
                return new_info
        return None

    def get_graph_stats(self):
        """Return basic graph statistics."""
        if not self.available:
            return {"available": False, "nodes": 0, "edges": 0}
        return {
            "available": True,
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "legislation_count": sum(
                1 for _, d in self.G.nodes(data=True) if d.get("label") == "Legislation"
            ),
        }

if __name__ == "__main__":
    # Test
    gm = GraphManager()
    print(f"Graph loaded. Testing with IPC Section 302...")
    
    # 1. Find IPC
    ipc = gm.get_legislation_by_title_fragment("PENAL CODE")
    if ipc:
        ipc_id = ipc[0]['id']
        print(f"Found IPC: {ipc_id}")
        
        # 2. Find Section 302
        sec_302 = gm.get_section_node(ipc_id, "302")
        print(f"Found Section 302 node: {sec_302}")
        
        if sec_302:
            # 3. Get version
            ver = gm.get_version_at_year(sec_302, 2020)
            print(f"Version at 2020: {ver.get('text')[:100]}...")
            
            # 4. Check BNS
            bns = gm.get_bns_equivalent("302")
            print(f"BNS Equivalent: {bns}")
