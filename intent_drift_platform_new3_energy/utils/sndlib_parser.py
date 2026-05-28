# utils/sndlib_parser.py

import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from glob import glob


class SNDlibParser:
    """SNDlib Network Topology and Traffic Matrix Analyzer"""
    
    # SNDlib XML Namespaces
    NS = {'sndlib': 'http://sndlib.zib.de/network'}
    
    def __init__(self, sndlib_dir='data/real_traces/sndlib'):
        self.sndlib_dir = sndlib_dir
    
    def list_available_networks(self):
        """List available networks"""
        if not os.path.exists(self.sndlib_dir):
            print(f"SNDlib directory not found: {self.sndlib_dir}")
            return []
        
        networks = []
        for item in os.listdir(self.sndlib_dir):
            item_path = os.path.join(self.sndlib_dir, item)
            if os.path.isdir(item_path):
                networks.append(item)
        
        return networks
    
    def parse_network(self, network_name):
        """Parse SNDlib network"""
        network_dir = self._find_network_dir(network_name)
        if not network_dir:
            print(f"Network directory not found for: {network_name}")
            return None
        
        print(f"Parsing network from: {network_dir}")
        
        result = {
            'name': network_name,
            'nodes': [],
            'links': [],
            'traffic_matrices': [],
            'matrix_timestamps': []
        }
        
        # Parse the topology file (typically an XML file that does not contain “demand”)
        topo_file = self._find_topology_file(network_dir)
        if topo_file:
            print(f"  Found topology file: {os.path.basename(topo_file)}")
            self._parse_xml_file(topo_file, result)
        
        # Parse traffic matrix files
        demand_files = self._find_demand_files(network_dir)
        if demand_files:
            print(f"  Found {len(demand_files)} traffic matrix files")
            for df in demand_files[:50]:
                matrix = self._parse_demand_matrix(df, result['nodes'])
                if matrix is not None and np.count_nonzero(matrix) > 0:
                    result['traffic_matrices'].append(matrix)
                    result['matrix_timestamps'].append(self._extract_timestamp(df))
        
        # If the topology file itself contains a “demand” element, parse it as well
        if topo_file and not result['traffic_matrices']:
            matrix = self._parse_demand_matrix(topo_file, result['nodes'])
            if matrix is not None and np.count_nonzero(matrix) > 0:
                result['traffic_matrices'].append(matrix)
                result['matrix_timestamps'].append('topo_file')
        
        print(f"  Result: {len(result['nodes'])} nodes, {len(result['links'])} links, "
              f"{len(result['traffic_matrices'])} traffic matrices")
        
        return result
    
    def _find_network_dir(self, network_name):
        """Browse the network directory"""
        direct_path = os.path.join(self.sndlib_dir, network_name)
        if os.path.isdir(direct_path):
            return direct_path
        
        for item in os.listdir(self.sndlib_dir):
            if network_name.lower() in item.lower():
                item_path = os.path.join(self.sndlib_dir, item)
                if os.path.isdir(item_path):
                    return item_path
        
        return None
    
    def _find_topology_file(self, network_dir):
        """Find the topology file (excluding the XML file containing the demandMatrix)"""
        for f in os.listdir(network_dir):
            if f.endswith('.xml') and 'demandMatrix' not in f:
                return os.path.join(network_dir, f)
        return None
    
    def _find_demand_files(self, network_dir):
        """Find all traffic matrix files"""
        files = glob(os.path.join(network_dir, '*demandMatrix*.xml'))
        # Filter out small files (possibly empty)
        valid_files = [f for f in files if os.path.getsize(f) > 10000]
        return sorted(valid_files)
    
    def _extract_timestamp(self, filepath):
        """Extract timestamp from filename"""
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{8})-(\d{4})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return filename
    
    def _parse_xml_file(self, xml_file, result):
        """Parse XML file, extract nodes and links"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Iterate through all elements and remove the namespace before matching
            for elem in root.iter():
                # Remove namespace prefix
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                # Parsing Node
                if tag == 'node' and elem.get('id'):
                    node_id = elem.get('id')
                    x, y = 0.0, 0.0
                    
                    for child in elem:
                        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if child_tag == 'coordinates':
                            for coord in child:
                                coord_tag = coord.tag.split('}')[-1] if '}' in coord.tag else coord.tag
                                if coord_tag == 'x' and coord.text:
                                    try:
                                        x = float(coord.text)
                                    except:
                                        pass
                                elif coord_tag == 'y' and coord.text:
                                    try:
                                        y = float(coord.text)
                                    except:
                                        pass
                    
                    if node_id not in [n['id'] for n in result['nodes']]:
                        result['nodes'].append({'id': node_id, 'x': x, 'y': y})
                
                # Parse the link
                elif tag == 'link' and elem.get('id'):
                    source, target, capacity = None, None, 100.0
                    
                    for child in elem.iter():
                        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        
                        if child_tag == 'source' and child.text:
                            source = child.text.strip()
                        elif child_tag == 'target' and child.text:
                            target = child.text.strip()
                        elif child_tag == 'capacity' and child.text:
                            try:
                                capacity = float(child.text)
                            except:
                                pass
                    
                    if source and target:
                        # Avoid adding the same item twice
                        link_exists = any(
                            l['source'] == source and l['target'] == target 
                            for l in result['links']
                        )
                        if not link_exists:
                            result['links'].append({
                                'source': source,
                                'target': target,
                                'capacity': capacity
                            })
            
            print(f"    Parsed {len(result['nodes'])} nodes, {len(result['links'])} links")
            
        except Exception as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_demand_matrix(self, xml_file, nodes):
        """Analyzing the Traffic Matrix"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # If `nodes` is empty, first extract from the file
            if not nodes:
                for elem in root.iter():
                    tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    if tag == 'node' and elem.get('id'):
                        node_id = elem.get('id')
                        if node_id not in [n['id'] for n in nodes]:
                            nodes.append({'id': node_id, 'x': 0, 'y': 0})
            
            node_ids = [n['id'] for n in nodes]
            n = len(node_ids)
            
            if n == 0:
                return None
            
            node_to_idx = {node: i for i, node in enumerate(node_ids)}
            matrix = np.zeros((n, n))
            
            # Analysis of the `demand` element
            demand_count = 0
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                
                if tag == 'demand' and elem.get('id'):
                    source, target, value = None, None, 0.0
                    
                    for child in elem:
                        child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        
                        if child_tag == 'source' and child.text:
                            source = child.text.strip()
                        elif child_tag == 'target' and child.text:
                            target = child.text.strip()
                        elif child_tag == 'demandValue' and child.text:
                            try:
                                value = float(child.text.strip())
                            except:
                                pass
                    
                    if source in node_to_idx and target in node_to_idx:
                        matrix[node_to_idx[source], node_to_idx[target]] = value
                        demand_count += 1
            
            if demand_count > 0:
                print(f"    Parsed {demand_count} demands from {os.path.basename(xml_file)}")
            
            return matrix
            
        except Exception as e:
            print(f"Error parsing demand matrix {xml_file}: {e}")
            return None
    
    def get_random_traffic_matrix(self, network_data, scale_factor=0.001):
        """Generate a scaled traffic matrix at random"""
        matrices = network_data.get('traffic_matrices', [])
        if matrices:
            idx = np.random.randint(len(matrices))
            return matrices[idx] * scale_factor
        return None
    
    def convert_to_mininet_topology(self, network_data, scale_factor=0.001):
        """Convert to Mininet topology configuration"""
        if not network_data or not network_data['nodes']:
            print("Warning: No nodes in network data")
            return None, None
        
        nodes = network_data['nodes']
        links = network_data['links']
        
        print(f"  Converting to Mininet: {len(nodes)} nodes, {len(links)} links")
        
        topo_config = {
            'name': network_data['name'].upper(),
            'description': f"SNDlib {network_data['name']} topology",
            'nodes': [],
            'links': [],
            'hosts': []
        }
        
        node_map = {}
        for i, node in enumerate(nodes):
            switch_id = f"s{i+1}"
            node_map[node['id']] = switch_id
            topo_config['nodes'].append({
                'id': switch_id,
                'name': node['id'],
                'type': 'switch'
            })
            topo_config['hosts'].append({
                'id': f"h{i+1}",
                'connected_to': switch_id
            })
        
        for link in links:
            src_id = node_map.get(link['source'])
            dst_id = node_map.get(link['target'])
            
            if src_id and dst_id:
                # Calculate latency based on node coordinates
                src_node = next((n for n in nodes if n['id'] == link['source']), None)
                dst_node = next((n for n in nodes if n['id'] == link['target']), None)
                
                delay = 2  # Default: 2 ms
                if src_node and dst_node:
                    dist = np.sqrt((src_node['x'] - dst_node['x'])**2 + 
                                   (src_node['y'] - dst_node['y'])**2)
                    delay = max(1, int(dist * 0.5))
                
                bandwidth = int(link['capacity'] * scale_factor)
                bandwidth = max(10, min(1000, bandwidth))
                
                topo_config['links'].append({
                    'src': src_id,
                    'dst': dst_id,
                    'bandwidth': bandwidth,
                    'delay': delay
                })
        
        return topo_config, node_map


# Test Function
if __name__ == '__main__':
    import sys
    
    parser = SNDlibParser('data/real_traces/sndlib')
    
    print("Available networks:", parser.list_available_networks())
    
    networks = parser.list_available_networks()
    if networks:
        result = parser.parse_network(networks[0])
        if result:
            print(f"\nParsed {result['name']}:")
            print(f"  Nodes: {len(result['nodes'])}")
            print(f"  Links: {len(result['links'])}")
            print(f"  Matrices: {len(result['traffic_matrices'])}")
