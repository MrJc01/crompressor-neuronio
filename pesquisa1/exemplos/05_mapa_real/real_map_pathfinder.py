import urllib.request
import urllib.parse
import json
import math
import time
import heapq

# ==============================================================================
# CROM MAP ROUTER: DADOS REAIS (OPENSTREETMAP) VS DIJKSTRA
# ==============================================================================

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def haversine(lat1, lon1, lat2, lon2):
    """Calcula a distância real na Terra em metros entre duas coordenadas."""
    R = 6371000 # Raio da Terra em metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_real_map_data():
    """Baixa as ruas REAIS da Avenida Paulista (SP) via Overpass API (OpenStreetMap)."""
    print(f"[SYS] Conectando ao Satélite (OpenStreetMap Overpass API)...")
    # Bounding box: MinLat, MinLon, MaxLat, MaxLon (Região da Av. Paulista, SP)
    bbox = "-23.565,-46.660,-23.555,-46.650"
    query = f"""
    [out:json];
    (
      way["highway"]({bbox});
    );
    (._;>;);
    out;
    """
    
    url = "http://overpass-api.de/api/interpreter"
    data = query.encode('utf-8')
    headers = {'User-Agent': 'CrompressorResearchBot/1.0 (https://github.com/crompressor)'}
    req = urllib.request.Request(url, data=data, headers=headers)
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"{Colors.FAIL}Erro ao baixar mapa real: {e}{Colors.ENDC}")
        print("Verifique sua conexão ou tente novamente (A API pode estar limitando).")
        exit(1)
        
    print(f"[SYS] Mapa Real baixado em {time.time()-start_time:.2f}s.")
    return result

def build_graph(osm_data):
    """Converte o JSON do satélite em um Grafo de navegação."""
    nodes = {}
    graph = {}
    
    # 1. Extrair nós (Coordenadas reais)
    for element in osm_data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = {'lat': element['lat'], 'lon': element['lon']}
            graph[element['id']] = []
            
    # 2. Extrair ruas (Arestas)
    street_count = 0
    for element in osm_data['elements']:
        if element['type'] == 'way' and 'nodes' in element:
            way_nodes = element['nodes']
            street_count += 1
            for i in range(len(way_nodes) - 1):
                n1 = way_nodes[i]
                n2 = way_nodes[i+1]
                
                if n1 in nodes and n2 in nodes:
                    dist = haversine(nodes[n1]['lat'], nodes[n1]['lon'], nodes[n2]['lat'], nodes[n2]['lon'])
                    # Grafo bidirecional para simplificar a navegação de ruas
                    graph[n1].append({'target': n2, 'cost': dist})
                    graph[n2].append({'target': n1, 'cost': dist})
                    
    # Remover nós isolados
    graph = {k: v for k, v in graph.items() if len(v) > 0}
    print(f"[SYS] Grafo construído: {len(graph)} cruzamentos (nós) e {street_count} ruas reais mapeadas.")
    return graph, nodes

def dijkstra_real(graph, start, target):
    """Algoritmo Clássico. Busca exaustiva pela perfeição matemática."""
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = 0
    
    start_time = time.time()
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)
        visited += 1
        
        if curr_node == target:
            break
            
        if curr_dist > distances[curr_node]:
            continue
            
        for edge in graph[curr_node]:
            dist = curr_dist + edge['cost']
            if dist < distances[edge['target']]:
                distances[edge['target']] = dist
                heapq.heappush(pq, (dist, edge['target']))
                
    elapsed = (time.time() - start_time) * 1000
    return distances[target], visited, elapsed

def crom_active_inference_real(graph, nodes, start, target):
    """
    Agente CROM (Active Inference).
    Usa a distância geográfica em linha reta como 'Prior Expectation' (Heurística).
    A Free Energy = Custo gasto + Incerteza (Distância visual até o alvo).
    Colapsa o universo buscando sempre minimizar essa energia livre.
    """
    target_lat = nodes[target]['lat']
    target_lon = nodes[target]['lon']
    
    pq = [(0, 0, start)] # (Free_Energy, Current_Cost, Node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = 0
    
    start_time = time.time()
    while pq:
        curr_free_energy, curr_cost, curr_node = heapq.heappop(pq)
        visited += 1
        
        if curr_node == target:
            break
            
        for edge in graph[curr_node]:
            new_cost = curr_cost + edge['cost']
            
            if new_cost < distances[edge['target']]:
                distances[edge['target']] = new_cost
                
                # Active Inference Core: Calcular a energia livre (Expectativa)
                lat_n = nodes[edge['target']]['lat']
                lon_n = nodes[edge['target']]['lon']
                heuristic = haversine(lat_n, lon_n, target_lat, target_lon)
                
                free_energy = new_cost + heuristic
                
                heapq.heappush(pq, (free_energy, new_cost, edge['target']))
                
    elapsed = (time.time() - start_time) * 1000
    return distances[target], visited, elapsed


def main():
    print(f"{Colors.HEADER}======================================================================================{Colors.ENDC}")
    print(f"  05. The Map Router | DADOS REAIS (Avenida Paulista, SP)")
    print(f"{Colors.HEADER}======================================================================================{Colors.ENDC}")
    
    osm_data = get_real_map_data()
    graph, nodes = build_graph(osm_data)
    
    node_ids = list(graph.keys())
    start_node = node_ids[0]
    target_node = node_ids[-1]
    
    dist_reta = haversine(nodes[start_node]['lat'], nodes[start_node]['lon'], 
                          nodes[target_node]['lat'], nodes[target_node]['lon'])
    
    print(f"\n📍 MISSÃO: Navegar do cruzamento {start_node} para o {target_node}.")
    print(f"Distância real em linha reta (pássaro): {dist_reta:.2f} metros.")
    print(f"{Colors.OKBLUE}--------------------------------------------------------------------------------------{Colors.ENDC}")
    
    # Dijkstra
    print(f"\n{Colors.WARNING}🏁 CORRIDA 1: Dijkstra (Guloso por Custo Exato){Colors.ENDC}")
    cost_dij, visited_dij, time_dij = dijkstra_real(graph, start_node, target_node)
    print(f" - Distância da Rota: {cost_dij:.2f} metros")
    print(f" - Cruzamentos Checados: {visited_dij} nós na RAM.")
    print(f" - Tempo Execução: {time_dij:.2f} ms")
    
    # CROM (Active Inference / A*)
    print(f"\n{Colors.OKGREEN}🏁 CORRIDA 2: Agente CROM (Active Inference){Colors.ENDC}")
    cost_crom, visited_crom, time_crom = crom_active_inference_real(graph, nodes, start_node, target_node)
    print(f" - Distância da Rota: {cost_crom:.2f} metros")
    print(f" - Cruzamentos Checados: {visited_crom} nós na RAM.")
    print(f" - Tempo Execução: {time_crom:.2f} ms")
    
    print(f"\n{Colors.OKCYAN}🔬 RESPOSTA AO CIENTISTA:{Colors.ENDC}")
    print(f"Você perguntou: 'Isso não daria um Prêmio Nobel se achasse o caminho mais curto e mais rápido?'")
    print(f"A matemática não permite mágica: Dijkstra SEMPRE acha a rota matematicamente perfeita (100% de otimização).")
    print(f"O problema do Dijkstra é que ele varre ruas para trás e para os lados cegamente até ter certeza absoluta.")
    print(f"A revolução do Agente CROM (Active Inference) não é quebrar a matemática para achar uma rota menor que a mínima.")
    print(f"A revolução é usar a Heurística (Free Energy) para focar apenas nas ruas que fazem sentido, achando uma rota quase perfeita ou até a mesma rota que o Dijkstra, mas visitando {visited_dij/visited_crom:.1f}x MENOS cruzamentos.")
    print(f"Na navegação de uma cidade isso poupa milissegundos. Na rede neural de um LLM com 50.000 tokens no vocabulário, isso poupa Gigabytes de VRAM e impede que o servidor trave.")

if __name__ == "__main__":
    main()
