from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import osmnx as ox
import pybdshadow as bd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import json
import networkx as nx

# --- 1. INITIALISATION DE L'APPLICATION API ---
app = FastAPI(
    title="Shady Finder API",
    description="Une API pour trouver l'itinéraire ombragé le plus fiable.",
    version="3.0.0" # Version simplifiée et robuste
)

# --- CONFIGURATION DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- POINT D'ACCÈS POUR LE CALCUL D'ITINÉRAIRE (LOGIQUE OPTIMISÉE) ---
@app.get("/calculer-itineraire")
async def calculer_itineraire(adresse_depart: str, adresse_arrivee: str):
    print(f"--- DÉBUT DU CALCUL D'ITINÉRAIRE ROBUSTE DE '{adresse_depart}' À '{adresse_arrivee}' ---")

    try:
        # 1. GÉOCODAGE ET CORRIDOR
        print("Géocodage et création du corridor...")
        start_coords = ox.geocode(adresse_depart)
        end_coords = ox.geocode(adresse_arrivee)
        line = LineString([(start_coords[1], start_coords[0]), (end_coords[1], end_coords[0])])
        gdf_line = gpd.GeoDataFrame([{'geometry': line}], crs='EPSG:4326')
        target_crs = 'EPSG:2154'
        gdf_line_proj = gdf_line.to_crs(target_crs)
        analysis_polygon_proj = gdf_line_proj.buffer(750).unary_union
        gdf_poly_proj = gpd.GeoDataFrame(geometry=[analysis_polygon_proj], crs=target_crs)
        analysis_polygon = gdf_poly_proj.to_crs('EPSG:4326').iloc[0]['geometry']

        # 2. TÉLÉCHARGEMENT DES DONNÉES
        print("Téléchargement des données...")
        G = ox.graph_from_polygon(analysis_polygon, network_type='walk')
        tags_buildings = {"building": True}
        gdf_buildings = ox.features_from_polygon(analysis_polygon, tags=tags_buildings)
        tags_water = {"natural": "water", "waterway": "riverbank"}
        gdf_water = ox.features_from_polygon(analysis_polygon, tags=tags_water)

        # 3. ANALYSE DES OMBRES
        print("Analyse des ombres...")
        if 'bridge' in gdf_buildings.columns:
            gdf_buildings = gdf_buildings[gdf_buildings['bridge'].isnull()]
        gdf_buildings = gdf_buildings[gdf_buildings['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
        gdf_buildings = gdf_buildings[~gdf_buildings.is_empty]
        gdf_buildings = gdf_buildings[gdf_buildings.is_valid]
        gdf_buildings = gdf_buildings.explode(index_parts=True)
        gdf_buildings = gdf_buildings.reset_index(drop=True)
        gdf_buildings['building_id'] = gdf_buildings.index
        gdf_buildings['height'] = 15
        sun_time_obj = datetime.now()
        sun_time_str = sun_time_obj.strftime('%Y-%m-%d %H:%M:%S')
        shadows = bd.cal_sunshadows(gdf_buildings, sun_time_str)
        shadows.crs = gdf_buildings.crs

        # 4. MODIFICATION DU GRAPHE AVEC LES COÛTS
        print("Application des coûts au graphe...")
        G_proj = ox.project_graph(G, to_crs=target_crs)
        shadows_proj = shadows.to_crs(target_crs)
        water_proj = gdf_water.to_crs(target_crs)
        total_shadow_area = unary_union(shadows_proj.geometry)
        total_water_area = unary_union(water_proj.geometry)

        for u, v, key, data in G_proj.edges(keys=True, data=True):
            length = data.get('length', 0)
            cost = length
            shade_percent = 0
            is_bridge = data.get('bridge') in ['yes', 'aqueduct', 'viaduct']

            if 'geometry' in data:
                edge_geom = data['geometry']
                if edge_geom.intersects(total_water_area) and not is_bridge:
                    cost = length * 1000
                else:
                    intersection = edge_geom.intersection(total_shadow_area)
                    shade_length = intersection.length
                    shade_percent = (shade_length / length) if length > 0 else 0
                    cost = length * (1 + (1 - shade_percent) * 2)
            else:
                 cost = length * 1000
            
            G_proj.edges[u, v, key]['cost'] = cost
            G_proj.edges[u, v, key]['shade_percent'] = shade_percent

        # 5. CALCUL DE L'ITINÉRAIRE
        print("Calcul de l'itinéraire...")
        start_node = ox.nearest_nodes(G, X=start_coords[1], Y=start_coords[0])
        end_node = ox.nearest_nodes(G, X=end_coords[1], Y=end_coords[0])
        
        try:
            route_nodes = nx.shortest_path(G_proj, source=start_node, target=end_node, weight='cost')
        except nx.NetworkXNoPath:
            print("AVERTISSEMENT: Aucun chemin ombragé trouvé. Utilisation du chemin le plus court comme alternative.")
            try:
                route_nodes = nx.shortest_path(G_proj, source=start_node, target=end_node, weight='length')
            except nx.NetworkXNoPath:
                raise HTTPException(status_code=404, detail="Impossible de trouver un chemin. Les points sont peut-être dans des zones non connectées.")

        # 6. FORMATAGE DE LA RÉPONSE
        print("Formatage de la réponse GeoJSON...")
        features = []
        for u, v in zip(route_nodes[:-1], route_nodes[1:]):
            edge_data = min(G_proj.get_edge_data(u, v).values(), key=lambda d: d.get('cost', float('inf')))
            if 'geometry' in edge_data:
                geom = edge_data['geometry']
                shade_percent = edge_data.get('shade_percent', 0)
                length = edge_data.get('length', 0)
                features.append({'type': 'Feature', 'geometry': geom, 'properties': {'shade_percent': shade_percent, 'length': length}})
        
        if not features:
            raise HTTPException(status_code=404, detail="L'itinéraire calculé est vide.")

        gdf_route = gpd.GeoDataFrame.from_features(features, crs=target_crs)
        web_crs = 'EPSG:4326'
        gdf_route_web = gdf_route.to_crs(web_crs)

        return JSONResponse(content={
            "heure_analyse": sun_time_str,
            "itineraire_segmente": json.loads(gdf_route_web.to_json())
        })

    except Exception as e:
        print(f"ERREUR lors du calcul de l'itinéraire: {e}")
        raise HTTPException(status_code=500, detail=str(e))
