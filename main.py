from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import osmnx as ox
import pybdshadow as bd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import json
import networkx as nx
import os
import pickle

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
        analysis_polygon_proj = gdf_line_proj.buffer(500).unary_union
        gdf_poly_proj = gpd.GeoDataFrame(geometry=[analysis_polygon_proj], crs=target_crs)
        analysis_polygon = gdf_poly_proj.to_crs('EPSG:4326').iloc[0]['geometry']

        # 2. GESTION DES DONNÉES (CACHE)
        bounds = analysis_polygon.bounds
        cache_key = "_".join([f"{b:.5f}" for b in bounds])
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)

        graph_path = os.path.join(cache_dir, f"graph_{cache_key}.pkl")
        buildings_path = os.path.join(cache_dir, f"buildings_{cache_key}.parquet")
        water_path = os.path.join(cache_dir, f"water_{cache_key}.parquet")

        if os.path.exists(graph_path) and os.path.exists(buildings_path) and os.path.exists(water_path):
            print("Chargement des données depuis le cache...")
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            gdf_buildings = gpd.read_parquet(buildings_path)
            gdf_water = gpd.read_parquet(water_path)
        else:
            print("Téléchargement des données (cache non trouvé)...")
            G = ox.graph_from_polygon(analysis_polygon, network_type='walk')
            tags_buildings = {"building": True}
            gdf_buildings = ox.features_from_polygon(analysis_polygon, tags_buildings)
            tags_water = {"natural": "water", "waterway": "riverbank"}
            gdf_water = ox.features_from_polygon(analysis_polygon, tags_water)

            print("Sauvegarde des données dans le cache...")
            with open(graph_path, 'wb') as f:
                pickle.dump(G, f)
            gdf_buildings.to_parquet(buildings_path)
            gdf_water.to_parquet(water_path)

        # 3. NETTOYAGE DES DONNÉES DE BÂTIMENTS
        print("Nettoyage des données de bâtiments...")
        if 'bridge' in gdf_buildings.columns:
            gdf_buildings = gdf_buildings[gdf_buildings['bridge'].isnull()]
        gdf_buildings = gdf_buildings[gdf_buildings['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
        gdf_buildings = gdf_buildings[~gdf_buildings.is_empty & gdf_buildings.is_valid]
        if not gdf_buildings.empty:
            gdf_buildings = gdf_buildings.explode(index_parts=True).reset_index(drop=True)
        gdf_buildings['height'] = 15

        # 4. ANALYSE DES OMBRES (AVEC CACHE)
        print("Analyse des ombres...")
        sun_time_obj = datetime.now()
        sun_time_str = sun_time_obj.strftime('%Y-%m-%d %H:%M:%S')
        sun_time_hour_str = sun_time_obj.strftime('%Y-%m-%d-%H')
        shadow_cache_dir = "shadow_cache"
        os.makedirs(shadow_cache_dir, exist_ok=True)
        shadow_cache_path = os.path.join(shadow_cache_dir, f"shadows_{cache_key}_{sun_time_hour_str}.parquet")

        if os.path.exists(shadow_cache_path):
            print("Chargement des ombres depuis le cache...")
            shadows = gpd.read_parquet(shadow_cache_path)
            shadows.crs = gdf_buildings.crs # Assurez-vous que le CRS est défini
        else:
            print("Calcul des ombres (cache non trouvé)...")
            shadows = bd.cal_sunshadows(gdf_buildings, sun_time_str)
            shadows.crs = gdf_buildings.crs
            print("Sauvegarde des ombres dans le cache...")
            shadows.to_parquet(shadow_cache_path)

        # 5. MODIFICATION DU GRAPHE AVEC LES COÛTS (OPTIMISÉ)
        print("Application des coûts au graphe (optimisé)...")
        G_proj = ox.project_graph(G, to_crs=target_crs)
        shadows_proj = shadows.to_crs(target_crs)
        water_proj = gdf_water.to_crs(target_crs)

        # Création des index spatiaux pour une recherche rapide
        shadows_sindex = shadows_proj.sindex
        water_sindex = water_proj.sindex

        for u, v, key, data in G_proj.edges(keys=True, data=True):
            length = data.get('length', 0)
            cost = length
            shade_percent = 0
            is_bridge = data.get('bridge') in ['yes', 'aqueduct', 'viaduct']

            if 'geometry' in data:
                edge_geom = data['geometry']

                # Vérification rapide de l'intersection avec l'eau
                possible_water_matches_idx = list(water_sindex.intersection(edge_geom.bounds))
                if not is_bridge and possible_water_matches_idx:
                    if any(water_proj.iloc[possible_water_matches_idx].intersects(edge_geom)):
                        cost = length * 1000  # Pénalité forte pour l'eau
                    else:
                        # Calcul de l'ombre uniquement si pas dans l'eau
                        possible_shadow_matches_idx = list(shadows_sindex.intersection(edge_geom.bounds))
                        if possible_shadow_matches_idx:
                            intersecting_shadows = shadows_proj.iloc[possible_shadow_matches_idx]
                            # Union unaire uniquement sur les ombres candidates
                            nearby_shadow_union = unary_union(intersecting_shadows.geometry)
                            intersection = edge_geom.intersection(nearby_shadow_union)
                            shade_length = intersection.length
                            shade_percent = (shade_length / length) if length > 0 else 0
                        cost = length * (1 + (1 - shade_percent) * 2)
                else: # Pas d'intersection avec l'eau ou c'est un pont
                    possible_shadow_matches_idx = list(shadows_sindex.intersection(edge_geom.bounds))
                    if possible_shadow_matches_idx:
                        intersecting_shadows = shadows_proj.iloc[possible_shadow_matches_idx]
                        nearby_shadow_union = unary_union(intersecting_shadows.geometry)
                        intersection = edge_geom.intersection(nearby_shadow_union)
                        shade_length = intersection.length
                        shade_percent = (shade_length / length) if length > 0 else 0
                    cost = length * (1 + (1 - shade_percent) * 2)
            else:
                 cost = length * 1000  # Pas de géométrie, pénalité forte
            
            G_proj.edges[u, v, key]['cost'] = cost
            G_proj.edges[u, v, key]['shade_percent'] = shade_percent

        # 6. CALCUL DE L'ITINÉRAIRE
        print("Calcul de l'itinéraire...")
        # On projette les coordonnées de départ et d'arrivée dans le CRS du graphe projeté
        start_geom_proj = gpd.GeoSeries([Point(start_coords[::-1])], crs='EPSG:4326').to_crs(target_crs).iloc[0]
        end_geom_proj = gpd.GeoSeries([Point(end_coords[::-1])], crs='EPSG:4326').to_crs(target_crs).iloc[0]

        # On cherche les noeuds les plus proches sur le graphe projeté
        start_node = ox.nearest_nodes(G_proj, X=start_geom_proj.x, Y=start_geom_proj.y)
        end_node = ox.nearest_nodes(G_proj, X=end_geom_proj.x, Y=end_geom_proj.y)

        try:
            route_nodes = nx.shortest_path(G_proj, source=start_node, target=end_node, weight='cost')
        except nx.NetworkXNoPath:
            print("AVERTISSEMENT: Aucun chemin ombragé trouvé. Utilisation du chemin le plus court comme alternative.")
            try:
                route_nodes = nx.shortest_path(G_proj, source=start_node, target=end_node, weight='length')
            except nx.NetworkXNoPath:
                raise HTTPException(status_code=404, detail="Impossible de trouver un chemin. Les points sont peut-être dans des zones non connectées.")

        # 7. FORMATAGE DE LA RÉPONSE
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

        # Log pour vérification
        print("--- GeoJSON de la réponse ---")
        print(gdf_route_web.to_json())
        print("-----------------------------")

        return JSONResponse(content={
            "heure_analyse": sun_time_str,
            "itineraire_segmente": json.loads(gdf_route_web.to_json())
        })

    except Exception as e:
        print(f"ERREUR lors du calcul de l'itinéraire: {e}")
        raise HTTPException(status_code=500, detail=str(e))
