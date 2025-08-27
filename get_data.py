import osmnx as ox
import matplotlib
matplotlib.use('Agg') # Utilise un backend non-interactif, idéal pour les serveurs
import matplotlib.pyplot as plt
import pybdshadow as bd
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# --- MESSAGE DE VÉRIFICATION ---
print("--- VERSION DU SCRIPT AVEC ANALYSE DES RUES OMBRAGÉES ---")

# --- 1. DÉFINIR LA ZONE D'ÉTUDE ---
place_name = "4th arrondissement of Paris, France"

# --- 2. ACQUÉRIR ET PRÉPARER LES DONNÉES DES BÂTIMENTS ---
print("Téléchargement des données des bâtiments...")
tags = {"building": True}
gdf_buildings = ox.features_from_place(place_name, tags)

print("Nettoyage des données géométriques...")
initial_count = len(gdf_buildings)
gdf_buildings = gdf_buildings[gdf_buildings['geometry'].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
gdf_buildings = gdf_buildings[~gdf_buildings.is_empty]
gdf_buildings = gdf_buildings[gdf_buildings.is_valid]
print(f"Après vérifications, il reste {len(gdf_buildings)} bâtiments valides.")

print("Décomposition des MultiPolygones...")
gdf_buildings = gdf_buildings.explode(index_parts=True)

print("Création d'un identifiant unique ('building_id')...")
gdf_buildings = gdf_buildings.reset_index(drop=True)
gdf_buildings['building_id'] = gdf_buildings.index
gdf_buildings['height'] = 15

# --- 3. ACQUÉRIR LE RÉSEAU VIAIRE ---
print("Téléchargement du réseau viaire...")
graph_streets = ox.graph_from_place(place_name, network_type='walk')
gdf_streets = ox.graph_to_gdfs(graph_streets, nodes=False, edges=True)

# --- 4. CALCULER LES OMBRES ---
sun_time_obj = datetime.now()
sun_time_str = sun_time_obj.strftime('%Y-%m-%d %H:%M:%S')
print(f"Calcul des ombres pour le {sun_time_str}...")
shadows = bd.cal_sunshadows(gdf_buildings, sun_time_str)
shadows.crs = gdf_buildings.crs
print("Calcul des ombres terminé.")

# --- 5. REPROJETER TOUTES LES DONNÉES EN MÈTRES POUR L'ANALYSE ---
print("Reprojection de toutes les données en Lambert-93...")
target_crs = 'EPSG:2154'
gdf_buildings_proj = gdf_buildings.to_crs(target_crs)
gdf_streets_proj = gdf_streets.to_crs(target_crs)
shadows_proj = shadows.to_crs(target_crs)
print("Reprojection terminée.")

# --- 6. ANALYSER L'INTERSECTION RUES / OMBRES ---
print("Analyse de l'intersection entre les rues et les zones d'ombre...")
# On fusionne toutes les ombres en une seule grande géométrie pour plus d'efficacité
total_shadow_area = unary_union(shadows_proj.geometry)

# On calcule les portions de rues qui sont DANS l'ombre (intersection)
gdf_streets_shady = gpd.overlay(gdf_streets_proj, gpd.GeoDataFrame(geometry=[total_shadow_area], crs=target_crs), how='intersection')

# On calcule les portions de rues qui sont HORS de l'ombre (différence)
gdf_streets_sunny = gpd.overlay(gdf_streets_proj, gpd.GeoDataFrame(geometry=[total_shadow_area], crs=target_crs), how='difference')
print("Analyse terminée.")

# --- 7. VISUALISER LE RÉSULTAT DE L'ANALYSE ---
print("Génération de la carte d'analyse...")
fig, ax = plt.subplots(figsize=(15, 15))

# Afficher les bâtiments en fond
gdf_buildings_proj.plot(ax=ax, color='grey', alpha=0.5)

# Afficher les portions de rues ensoleillées en jaune
gdf_streets_sunny.plot(ax=ax, color='yellow', linewidth=2, label='Rues Ensoleillées')

# Afficher les portions de rues ombragées en bleu
gdf_streets_shady.plot(ax=ax, color='blue', linewidth=2, label='Rues Ombragées')

ax.set_title(f"Analyse de l'Ombre sur les Rues - {place_name}\nle {sun_time_obj.strftime('%Y-%m-%d à %H:%M')}", fontsize=20)
ax.set_xlabel("Coordonnée Est (mètres)")
ax.set_ylabel("Coordonnée Ouest (mètres)")
ax.legend()
ax.tick_params(axis='x', rotation=45)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax.set_facecolor('black') # Fond noir pour mieux voir le jaune

# --- 8. ENREGISTRER LA CARTE DANS UN FICHIER ---