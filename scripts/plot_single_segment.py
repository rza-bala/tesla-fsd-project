import folium

# Define GPS bounding box
lat_min, lat_max = 37.64392865319225, 37.64478055776392
lon_min, lon_max = -122.45102412071952, -122.44808482385491
# Center of the rectangle
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2

# Create a map centered on the segment
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Add rectangle to the map
folium.Rectangle(
    bounds=[[lat_min, lon_min], [lat_max, lon_max]],
    color="orange",
    fill=True,
    fill_opacity=0.4,
    tooltip="Merge onto I-280 South",
).add_to(m)

# Save map to file
m.save("outputs/merge_segment_map.html")
print("✅ Map saved as: outputs/merge_segment_map.html")
