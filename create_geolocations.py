import pandas as pd
import re

# Read the CSV files
df_cis = pd.read_csv('city_codes_cis.csv')
df_numbers = pd.read_csv('city_numbers.csv')
df_world = pd.read_csv('world_city_codes.csv')

# Create sets of unique locations from each file
locations_cis = set(df_cis['Страна'].unique()) | set(df_cis['Город'].unique())
locations_numbers = set(df_numbers['Город'].unique())
locations_world = set(df_world['Город'].unique())

# Combine all unique locations
all_locations = locations_cis | locations_numbers | locations_world

# Convert the set to a list
geolocations_list = list(all_locations)



def normalize_location(location):
    # Remove content in parentheses
    location = re.sub(r'\([^)]*\)', '', location)
    # Split by '//' and take both parts if available
    parts = [part.strip() for part in location.split('//')]
    # Remove empty strings
    parts = [part for part in parts if part]
    return parts

normalized_list = []
for item in geolocations_list:
    normalized_list.extend(normalize_location(item))

# Remove duplicates and sort
normalized_list = sorted(set(normalized_list))

# print(normalized_list)
if any(location.lower() == "Москва" for location in normalized_list):
    print("'крым' найден в списке геолокаций.")
else:
    print("'крым' не найден в списке геолокаций.")