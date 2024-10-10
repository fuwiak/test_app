import pandas as pd
import re


class GeoLocationNormalizer:
    def __init__(self, cis_file, numbers_file, world_file):
        self.cis_file = cis_file
        self.numbers_file = numbers_file
        self.world_file = world_file

    def read_csv_files(self):
        df_cis = pd.read_csv(self.cis_file)
        df_numbers = pd.read_csv(self.numbers_file)
        df_world = pd.read_csv(self.world_file)
        return df_cis, df_numbers, df_world

    def get_unique_locations(self, df_cis, df_numbers, df_world):
        locations_cis = set(df_cis['Страна'].unique()) | set(
            df_cis['Город'].unique())
        locations_numbers = set(df_numbers['Город'].unique())
        locations_world = set(df_world['Город'].unique())
        return locations_cis | locations_numbers | locations_world

    def normalize_location(self, location):
        location = re.sub(r'\([^)]*\)', '', location)
        parts = [part.strip() for part in location.split('//')]
        return [part for part in parts if part]

    def get_normalized_list(self):
        df_cis, df_numbers, df_world = self.read_csv_files()
        all_locations = self.get_unique_locations(df_cis, df_numbers, df_world)

        normalized_list = []
        for item in all_locations:
            normalized_list.extend(self.normalize_location(item))

        normalized_list = sorted(set(normalized_list))

        # Create a list with both lowercase and uppercase starts
        final_list = []
        for location in normalized_list:
            final_list.append(location.lower())
            final_list.append(location.capitalize())

        return sorted(set(final_list))


# Usage:
# normalizer = GeoLocationNormalizer('city_codes_cis.csv', 'city_numbers.csv',
#                                    'world_city_codes.csv')
# result = normalizer.get_normalized_list()
# print(result)