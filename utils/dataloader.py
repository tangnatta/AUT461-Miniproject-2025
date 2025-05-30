import pandas as pd
import os
from typing import Dict, Optional, List, Union


class Dataloader:
    western_europe_countries: set[str] = {
        'Belgium', 'France', 'Ireland', 'Luxembourg', 'Monaco',
        'Netherlands', 'United Kingdom'
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.country_name_mapping = {
            "Republic Of Ireland": "Ireland",
        }

    def normalize_country_name(self, country: str) -> str:
        """
        Normalize the country name to a standard format.

        Args:
            country (str): The name of the country.

        Returns:
            str: The normalized country name.
        """
        country = country.title().strip()
        if country in self.country_name_mapping:
            return self.country_name_mapping[country].title()

        return country

    def is_western_europe(self, country: str) -> bool:
        """
        Check if a country is in Western Europe.

        Args:
            country (str): The name of the country.

        Returns:
            bool: True if the country is in Western Europe, False otherwise.
        """

        return country.title() in self.western_europe_countries

    def _get_file_path(self, file_name: str) -> str:
        """
        Get the full file path and verify the file exists.

        Args:
            file_name (str): The name of the file to load.

        Returns:
            str: The validated full file path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist.")
        return file_path

    def fill_missing_dates_in_df_of_every_country(self, df: pd.DataFrame, date_col: str = 'date', group_by: Union[str, List[str]] = 'country') -> pd.DataFrame:
        """
        Fill missing dates in the DataFrame for each country or group combination.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            date_col (str): The column containing date values.
            group_by (str or List[str]): The column(s) to group by for filling missing dates.

        Returns:
            pd.DataFrame: The DataFrame with filled missing dates.
        """
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        # Create a list to hold the filled DataFrames
        filled_dfs = []

        # Process each group separately
        for name, group in df.groupby(group_by):
            # Ensure the group has a date column
            min_date = group[date_col].min()
            max_date = group[date_col].max()

            # Get full date range
            full_date_range = pd.date_range(
                start=min_date, end=max_date, freq='D')

            # Create a DataFrame with all dates
            date_df = pd.DataFrame({date_col: full_date_range})

            # Merge with original data to get missing dates
            merged = pd.merge(date_df, group, on=date_col, how='left')

            # Fill the group_by column(s) with the current group name(s)
            if isinstance(group_by, list):
                for i, col in enumerate(group_by):
                    merged[col] = name[i] if isinstance(name, tuple) else name
            else:
                merged[group_by] = name

            # Add to our list
            filled_dfs.append(merged)

        # Combine all processed groups back into a single DataFrame
        df = pd.concat(filled_dfs, ignore_index=True)
        return df

    def interpolate_columns(self, df: pd.DataFrame, cols_to_interpolate: List[str], group_by: str) -> pd.DataFrame:
        """
        Interpolate missing values in specified columns grouped by a given column.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            cols_to_interpolate (List[str]): List of columns to interpolate.
            group_by (str): The column to group by for interpolation.

        Returns:
            pd.DataFrame: The DataFrame with interpolated values.
        """
        for col in cols_to_interpolate:
            if col in df.columns:
                df[col] = df.groupby(group_by)[col].transform(
                    lambda x: x.interpolate(
                        method='linear', limit_direction='both')
                )
        return df

    def load_comprehensive_data(self, file_name: str = "Comprehensive_Global_COVID-19_Dataset.csv") -> pd.DataFrame:
        """
        Load the comprehensive COVID-19 dataset from a CSV file.

        Args:
            file_name (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = self._get_file_path(file_name)

        df = pd.read_csv(file_path)

        # dropna
        df.dropna(how="all", inplace=True)

        # replace column names with more readable ones
        df.rename(columns={
            'S. No.': 'ID',
            'Country Name': 'country',
            'Cases': 'confirmed_cases',
            'Deaths': 'deaths_cases',
            'Recovered': 'recovered_cases'
        }, inplace=True)

        df['country'] = df['country'].apply(self.normalize_country_name)
        df['is_western_europe'] = df['country'].apply(self.is_western_europe)

        return df

    def load_covid19_testing_record(self, file_name: str = "Covid19-TestingRecord.csv") -> pd.DataFrame:
        """
        Load the COVID-19 testing record dataset from a CSV file.

        Args:
            file_name (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = self._get_file_path(file_name)

        df = pd.read_csv(file_path)

        # dropna
        df.dropna(how="all", inplace=True)

        # replace column names with more readable ones (snake casing)
        df.rename(columns={
            'Entity': 'entity',
            'ISO code': 'iso_code',
            'Date': 'date',
            'Source URL': 'source_url',
            'Source label': 'source_label',
            'Notes': 'notes',
            'Daily change in cumulative total': 'daily_change',
            'Cumulative total': 'total_tests',
            'Cumulative total per thousand': 'total_tests_per_thousand',
            'Daily change in cumulative total per thousand': 'daily_change_per_thousand',
            '7-day smoothed daily change': 'smoothed_daily_change',
            '7-day smoothed daily change per thousand': 'smoothed_daily_change_per_thousand',
            'Short-term positive rate': 'positive_rate',
            'Short-term tests per case': 'tests_per_confirm_case'
        }, inplace=True)

        # convert date to datetime
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce')

        df['country'] = df['entity'].str.split(" - ").str[0]
        df['tested_type'] = df['entity'].str.split(" - ").str[1]

        df['country'] = df['country'].apply(self.normalize_country_name)

        # Fill missing dates for each country
        df = self.fill_missing_dates_in_df_of_every_country(
            df, date_col='date', group_by='country')

        df['is_western_europe'] = df['country'].apply(self.is_western_europe)

        # Columns to interpolate
        cols_to_interpolate = [
            'daily_change', 'total_tests', 'total_tests_per_thousand',
            'daily_change_per_thousand', 'smoothed_daily_change',
            'smoothed_daily_change_per_thousand', 'positive_rate',
            'tests_per_confirm_case'
        ]

        # Group by country and interpolate the missing values
        df = self.interpolate_columns(
            df, cols_to_interpolate, group_by='country')

        return df

    def load_covid19_variants_found(self, file_name: str = "Covid19-VariantsFound.csv") -> pd.DataFrame:
        """
        Load the COVID-19 variants found dataset from a CSV file.

        Args:
            file_name (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = self._get_file_path(file_name)

        df = pd.read_csv(file_path)

        # dropna
        df.dropna(how="all", inplace=True)

        # replace column names with more readable ones (snake casing)
        df.rename(columns={
            'location': 'country',
            'date': 'date',
            'variant': 'variant',
            'num_sequences': 'number_of_sequences',
            'perc_sequences': 'percentage_of_sequences',
            'num_sequences_total': 'total_sequences'
        }, inplace=True)

        # convert date to datetime
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce')

        df['country'] = df['country'].apply(self.normalize_country_name)

        # remove duplication of "other" variants and "non-who" variants
        # df['variant'] = df['variant'].replace(
        #     {'other': 'other', 'non-who': 'other'})
        # df.drop_duplicates(subset=['country', 'date', 'variant', 'number_of_sequences',
        #                    'percentage_of_sequences', 'total_sequences'], inplace=True)

        # Fill missing dates for each country-variant combination
        # Note: For variants, we need to group by both country and variant
        df = self.fill_missing_dates_in_df_of_every_country(
            df, date_col='date', group_by=['country', 'variant'])
        df['is_western_europe'] = df['country'].apply(self.is_western_europe)

        # Define columns to interpolate
        cols_to_interpolate = [
            'number_of_sequences', 'percentage_of_sequences', 'total_sequences'
        ]

        # Interpolate missing values for each country-variant combination
        df = self.interpolate_columns(
            df, cols_to_interpolate, group_by=['country', 'variant'])

        return df

    # TODO: Transform the data in to better show the vaccines used
    def load_vaccinations_by_country(self, file_name: str = "Vaccinations_ByCountry.csv") -> pd.DataFrame:
        """
        Load the vaccinations by country dataset from a CSV file.

        Args:
            file_name (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = self._get_file_path(file_name)

        df = pd.read_csv(file_path)

        # dropna
        df.dropna(how="all", inplace=True)

        # replace column names with more readable ones (snake casing)
        df.rename(columns={
            'country': 'country',
            'iso_code': 'iso_code',
            'date': 'date',
            'total_vaccinations': 'total_vaccinations',
            'people_vaccinated': 'people_vaccinated',
            'people_fully_vaccinated': 'people_fully_vaccinated',
            'daily_vaccinations_raw': 'daily_vaccinations_raw',
            'daily_vaccinations': 'daily_vaccinations',
            'total_vaccinations_per_hundred': 'total_vaccinations_per_hundred',
            'people_vaccinated_per_hundred': 'people_vaccinated_per_hundred',
            'people_fully_vaccinated_per_hundred': 'people_fully_vaccinated_per_hundred',
            'daily_vaccinations_per_million': 'daily_vaccinations_per_million',
            'vaccines': 'manufacturers',
            'source_name': 'source_name',
            'source_website': 'source_website'
        }, inplace=True)

        # convert date to datetime
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce')

        df['country'] = df['country'].apply(self.normalize_country_name)

        # Fill missing dates for each country
        df = self.fill_missing_dates_in_df_of_every_country(
            df, date_col='date', group_by='country')

        df['is_western_europe'] = df['country'].apply(self.is_western_europe)

        # Columns to interpolate
        cols_to_interpolate = [
            'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
            'daily_vaccinations_raw', 'daily_vaccinations',
            'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
            'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million'
        ]

        # Group by country and interpolate the missing values
        df = self.interpolate_columns(
            df, cols_to_interpolate, group_by='country')

        return df

    def load_vaccination_by_manufacturer(self, file_name: str = "Vaccinations_ByCountry_ByManufacturer.csv") -> pd.DataFrame:
        """
        Load the vaccinations by manufacturer dataset from a CSV file.

        Args:
            file_name (str): The name of the CSV file to load.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        file_path = self._get_file_path(file_name)

        df = pd.read_csv(file_path)

        # dropna
        df.dropna(how="all", inplace=True)

        # replace column names with more readable ones (snake casing)
        df.rename(columns={
            'location': 'country',
            'date': 'date',
            'vaccine': 'manufacturer',
            'total_vaccinations': 'total_vaccinations'
        }, inplace=True)

        # convert date to datetime
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce')

        df['country'] = df['country'].apply(self.normalize_country_name)

        # Fill missing dates for each country-manufacturer combination
        df = self.fill_missing_dates_in_df_of_every_country(
            df, date_col='date', group_by=['country', 'manufacturer'])

        df['is_western_europe'] = df['country'].apply(self.is_western_europe)

        # Interpolate total_vaccinations for each country-manufacturer combination
        df = self.interpolate_columns(
            df, ['total_vaccinations'], group_by=['country', 'manufacturer'])

        return df

    def load_all(self):
        """
        Load all datasets and return them as a dictionary.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing all loaded DataFrames.
        """
        return {
            "comprehensive_data": self.load_comprehensive_data(),
            "covid19_testing_record": self.load_covid19_testing_record(),
            "covid19_variants_found": self.load_covid19_variants_found(),
            "vaccinations_by_country": self.load_vaccinations_by_country(),
            "vaccination_by_manufacturer": self.load_vaccination_by_manufacturer()
        }


if __name__ == "__main__":
    # Replace with your actual data directory
    data_dir = r"D:\\Github\\AUT461-Covid-2025\\data"
    dataloader = Dataloader(data_dir)
    comprehensive_data = dataloader.load_comprehensive_data()
    covid19_testing_record_data = dataloader.load_covid19_testing_record()
    covid19_variants_found_data = dataloader.load_covid19_variants_found()
    vaccinations_by_country_data = dataloader.load_vaccinations_by_country()
    vaccination_by_manufacturer_data = dataloader.load_vaccination_by_manufacturer()
    print(comprehensive_data.head())
    print(covid19_testing_record_data.head())
    print(covid19_variants_found_data.head())
    print(vaccinations_by_country_data.head())
    print(vaccination_by_manufacturer_data.head())

    # to csv
    comprehensive_data.to_csv(
        os.path.join(data_dir, "processed", "comprehensive_data.csv"), index=False)
    covid19_testing_record_data.to_csv(
        os.path.join(data_dir, "processed", "covid19_testing_record_data.csv"), index=False)
    covid19_variants_found_data.to_csv(
        os.path.join(data_dir, "processed", "covid19_variants_found_data.csv"), index=False)
    vaccinations_by_country_data.to_csv(
        os.path.join(data_dir, "processed", "vaccinations_by_country_data.csv"), index=False)
    vaccination_by_manufacturer_data.to_csv(
        os.path.join(data_dir, "processed", "vaccination_by_manufacturer_data.csv"), index=False)
    print("Data loaded and saved to CSV files.")
