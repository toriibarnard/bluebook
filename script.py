# File: cbb_price_fetcher_fixed.py
# Canadian Blue Book API Price Fetcher - WORKING VERSION with Rough Price

import pandas as pd
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CanadianBlueBookPriceFetcher:
    """
    Fetches wholesale prices from Canadian Blue Book API for vehicles in Excel file.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://canadianbluebook.com/api'
        self.request_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

        # Vehicle type mapping
        self.motorcycle_makes = {
            'YAMAHA', 'KAWASAKI', 'HONDA', 'SUZUKI', 'BMW', 'TRIUMPH',
            'HARLEY DAVIDSON', 'INDIAN', 'VICTORY', 'KTM', 'HUSQVARNA'
        }
        self.atv_makes = {
            'POLARIS', 'CAN-AM', 'CANAM', 'CFMOTO', 'ARCTIC CAT', 'KUBOTA'
        }
        self.scooter_makes = {'PIAGGIO', 'SYM'}

    def api_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Tuple[bool, Dict]:
        """Make authenticated API request with error handling."""
        self.request_count += 1
        url = f"{self.base_url}{endpoint}"

        logger.info(f"API Request #{self.request_count}: {method} {endpoint}")

        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            # Rate limiting
            time.sleep(1)

            return True, response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API Error for {endpoint}: {str(e)}")
            return False, {'error': str(e)}

    def get_by_vin(self, vin: str) -> Dict:
        """Try to get vehicle data by VIN (preferred method)."""
        # Try v3 first (more precise)
        success, data = self.api_request(f'/v3/vin/{vin}')
        if success:
            return {'success': True, 'data': data, 'method': 'VIN_V3'}

        # Fallback to v1 (fuzzy search)
        success, data = self.api_request(f'/v1/vin/{vin}')
        if success:
            return {'success': True, 'data': data, 'method': 'VIN_V1'}

        return {'success': False, 'error': data.get('error', 'VIN lookup failed'), 'method': 'VIN_FAILED'}

    def determine_vehicle_type(self, make: str) -> str:
        """Determine vehicle type based on manufacturer."""
        make_upper = make.upper()

        if make_upper in self.motorcycle_makes:
            return 'Motorcycle'
        elif make_upper in self.atv_makes:
            return 'ATV'
        elif make_upper in self.scooter_makes:
            return 'Scooter'
        else:
            # Default to motorcycle for unknown makes
            return 'Motorcycle'

    def find_best_model_match(self, results: List[Dict], target_model: str) -> Optional[Dict]:
        """Find best model match from search results."""
        if not results:
            return None

        target_lower = target_model.lower()

        # Look for exact match first
        for result in results:
            if result.get('model', '').lower() == target_lower:
                return result

        # Look for partial match
        for result in results:
            model = result.get('model', '').lower()
            if target_lower in model or model in target_lower:
                return result

        # Return first result if no good match found
        return results[0]

    def search_by_year_make_model(self, year: int, make: str, model: str) -> Dict:
        """Fuzzy search by year, make, model using POST query."""
        vehicle_type = self.determine_vehicle_type(make)

        query_data = {
            'type': vehicle_type,
            'manu': make,
            'year': int(year)
        }

        # Try v1 first (allows partial model matching)
        success, data = self.api_request('/v1/query', 'POST', query_data)

        if success and data:
            # Try to find exact or close model match
            model_match = self.find_best_model_match(data, model)
            if model_match:
                return {'success': True, 'data': model_match, 'method': 'SEARCH_V1'}

        return {'success': False, 'error': 'No matching models found', 'method': 'SEARCH_FAILED'}

    def extract_wholesale_price(self, data) -> Optional[float]:
        """Extract wholesale price from API response - FIXED to handle lists and get rough price."""
        if not data:
            return None

        # If data is a list, take the first item
        if isinstance(data, list):
            if not data:  # Empty list
                return None
            data = data[0]  # Take first vehicle

        # If data is not a dict after handling list, return None
        if not isinstance(data, dict):
            return None

        # Canadian Blue Book specific fields - PRIORITIZE ROUGH PRICE as requested
        price_fields = [
            'w_rgh',  # wholesale rough condition (user wants this)
            'w_avg',  # wholesale average condition
            'w_clean',  # wholesale clean condition
        ]

        # Check CBB-specific fields
        for field in price_fields:
            if field in data and data[field] is not None:
                try:
                    price = float(data[field])
                    if price > 0:  # Must be positive
                        logger.info(f"Found CBB wholesale price ${price} in field '{field}'")
                        return price
                except (ValueError, TypeError):
                    continue

        # Fallback to other possible price fields
        other_price_fields = [
            'wholesale_price', 'wholesale', 'trade_value', 'tradeValue',
            'value', 'price', 'amount'
        ]

        for field in other_price_fields:
            if field in data and data[field] is not None:
                try:
                    price = float(data[field])
                    if price > 0:
                        logger.info(f"Found price ${price} in field '{field}'")
                        return price
                except (ValueError, TypeError):
                    continue

        logger.debug(f"No price found in response")
        return None

    def process_vehicle(self, vehicle: Dict) -> Dict:
        """Process a single vehicle record."""
        logger.info(f"Processing: {vehicle['year']} {vehicle['make']} {vehicle['model']} (VIN: {vehicle['vin']})")

        result = {
            **vehicle,
            'cbb_wholesale_price': None,
            'lookup_method': None,
            'api_response': None,
            'error': None
        }

        price_found = False

        try:
            # Try VIN lookup first
            if vehicle['vin'] and len(str(vehicle['vin'])) > 10:
                vin_result = self.get_by_vin(vehicle['vin'])

                if vin_result['success']:
                    price = self.extract_wholesale_price(vin_result['data'])
                    if price is not None and price > 0:  # Only count as success if we get a valid price
                        result['cbb_wholesale_price'] = price
                        result['lookup_method'] = vin_result['method']
                        result['api_response'] = str(vin_result['data'])[:200]  # Truncate for storage
                        price_found = True
                        logger.info(f"✅ VIN lookup found price: ${price}")
                    else:
                        logger.info(f"VIN lookup successful but no price found, trying year/make/model...")
                else:
                    logger.info(f"VIN lookup failed, trying year/make/model...")

            # Try year/make/model search if VIN didn't give us a price
            if not price_found:
                search_result = self.search_by_year_make_model(vehicle['year'], vehicle['make'], vehicle['model'])

                if search_result['success']:
                    price = self.extract_wholesale_price(search_result['data'])
                    if price is not None and price > 0:
                        result['cbb_wholesale_price'] = price
                        result['lookup_method'] = search_result['method']
                        result['api_response'] = str(search_result['data'])[:200]  # Truncate for storage
                        price_found = True
                        logger.info(f"✅ Search lookup found price: ${price}")
                    else:
                        result['error'] = 'Search found vehicle but no price available'
                        result['lookup_method'] = search_result['method']
                        logger.info(f"Search found vehicle but no price available")
                else:
                    result['error'] = search_result['error']
                    result['lookup_method'] = search_result['method']
                    logger.info(f"Search failed: {search_result['error']}")

            if not price_found and not result['error']:
                result['error'] = 'No price found via any method'
                result['lookup_method'] = 'ALL_METHODS_FAILED'

        except Exception as e:
            result['error'] = str(e)
            result['lookup_method'] = 'EXCEPTION'
            logger.error(f"Exception processing vehicle: {str(e)}")

        return result

    def process_excel_file(self, input_filename: str,
                           output_filename: str = 'cbb_wholesale_prices.xlsx') -> pd.DataFrame:
        """Process all vehicles from Excel file and create output file."""
        logger.info(f"Reading Excel file: {input_filename}")

        # Read Excel file
        df = pd.read_excel(input_filename)

        # Extract relevant columns
        vehicle_data = []
        for index, row in df.iterrows():
            if pd.notna(row.get('MODELYEAR')) and pd.notna(row.get('MAKEDESCRIPTION')) and pd.notna(
                    row.get('MODELDESCRIPTION')):
                vehicle_data.append({
                    'row_number': index + 2,  # Excel row number
                    'year': int(row['MODELYEAR']),
                    'make': str(row['MAKEDESCRIPTION']),
                    'model': str(row['MODELDESCRIPTION']),
                    'vin': str(row.get('VIN', '')),
                    'current_wholesale_value': row.get('AVERAGEWHOLESALEVALUE')
                })

        logger.info(f"Found {len(vehicle_data)} vehicles to process")

        # Process each vehicle
        results = []
        for i, vehicle in enumerate(vehicle_data):
            logger.info(f"\n--- Processing {i + 1}/{len(vehicle_data)} ---")

            try:
                result = self.process_vehicle(vehicle)
                results.append(result)

                # Log progress
                if result['cbb_wholesale_price']:
                    logger.info(f"✅ Success: ${result['cbb_wholesale_price']} ({result['lookup_method']})")
                else:
                    logger.info(f"❌ Failed: {result['error'] or 'No price found'}")

            except Exception as e:
                logger.error(f"Error processing vehicle {i + 1}: {str(e)}")
                results.append({
                    **vehicle,
                    'cbb_wholesale_price': None,
                    'lookup_method': 'ERROR',
                    'error': str(e)
                })

            # Extra delay every 10 requests
            if (i + 1) % 10 == 0:
                logger.info('Pausing for rate limiting...')
                time.sleep(2)

        # Create output DataFrame
        output_df = self.create_output_dataframe(results)

        # Save to Excel
        self.save_to_excel(output_df, results, output_filename)

        # Print summary
        self.print_summary(results)

        return output_df

    def create_output_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create output DataFrame from results."""
        output_data = []
        for result in results:
            output_data.append({
                'Row Number': result['row_number'],
                'Year': result['year'],
                'Make': result['make'],
                'Model': result['model'],
                'VIN': result['vin'],
                'Original Wholesale Value': result.get('current_wholesale_value', ''),
                'CBB Wholesale Price (Rough)': result['cbb_wholesale_price'],
                'Lookup Method': result['lookup_method'],
                'Error': result['error']
            })

        return pd.DataFrame(output_data)

    def save_to_excel(self, output_df: pd.DataFrame, results: List[Dict], filename: str):
        """Save results to Excel file with multiple sheets."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results sheet
            output_df.to_excel(writer, sheet_name='CBB Results', index=False)

            # Summary sheet
            total = len(results)
            successful = len([r for r in results if r['cbb_wholesale_price'] is not None])
            failed = total - successful

            # Method breakdown
            method_counts = {}
            for result in results:
                method = result['lookup_method']
                method_counts[method] = method_counts.get(method, 0) + 1

            summary_data = [
                ['Metric', 'Value'],
                ['Total Vehicles Processed', total],
                ['Successful Lookups', successful],
                ['Failed Lookups', failed],
                ['Success Rate', f"{(successful / total * 100):.1f}%"],
                ['', ''],
                ['Lookup Methods:', '']
            ]

            for method, count in method_counts.items():
                summary_data.append([method, count])

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)

        logger.info(f"Results saved to {filename}")

    def print_summary(self, results: List[Dict]):
        """Print processing summary."""
        total = len(results)
        successful = len([r for r in results if r['cbb_wholesale_price'] is not None])
        failed = total - successful

        method_stats = {}
        for result in results:
            method = result['lookup_method']
            method_stats[method] = method_stats.get(method, 0) + 1

        print('\n=== PROCESSING SUMMARY ===')
        print(f'Total vehicles processed: {total}')
        print(f'Successful price lookups: {successful} ({successful / total * 100:.1f}%)')
        print(f'Failed lookups: {failed} ({failed / total * 100:.1f}%)')
        print(f'Total API requests made: {self.request_count}')

        print('\nLookup method breakdown:')
        for method, count in method_stats.items():
            print(f'- {method}: {count}')

        # Price statistics
        prices = [r['cbb_wholesale_price'] for r in results if r['cbb_wholesale_price'] is not None]
        if prices:
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)

            print('\nPrice statistics:')
            print(f'- Average: ${avg_price:.2f}')
            print(f'- Minimum: ${min_price:.2f}')
            print(f'- Maximum: ${max_price:.2f}')


def main():
    """Example usage of the CanadianBlueBookPriceFetcher."""
    # Replace with your actual API key
    API_KEY = '144|inuYwvGhZePXzVmR3ZbRzH6dsRXs5UHYPtmvv9uX'

    if API_KEY == '1':
        print("⚠️  Please set your Canadian Blue Book API key in the API_KEY variable")
        print("Contact Canadian Blue Book to obtain API credentials")
        return

    # Initialize the fetcher
    fetcher = CanadianBlueBookPriceFetcher(API_KEY)

    # Process the Excel file
    try:
        print("🚀 Starting Canadian Blue Book price lookup...")
        print("📋 Strategy: VIN lookup first, then fallback to year/make/model search")
        print("💰 Looking for ROUGH wholesale price (w_rgh field)")
        print("")

        results_df = fetcher.process_excel_file(
            input_filename='bluebooktest.xlsx',
            output_filename='cbb_wholesale_prices.xlsx'
        )

        print(f"\n✅ Processing complete!")
        print(f"📁 Results saved to: cbb_wholesale_prices.xlsx")
        print(f"📊 Processed {len(results_df)} vehicles")

    except FileNotFoundError:
        print("❌ Error: bluebooktest.xlsx not found")
        print("Please ensure the input Excel file is in the same directory")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()