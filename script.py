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
        """Find best model match from search results with proper type checking."""
        if not results:
            return None

        # Ensure results is a list and contains dictionaries
        if not isinstance(results, list):
            logger.warning(f"Expected list of results, got {type(results)}")
            return None

        target_lower = target_model.lower().strip()
        target_clean = ''.join(c for c in target_lower if c.isalnum())

        best_match = None
        best_score = 0

        for result in results:
            # Skip if result is not a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Expected dict result, got {type(result)}: {result}")
                continue

            # Get model name safely
            api_model = result.get('model', '')
            if not api_model:
                continue

            api_lower = str(api_model).lower().strip()
            api_clean = ''.join(c for c in api_lower if c.isalnum())

            score = 0

            # Exact match (highest priority)
            if api_lower == target_lower:
                logger.info(f"Exact model match found: '{api_model}' = '{target_model}'")
                return result

            # Exact match without spaces/special chars
            if api_clean == target_clean:
                score = 95
                logger.info(f"Clean exact match: '{api_model}' ≈ '{target_model}' (score: {score})")

            # Target model is contained in API model (e.g., "fly 150" in "Fly 150 Sport")
            elif target_lower in api_lower:
                score = 90 - (len(api_lower) - len(target_lower))  # Prefer shorter matches
                logger.info(f"Target in API: '{target_model}' in '{api_model}' (score: {score})")

            # API model is contained in target (e.g., "fly" in "fly 150")
            elif api_lower in target_lower:
                score = 85 - (len(target_lower) - len(api_lower))
                logger.info(f"API in target: '{api_model}' in '{target_model}' (score: {score})")

            # Partial alphanumeric match
            elif target_clean in api_clean or api_clean in target_clean:
                overlap = min(len(target_clean), len(api_clean))
                total = max(len(target_clean), len(api_clean))
                score = (overlap / total) * 80
                logger.info(f"Partial match: '{api_model}' ≈ '{target_model}' (score: {score:.1f})")

            # Word-by-word matching for complex models
            else:
                target_words = target_clean.split()
                api_words = api_clean.split()

                matching_words = 0
                for t_word in target_words:
                    for a_word in api_words:
                        if len(t_word) >= 3 and len(a_word) >= 3:
                            if t_word == a_word or t_word in a_word or a_word in t_word:
                                matching_words += 1
                                break

                if matching_words > 0:
                    score = (matching_words / max(len(target_words), len(api_words))) * 70
                    logger.info(f"Word match: '{api_model}' ≈ '{target_model}' (score: {score:.1f})")

            # Keep track of best match
            if score > best_score:
                best_score = score
                best_match = result

        if best_match and best_score >= 50:  # Only return if score is decent
            logger.info(f"Best match selected (score: {best_score:.1f}): '{best_match.get('model', 'N/A')}'")
            return best_match

        logger.info(f"No good matches found for '{target_model}' (best score: {best_score:.1f})")
        return None

    def search_by_year_make_model(self, year: int, make: str, model: str) -> Dict:
        """Search by year, make, model using the building blocks approach."""
        vehicle_type = self.determine_vehicle_type(make)
        clean_make = make.upper().strip()

        # Strategy 1: Try to get full records using building blocks approach
        # This mimics the API documentation workflow

        logger.info(f"Trying building blocks approach for {vehicle_type} - {clean_make} {year}")

        # Step 1: Get models for this year/make using the building blocks API
        try:
            # Use the v1 building blocks approach: /api/v1/type/{type}/manu/{manu}/year/{year}/model
            endpoint = f'/v1/type/{vehicle_type}/manu/{clean_make}/year/{year}/model'
            logger.info(f"Building blocks request: {endpoint}")

            success, models_data = self.api_request(endpoint, 'GET')

            if success and models_data:
                logger.info(f"Building blocks found {len(models_data) if isinstance(models_data, list) else 1} models")

                # Now we should have a list of model names or IDs
                best_model_id = None
                best_score = 0

                if isinstance(models_data, list):
                    for model_item in models_data:
                        if isinstance(model_item, str):
                            # Score this model name
                            score = self.calculate_model_similarity(model, model_item)
                            logger.info(f"Model match score: '{model_item}' vs '{model}' = {score:.1f}%")

                            if score > best_score and score >= 50:  # At least 50% match
                                best_score = score
                                best_model_id = model_item
                        elif isinstance(model_item, dict) and 'model' in model_item:
                            # Already a full record
                            score = self.calculate_model_similarity(model, model_item['model'])
                            if score >= 50:
                                logger.info(f"Found full record match: {model_item['model']} (score: {score:.1f}%)")
                                return {'success': True, 'data': model_item, 'method': 'BUILDING_BLOCKS_FULL'}

                # If we found a good model match, try to get the full record
                if best_model_id and best_score >= 50:
                    logger.info(f"Best model match: '{best_model_id}' (score: {best_score:.1f}%)")

                    # Try to get full record using: /api/v1/type/{type}/manu/{manu}/year/{year}/model/{model}/go
                    full_endpoint = f'/v1/type/{vehicle_type}/manu/{clean_make}/year/{year}/model/{best_model_id}/go'
                    logger.info(f"Getting full record: {full_endpoint}")

                    success, full_data = self.api_request(full_endpoint, 'GET')

                    if success and full_data:
                        logger.info(f"✅ Got full record for {best_model_id}")
                        # Handle if it returns a list
                        if isinstance(full_data, list) and len(full_data) > 0:
                            full_data = full_data[0]

                        return {'success': True, 'data': full_data, 'method': f'BUILDING_BLOCKS_{vehicle_type}'}
                    else:
                        logger.info(f"Failed to get full record for {best_model_id}")
                else:
                    logger.info(f"No good model matches found (best score: {best_score:.1f}%)")
            else:
                logger.info(f"Building blocks approach found no models for {clean_make} {year}")

        except Exception as e:
            logger.warning(f"Building blocks approach failed: {str(e)}")

        # Strategy 2: Try different vehicle type classifications
        alternative_types = []
        if vehicle_type == 'Scooter':
            alternative_types = ['Motorcycle']
        elif vehicle_type == 'Motorcycle':
            alternative_types = ['Scooter']
        elif make.upper() in ['POLARIS', 'CAN-AM', 'CANAM']:
            alternative_types = ['UTV', 'Side-by-Side', 'Motorcycle']

        for alt_type in alternative_types:
            logger.info(f"Trying alternative type: {alt_type}")
            try:
                endpoint = f'/v1/type/{alt_type}/manu/{clean_make}/year/{year}/model'
                success, models_data = self.api_request(endpoint, 'GET')

                if success and models_data and isinstance(models_data, list):
                    for model_item in models_data:
                        if isinstance(model_item, str):
                            score = self.calculate_model_similarity(model, model_item)
                            if score >= 70:  # Higher threshold for alt types
                                # Try to get full record
                                full_endpoint = f'/v1/type/{alt_type}/manu/{clean_make}/year/{year}/model/{model_item}/go'
                                success, full_data = self.api_request(full_endpoint, 'GET')

                                if success and full_data:
                                    if isinstance(full_data, list) and len(full_data) > 0:
                                        full_data = full_data[0]
                                    return {'success': True, 'data': full_data, 'method': f'ALT_TYPE_{alt_type}'}
            except Exception as e:
                logger.warning(f"Alternative type {alt_type} failed: {str(e)}")

        return {'success': False, 'error': f'No full vehicle records found for {make} {year} {model}',
                'method': 'SEARCH_FAILED'}

    def calculate_model_similarity(self, target_model: str, api_model: str) -> float:
        """Calculate similarity percentage between two model names."""
        if not target_model or not api_model:
            return 0.0

        target_clean = ''.join(c.lower() for c in target_model if c.isalnum())
        api_clean = ''.join(c.lower() for c in api_model if c.isalnum())

        if not target_clean or not api_clean:
            return 0.0

        # Exact match
        if target_clean == api_clean:
            return 100.0

        # Full substring match
        if target_clean in api_clean or api_clean in target_clean:
            overlap = min(len(target_clean), len(api_clean))
            total = max(len(target_clean), len(api_clean))
            return (overlap / total) * 95.0

        # Calculate character overlap
        target_set = set(target_clean)
        api_set = set(api_clean)

        if not target_set or not api_set:
            return 0.0

        intersection = len(target_set.intersection(api_set))
        union = len(target_set.union(api_set))

        return (intersection / union) * 80.0

    def extract_wholesale_price(self, data) -> Optional[float]:
        """Extract wholesale price from API response with robust type checking."""
        if not data:
            return None

        # Handle different data types
        if isinstance(data, str):
            logger.warning(f"Expected dict/list but got string: {data[:100]}...")
            return None

        # If data is a list, take the first item
        if isinstance(data, list):
            if not data:  # Empty list
                return None

            # Take first item that looks like a vehicle record
            for item in data:
                if isinstance(item, dict):
                    data = item
                    break
            else:
                logger.warning(f"No dictionary found in list: {data}")
                return None

        # If data is still not a dict, give up
        if not isinstance(data, dict):
            logger.warning(f"Expected dict after processing, got {type(data)}: {data}")
            return None

        # Canadian Blue Book specific fields - PRIORITIZE ROUGH PRICE as requested
        price_fields = [
            'w_rgh',  # wholesale rough condition (user wants this)
            'w_avg',  # wholesale average condition
            'w_clean',  # wholesale clean condition
        ]

        # Check CBB-specific fields first
        for field in price_fields:
            value = data.get(field)
            if value is not None:
                try:
                    price = float(value)
                    if price > 0:  # Must be positive
                        logger.info(f"Found CBB wholesale price ${price} in field '{field}'")
                        return price
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {field}={value} to float")
                    continue

        # Fallback to other possible price fields
        other_price_fields = [
            'wholesale_price', 'wholesale', 'trade_value', 'tradeValue',
            'value', 'price', 'amount', 'cost'
        ]

        for field in other_price_fields:
            value = data.get(field)
            if value is not None:
                try:
                    price = float(value)
                    if price > 0:
                        logger.info(f"Found price ${price} in field '{field}'")
                        return price
                except (ValueError, TypeError):
                    continue

        # Log what fields are available for debugging
        available_fields = list(data.keys()) if isinstance(data, dict) else []
        logger.debug(f"No price found. Available fields: {available_fields[:10]}")
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

    # Enable debug mode to catch issues
    DEBUG_MODE = True  # Temporarily enable to debug the 'str' error

    if DEBUG_MODE:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Debug mode enabled to track down the 'str' error")

    if API_KEY == 'YOUR_API_KEY_HERE':
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
        print("🔧 Fixed: 'str' object error and improved exact model matching")
        print("✨ Special handling for PIAGGIO and other edge cases")
        print("")

        results_df = fetcher.process_excel_file(
            input_filename='bluebooktest.xlsx',
            output_filename='cbb_wholesale_prices_fixed.xlsx'
        )

        print(f"\n✅ Processing complete!")
        print(f"📁 Results saved to: cbb_wholesale_prices_fixed.xlsx")
        print(f"📊 Processed {len(results_df)} vehicles")

    except FileNotFoundError:
        print("❌ Error: bluebooktest.xlsx not found")
        print("Please ensure the input Excel file is in the same directory")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()