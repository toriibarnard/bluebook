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

    def normalize_manufacturer_name(self, make: str) -> List[str]:
        """Generate all possible manufacturer name variations for better matching."""
        make_clean = make.upper().strip()
        variations = [make_clean]

        # Common manufacturer variations
        variations_map = {
            'CANAM': ['CAN-AM', 'CAN AM', 'CANAM'],
            'CAN-AM': ['CAN-AM', 'CAN AM', 'CANAM'],
            'CAN AM': ['CAN-AM', 'CAN AM', 'CANAM'],
            'HARLEY DAVIDSON': ['HARLEY DAVIDSON', 'HARLEY-DAVIDSON', 'HARLEY', 'HD'],
            'HARLEY-DAVIDSON': ['HARLEY DAVIDSON', 'HARLEY-DAVIDSON', 'HARLEY', 'HD'],
            'HARLEY': ['HARLEY DAVIDSON', 'HARLEY-DAVIDSON', 'HARLEY', 'HD'],
            'PIAGGO': ['PIAGGIO'],  # Common misspelling
            'PIAGGIO': ['PIAGGIO'],
            'BMW': ['BMW', 'B.M.W.'],
            'KTM': ['KTM', 'K.T.M.'],
            'CFMOTO': ['CFMOTO', 'CF MOTO', 'CF-MOTO'],
            'CF MOTO': ['CFMOTO', 'CF MOTO', 'CF-MOTO'],
        }

        if make_clean in variations_map:
            variations.extend(variations_map[make_clean])

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)

        return unique_variations

    def advanced_model_similarity(self, target_model: str, api_model: str) -> float:
        """Advanced fuzzy matching algorithm with multiple scoring methods."""
        if not target_model or not api_model:
            return 0.0

        # Normalize inputs
        target = target_model.strip()
        api = api_model.strip()
        target_lower = target.lower()
        api_lower = api.lower()

        # Remove special characters for clean comparison
        target_clean = ''.join(c for c in target_lower if c.isalnum() or c.isspace())
        api_clean = ''.join(c for c in api_lower if c.isalnum() or c.isspace())

        # Remove extra spaces
        target_clean = ' '.join(target_clean.split())
        api_clean = ' '.join(api_clean.split())

        scores = []

        # 1. EXACT MATCH (100%)
        if target_lower == api_lower or target_clean == api_clean:
            return 100.0

        # 2. PREFIX MATCHING (95-98%)
        if target_clean and api_clean:
            if api_clean.startswith(target_clean):
                # API model starts with our search term - very high confidence
                # E.g., "FLHTKL" matches "FLHTKL Electra glide ultra LTD low ABS 103 LC"
                ratio = len(target_clean) / len(api_clean)
                score = 95 + (ratio * 3)  # 95-98%
                scores.append(('prefix_api', score))
            elif target_clean.startswith(api_clean):
                # Our search term starts with API model
                # E.g., "SPYDER RT LIMITED" starts with "Spyder"
                ratio = len(api_clean) / len(target_clean)
                score = 92 + (ratio * 3)  # 92-95%
                scores.append(('prefix_target', score))

        # 3. ALPHANUMERIC ONLY MATCHING (90-94%)
        target_alphanum = ''.join(c for c in target_lower if c.isalnum())
        api_alphanum = ''.join(c for c in api_lower if c.isalnum())

        if target_alphanum and api_alphanum:
            if target_alphanum == api_alphanum:
                scores.append(('alphanum_exact', 94.0))
            elif api_alphanum.startswith(target_alphanum):
                ratio = len(target_alphanum) / len(api_alphanum)
                score = 88 + (ratio * 6)  # 88-94%
                scores.append(('alphanum_prefix', score))
            elif target_alphanum.startswith(api_alphanum):
                ratio = len(api_alphanum) / len(target_alphanum)
                score = 85 + (ratio * 6)  # 85-91%
                scores.append(('alphanum_prefix_rev', score))

        # 4. SUBSTRING CONTAINMENT (80-89%)
        if target_clean in api_clean:
            # Our search term is contained in API result
            # E.g., "KLX110" in "KLX110R"
            if len(target_clean) >= 5:  # Longer terms get higher scores
                ratio = len(target_clean) / len(api_clean)
                score = 82 + (ratio * 7)  # 82-89%
                scores.append(('substring_long', score))
            else:
                ratio = len(target_clean) / len(api_clean)
                score = 75 + (ratio * 7)  # 75-82%
                scores.append(('substring_short', score))
        elif api_clean in target_clean:
            # API result is contained in our search term
            ratio = len(api_clean) / len(target_clean)
            score = 78 + (ratio * 7)  # 78-85%
            scores.append(('substring_rev', score))

        # 5. WORD-LEVEL MATCHING (70-85%)
        target_words = set(target_clean.split())
        api_words = set(api_clean.split())

        if target_words and api_words:
            common_words = target_words.intersection(api_words)
            if common_words:
                # Calculate word overlap scores
                target_coverage = len(common_words) / len(target_words)
                api_coverage = len(common_words) / len(api_words)
                avg_coverage = (target_coverage + api_coverage) / 2

                # Bonus for longer words
                avg_word_len = sum(len(word) for word in common_words) / len(common_words)
                word_bonus = min(avg_word_len / 4, 1.0)  # Max 1.0 bonus

                score = 65 + (avg_coverage * 15) + (word_bonus * 5)  # 65-85%
                scores.append(('word_overlap', score))

        # 6. CHARACTER SEQUENCE MATCHING (60-75%)
        if target_alphanum and api_alphanum:
            # Find longest common subsequence
            common_chars = 0
            min_len = min(len(target_alphanum), len(api_alphanum))

            # Count matching characters from the start
            for i in range(min_len):
                if target_alphanum[i] == api_alphanum[i]:
                    common_chars += 1
                else:
                    break

            if common_chars >= 3:  # At least 3 chars match from start
                ratio = common_chars / max(len(target_alphanum), len(api_alphanum))
                score = 55 + (ratio * 20)  # 55-75%
                scores.append(('char_sequence', score))

        # 7. JACCARD SIMILARITY (50-70%)
        target_chars = set(target_alphanum)
        api_chars = set(api_alphanum)

        if target_chars and api_chars:
            intersection = len(target_chars.intersection(api_chars))
            union = len(target_chars.union(api_chars))
            jaccard = intersection / union
            score = 40 + (jaccard * 30)  # 40-70%
            scores.append(('jaccard', score))

        # 8. EDIT DISTANCE (30-60%)
        if len(target_clean) <= 20 and len(api_clean) <= 20:  # Only for short strings
            max_len = max(len(target_clean), len(api_clean))
            if max_len > 0:
                edit_dist = self.levenshtein_distance(target_clean, api_clean)
                similarity = 1 - (edit_dist / max_len)
                score = similarity * 60  # 0-60%
                scores.append(('edit_distance', score))

        # Return the highest score
        if scores:
            best_method, best_score = max(scores, key=lambda x: x[1])
            logger.debug(f"Model match '{target}' vs '{api}': {best_score:.1f}% (method: {best_method})")
            return best_score

        return 0.0

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def find_best_model_match(self, results: List[Dict], target_model: str) -> Optional[Dict]:
        """Find best model match using advanced fuzzy matching."""
        if not results:
            return None

        # Ensure results is a list and contains dictionaries
        if not isinstance(results, list):
            logger.warning(f"Expected list of results, got {type(results)}")
            return None

        best_match = None
        best_score = 0.0
        all_matches = []

        for result in results:
            # Skip if result is not a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Expected dict result, got {type(result)}: {result}")
                continue

            # Get model name safely
            api_model = result.get('model', '')
            if not api_model:
                continue

            # Calculate similarity score
            score = self.advanced_model_similarity(target_model, str(api_model))

            if score > 0:
                all_matches.append((score, api_model, result))

                if score > best_score:
                    best_score = score
                    best_match = result

        # Log all matches for debugging
        if all_matches:
            logger.info(f"Model matching results for '{target_model}':")
            sorted_matches = sorted(all_matches, key=lambda x: x[0], reverse=True)
            for score, model, _ in sorted_matches[:5]:  # Show top 5
                logger.info(f"  {score:.1f}%: '{model}'")

        # Use more lenient threshold - accept matches >= 60%
        if best_match and best_score >= 60.0:
            logger.info(f"✅ Selected match: '{best_match.get('model', 'N/A')}' (score: {best_score:.1f}%)")
            return best_match
        elif best_match:
            logger.info(
                f"❌ Best match below threshold: '{best_match.get('model', 'N/A')}' (score: {best_score:.1f}% < 60%)")
        else:
            logger.info(f"❌ No model matches found for '{target_model}'")

        return None

    def search_by_year_make_model(self, year: int, make: str, model: str) -> Dict:
        """Search by year, make, model using improved fuzzy matching."""
        vehicle_type = self.determine_vehicle_type(make)

        # Get all manufacturer name variations
        make_variations = self.normalize_manufacturer_name(make)

        logger.info(f"Searching for {vehicle_type}: {make} {year} {model}")
        logger.info(f"Manufacturer variations: {make_variations}")

        # Try multiple vehicle type classifications for edge cases
        vehicle_types = [vehicle_type]
        if vehicle_type == 'Scooter':
            vehicle_types.extend(['Motorcycle'])
        elif vehicle_type == 'Motorcycle':
            vehicle_types.extend(['Scooter'])
        elif make.upper() in ['POLARIS', 'CAN-AM', 'CANAM']:
            vehicle_types.extend(['UTV', 'Side-by-Side'])

        # Strategy 1: Building blocks approach with all variations
        for v_type in vehicle_types:
            for make_var in make_variations:
                logger.info(f"Trying building blocks: {v_type} - {make_var} {year}")

                try:
                    # Get models for this year/make
                    endpoint = f'/v1/type/{v_type}/manu/{make_var}/year/{year}/model'
                    success, models_data = self.api_request(endpoint, 'GET')

                    if success and models_data:
                        logger.info(
                            f"Found {len(models_data) if isinstance(models_data, list) else 1} models for {make_var} {year}")

                        # Find best model match using advanced fuzzy matching
                        best_model_name = None
                        best_score = 0.0

                        if isinstance(models_data, list):
                            for model_item in models_data:
                                if isinstance(model_item, str):
                                    score = self.advanced_model_similarity(model, model_item)
                                    logger.debug(f"Model score: '{model_item}' = {score:.1f}%")

                                    if score > best_score:
                                        best_score = score
                                        best_model_name = model_item
                                elif isinstance(model_item, dict) and 'model' in model_item:
                                    # Already a full record
                                    score = self.advanced_model_similarity(model, model_item['model'])
                                    if score >= 60.0:  # Lower threshold for full records
                                        logger.info(
                                            f"Found full record match: {model_item['model']} (score: {score:.1f}%)")
                                        return {'success': True, 'data': model_item,
                                                'method': f'BUILDING_BLOCKS_FULL_{v_type}'}

                        # If we found a good model match, get the full record
                        if best_model_name and best_score >= 60.0:  # Lowered threshold from 50% to 60%
                            logger.info(f"Best model match: '{best_model_name}' (score: {best_score:.1f}%)")

                            # Get full record
                            full_endpoint = f'/v1/type/{v_type}/manu/{make_var}/year/{year}/model/{best_model_name}/go'
                            logger.info(f"Getting full record: {full_endpoint}")

                            success, full_data = self.api_request(full_endpoint, 'GET')

                            if success and full_data:
                                logger.info(f"✅ Got full record for {best_model_name}")
                                # Handle if it returns a list
                                if isinstance(full_data, list) and len(full_data) > 0:
                                    full_data = full_data[0]

                                return {'success': True, 'data': full_data,
                                        'method': f'BUILDING_BLOCKS_{v_type}_{make_var}'}
                            else:
                                logger.info(f"Failed to get full record for {best_model_name}")
                        else:
                            if best_model_name:
                                logger.info(
                                    f"Best match below threshold: '{best_model_name}' (score: {best_score:.1f}% < 60%)")
                            else:
                                logger.info(f"No model matches found for '{model}'")
                    else:
                        logger.debug(f"No models found for {v_type} - {make_var} {year}")

                except Exception as e:
                    logger.warning(f"Building blocks failed for {v_type} - {make_var}: {str(e)}")

        # Strategy 2: Try even more vehicle types for edge cases
        extended_types = []
        make_upper = make.upper()

        if 'POLARIS' in make_upper:
            extended_types = ['Snowmobile', 'Watercraft']
        elif any(x in make_upper for x in ['YAMAHA', 'KAWASAKI', 'HONDA']):
            extended_types = ['ATV', 'Watercraft', 'Snowmobile']
        elif 'BMW' in make_upper:
            extended_types = ['ATV']

        for ext_type in extended_types:
            for make_var in make_variations[:2]:  # Only try first 2 variations for extended types
                logger.info(f"Trying extended type: {ext_type} - {make_var}")
                try:
                    endpoint = f'/v1/type/{ext_type}/manu/{make_var}/year/{year}/model'
                    success, models_data = self.api_request(endpoint, 'GET')

                    if success and models_data and isinstance(models_data, list):
                        for model_item in models_data:
                            if isinstance(model_item, str):
                                score = self.advanced_model_similarity(model, model_item)
                                if score >= 70.0:  # Higher threshold for extended types
                                    # Try to get full record
                                    full_endpoint = f'/v1/type/{ext_type}/manu/{make_var}/year/{year}/model/{model_item}/go'
                                    success, full_data = self.api_request(full_endpoint, 'GET')

                                    if success and full_data:
                                        if isinstance(full_data, list) and len(full_data) > 0:
                                            full_data = full_data[0]
                                        logger.info(
                                            f"✅ Found via extended type {ext_type}: {model_item} (score: {score:.1f}%)")
                                        return {'success': True, 'data': full_data,
                                                'method': f'EXTENDED_{ext_type}_{make_var}'}
                except Exception as e:
                    logger.debug(f"Extended type {ext_type} failed: {str(e)}")

        return {'success': False,
                'error': f'No matching vehicles found for {make} {year} {model} (tried all variations)',
                'method': 'SEARCH_FAILED'}

    def calculate_model_similarity(self, target_model: str, api_model: str) -> float:
        """Wrapper for backward compatibility."""
        return self.advanced_model_similarity(target_model, api_model)

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
    """Example usage of the CanadianBlueBookPriceFetcher with improved fuzzy matching."""
    API_KEY = '144|inuYwvGhZePXzVmR3ZbRzH6dsRXs5UHYPtmvv9uX'

    # Reduced debug mode - only show important matching info
    DEBUG_MODE = False  # Set to True for detailed API debugging

    if DEBUG_MODE:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Debug mode enabled - will show detailed API responses")

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
        print("🧠 IMPROVED FUZZY MATCHING:")
        print("   ✅ Advanced similarity scoring (8 different algorithms)")
        print("   ✅ Manufacturer name normalization (CANAM ↔ CAN-AM)")
        print("   ✅ Prefix matching (FLHTKL matches 'FLHTKL Electra glide...')")
        print("   ✅ Model code variations (KLX110 matches KLX110R)")
        print("   ✅ Case-insensitive matching")
        print("   ✅ Lowered threshold to 60% for better coverage")
        print("")

        results_df = fetcher.process_excel_file(
            input_filename='bluebooktest.xlsx',
            output_filename='cbb_wholesale_prices_improved.xlsx'
        )

        print(f"\n✅ Processing complete!")
        print(f"📁 Results saved to: cbb_wholesale_prices_improved.xlsx")
        print(f"📊 Processed {len(results_df)} vehicles")
        print("")
        print("🎯 EXPECTED IMPROVEMENTS:")
        print("   • 2015 Harley Davidson FLHTKL → Should find 'FLHTKL Electra glide ultra LTD low ABS 103 LC'")
        print("   • 2022 Kawasaki KLX110 → Should find 'KLX110R' or 'KLX 110CNF KLX110R'")
        print("   • 2021 Can-Am Spyder RT Limited → Should find 'Spyder RT Limited SE6'")
        print("   • Better success rate overall with improved matching")

    except FileNotFoundError:
        print("❌ Error: bluebooktest.xlsx not found")
        print("Please ensure the input Excel file is in the same directory")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.exception("Full error details:")


if __name__ == "__main__":
    main()