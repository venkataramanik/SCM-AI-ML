import random
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ======================================================================
# CORE TMS SIMULATOR (EXPANDED END-TO-END)
# Implements 5 Advanced Rating Cases + Execution Cycle
# ======================================================================

class TMS:
    """
    Simulates the core procure-to-pay and execution cycle of a TMS, 
    including advanced rating for Parcel, LTL, Multistop TL, Hub-Spoke, and Multimodal.
    """
    def __init__(self):
        """
        #### HIGHLIGHT: Core Initialization & Advanced Rate Tables ####
        Initializes databases and adds complex data structures for real-world rating:
        Zones (for Parcel), Classes (for LTL), Hubs (for network optimization), and Multimodal costs.
        """
        # --- Internal Databases ---
        self.orders = {}
        self.loads = {}
        self.invoices = {}
        self.next_order_id = 1
        self.next_load_id = 1
        self.next_invoice_id = 1
        
        # --- Geospatial and Carrier Data ---
        self.carriers = {
            'SwiftLogistics': {'base_ftr_rate_per_km': 0.75, 'capacity': 100},
            'ApexFreight': {'base_ftr_rate_per_km': 0.90, 'capacity': 80},
            'GlobalHaulers': {'base_ftr_rate_per_km': 0.65, 'capacity': 120},
        }
        self.locations = {
            'NYC': (40.7128, -74.0060), 'CHI': (41.8781, -87.6298), # Hub
            'LA': (34.0522, -118.2437), 'DAL': (32.7767, -96.7970),
            'MIA': (25.7617, -80.1918), 'SEA': (47.6062, -122.3321),
            'Port_LongBeach': (33.75, -118.27) # For Multimodal Ocean
        }
        self.hubs = {'CHI'} # Central Hub for Hub & Spoke analysis

        # --- Case 1: Parcel Rating (Zone-based) ---
        # Look-up table based on weight (kg) and distance zone
        self.parcel_rates = {
            1: {1: 10, 5: 18, 10: 25}, # Zone 1 (Local)
            3: {1: 15, 5: 25, 10: 35}, # Zone 3 (Regional)
            8: {1: 25, 5: 40, 10: 60}  # Zone 8 (Cross-country)
        }
        # Simplified Zone Mapping: NYC/SEA to LA is 8. NYC to DAL/CHI is 3. Same region is 1.
        self._zone_map = {('NYC', 'LA'): 8, ('NYC', 'CHI'): 3, ('SEA', 'LA'): 8, ('NYC', 'DAL'): 3, ('LA', 'LA'): 1}

        # --- Case 2: LTL Rating (Freight Class/Weight) ---
        # Tariff structure: (Freight Class: Base Rate per 100kg)
        self.ltl_tariff = {
            50: 150.00,  # Easy freight (e.g., densely packed items)
            100: 250.00, # Standard freight
            250: 450.00  # High-value/bulky/difficult freight
        }
        self.ltl_discount = 0.65 # 35% off LTL base rate

        # --- Case 5: Multimodal Transit Data ---
        # Time and Cost penalty for mode transfers
        self.multimodal_factors = {
            'TRUCK-RAIL': {'cost_per_transfer': 500, 'time_penalty_days': 2},
            'TRUCK-OCEAN': {'cost_per_transfer': 1500, 'time_penalty_days': 10},
        }

    # --- UTILITY FUNCTIONS ---
    
    def _calculate_haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculates the **great-circle distance (in km)** between two points 
        (as-the-crow-flies). Used as the reliable fallback.
        """
        R = 6371 
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _get_api_distance(self, origin, destination):
        """
        #### HIGHLIGHT: Mocked API Call for Real-World Distance ####
        Mocks a call to a geospatial routing API (like Google Maps) to get 
        more accurate driving distance, which is typically longer than Haversine.
        """
        try:
            o_coords = self.locations[origin]
            d_coords = self.locations[destination]
        except KeyError:
            # Handle case where location isn't recognized
            return None 

        haversine_dist = self._calculate_haversine(o_coords[0], o_coords[1], d_coords[0], d_coords[1])
        
        # Simulates real-world road miles: 10-20% longer than straight-line distance
        road_factor = random.uniform(1.10, 1.20)
        api_distance_km = haversine_dist * road_factor
        
        return api_distance_km

    def _calculate_distance(self, origin, destination):
        """
        Primary distance calculation function. Prefers API distance, falls back to Haversine.
        """
        # Attempt to get the accurate (mocked) API distance
        distance = self._get_api_distance(origin, destination)

        if distance is None:
            # Fallback to the reliable Haversine calculation
            o_coords = self.locations[origin]
            d_coords = self.locations[destination]
            distance = self._calculate_haversine(o_coords[0], o_coords[1], d_coords[0], d_coords[1])
            print(f"      [Distance Warning] Used Haversine for {origin} to {destination} distance.")
        
        return distance

    def _calculate_transit_time(self, distance_km, mode='FTL'):
        """
        Estimates shipment duration (ETA) based on distance and mode.
        """
        speed_map = {'FTL': 800, 'LTL': 600, 'PARCEL': 1000, 'RAIL': 500, 'OCEAN': 200}
        days = distance_km / speed_map.get(mode, 800)
        return max(1, round(days)) 

    def display_network_and_load_path(self, load_id):
        """
        Displays a structured, text-based map of the network locations 
        and visualizes the path of a specific load, highlighting hubs.
        """
        if load_id not in self.loads:
            print(f"Error: Load {load_id} not found for visualization.")
            return

        load = self.loads[load_id]
        origin = load['origin']
        destination = load['destination']
        mode = load['mode']
        
        # 1. Network Overview
        print("\n\n--- Network Map Overview ---")
        for loc, coords in self.locations.items():
            is_hub = " [HUB]" if loc in self.hubs else ""
            print(f"Location: {loc}{is_hub} ({coords[0]:.2f}, {coords[1]:.2f})")
        
        # 2. Path Visualization
        path = [origin]
        path_description = f"Path for Load {load_id} ({mode}):\n"

        if mode == 'HUB_SPOKE' and self.hubs:
            hub = list(self.hubs)[0]
            path.append(hub)
            path_description += f"  {origin} (Spoke) -> {hub} (HUB) -> {destination} (Spoke)\n"
        elif mode == 'MULTISTOP' and load.get('stop_locations'):
            stops = load['stop_locations']
            path.extend(stops)
            path_description += f"  {origin} (O) -> " + " -> ".join(stops) + f" (Stops) -> {destination} (D)\n"
        elif mode == 'MULTIMODAL':
            # Simplified path through the implicit transload point (CHI)
            path.append('CHI')
            path_description += f"  {origin} (Truck) -> CHI (Transload/Rail) -> {destination} (Destination)\n"
        else:
            path_description += f"  {origin} -> {destination} (Direct Route)\n"
        
        path.append(destination)
        
        # 3. Simple Visual Path Diagram
        diagram = ""
        
        for i, loc in enumerate(path):
            is_start_end = "(O)" if i == 0 else ("(D)" if i == len(path) - 1 else "")
            is_hub_mark = "*" if loc in self.hubs else " "
            
            diagram += f" {is_hub_mark}--- {loc}{is_start_end}\n"
            if i < len(path) - 1:
                # Add a vertical connector line
                arrow = "|" if load['mode'] in ['MULTIMODAL', 'HUB_SPOKE'] else "v"
                diagram += f"  {arrow}\n"
        
        print("\n--- Visual Path Diagram ---")
        print(diagram)
        print("---------------------------")


    # --- RATING FUNCTIONS (Cases 1, 2, 3, 4, 5) ---

    def _get_zone(self, origin, destination):
        """Helper for Case 1: Determines the parcel rating zone."""
        key = tuple(sorted((origin, destination))) # Sort to handle O/D independently
        return self._zone_map.get((origin, destination), 5) # Default to Zone 5

    def rate_parcel(self, origin, destination, weight_kg):
        """
        #### Case 1: Parcel Rating based on Zone ####
        Looks up cost based on destination zone and weight tiers (kg).
        """
        zone = self._get_zone(origin, destination)
        
        # Find the highest weight tier the shipment fits into
        tier_weights = sorted([w for w in self.parcel_rates.get(zone, {}).keys() if w >= weight_kg])
        
        if tier_weights:
            rate = self.parcel_rates[zone][tier_weights[0]]
            return rate
        
        return 999.00 # High default cost if weight is too large

    def rate_ltl(self, load_weight, freight_class=100):
        """
        #### Case 2: LTL Rating based on Freight Class and Discount Tariff ####
        Calculates cost using NMFC-like freight class (density/stowability) and applies 
        a negotiated discount to the published tariff.
        """
        base_rate_per_100kg = self.ltl_tariff.get(freight_class, self.ltl_tariff[100])
        total_rate = (load_weight / 100) * base_rate_per_100kg
        final_cost = total_rate * (1 - self.ltl_discount)
        
        # Add a fixed linehaul component based on distance (simplified)
        linehaul_add = 0.50 * self._calculate_distance('NYC', 'LA') # Use the primary distance function
        
        return round(final_cost + linehaul_add, 2)


    def rate_tl_multistop(self, load, stop_locations):
        """
        #### Case 3: TL Consolidation with Stops (Multi-stop Routing) ####
        Calculates the Full Truckload (FTL) cost plus a fixed fee for each intermediate 
        stop-off, simulating Milk-Run or dedicated delivery routes.
        """
        if not stop_locations:
            # If no stops, fall back to simple FTL linehaul
            return self.rate_ftl(load)

        # Base FTL linehaul cost (O to D distance)
        base_cost = self.rate_ftl(load)
        
        # Cost per intermediate stop (fixed fee)
        stop_off_fee = 250.00
        total_stop_fee = len(stop_locations) * stop_off_fee
        
        # In a real TMS, this would calculate actual mileage between O -> Stop1 -> Stop2 -> D
        # Here, we only add the stop fee.
        
        return round(base_cost + total_stop_fee, 2)

    def rate_hub_spoke(self, load, hub='CHI'):
        """
        #### Case 4: Hub and Spoke Network Rating ####
        Calculates the cost of routing a shipment from Origin to a central Hub, and 
        then from the Hub to the Destination, using a slightly cheaper Hub-to-Hub rate.
        """
        origin = load['origin']
        destination = load['destination']
        
        if hub not in self.locations:
            return self.rate_ftl(load) # Fallback

        # Use the primary distance function for both legs
        dist_o_to_h = self._calculate_distance(origin, hub)
        dist_h_to_d = self._calculate_distance(hub, destination)
        total_distance = dist_o_to_h + dist_h_to_d

        # Assume Hub-to-Hub/Spoke-to-Hub leg is 10% cheaper per km (better network efficiency)
        hub_rate = self.carriers['SwiftLogistics']['base_ftr_rate_per_km'] * 0.90 
        
        cost = total_distance * hub_rate + random.uniform(50, 100)
        
        return round(cost, 2)

    def rate_multimodal(self, load, mode_sequence=['TRUCK', 'RAIL', 'TRUCK']):
        """
        #### Case 5: Multimodal Shipments (Intermodal) ####
        Calculates the combined cost and transit time of using multiple modes (e.g., Truck/Rail) 
        and adds penalties for transloading (hand-off) between modes.
        """
        total_cost = 0
        total_days = 0
        
        # Simple simulation: Assume NYC to LA is split between Truck and Rail at CHI Hub
        # Use the primary distance function for both legs
        dist_truck = self._calculate_distance(load['origin'], 'CHI')
        dist_rail = self._calculate_distance('CHI', load['destination'])
        
        # Truck Cost/Time
        total_cost += dist_truck * self.carriers['SwiftLogistics']['base_ftr_rate_per_km']
        total_days += self._calculate_transit_time(dist_truck, 'FTL')

        # Rail Cost/Time
        rail_rate_per_km = 0.35 # Cheaper per km
        total_cost += dist_rail * rail_rate_per_km
        total_days += self._calculate_transit_time(dist_rail, 'RAIL')
        
        # Add Transload Penalty (Truck to Rail hand-off)
        penalty = self.multimodal_factors['TRUCK-RAIL']
        total_cost += penalty['cost_per_transfer']
        total_days += penalty['time_penalty_days']
        
        load['status'] = 'MULTIMODAL_PLANNED'
        load['estimated_delivery'] = datetime.now().date() + timedelta(days=total_days)
        
        print(f"      Multimodal ETA: {total_days} days (Truck + Rail + Transfer).")
        return round(total_cost, 2)

    def rate_ftl(self, load):
        """Standard FTL rating (baseline for comparison)."""
        distance = load['distance_km']
        base_rate = self.carriers['GlobalHaulers']['base_ftr_rate_per_km'] # Lowest base rate
        surcharge = (load['total_weight_kg'] * 0.05) 
        cost = (base_rate * distance) + surcharge + random.uniform(50, 100)
        return round(cost, 2)

    # --- 1. ORDER MANAGEMENT AND LOAD CONSOLIDATION ---

    def create_order(self, origin, destination, weight_kg, volume_m3, requested_mode='FTL', freight_class=None):
        """
        #### HIGHLIGHT: Captures New Customer Requests (Mode Aware) ####
        Generates order, accepts requested transport mode, and calculates distance/ETA.
        """
        if origin not in self.locations or destination not in self.locations:
            print(f"Error: City {origin} or {destination} not recognized.")
            return None
            
        order_id = f"TMS-O{self.next_order_id:04d}"
        
        # Calculate distance using the new primary function (API preferred)
        distance_km = self._calculate_distance(origin, destination)
        
        # Initial ETA based on requested mode (e.g., Parcel is faster than FTL)
        transit_days = self._calculate_transit_time(distance_km, requested_mode)

        self.orders[order_id] = {
            'id': order_id, 'origin': origin, 'destination': destination, 
            'weight_kg': weight_kg, 'volume_m3': volume_m3, 'distance_km': round(distance_km, 2),
            'status': 'NEW_ORDER', 'estimated_delivery': datetime.now().date() + timedelta(days=transit_days),
            'load_id': None, 'requested_mode': requested_mode, 'freight_class': freight_class
        }
        self.next_order_id += 1
        return order_id

    def consolidate_orders_into_load(self, order_ids, stop_locations=[]):
        """
        #### HIGHLIGHT: Crucial TMS Optimization Step ####
        Groups orders into a Load. Now accepts optional `stop_locations` 
        for Case 3 (Multistop TL).
        """
        if not order_ids: return None

        first_order = self.orders.get(order_ids[0])
        if not first_order: return None
        
        # Basic aggregation
        total_weight = sum(self.orders[oid]['weight_kg'] for oid in order_ids)
        total_volume = sum(self.orders[oid]['volume_m3'] for oid in order_ids)
        
        # Determine the primary mode and class for the load
        primary_mode = first_order['requested_mode']
        primary_class = first_order['freight_class']

        load_id = f"TMS-L{self.next_load_id:04d}"
        
        load_details = {
            'id': load_id, 'orders': order_ids, 'origin': first_order['origin'], 'destination': first_order['destination'],
            'total_weight_kg': total_weight, 'total_volume_m3': total_volume, 
            'distance_km': first_order['distance_km'], 'mode': primary_mode, 
            'freight_class': primary_class, 'stop_locations': stop_locations,
            'planned_cost': None, 'actual_cost': None,
            'carrier': None, 'status': 'CONSOLIDATED',
            'delivery_date_actual': None,
            'estimated_delivery': first_order['estimated_delivery']
        }
        
        self.loads[load_id] = load_details
        
        for oid in order_ids:
            self.orders[oid]['load_id'] = load_id
            self.orders[oid]['status'] = 'CONSOLIDATED'

        self.next_load_id += 1
        print(f"\n[+] Load {load_id} created, mode: {primary_mode}, stops: {len(stop_locations)}.")
        return load_id

    # --- 2. TENDERING AND CARRIER SELECTION ---
    
    def tender_load(self, load_id):
        """
        #### HIGHLIGHT: Tendering and Dynamic Rating ####
        Selects the correct rating function based on the load's requested `mode`. 
        This is where the TMS logic determines the cost profile.
        """
        if load_id not in self.loads or self.loads[load_id]['carrier'] is not None:
            print(f"Warning: Load {load_id} not found or already tendered.")
            return

        load = self.loads[load_id]
        mode = load['mode']
        
        # 1. Determine Planned Cost based on mode
        if mode == 'PARCEL':
            planned_cost = self.rate_parcel(load['origin'], load['destination'], load['total_weight_kg'])
            best_carrier = 'FedEx/UPS_Simulator' # Parcel is typically single-source
        elif mode == 'LTL':
            planned_cost = self.rate_ltl(load['total_weight_kg'], load['freight_class'])
            best_carrier = 'LTL_Carrier_Sim' 
        elif mode == 'MULTISTOP': # Case 3
            planned_cost = self.rate_tl_multistop(load, load['stop_locations'])
            best_carrier = 'ApexFreight'
        elif mode == 'HUB_SPOKE': # Case 4
            planned_cost = self.rate_hub_spoke(load, hub='CHI')
            best_carrier = 'GlobalHaulers'
        elif mode == 'MULTIMODAL': # Case 5
            planned_cost = self.rate_multimodal(load)
            best_carrier = 'Intermodal_Partner'
        else: # Standard FTL
            planned_cost = self.rate_ftl(load)
            best_carrier = 'SwiftLogistics' # Simple FTL default

        # 2. Update Load Status
        load['carrier'] = best_carrier
        load['planned_cost'] = planned_cost
        load['status'] = 'TENDERED_ACCEPTED'
        
        self.loads[load_id] = load
        
        for oid in load['orders']:
            self.orders[oid]['status'] = 'TENDERED_ACCEPTED'
        
        print(f"[!] Load {load_id} ({mode}) Tendered and Accepted by {best_carrier}.")
        print(f"    Planned Cost: ${planned_cost:,.2f}")

    # --- 3. SHIPMENT EVENTS AND TRACKING ---

    def process_shipment_events(self, load_id, event_type):
        """
        #### HIGHLIGHT: Real-Time Execution and Tracking ####
        Simulates dynamic tracking updates (e.g., from EDI feeds or driver apps). 
        """
        if load_id not in self.loads:
            print(f"Error: Load {load_id} not found.")
            return

        load = self.loads[load_id]
        
        if event_type == 'PICKED_UP':
            load['status'] = 'IN_TRANSIT'
        elif event_type == 'DELIVERED':
            load['status'] = 'DELIVERED'
            load['delivery_date_actual'] = datetime.now().date()
        elif event_type == 'DELAYED':
            load['status'] = 'DELAYED'
        
        for oid in load['orders']:
            self.orders[oid]['status'] = load['status']
        
        print(f"[#] Event: Load {load_id} status updated to {load['status']} by {load['carrier']}.")


    # --- 4 & 5. FREIGHT INVOICE MATCH AND PAY ---

    def generate_freight_invoice(self, load_id):
        """
        #### HIGHLIGHT: Invoice Creation with Variance ####
        Creates the financial document from the carrier. Generates an 
        `invoiced_amount` with a random variance compared to the `planned_cost`.
        """
        if load_id not in self.loads or self.loads[load_id]['status'] != 'DELIVERED':
            print(f"Error: Invoice can only be generated for delivered load {load_id}.")
            return None
        
        load = self.loads[load_id]
        planned_cost = load['planned_cost']
        
        # Simulate invoice variance (e.g., accessorial fees, detention)
        variance_factor = random.uniform(-0.01, 0.08) # -1% to +8% variance for complex loads
        invoiced_amount = planned_cost * (1 + variance_factor)
        
        invoice_id = f"INV-{self.next_invoice_id:04d}"
        
        invoice = {
            'id': invoice_id, 'load_id': load_id, 'carrier': load['carrier'],
            'planned_cost': round(planned_cost, 2), 'invoiced_amount': round(invoiced_amount, 2),
            'variance_amount': round(invoiced_amount - planned_cost, 2), 'status': 'PENDING_MATCH'
        }
        
        self.invoices[invoice_id] = invoice
        self.loads[load_id]['actual_cost'] = round(invoiced_amount, 2)
        self.next_invoice_id += 1
        
        print(f"\n[$$] Invoice {invoice_id} generated for Load {load_id} (Mode: {load['mode']}).")
        return invoice_id
    
    def match_and_pay_invoice(self, invoice_id):
        """
        #### HIGHLIGHT: Financial Audit (Match & Pay) ####
        Compares the `invoiced_amount` against the `planned_cost`. If variance is
        outside the 3% tolerance, it's flagged for **`AUDIT_REQUIRED`**.
        """
        if invoice_id not in self.invoices:
            print(f"Error: Invoice {invoice_id} not found.")
            return

        invoice = self.invoices[invoice_id]
        
        # Need to protect division by zero here if planned_cost was 0, but it should be > 0
        if invoice['planned_cost'] == 0:
             variance_pc = 0.0
        else:
             variance_pc = (invoice['variance_amount'] / invoice['planned_cost']) * 100
        
        print(f"\n[PAY] Processing Invoice {invoice_id}...")
        print(f"      Variance: ${invoice['variance_amount']:,.2f} ({variance_pc:+.2f}%)")
        
        # Financial rule: auto-pay if variance is less than 3%
        if abs(variance_pc) <= 3:
            invoice['status'] = 'PAID_AUTO_MATCH'
            print("      Action: AUTO-PAYMENT processed.")
        else:
            invoice['status'] = 'AUDIT_REQUIRED'
            print("      Action: AUDIT REQUIRED (Variance exceeds 3% threshold).")
            
        self.invoices[invoice_id] = invoice


    # --- 6. TMS ANALYTICS ---

    def generate_analytics(self):
        """
        #### HIGHLIGHT: Business Intelligence (KPIs) ####
        Produces key performance indicators (KPIs) for OTD and Financial Performance.
        """
        print("\n==========================================================")
        print("                   TMS PERFORMANCE ANALYTICS")
        print("==========================================================")

        # A. On-Time Delivery (OTD) Rate
        delivered_loads = [l for l in self.loads.values() if l['status'] == 'DELIVERED']
        if delivered_loads:
            # Randomly select if a load was delayed or not
            otd_count = sum(1 for load in delivered_loads if load['status'] != 'DELAYED' and random.choice([True, True, False]))
            otd_rate = (otd_count / len(delivered_loads)) * 100
            print(f"1. On-Time Delivery (OTD) Rate: {otd_rate:.2f}%")
        else:
            print("1. OTD Rate: No loads delivered yet.")
        
        # B. Cost Variance Analysis (CVA)
        df_invoices = pd.DataFrame(self.invoices.values())
        if not df_invoices.empty:
            total_planned_cost = df_invoices['planned_cost'].sum()
            total_invoiced_cost = df_invoices['invoiced_amount'].sum()
            total_variance = total_invoiced_cost - total_planned_cost
            
            # Safety Check: Prevent division by zero if no costs were planned
            if total_planned_cost != 0:
                variance_pc = (total_variance / total_planned_cost) * 100
            else:
                variance_pc = 0.0
            
            # Round variables for clean display and robustness
            total_planned_cost_safe = round(total_planned_cost, 2)
            total_variance_safe = round(total_variance, 2)
            variance_pc_safe = round(variance_pc, 2)

            # NOTE: Using only the basic '.2f' or '+.2f' formatting to avoid the locale issues
            # caused by the non-standard '+, .2f' specifier.
            print(f"2. Total Planned Cost: ${total_planned_cost_safe:,.2f}")
            
            # FIX applied here: Changed problematic format specifier to the safe '+.2f'
            print(f"3. Net Cost Variance: ${total_variance_safe:+.2f} ({variance_pc_safe:+.2f}%)")
        else:
            print("2-3. Cost Variance Analysis: No invoices processed yet.")

        # C. Detailed Load Status Summary
        status_counts = pd.Series([load['status'] for load in self.loads.values()]).value_counts()
        print("\n4. Load Status Breakdown:")
        print(status_counts.to_string())
        print("==========================================================")


# ======================================================================
# EXECUTION DEMONSTRATION: 5 Advanced Cases
# ======================================================================

if __name__ == "__main__":
    
    tms_system = TMS()
    print("TMS System Initialized. Running 5 Advanced Rating Case Studies.")
    print("------------------------------------------------------------------")
    
    # 1. PARCEL RATING (Zone-based) - NYC to LA (Zone 8, Light Weight)
    print("\n--- CASE 1: Parcel Rating (Zone-Based) ---")
    o_parcel = tms_system.create_order('NYC', 'LA', 7.5, 0.5, requested_mode='PARCEL')
    l_parcel = tms_system.consolidate_orders_into_load([o_parcel])
    tms_system.tender_load(l_parcel)
    
    # 2. LTL RATING (Freight Class) - SEA to DAL (Heavy, Class 250)
    print("\n--- CASE 2: LTL Rating (Class/Tariff) ---")
    o_ltl_heavy = tms_system.create_order('SEA', 'DAL', 500, 5.0, requested_mode='LTL', freight_class=250)
    l_ltl = tms_system.consolidate_orders_into_load([o_ltl_heavy])
    tms_system.tender_load(l_ltl)

    # 3. TL CONSOLIDATION WITH STOPS (Multi-Stop FTL) - NYC to DAL with a CHI Stop
    print("\n--- CASE 3: TL Multi-Stop Rating ---")
    o_stop_a = tms_system.create_order('NYC', 'DAL', 1000, 20.0, requested_mode='MULTISTOP')
    o_stop_b = tms_system.create_order('NYC', 'DAL', 500, 10.0, requested_mode='MULTISTOP')
    # Consolidate, defining an intermediate stop in Chicago (CHI)
    l_multistop = tms_system.consolidate_orders_into_load([o_stop_a, o_stop_b], stop_locations=['CHI'])
    tms_system.tender_load(l_multistop)
    
    # 4. HUB AND SPOKE RATING - LA to MIA (routed via CHI Hub)
    print("\n--- CASE 4: Hub and Spoke Rating ---")
    o_hub_spoke = tms_system.create_order('LA', 'MIA', 1200, 25.0, requested_mode='HUB_SPOKE')
    l_hub_spoke = tms_system.consolidate_orders_into_load([o_hub_spoke])
    tms_system.tender_load(l_hub_spoke)
    
    # 5. MULTIMODAL SHIPMENT (TRUCK + RAIL) - NYC to LA
    print("\n--- CASE 5: Multimodal (Truck/Rail) Rating ---")
    o_multi = tms_system.create_order('NYC', 'LA', 2000, 35.0, requested_mode='MULTIMODAL')
    l_multi = tms_system.consolidate_orders_into_load([o_multi])
    tms_system.tender_load(l_multi)
    
    # --- VISUALIZATION ---
    # Displaying the network and the route for the Hub & Spoke load
    tms_system.display_network_and_load_path(l_hub_spoke)
    
    # --- Execute and Finalize All Shipments ---
    
    all_loads = list(tms_system.loads.keys())
    print("\n--- Execution & Financial Settlement ---")
    
    for load_id in all_loads:
        # Simulate pick up, transit, and delivery (with a forced delay on the multimodal load)
        tms_system.process_shipment_events(load_id, 'PICKED_UP')
        if load_id == l_multi:
            tms_system.process_shipment_events(load_id, 'DELAYED') # Multimodal delay due to transload
        tms_system.process_shipment_events(load_id, 'DELIVERED')
        
        # Generate Invoice
        inv_id = tms_system.generate_freight_invoice(load_id)
        
        # Match and Pay
        tms_system.match_and_pay_invoice(inv_id)

    # 7. Analytics and Reporting
    print("\n--- Final Reporting ---")
    tms_system.generate_analytics()
    
    print("\n--- All 5 Advanced Rating Cases Simulated Successfully ---")
