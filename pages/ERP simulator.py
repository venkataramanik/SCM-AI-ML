"""
### ERP SIMULATOR BLURB: END-TO-END MANUFACTURING CYCLE

This single-file Python program simulates the core **Enterprise Resource Planning (ERP)** cycle for a small manufacturing company making custom wooden chairs.

It covers the complete procure-to-pay and order-to-cash process:

1.  **Sales Order (SO):** Creates customer demand.
2.  **MRP (Planning):** Checks stock, calculates material shortages based on the **BOM**.
3.  **Procurement (PO & Receipt):** Creates and executes **Purchase Orders** for raw materials, **increasing inventory** and logging the supplier **EXPENSE**.
4.  **Manufacturing:** **Consumes raw materials**, incurs **labor costs**, and creates **Finished Goods (FG)** stock.
5.  **Shipping & Customer Invoice:** **Reduces FG stock**, logs the unit cost as **COGS**, and recognizes the sale as **REVENUE**.
6.  **Financials:** Reconciles the entire ledger to generate the final **Profit & Loss (P&L) Report**.
"""
import uuid
import time
from datetime import datetime
from collections import defaultdict
import pandas as pd

# ====================================================================
# MANUFACTURING ERP SIMULATOR (Five Core Modules)
# ====================================================================

class ERP_Simulator:
    """
    Simulates the core functions of an ERP system for a small company 
    manufacturing a single product: the 'Custom Wooden Chair'.
    
    Modules simulated: Sales, Inventory (WMS), Production Planning (MRP), 
    Procurement (MM), and Financials (FI).
    """
    
    def __init__(self):
        """Initializes all data structures, including BOM and starting inventory."""
        self.inventory = {
            'FG-CHAIR': {'type': 'Finished Goods', 'stock': 10, 'uom': 'EA', 'cost': 120.00},
            'RM-WOOD': {'type': 'Raw Material', 'stock': 100, 'uom': 'PLANK', 'cost': 10.00},
            'RM-SCREW': {'type': 'Raw Material', 'stock': 500, 'uom': 'UNIT', 'cost': 0.10}
        }
        
        # Bill of Materials (BOM): What it takes to make one finished product
        # 1 FG-CHAIR requires: 5 Wood Planks and 20 Screws
        self.bom = {
            'FG-CHAIR': [
                {'material_id': 'RM-WOOD', 'quantity': 5},
                {'material_id': 'RM-SCREW', 'quantity': 20}
            ]
        }
        
        # Transactional Records
        self.sales_orders = []       # Customer demand
        self.purchase_orders = []    # Supplier orders
        self.production_orders = []  # Manufacturing runs
        
        # Financial Ledger (Simple double-entry concept for P&L)
        self.ledger = [] 
        
        self.labor_rate_per_chair = 50.00
        self.sales_price_per_chair = 250.00
        print("ERP System Initialized with default Inventory.")
        self._print_inventory()

    def _log_transaction(self, entry_type, amount, related_id, details=""):
        """Internal function to log all financial activity to the ledger."""
        self.ledger.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': entry_type,
            'amount': amount,
            'related_id': related_id,
            'details': details
        })

    def _print_inventory(self):
        """Helper to display current inventory status clearly."""
        print("\n--- Current Inventory Status ---")
        for item, data in self.inventory.items():
            print(f"  {item} ({data['type']}): {data['stock']} {data['uom']} @ ${data['cost']:.2f}")
        print("--------------------------------")

    # --- 1. SALES / ORDER MANAGEMENT (OM) ---

    def create_sales_order(self, quantity):
        """Creates a new customer order and logs demand."""
        order_id = f"SO-{uuid.uuid4().hex[:6]}"
        order = {
            'id': order_id, 
            'product': 'FG-CHAIR', 
            'quantity': quantity,
            'status': 'DEMAND_CREATED',
            'required_by': datetime.now() + pd.Timedelta(days=7) # Delivery in 7 days
        }
        self.sales_orders.append(order)
        print(f"\n[SALES] New Sales Order {order_id} created for {quantity} Chairs.")
        return order_id

    # --- 2. PRODUCTION PLANNING / MRP ---

    def run_mrp(self):
        """
        Runs the Material Requirements Planning (MRP) check.
        Determines: 1. What needs to be produced. 2. What raw materials need to be purchased.
        """
        total_demand = sum(so['quantity'] for so in self.sales_orders if so['status'] == 'DEMAND_CREATED')
        fg_stock = self.inventory['FG-CHAIR']['stock']
        
        # 1. Determine Production Requirement
        production_required = max(0, total_demand - fg_stock)
        print(f"\n[MRP] Total Demand: {total_demand}. On Hand: {fg_stock}. Production Required: {production_required}")
        
        if production_required > 0:
            po_id = f"PO-{uuid.uuid4().hex[:6]}"
            prod_order = {
                'id': po_id, 
                'product': 'FG-CHAIR', 
                'quantity': production_required,
                'status': 'PLANNED',
                'materials_needed': defaultdict(int)
            }
            
            # 2. Determine Raw Material Requirements (Based on BOM)
            print("[MRP] Checking Raw Material Needs...")
            for component in self.bom['FG-CHAIR']:
                mat_id = component['material_id']
                qty_per_unit = component['quantity']
                total_needed = production_required * qty_per_unit
                stock = self.inventory[mat_id]['stock']
                
                # Determine Purchase Requirement
                purchase_needed = max(0, total_needed - stock)
                
                prod_order['materials_needed'][mat_id] = total_needed
                
                if purchase_needed > 0:
                    print(f"  > Material {mat_id}: Needed={total_needed}, Stock={stock}. PURCHASING {purchase_needed}.")
                    # Generate Purchase Order Proposal
                    self.create_purchase_order(mat_id, purchase_needed, po_id)
                else:
                    print(f"  > Material {mat_id}: Sufficient stock.")

            self.production_orders.append(prod_order)
            print(f"[MRP] Production Order {po_id} created for {production_required} Chairs.")
        
        # Update Sales Order Statuses (simple scenario: assume all demand is addressed)
        for so in self.sales_orders:
            if so['status'] == 'DEMAND_CREATED':
                so['status'] = 'MRP_PROCESSED'

    # --- 3. PROCUREMENT / PURCHASING (MM) ---

    def create_purchase_order(self, material_id, quantity, related_prod_id):
        """Generates a Purchase Order for required raw materials."""
        po_id = f"MM-{uuid.uuid4().hex[:6]}"
        po = {
            'id': po_id, 
            'material': material_id, 
            'quantity': quantity, 
            'unit_cost': self.inventory[material_id]['cost'],
            'total_cost': quantity * self.inventory[material_id]['cost'],
            'status': 'SENT_TO_SUPPLIER',
            'related_prod_id': related_prod_id
        }
        self.purchase_orders.append(po)
        print(f"  [MM] Purchase Order {po_id} created for {quantity} x {material_id}. Cost: ${po['total_cost']:.2f}")
        return po_id

    def execute_procurement(self, po_id):
        """Simulates receiving goods and paying the supplier."""
        po = next((p for p in self.purchase_orders if p['id'] == po_id and p['status'] == 'SENT_TO_SUPPLIER'), None)
        if not po: return print(f"  [MM] Error: PO {po_id} not found or already received.")
        
        self.inventory[po['material']]['stock'] += po['quantity']
        po['status'] = 'RECEIVED'
        
        # Log payment (debit cash) - This is the Supplier Invoice payment
        self._log_transaction('EXPENSE', po['total_cost'], po_id, f"Payment for {po['material']} procurement (Supplier Invoice).")
        print(f"  [MM] Goods received for PO {po_id}. Inventory updated. Cost: ${po['total_cost']:.2f} logged.")

    # --- 4. MANUFACTURING / PRODUCTION ---

    def execute_production(self, prod_id):
        """
        Executes a production run: 
        1. Consumes Raw Materials (RM). 
        2. Incurs Labor Cost.
        3. Creates Finished Goods (FG).
        """
        prod_order = next((p for p in self.production_orders if p['id'] == prod_id and p['status'] == 'PLANNED'), None)
        if not prod_order: return print(f"[MFG] Error: Production Order {prod_id} not found or already completed.")
        
        product_quantity = prod_order['quantity']
        total_rm_cost = 0.0
        
        # 1. Consume Raw Materials (RM) and track cost
        print(f"[MFG] Starting production of {product_quantity} chairs...")
        for mat_id, qty_needed in prod_order['materials_needed'].items():
            cost_per_unit = self.inventory[mat_id]['cost']
            cost_consumed = qty_needed * cost_per_unit
            total_rm_cost += cost_consumed
            
            self.inventory[mat_id]['stock'] -= qty_needed
            self._log_transaction('COGS_RM', cost_consumed, prod_id, f"Consumed {qty_needed} of {mat_id} for production.")
            print(f"  > Consumed {qty_needed} {mat_id}. Cost: ${cost_consumed:.2f}")

        # 2. Incur Labor Cost
        total_labor_cost = product_quantity * self.labor_rate_per_chair
        self._log_transaction('COGS_LABOR', total_labor_cost, prod_id, "Incurred labor and overhead cost.")
        print(f"  > Incurred Labor/Overhead Cost: ${total_labor_cost:.2f}")

        # 3. Create Finished Goods (FG)
        self.inventory['FG-CHAIR']['stock'] += product_quantity
        prod_order['status'] = 'COMPLETED'
        
        total_production_cost = total_rm_cost + total_labor_cost
        cost_per_chair = total_production_cost / product_quantity
        
        # Update FG cost (simple average cost calculation for simplicity)
        self.inventory['FG-CHAIR']['cost'] = cost_per_chair # Note: This overwrites old cost for simplicity
        
        print(f"[MFG] Production completed. {product_quantity} Chairs added to stock.")
        print(f"      Total Production Cost: ${total_production_cost:,.2f} | Unit Cost: ${cost_per_chair:.2f}")
        
    def ship_order(self, so_id):
        """Simulates shipping to the customer and recognizing revenue and COGS."""
        so = next((s for s in self.sales_orders if s['id'] == so_id and s['status'] == 'MRP_PROCESSED'), None)
        if not so: return print(f"[SHIPPING] Error: Sales Order {so_id} not found or not ready.")

        if self.inventory[so['product']]['stock'] < so['quantity']:
            return print(f"[SHIPPING] Error: Not enough stock to ship SO {so_id}.")
        
        # 1. Update Inventory and COGS
        self.inventory[so['product']]['stock'] -= so['quantity']
        unit_cost = self.inventory[so['product']]['cost']
        cogs = so['quantity'] * unit_cost
        
        # 2. Recognize Revenue (Customer Invoice)
        revenue = so['quantity'] * self.sales_price_per_chair
        
        # 3. Log Financials
        self._log_transaction('REVENUE', revenue, so_id, "Sale of Finished Goods (Customer Invoice).")
        self._log_transaction('COGS', cogs, so_id, "Cost of Goods Sold.")
        
        so['status'] = 'SHIPPED_INVOICED'
        print(f"\n[SHIPPING] SO {so_id} shipped. {so['quantity']} Chairs sold.")
        print(f"      Revenue: ${revenue:,.2f} | COGS: ${cogs:,.2f}")

    # --- 5. FINANCIALS / REPORTING (FI) ---

    def generate_report(self):
        """Calculates a simple Profit & Loss (P&L) statement."""
        
        df = pd.DataFrame(self.ledger)
        if df.empty:
            print("\n[FINANCE] Ledger is empty. No financial activity recorded.")
            return

        print("\n========================================================")
        print(f"           FINANCIAL REPORT (P&L) - {datetime.now().year}")
        print("========================================================")
        
        # Calculate key financial metrics
        total_revenue = df[df['type'] == 'REVENUE']['amount'].sum()
        total_cogs = df[df['type'].str.contains('COGS')]['amount'].sum()
        total_expense = df[df['type'] == 'EXPENSE']['amount'].sum()

        gross_profit = total_revenue - total_cogs
        net_profit = gross_profit - total_expense
        
        # Display P&L Summary
        print(f"1. Total Revenue:                ${total_revenue:,.2f}")
        print(f"2. Cost of Goods Sold (COGS):   -${total_cogs:,.2f}")
        print("--------------------------------------------------------")
        print(f"**Gross Profit**:                ${gross_profit:,.2f}")
        print(f"3. Operating Expenses:          -${total_expense:,.2f}")
        print("========================================================")
        print(f"**NET PROFIT**:                  ${net_profit:,.2f}")
        print("========================================================")

        # Detailed Ledger for Reconciliation
        print("\n--- Detailed Financial Ledger (for Reconciliation) ---")
        if not df.empty:
            print(df.to_string(index=False)) # Display the full ledger for audit
        print("------------------------------------------------------")
        
        return df


# ====================================================================
# SIMULATION EXECUTION
# ====================================================================

if __name__ == "__main__":
    
    erp = ERP_Simulator()
    print("\n\n--- SIMULATION START: END-TO-END ERP CYCLE ---")
    time.sleep(1)

    # 1. SALES: Customer places a large order (15 chairs)
    # Note: Initial stock is 10 chairs. This will trigger production.
    order_id = erp.create_sales_order(quantity=15)

    # 2. PRODUCTION PLANNING (MRP): Checks demand vs. stock
    erp.run_mrp() 
    
    # Identify the generated production and purchase orders
    prod_order = erp.production_orders[-1]
    purchase_orders = [po for po in erp.purchase_orders if po['related_prod_id'] == prod_order['id']]
    
    # 3. PROCUREMENT (MM): Executes the purchase of needed raw materials
    print("\n--- Procurement Phase (Inventory Replenishment) ---")
    time.sleep(1)
    for po in purchase_orders:
        erp.execute_procurement(po['id'])
    
    erp._print_inventory() # Should show increased raw materials

    # 4. MANUFACTURING (MFG): Executes the production run
    print("\n--- Manufacturing Phase ---")
    time.sleep(1)
    # The MRP determined we need to produce 5 new chairs (15 demanded - 10 in stock)
    erp.execute_production(prod_order['id']) 
    
    erp._print_inventory() # Should show reduced raw materials and new finished goods (10 + 5 = 15)

    # 5. SHIPPING: Ship the completed sales order
    erp.ship_order(order_id)
    
    erp._print_inventory() # Finished goods stock should be 0 (15 - 15)

    # 6. FINANCIALS (FI): Generate the P&L Report
    erp.generate_report()
    
    print("\n--- SIMULATION COMPLETE: ERP cycle finished ---")
    print("The company successfully fulfilled the order, logged all costs and revenue.")
