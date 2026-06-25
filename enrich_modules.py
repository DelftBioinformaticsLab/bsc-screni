import gseapy as gp
import os

# 1. Download the database ONCE outside of the loops
print("Connecting to server to download WikiPathways database...")
try:
    pathway_db = gp.get_library(name='WikiPathways_2024_Mouse', organism='Mouse')
    print("Database downloaded successfully! Running local enrichment loops...\n")
except Exception as e:
    print("Could not connect to the server to download the database.")
    raise e

# Define the path for your concatenated output file
summary_file_path = './data/results/enrichment_summary_py.txt'

# Open the summary file once in write mode ('w') before entering the loops
with open(summary_file_path, 'w') as out_file:
    
    # 2. Run your nested loops completely offline
    for module in range(10):
        for cell_type in ['MG', 'RPC1', 'RPC2', 'RPC3']: 
            path = f'./data/results/modules/module_{module}_{cell_type}_py.txt'
            background = f'./data/results/background_genes_{cell_type}_py.txt'
            
            # Check if BOTH files exist before proceeding
            if not os.path.isfile(path) or not os.path.isfile(background):
                continue
                
            # Read the files into lists to avoid parsing or whitespace bugs
            with open(path, 'r') as f:
                my_genes = [line.strip() for line in f if line.strip()]
            with open(background, 'r') as f:
                bg_genes = [line.strip() for line in f if line.strip()]
                

            # 3. Run the enrichment using the pre-loaded dictionary
            enr = gp.enrich(
                gene_list=my_genes,
                gene_sets=pathway_db,  
                background=bg_genes,
                outdir='./data/results/enrichment',
                no_plot=True
            )
            
            # 4. Write results to the file if any pathways were found
            # FIXED: Check that res2d is not None before checking if it is empty
            if enr.res2d is not None and not enr.res2d.empty:
                print(f"Logging sorted results for Module {module} ({cell_type})...")
                
                # Sort: Lowest Adjusted P-value first, Highest Odds Ratio second
                sorted_df = enr.res2d.sort_values(
                    by=['Adjusted P-value'], 
                    ascending=[True]
                )
                
                # Write the header and the top 5 sorted rows into the text file
                out_file.write(f"=== Module {module} | Type: {cell_type} ===\n")
                out_file.write(sorted_df[['Term', 'Adjusted P-value', 'Odds Ratio', 'Genes']].head(5).to_string())
                out_file.write("\n" + "-" * 50 + "\n\n")
            else:
                # Optional: tracking empty datasets in terminal
                print(f"Module {module} ({cell_type}): No enriched terms found.")

print(f"\nProcess complete! All sorted and concatenated outputs have been saved to: {summary_file_path}")