import os
import pandas as pd

def transform_single_mutants():
    input_path = "data/raw/megascale_d/megascale_single_mutants_trimmed.csv"
    output_path = "data/raw/megascale_d/single_mutants_transformed.parquet"
    
    print(f"Reading input CSV from {input_path}...")
    df_in = pd.read_csv(input_path)
    
    print("Parsing mut_type column...")
    # Extract wildtype, position, mutant
    wt = df_in["mut_type"].str[0]
    pos = df_in["mut_type"].str[1:-1]
    mut = df_in["mut_type"].str[-1]
    
    # Combine into single-row representation
    df_out = pd.DataFrame()
    df_out["mutation_id"] = df_in["mut_type"].astype(str)
    df_out["protein_id"] = df_in["WT_name"].astype(str)
    df_out["position"] = pos.astype(str)
    df_out["wildtype"] = wt.astype(str)
    df_out["mutant"] = mut.astype(str)
    
    # Explicitly convert experimental_ddg using pd.to_numeric to keep it numeric float
    df_out["experimental_ddg"] = pd.to_numeric(df_in["ddG_ML"], errors="coerce")
    
    # Tracks row drop counts
    initial_count = len(df_out)
    
    # 1. Make mutation_id UNIQUE across proteins
    df_out["mutation_id"] = df_out["protein_id"] + "_" + df_out["mutation_id"]
    
    # 2. Normalize mutation_id formatting
    df_out["mutation_id"] = df_out["mutation_id"].str.replace(" ", "", regex=False).str.replace(",", "+", regex=False).str.replace("_", "+", regex=False)
    
    # 3. Remove invalid or impossible amino acids
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    is_valid = lambda s: all(c in valid_aas for c in s if c != "_")
    
    valid_wt_mask = df_out["wildtype"].apply(is_valid)
    valid_mut_mask = df_out["mutant"].apply(is_valid)
    df_out = df_out[valid_wt_mask & valid_mut_mask]
    count_after_aa = len(df_out)
    dropped_aa = initial_count - count_after_aa
    
    # 4. Remove rows where experimental_ddg is NaN or infinite / out of bounds [-100, 100]
    valid_ddg_mask = df_out["experimental_ddg"].notna() & (df_out["experimental_ddg"] >= -100) & (df_out["experimental_ddg"] <= 100)
    df_out = df_out[valid_ddg_mask]
    count_after_ddg = len(df_out)
    dropped_ddg = count_after_aa - count_after_ddg
    
    # 5. Remove duplicate rows after mutation_id is updated
    df_out = df_out.drop_duplicates(subset=["protein_id", "mutation_id"])
    
    # 6. Ensure results output directory exists
    os.makedirs("results/sparsity/empirical_single", exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Print requested stats
    print("\n--- First 10 mutation_id values ---")
    print(df_out["mutation_id"].head(10).to_string(index=False))
    print("-----------------------------------\n")
    
    print("--- Transformed DataFrame Column Data Types ---")
    print(df_out.dtypes)
    print("-----------------------------------------------\n")
    
    print(f"Dropped {dropped_aa} rows due to invalid amino acids.")
    print(f"Dropped {dropped_ddg} rows due to invalid or extreme experimental_ddg.")
    print(f"Total rows dropped: {dropped_aa + dropped_ddg}\n")
    
    print(f"Saving transformed dataframe with {len(df_out)} rows to {output_path}...")
    
    # 7. Save parquet using snappy compression
    df_out.to_parquet(output_path, index=False, compression="snappy")
    print("Transformation completed successfully!")

if __name__ == "__main__":
    transform_single_mutants()
