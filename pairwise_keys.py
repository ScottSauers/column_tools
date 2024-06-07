import os
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from datetime import datetime

def select_file_and_column(temp_dir):
    root = tk.Tk()
    root.withdraw()

    # Prompt user to select a file
    file_path = filedialog.askopenfilename(initialdir=temp_dir, title="Select file",
                                           filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*")))
    if not file_path:
        raise ValueError("No file selected")

    # Load the selected file
    data = pd.read_excel(file_path)

    # Create a new window to select column
    def on_select(event, col):
        global selected_file_column
        selected_file_column = (file_path, col)
        root.quit()

    root = tk.Tk()
    root.title("Select Column")

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for col in data.columns:
        preview = data[col].dropna().head(2).to_list()
        preview_text = f"{col}: Example Values: {preview}"
        btn = ttk.Button(scrollable_frame, text=preview_text, style="TButton")
        btn.pack(fill='x', padx=10, pady=5)
        btn.bind("<Double-1>", lambda e, col=col: on_select(e, col))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.update_idletasks()
    root.geometry(f"{scrollable_frame.winfo_width() + scrollbar.winfo_width()}x{scrollable_frame.winfo_height()}")
    root.deiconify()
    root.mainloop()
    root.destroy()

    return selected_file_column

def is_numeric(series):
    return pd.to_numeric(series, errors='coerce').notna().all()

def find_similar_columns(selected_files, temp_dir, master_key_file, key_column):
    master_key_path = os.path.join(temp_dir, master_key_file)

    # Attempt to load the master key file based on its extension
    try:
        if master_key_path.lower().endswith('.csv'):
            key_data = pd.read_csv(master_key_path)
        elif master_key_path.lower().endswith('.xlsx'):
            key_data = pd.read_excel(master_key_path, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file format for the master key file: {master_key_file}")
    except Exception as e:
        raise ValueError(f"Failed to load master key file due to: {str(e)}")

    if key_column not in key_data.columns:
        raise ValueError(f"Key column '{key_column}' not found in the file {master_key_file}")

    # Consider only unique non-missing values
    key_data = key_data[key_column].dropna().drop_duplicates()
    if not is_numeric(key_data):
        raise ValueError("Key column contains non-numeric data")

    key_hist = key_data.value_counts(normalize=True)
    key_std = key_data.std()
    results = []

    for filename in selected_files:
        if filename.startswith("~$"):  # Skip temporary or hidden files
            continue
        file_path = os.path.join(temp_dir, filename)
        try:
            if file_path.lower().endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.lower().endswith('.xlsx'):
                data = pd.read_excel(file_path, engine='openpyxl')
            else:
                continue  # Skip files with unsupported formats
        except Exception as e:
            raise ValueError(f"Failed to load data from {filename} due to: {str(e)}")

        columns = data.columns.tolist()
        selected_columns = columns[:20] + columns[-5:]

        max_similarity = 0
        most_similar_column = None

        for column in selected_columns:
            if column in data.columns:
                column_data = data[column].dropna().drop_duplicates()
                if not is_numeric(column_data):
                    continue
                common_values = len(set(key_data).intersection(set(column_data)))
                column_std = column_data.std()
                if common_values / len(key_data) >= 0.25 and (0.5 * key_std) <= column_std <= (2 * key_std):
                    column_hist = column_data.value_counts(normalize=True)
                    if column_hist.empty:
                        continue
                    emd = wasserstein_distance(key_hist, column_hist)
                    similarity = 1 / (1 + emd)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_column = column

        results.append((filename, master_key_file, key_column, most_similar_column, max_similarity))

    return results

def pairwise_heatmap_for_ids(results, temp_dir, ax, title_suffix):
    all_ids = {}

    for filename, key_file, key_column, column, similarity in results:
        if filename.startswith("~$"):  # Skip temporary or hidden files
            continue
        file_path = os.path.join(temp_dir, filename)
        data = pd.read_excel(file_path, engine='openpyxl')

        if column in data.columns:
            ids = data[column].dropna().drop_duplicates().unique()
            all_ids[filename] = set(ids)

    files = list(all_ids.keys())
    n = len(files)

    # Initialize matrices
    upper_triangle = pd.DataFrame(index=files, columns=files, dtype=float)
    lower_triangle = pd.DataFrame(index=files, columns=files, dtype=float)

    # Calculate pairwise percentages
    for i in range(n):
        for j in range(n):
            if i != j:
                ids_i = all_ids[files[i]]
                ids_j = all_ids[files[j]]
                percentage_i_in_j = len(ids_i.intersection(ids_j)) / len(ids_i) * 100
                percentage_j_in_i = len(ids_j.intersection(ids_i)) / len(ids_j) * 100
                upper_triangle.iloc[i, j] = percentage_i_in_j
                lower_triangle.iloc[i, j] = percentage_j_in_i
            else:
                upper_triangle.iloc[i, j] = 100
                lower_triangle.iloc[i, j] = 100

    # Combine upper and lower triangles into one matrix
    combined_matrix = upper_triangle.where(~upper_triangle.isna(), lower_triangle)

    # Plot heatmap
    sns.heatmap(combined_matrix, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
    ax.set_title(f"Pairwise Heatmap of ID Percentages ({title_suffix})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

def save_results_to_csv(results, output_file):
    data = []
    for result in results:
        file, key_file, key_column, column, _ = result
        data.append([file, key_file, key_column, column])

    df = pd.DataFrame(data, columns=["File Path", "Key File", "Key Column", "Matched Column"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    temp_dir = "/Users/scott/Downloads/merger"
    selected_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.xlsx')]

    # Prompt the user to select the first file and column
    selected_file1, selected_column1 = select_file_and_column(temp_dir)
    
    # Prompt the user to select the second file and column
    selected_file2, selected_column2 = select_file_and_column(temp_dir)
    
    # Find similar columns in the selected files for the first selection
    results1 = find_similar_columns(selected_files, temp_dir, selected_file1, selected_column1)
    
    # Print the results for the first selection
    print("Most similar columns found for the first selection:")
    for result in results1:
        print(f"File: {result[0]}, Column: {result[3]}, Similarity: {result[4]}")
    
    # Find similar columns in the selected files for the second selection
    results2 = find_similar_columns(selected_files, temp_dir, selected_file2, selected_column2)
    
    # Print the results for the second selection
    print("\nMost similar columns found for the second selection:")
    for result in results2:
        print(f"File: {result[0]}, Column: {result[3]}, Similarity: {result[4]}")
    
    # Plot heatmaps side by side
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))

    pairwise_heatmap_for_ids(results1, temp_dir, axs[0], "First Selection")
    pairwise_heatmap_for_ids(results2, temp_dir, axs[1], "Second Selection")

    plt.tight_layout()
    plt.show()

    # Compare similarities
    print("\nComparing maximum similarities for each file:")
    final_results = []
    for res1, res2 in zip(results1, results2):
        file1, key_file1, key_column1, column1, sim1 = res1
        file2, key_file2, key_column2, column2, sim2 = res2
        best_res = res1 if sim1 > sim2 else res2
        file, key_file, key_column, column, best_sim = best_res
        log_scale_sim = -np.log10(1 - best_sim)
        print(f"File: {file}, Key File: {key_file}, Key Column: {key_column}, Column: {column}, Similarity: {best_sim} (Log Scale: {log_scale_sim})")
        final_results.append((file, key_file, key_column, column, best_sim))

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(temp_dir, f"key_columns_{timestamp}.csv")
    save_results_to_csv(final_results, output_file)

if __name__ == "__main__":
    main()
