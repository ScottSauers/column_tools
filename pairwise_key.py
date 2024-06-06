import os
import pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox

def select_file_and_column(files, temp_dir):
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
    root.deiconify()
    columns = data.columns.tolist()
    
    def on_select():
        selected_column = lb.get(lb.curselection())
        root.destroy()
        global selected_file_column
        selected_file_column = (file_path, selected_column)
    
    lb = tk.Listbox(root, selectmode=tk.SINGLE)
    for col in columns:
        lb.insert(tk.END, col)
    lb.pack()
    
    select_button = tk.Button(root, text="Select Column", command=on_select)
    select_button.pack()
    
    root.mainloop()
    
    return selected_file_column

def find_similar_columns(selected_files, temp_dir, master_key_file, key_column):
    master_key_path = os.path.join(temp_dir, master_key_file)

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

    key_hist = key_data[key_column].drop_duplicates().value_counts(normalize=True)
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
        selected_columns = columns[:20] + columns[-5:]  # example slice, adjust as necessary

        max_similarity = 0
        most_similar_column = None

        for column in selected_columns:
            if column in data.columns:
                column_hist = data[column].drop_duplicates().value_counts(normalize=True)
                if column_hist.empty:
                    continue
                emd = wasserstein_distance(key_hist, column_hist)
                similarity = 1 / (1 + emd)

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_column = column

        results.append((filename, most_similar_column, max_similarity))

    return results

def pairwise_heatmap_for_ids(results, temp_dir):
    all_ids = {}

    for filename, column, similarity in results:
        if filename.startswith("~$"):  # Skip temporary or hidden files
            continue
        file_path = os.path.join(temp_dir, filename)
        data = pd.read_excel(file_path, engine='openpyxl')
        
        if column in data.columns:
            ids = data[column].drop_duplicates().dropna().unique()
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
    plt.figure(figsize=(15, 10))
    sns.heatmap(combined_matrix, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f", annot_kws={"size": 8})
    plt.title("Pairwise Heatmap of ID Percentages")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def main():
    temp_dir = "/your/favorite/path/to/merger"
    selected_files = [f for f in os.listdir(temp_dir) if f.lower().endswith('.xlsx')]

    selected_file, selected_column = select_file_and_column(selected_files, temp_dir)
    results = find_similar_columns(selected_files, temp_dir, selected_file, selected_column)
    
    print("Most similar columns found:")
    for result in results:
        print(f"File: {result[0]}, Column: {result[1]}, Similarity: {result[2]}")

    # Generate pairwise heatmap for IDs based on similar columns
    pairwise_heatmap_for_ids(results, temp_dir)

if __name__ == "__main__":
    main()
