import os
import pandas as pd

# Specify the directory containing the NIfTI files
directory = r'D:\dataset\med\imageTBAD'

# Initialize an empty DataFrame
df = pd.DataFrame(columns=["img", "mask"])

# Traverse the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".nii.gz"):
        # Split the filename to get the file_file_id and file_type (image or label)
        file_id, file_type = filename.split("_")
        file_type = file_type.split(".")[0]

        # Construct the file path
        file_path = os.path.join(directory, filename)

        # Add the file path to the DataFrame
        if file_type == "image":
            df = df._append({"img": file_path, "mask": None}, ignore_index=True)
        elif file_type == "label":
            # Find the corresponding image row and add the label path
            img_row = df[df['img'].str.contains(f"{file_id}_image")]
            if not img_row.empty:
                df.loc[img_row.index, 'mask'] = file_path

# Specify the output path for saving the DataFrame as a CSV file
output_path = r'D:\dataset\med\imageTBAD\dataframe.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_path, index=False)

print(f"DataFrame has been saved to {output_path}")