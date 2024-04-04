import os
import pandas as pd
from rich.console import Console
from rich import inspect

console = Console()
# Specify the directory containing the NIfTI files
directory = r'D:\dataset\med\imageTBAD'

# Initialize an empty list to store the data
data = {'img': [], 'mask': []}

# Traverse the files in the directory
for fileitem in os.walk(directory):
    for filename in fileitem[2]:
        if filename.endswith('label.nii.gz'):
            # Append the file paths to the list as a dictionary
            fileid = filename.split('_')[0]
            print(fileid)
            data['img'].append(os.path.join(directory, f'{fileid}_image.nii.gz'))
            data['mask'].append(os.path.join(directory, f'{fileid}_label.nii.gz'))

        # Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Specify the output path for saving the DataFrame as a CSV file
output_path = r'D:\dataset\med\imageTBAD\dataframe.csv'

# Save the DataFrame as a CSV file
df.to_csv(output_path, index=False)

print(f"DataFrame has been saved to {output_path}")
