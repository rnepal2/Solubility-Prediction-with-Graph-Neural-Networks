# libraries
import os
import numpy as np
import pandas as pd
import deepchem as dc
import torch
from torch_geometric.data import Dataset


"""
Creating a custom dataset for the torch_geometric models
using aqueous solubility dataset from: https://doi.org/10.1038/s41597-019-0151-1
"""

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, length=0):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.length = length
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        processed_files = [f for f in os.listdir(self.processed_dir) if not f.startswith("pre")]
    
        if self.test:
            processed_files = [file for file in processed_files if "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            length = len(processed_files)
            self.length = length
            return [f'data_test_{i}.pt' for i in list(range(length))]  # TODO: make sure the presence of file names with all data_i
        else:
            processed_files = [file for file in processed_files if not "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            length = len(processed_files)
            self.length = length
            return [f'data_{i}.pt' for i in list(range(length))]
        

    def download(self):
        "Implement if needed to triger raw file download from the web."
        "Raw data file read from the raw directory."
        pass
    

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)  # node features from DeepChem MolGraphConvFeaturizer
        count = 0
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            try:
                f = featurizer.featurize(mol["SMILES"])
                data = f[0].to_pyg_graph()
                count += 1
            except:
                continue
            data.y = self.get_label(mol["isSoluble"])  # binary classification label
            data.smiles = mol["SMILES"]
            
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f"data_test_{count-1}.pt"))
            else:
                torch.save(data, os.path.join(self.processed_dir, f"data_{count-1}.pt"))
        print(f"Number of molecules included: {count}")


    def get_label(self, label):
        """Returns the label (0/1) for the model: data.y"""
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        """ 
        - Equivalent to __getitem__ in pytorch, not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data