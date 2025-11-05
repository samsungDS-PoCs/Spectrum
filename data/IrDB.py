import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
import torch.nn.functional as F
from torch_geometric.utils import one_hot, scatter

spectrum_target_dict = {
    0: 'A2',
    1: 'A3',
    2: 'B1',
    3: 'B2',
    4: 'B3',
    5: 'C1',
    6: 'C2',
    7: 'C3',
    8: 'S1',
    9: 'S2',
   10: 'S3',
   11: 'C',
   12: 'E0',
   13: 'h1',
   14: 'h2',
   15: 'h3',
   16: 'S1(2)',
   17: 'S2(2)',
   18: 'C(2)',
   19: 'E0(2)',
   20: 'h1(2)',
   21: 'h2(2)',
   22: 'S1(4)',
   23: 'S2(4)',
   24: 'S3(4)',
   25: 'S4(4)',
   26: 'C(4)',
   27: 'E0(4)',
   28: 'h1(4)',
   29: 'h2(4)',
   30: 'h3(4)',
   31: 'h4(4)',
   32: 'S1(5)',
   33: 'S2(5)',
   34: 'S3(5)',
   35: 'S4(5)',
   36: 'S5(5)',
   37: 'C(5)',
   38: 'E0(5)',
   39: 'h1(5)',
   40: 'h2(5)',
   41: 'h3(5)',
   42: 'h4(5)',
   43: 'h5(5)',
   44: 'S1(6)',
   45: 'S2(6)',
   46: 'S3(6)',
   47: 'S4(6)',
   48: 'S5(6)',
   49: 'S6(6)',
   50: 'C(6)',
   51: 'E0(6)',
   52: 'h1(6)',
   53: 'h2(6)',
   54: 'h3(6)',
   55: 'h4(6)',
   56: 'h5(6)',
   57: 'h6(6)',
} | {i + 58: f'y{i}' for i in range(800)}

class IrDB_org(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        import rdkit  # noqa
        return ['IrDB.sdf', 'IrDB.sdf.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType

        atom_types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'Ir': 7}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1]) as f:
            target = [[float(x) for x in line.split(',')[1:-3]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)

        with open(self.raw_paths[1]) as f:
            spec = [[line.split(',')[-3],line.split(',')[-2]] for line in f.read().split('\n')[1:-1]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=True,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            # build pair-wise edge graphs
            num_nodes = pos.shape[0]
            node_index = torch.tensor([i for i in range(num_nodes)])
            edge_d_dst_index = torch.repeat_interleave(node_index, repeats=num_nodes)
            edge_d_src_index = node_index.repeat(num_nodes)
            edge_d_attr = pos[edge_d_dst_index] - pos[edge_d_src_index]
            edge_d_attr = edge_d_attr.norm(dim=1, p=2)
            edge_d_dst_index = edge_d_dst_index.view(1, -1)
            edge_d_src_index = edge_d_src_index.view(1, -1)
            edge_d_index = torch.cat((edge_d_dst_index, edge_d_src_index), dim=0)
 
            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []

            n_atm = 0
            for atom in mol.GetAtoms():
                if not atom.GetSymbol() in atom_types:
                    continue
                n_atm += 1
                type_idx.append(atom_types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            # from torch geometric
            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * n_atm + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=n_atm, reduce='sum').tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            name = mol.GetProp('ID')
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            spec_x = torch.tensor(list(map(float, spec[i][0].split())), dtype=torch.float).unsqueeze(0)
            spec_y = torch.tensor(list(map(float, spec[i][1].split())), dtype=torch.float).unsqueeze(0)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                index=i,
                spec_x = spec_x,
                spec_y = spec_y,
                edge_d_index=edge_d_index,
                edge_d_attr=edge_d_attr,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

class IrDB(IrDB_org):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset_arg=None,
    ):

        if isinstance(dataset_arg, (list, tuple)):
            self.labels = list(dataset_arg)
        else:
            self.labels = [dataset_arg]
        label2idx = dict(zip(spectrum_target_dict.values(), spectrum_target_dict.keys()))
        self.label_idx = [label2idx[l] for l in self.labels]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(IrDB, self).__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def _filter_label(self, batch):
        if batch.y[:,self.label_idx].ndim == 1:
            batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        elif batch.y[:,self.label_idx].ndim == 2:
            batch.y = batch.y[:, self.label_idx]
        else:
            raise ValueError(f"Unsupported shape {batch.y[:,self.label_idx]}")
        return batch

