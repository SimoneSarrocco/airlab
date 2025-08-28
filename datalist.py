"""
Datalist class to create a json datalist from the brats data which is also suitable for the auto3dseg
"""
import os
from pathlib import Path
from typing import Dict
import random

import json
import logging

from lakefsloader import LakeFSLoader


class DataList():

    def __init__(
            self,
            data: Dict,
    ) -> None:
        """
        Initialize a json compatible datalist.
        """
        self.data = data

    @classmethod
    def from_json(
        cls,
        filepath: Path,
        lakefs_config: dict = {},
        ) -> None:
        """
        Create datalist from a json file. 
        """
        filepath = Path(filepath)
        with open(filepath) as json_file:
            data = json.load(json_file)

        # configure lakefs
        if lakefs_config:
            lakefs_loader = LakeFSLoader(
                local_cache_path=lakefs_config["cache_path"],
                repo_name=lakefs_config["data_repository"],
                branch_id=lakefs_config["branch"],
                ca_path=lakefs_config["ca_path"],
                endpoint=lakefs_config["s3_endpoint"],
                secret_key=lakefs_config["secret_key"],
                access_key=lakefs_config["access_key"],
            )

        # make sure that the file in the data exists
        logging.info(f"Checking that every entry in the data is valid.")
        for _, set_value in data.items():
            # each patient
            for patient in set_value:
                # split the keys
                for patient_key, patient_value in patient.items():
                    # only consider image and label
                    if patient_key == "image" or patient_key == "label":
                        # handle both lists (multichannel image) and single images
                        if isinstance(patient_value, str):
                            patient_value = [patient_value]
                        for pv in patient_value:
                            if lakefs_config:
                                _, extension = os.path.splitext(pv)
                                if extension:
                                    lakefs_loader.check_file(pv)
                                else:
                                    lakefs_loader.check_dir(pv)
                            elif not os.path.exists(pv):
                                logging.warning(f"{pv} was not found, make sure to use a valid datalist.json")
                                raise FileNotFoundError(f"{pv} could not be found, make sure that it exists or use the s3 storage")

        return cls(data=data)

    @classmethod
    def from_file(
        cls,
        datapath: Path,
        config: Dict,
        ):
        """
        Create datalist from a single datafile.
        """
        datapath = Path(datapath)
        img_tag = config["img_tag"]

        # get the file
        case = ''
        for file in datapath.iterdir():
            if file.is_file() and img_tag in file.name.lower():
                case = file

        # create the datalist
        if case == '':
            data = None
        else:
            data = {"testing": [], "training": []}
            data["testing"].append({"image": str(case)})

        return cls(data=data)

    @classmethod
    def from_directory(
            cls,
            datapath: Path,
            config: Dict,
            include_root: bool = True,
            shuffle: bool = True,
            image_modality: str = 'Spectralis_oct',
            ):
        """
        Create datalist from a directory. It is expected that the directory has a subdirectory for each patient.
        
        Args:
            datapath (str): The local path to the data
            config (dict): According to config.yaml - data
            include_root (bool): Whether or not to include the root in the datalist
            shuffle: (bool): Shuffle the dataset
        """
        datapath = Path(datapath)
        if image_modality == 'Spectralis_oct':
            img_tag = '.nii.gz'
        elif image_modality == 'Spectralis_slo':
            img_tag = '.tiff'
        else: 
            img_tag = config["img_tag"]
        lbl_tag = config["lbl_tag"]
        random.seed(42)

        # handle the train-test set splitting
        if config["train_test_already_split"]:
            patient_list = [f for f in datapath.iterdir() if f.is_dir() and f.name.startswith('OMEGA')]

        else:
            # put everything in the same list
            patient_list = [f for f in datapath.iterdir() if f.is_dir() and f.name.startswith('OMEGA')]
            # patient_list = [p for p in datapath.glob("*/*/*/Spectralis_oct") if p.is_dir()]
            # skip corrupted
            # patient_list = [p for p in patient_list if 'OMEGA04/L/V02' not in str(p)]           

        # create data skeleton
        data = {"full_dataset": []}

        # populate testing data
        for patient in sorted(patient_list):
            # OMEGAxx --> OS/OD --> V01/V02/V03/V04 --> Spectralis_oct
            # get the img and label
            # patient_test_img = ''
            for eye_dir in sorted(patient.glob('*')):
                if not eye_dir.is_dir():
                    continue
                for visit_dir in sorted(eye_dir.glob('*')):
                    if not visit_dir.is_dir():
                        continue
                    spec_modality = visit_dir / image_modality
                    if not spec_modality.is_dir():
                        continue
                    # Find image files matching img_tag
                    if image_modality == 'Spectralis_oct':
                        img_files = [f for f in spec_modality.glob("*.nii.gz") if img_tag in f.name.lower()]
                    elif image_modality == 'Spectralis_slo':
                        img_files = [f for f in spec_modality.glob("*.tiff") if img_tag in f.name.lower()]
                    else: 
                        img_files = [f for f in spec_modality.glob("*.dcm") if img_tag in f.name.lower()]
                    if not img_files:
                        continue
                    img_file = img_files[0]  # Take the first matching image file

                    if include_root:
                        img_path = str(img_file)
                    else:
                        img_path = '/'.join(img_file.parts[-6:])

                    # Add to data list
                    data["full_dataset"].append({"image": img_path})

            # for file in patient.iterdir():
            #    if img_tag in file.name.lower():
            #        patient_test_img = file
            #    elif lbl_tag in file.name.lower():
            #        patient_test_lbl = file

        # split the folds
        num_folds = config["folds"]
        if num_folds > 1:
            fold_size = len(data["full_dataset"]) // num_folds
            for fold_number in range(num_folds):
                for i in range(fold_size):
                    data["full_dataset"][fold_number * fold_size + i]["fold"] = fold_number
        
        return cls(data=data)
    
    @classmethod
    def from_lakefs(
        cls,
        data_config,
        lakefs_config,
        filepath: str = '',
        include_root: bool = True,
        shuffle: bool = True, 
        image_modality: str = 'Spectralis_oct',
    ):
        """
        Create the datalist from the s3 storage
        
        Args:
            data_config (dict): According to config.yaml - data
            lakefs_config (dict): According to lakefs_cfg.yaml
            filepath (str): The relative path, starting from within the lakefs branch
            include_root (bool): Whether or not to include the root in the datalist
            shuffle: (bool): Shuffle the dataset
        """
        # parse lakefs config
        lakefs_loader = LakeFSLoader(
                local_cache_path=lakefs_config["cache_path"],
                repo_name=lakefs_config["data_repository"],
                branch_id=lakefs_config["branch"],
                ca_path=lakefs_config["ca_path"],
                endpoint=lakefs_config["s3_endpoint"],
                secret_key=lakefs_config["secret_key"],
                access_key=lakefs_config["access_key"],
            )

        # iterate through the s3 objects, only considering the ones of interest
        if image_modality == 'Spectralis_oct':
            tags = '.nii.gz'
        elif image_modality == 'Spectralis_slo':
            tags = '.tiff'
        else:
            tags = data_config['img_tag']
        objects = lakefs_loader.read_s3_objects(filter=tags, prefix=filepath)
        logging.info(f"Creating a datalist from {len(objects)} files.")

        # check if they are available in the cache, otherwise download
        logging.info(f"Checking cache... Need to download {lakefs_loader.check_num_missing_files(objects)} files.")
        for obj in objects:
            lakefs_loader.check_file(obj)
        logging.info(f"Finished caching all objects into {lakefs_loader.get_branch_dir()}.")

        # create the datalist by calling from_directory on the local cache
        datapath = lakefs_loader.get_branch_dir() / filepath
        return cls.from_directory(
            datapath=datapath,
            config=data_config,
            include_root=include_root,
            shuffle=shuffle,
            image_modality=image_modality,
            )

    def save_datalist_to_json(self, path: Path, remember_path: bool = True) -> None:
        """
        Save the datalist to file.
        """
        # save the filepath
        if remember_path:
            self.filepath = path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        logging.info(f"Datalist saved to {path}")
        
