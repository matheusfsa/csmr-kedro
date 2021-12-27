"""``CSVDataSet`` loads/saves data from/to a CSV file using an underlying
filesystem (e.g.: local, S3, GCS). It uses pandas to handle the CSV file.
"""
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict
from importlib import import_module
import fsspec
import pandas as pd
import torch
from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)


def model_from_string(model_name, **model_args):
    model_class = getattr(
        import_module((".").join(model_name.split(".")[:-1])),
        model_name.rsplit(".")[-1],
    )
    return model_class(**model_args)


class TorchDataSet(AbstractVersionedDataSet):
    DEFAULT_LOAD_ARGS = {}  # type: Dict[str, Any]
    DEFAULT_SAVE_ARGS = {}  # type: Dict[str, Any]

    def __init__(
        self,
        filepath: str,
        model_class: str,
        device: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:
        _fs_args = deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop("open_args_load", {})
        _fs_open_args_save = _fs_args.pop("open_args_save", {})
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        self.device = device
        self.model_class = model_class
        self.model_params  = self._load_args.get("model_params", {})
        

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            load_args=self._load_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _load(self) -> torch.nn.Module:
        print(self.model_params)
        model = model_from_string(self.model_class, **self.model_params)
        
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        device = torch.device(self.device)
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)
        return model      

    def _save(self, data: torch.nn.Module) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        torch.save(data.state_dict(), save_path)

        self._invalidate_cache()

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)
