"""``CSVDataSet`` loads/saves data from/to a CSV file using an underlying
filesystem (e.g.: local, S3, GCS). It uses pandas to handle the CSV file.
"""
from copy import deepcopy
from datetime import datetime, timedelta
import imp
from pathlib import PurePosixPath
from typing import Any, Dict, List
from csmr_kedro.extras.twitter import TwitterExtractor
import fsspec
import pandas as pd
import pytz

from kedro.io.core import (
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

class TwitterDataSet(AbstractVersionedDataSet):
    def __init__(
        self,
        filepath: str,
        companies: List[str], 
        hours: int =24, 
        max_tweets_search=100,
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
        self._load_args = {} 
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = {"index": None} 
        if save_args is not None:
            self._save_args.update(save_args)

        _fs_open_args_save.setdefault("mode", "w")
        _fs_open_args_save.setdefault("newline", "")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

        self._companies = companies
        self._hours = hours
        self._max_tweets_search = max_tweets_search
        self._actual_datetime = datetime.now()
        self._extractor = TwitterExtractor(companies=self._companies, 
                                           api_token=credentials["token"], 
                                           hours=self._hours,
                                           max_tweets_search=self._max_tweets_search,
                                           actual_datetime=self._actual_datetime)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            load_args=self._load_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _load(self) -> pd.DataFrame:
        if self._exists():
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
            with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
                return pd.read_csv(fs_file, **self._load_args)

        data = self._extractor.extract_last_tweets()
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            data.to_csv(path_or_buf=fs_file, **self._save_args)

        self._invalidate_cache()
        
        return data

    def _save(self, data: pd.DataFrame) -> None:
        pass

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
            
        except DataSetError:
            return False
        if self._fs.exists(load_path):
            with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
                    data = pd.read_csv(fs_file, **self._load_args)
            data["created_at"] = pd.to_datetime(data["created_at"])
            min_date = data["created_at"].min()
            max_date_time = (self._actual_datetime - timedelta(hours=self._hours)).replace(tzinfo=pytz.timezone('America/Sao_Paulo'))
            return min_date < max_date_time
        else: 
            return False

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)

