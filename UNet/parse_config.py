# parse_config.py

import json
from pathlib import Path
from collections import OrderedDict

class ConfigParser:
    def __init__(self, args):
        """
        config 파일 경로와 커맨드라인 수정을 처리하는 클래스
        """
        args = args.parse_args()

        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
        else:
            assert args.config is not None, "설정 파일이 지정되지 않았습니다. '-c config.json'을 추가하세요."
            self.resume = None
            self.cfg_fname = Path(args.config)
        
        self.config = self._read_json(self.cfg_fname)
        self.config['resume'] = self.resume # resume 경로를 config에 추가

        if args.modification:
            self.config = self._modify_config(self.config, args.modification)

    def __getitem__(self, name):
        """dict처럼 config 항목에 접근할 수 있게 함"""
        return self.config[name]

    def _read_json(self, fname):
        """JSON 파일을 읽어 OrderedDict로 반환"""
        with fname.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)

    def _modify_config(self, config, modification):
        """커맨드라인에서 받은 인자로 config를 수정"""
        # 이 기능은 필요하다면 그대로 사용하거나, 더 단순하게 만들 수 있습니다.
        # 여기서는 기존 로직을 유지합니다.
        for key, value in modification.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        return config