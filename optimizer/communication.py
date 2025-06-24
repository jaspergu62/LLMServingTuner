# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import stat
import time
from pathlib import Path
 
from filelock import FileLock
 
 
class CustomCommand:
    cmd_eof = "eof"
 
    def __init__(self):
        self._start = "start"
        self._check_success = "check_success"
        self._process_poll = "process_poll"
        self._stop = "stop"
        self._history = []
        self._backup = "backup"
        self._init = "init"
 
    @property
    def backup(self):
        return f"{self._backup} {time.time_ns()}"
 
    @property
    def init(self):
        return f"{self._init} {time.time_ns()}"
 
    @property
    def start(self):
        return f"{self._start} {time.time_ns()}"
 
    @property
    def check_success(self):
        return f"{self._check_success} {time.time_ns()}"
 
    @property
    def process_poll(self):
        return f"{self._process_poll} {time.time_ns()}"
 
    @property
    def stop(self):
        return f"{self._stop} {time.time_ns()}"
 
    @property
    def history(self):
        return tuple(self._history)
 
    @history.setter
    def history(self, value):
        self._history.append(value)
 
    @history.deleter
    def history(self):
        self._history.clear()
 
 
class CommunicationForFile:
    def __init__(self, cmd_file: Path, res_file: Path, timeout=120):
        if not cmd_file.parent.exists():
            cmd_file.parent.mkdir(parents=True)
        if not res_file.parent.exists():
            res_file.parent.mkdir(parents=True)
        self.cmd_file = cmd_file
        self.cmd_file_lock = cmd_file.parent.joinpath(f"{cmd_file.name}.lock")
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IROTH | stat.S_IWOTH
        if not self.cmd_file_lock.exists():
            with os.fdopen(os.open(self.cmd_file_lock, flags, modes), "w") as f:
                pass
        self.res_file = res_file
        self.res_file_lock = res_file.parent.joinpath(f"{res_file.name}.lock")
        if not self.res_file_lock.exists():
            with os.fdopen(os.open(self.res_file_lock, flags, modes), "w") as f:
                pass
        self.timeout = timeout
 
    def send_command(self, cmd):
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IROTH
        with FileLock(self.cmd_file_lock):
            if self.cmd_file.exists():
                with open(self.cmd_file, "w") as fcmd:
                    fcmd.write(cmd)
            else:
                with os.fdopen(os.open(self.cmd_file, flags, modes), "w", buffering=1024) as fcmd:
                    fcmd.write(cmd)
 
    def recv_command(self):
        with FileLock(self.res_file_lock):
            if not self.res_file.exists():
                return ''
            with open(self.res_file, 'r', encoding="utf-8") as f:
                data = f.read()
        return data
 
    def clear_cmd(self, command):
        # 确认命令是否完成，进行清理相关命令。
        st = time.perf_counter()
        while True:
            if time.perf_counter() - st > self.timeout:
                raise TimeoutError("Timeout while getting command result. command {}", command)
            time.sleep(1)
            cmd_res = self.recv_command()
            if not cmd_res:
                continue
            if command not in cmd_res:
                continue
            res = cmd_res[len(command) + 1:].strip().lower()
            if res == "done":
                status = "done"
                break
            elif res == "true":
                status = True
                break
            elif res == "false":
                status = False
                break
            elif res == "none":
                status = None
                break
            elif 'error' in res:
                raise ValueError(f"Failed to start the program on another server. info: {cmd_res}")
            else:
                status = res
                break
        self.send_command(CustomCommand.cmd_eof)
        self.clear_res()
        return status
 
    def clear_res(self):
        while True:
            time.sleep(1)
            data = self.recv_command()
            if data.strip().lower() == CustomCommand.cmd_eof:
                self.send_command(CustomCommand.cmd_eof)
                break