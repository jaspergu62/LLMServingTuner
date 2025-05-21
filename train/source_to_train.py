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
import argparse
import ast
import csv
import json
import os
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd
from loguru import logger

from msserviceprofiler.modelevalstate.train.pretrain import pretrain


def fetch_rids_from_db(db_path):
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # 执行查询
        cursor.execute("SELECT * FROM batch WHERE name = 'modelExec';")
        batch_rows = cursor.fetchall()

        rids = []
        for row in batch_rows:
            if isinstance(row, str):
                data = ast.literal_eval(row)
                # 提取rid值
                rids.extend([item['rid'] for item in data])
            else:
                rids.append(row[1])
        return rids
    except sqlite3.Error as e:
        logger.error(f"数据库错误: {e}")
        return []
    except Exception as e:
        logger.error(f"其他错误: {e}")
        return []
    finally:
        # 确保关闭游标和连接
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


class DatabaseConnector:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接到数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return self.cursor
        except sqlite3.Error as e:
            raise ConnectionError(f"无法连接到数据库: {e}") from e

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


def read_batch_exec_data(cursor) -> List[Tuple]:
    """读取 batch_exec 表中的数据"""
    try:
        cursor.execute("SELECT * FROM batch_exec WHERE name = 'forward';")
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch_exec 表时出错: {e}") from e


def group_exec_data_by_pid(exec_rows: List[Tuple]) -> Dict[int, List[Tuple]]:
    """按 pid 分组 batch_exec 数据"""
    data_by_pid = {}
    for row in exec_rows:
        pid = row[2]
        if pid not in data_by_pid:
            data_by_pid[pid] = []
        data_by_pid[pid].append(row)
    return data_by_pid


def read_batch_data(cursor) -> List[Tuple]:
    """读取 batch 表中的数据"""
    try:
        cursor.execute("SELECT * FROM batch WHERE name = 'modelExec';")
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch 表时出错: {e}") from e


def read_batch_req_data(cursor) -> List[Tuple]:
    """读取 batch_req 表中的数据"""
    try:
        cursor.execute("SELECT * FROM batch_req;")
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise ValueError(f"读取 batch_req 表时出错: {e}") from e


def calculate_block_sums(req_rows: List[Tuple]) -> Dict[int, float]:
    """按 batch_id 分组并计算 block 的总和"""
    batch_id_block_sum = {}
    for row in req_rows:
        batch_id = row[0]
        block = row[4]
        if not isinstance(block, (int, float)):
            continue
        if batch_id in batch_id_block_sum:
            batch_id_block_sum[batch_id] += block
        else:
            batch_id_block_sum[batch_id] = block
    return batch_id_block_sum


def create_output_folder(input_path: str) -> str:
    """创建输出文件夹"""
    output_folder = os.path.join(input_path, 'output_csv')
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_batch_data(exec_data: List[Tuple], batch_rows: List[Tuple], current_batch_index: int) -> List[Tuple]:
    """处理批量数据"""
    num_exec_rows = len(exec_data)
    num_available_batch_rows = len(batch_rows) - current_batch_index

    if num_exec_rows > num_available_batch_rows:
        return batch_rows[current_batch_index:current_batch_index + num_available_batch_rows]
    else:
        return batch_rows[current_batch_index:current_batch_index + num_exec_rows]


def write_csv_header(csvfile) -> None:
    writer = csv.writer(csvfile)
    writer.writerow([
        ('batch_stage', 'batch_size', 'total_need_blocks', 'total_prefill_token', 'max_seq_len', 'model_execute_time'),
        ('input_length', 'need_blocks', 'output_length')
    ])


@dataclass
class ExecutionData:
    exec_data: List[Tuple]
    batch_data: List[Tuple]
    req_df: pd.DataFrame
    rids_ori: List[Any]
    index_dict: Dict[Tuple, Any]
    batch_id_block_sum: Dict[int, float]


def process_execution_data(csv_data: ExecutionData) -> List[Tuple]:

    processed_data = []
    for i, _ in enumerate(csv_data.exec_data):
        exec_row = csv_data.exec_data[i]
        batch_row = csv_data.batch_data[i]
        total_prefill_token = 0
        max_seq_len = 0
        total_req_info = []
        data = ast.literal_eval(csv_data.rids_ori[i])
        rids = []
        iters = []
        req_df = csv_data.req_df
        for item in data:
            try:
                rid = item['rid']
                rids.append(rid)
                iter_val = item['iter']
                iters.append(iter_val)
            except KeyError as e:
                logger.error(f"缺少键 '{e}'，跳过该条目")
        for j, _ in enumerate(rids):
            recv_token = req_df[req_df['http_rid'] == rids[j]]['recv_token_size'].values[0]
            total_prefill_token += recv_token
            if recv_token > max_seq_len:
                max_seq_len = recv_token
            req_blcok = csv_data.index_dict.get((rids[j], iters[j]), 0)
            req_info = (int(recv_token), req_blcok, iters[j])
            total_req_info.append(req_info)
        process_req_info = tuple(total_req_info)
        start = exec_row[3]
        end = exec_row[4]
        model_exec = end - start
        current_batch_id = i
        block_sum = csv_data.batch_id_block_sum.get(current_batch_id + 1, 0)

        combined_row = list(exec_row) + list(batch_row) + [model_exec, block_sum, total_prefill_token, max_seq_len]
        if combined_row[10] == 'Prefill':
            combined_row[10] = 'prefill'
        else:
            combined_row[10] = 'decode'
        tuple_elements = (
            combined_row[10],
            combined_row[9],
            combined_row[16],
            int(combined_row[17]),
            int(combined_row[18]),
            combined_row[15]
        )
        combined_row = tuple([tuple_elements]) + tuple([process_req_info])
        processed_data.append(combined_row)
    return processed_data


def write_csv_row(csvfile, row: Tuple) -> None:
    writer = csv.writer(csvfile)
    writer.writerow(row)


@dataclass
class ProcessedData:
    input_path: str
    data_by_pid: Dict[int, List[Tuple]]
    batch_rows: List[Tuple]
    batch_id_block_sum: Dict[int, float]
    req_df: pd.DataFrame
    rids_ori: List[Any]
    index_dict: Dict[Tuple, Any]


def save_processed_data_to_csv(
        processed_data: ProcessedData
) -> None:
    output_folder = create_output_folder(processed_data.input_path)
    batch_rows = processed_data.batch_rows
    for pid, exec_data in processed_data.data_by_pid.items():
        current_batch_index = 0
        batch_data = process_batch_data(exec_data, batch_rows, current_batch_index)

        parrent_path = os.path.join(output_folder, f'pid_{pid}')
        os.makedirs(parrent_path, exist_ok=True)
        file_path = os.path.join(parrent_path, 'feature.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            write_csv_header(csvfile)
            an_data = ExecutionData(exec_data, batch_data, processed_data.req_df, processed_data.rids_ori, 
                                    processed_data.index_dict, processed_data.batch_id_block_sum)
            feature_data = process_execution_data(an_data)
            for row in feature_data:
                write_csv_row(csvfile, row)


def source_to_model(input_path: str):
    ori_db_path = os.path.join(input_path, 'profiler.db')
    db_connector = DatabaseConnector(ori_db_path)
    cursor = db_connector.connect()

    try:
        exec_rows = read_batch_exec_data(cursor)
        batch_rows = read_batch_data(cursor)
        req_rows = read_batch_req_data(cursor)
        index_dict = {}
        for row in req_rows:
            key = (row[1], row[3])
            value = row[4]
            index_dict[key] = value
        data_by_pid = group_exec_data_by_pid(exec_rows)
        batch_id_block_sum = calculate_block_sums(req_rows)

        csv_file = os.path.join(input_path, 'request.csv')
        req_df = pd.read_csv(csv_file, header=0)

        rids_ori = fetch_rids_from_db(ori_db_path)
        csv_data = ProcessedData(input_path,
            data_by_pid,
            batch_rows,
            batch_id_block_sum,
            req_df,
            rids_ori,
            index_dict)
        save_processed_data_to_csv(csv_data)

    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise e
    finally:
        db_connector.close()


def req_decodetimes(input_path, output_path):
    csv_file = os.path.join(input_path, f'request.csv')
    json_file = os.path.join(output_path, f'req_id_and_decode_num.json')

    # 初始化一个空字典来存储数据
    data = {}

    # 打开并读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            http_reqid = row['http_rid']
            reply_token_size = int(float(row['reply_token_size']))
            data[http_reqid] = reply_token_size

    # 将字典写入JSON文件
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


parser = argparse.ArgumentParser(prog="Train Model.")
parser.add_argument("-i", "--input", default=None, type=Path, required=True)
parser.add_argument("-o", "--output", default=Path("output"), type=Path)


def main(args):
    input_path = args.input
    output_path = args.output
    # 读取输入文件
    try:
        source_to_model(input_path)
    except IOError as e:
        logger.error(f"无法读取输入文件: {e}")
        raise e
    # 确保输出目录存在
    input_csv_path = os.path.join(input_path, f'output_csv')
    pretrain(input_csv_path, output_path)
    req_decodetimes(input_path, output_path)


if __name__ == '__main__':
    _args = parser.parse_args()
    main(_args)
