# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


import importlib
import sys

from loguru import logger
from packaging import version

from msserviceprofiler.modelevalstate.inference.simulate import Simulate, ServiceField


def generate_token(self, input_metadata):
    from mindie_llm.modeling.backend_type import BackendType
    from mindie_llm.utils.env import ENV
    from mindie_llm.utils.prof.profiler import span_start, span_end, span_req

    logger.info("proprocess")
    prof = span_start("preprocess", True)
    self.plugin_data_param.q_len = None
    self.plugin_data_param.mask = None
    cache_ids, model_inputs, sampling_metadata, trace_ids = self.preprocess(input_metadata)
    logger.info("model_inputs_update_manager")
    model_inputs, qlen, mask = self.model_inputs_update_manager(model_inputs, input_metadata, cache_ids)
    self.plugin_data_param.q_len = qlen if qlen is not None else self.plugin_data_param.q_len
    self.plugin_data_param.mask = mask if mask is not None else self.plugin_data_param.mask
    span_req(prof, trace_ids)
    prof.attr("blocks", [int(x) for x in np.count_nonzero(input_metadata.block_tables > -1, axis=1)])
    logger.info("generate_token")
    logger.info("simulate init")
    try:
        Simulate.init(self)
    except Exception as e:
        logger.error(f"Failed in simulate init. error {e}")
        raise e
    try:
        Simulate.generate_features(self, input_metadata, cache_ids)
    except Exception as e:
        logger.error(f"Failed in generate features, error {e}")
        raise e
    prof = span_start("forward", True)
    logger.info("forward")
    if ServiceField.batch_field:
        Simulate.predict_and_save()
        result = Simulate.generate_logits(input_metadata.block_tables.shape[0],
                                          self.model_wrapper.config.vocab_size, self.model_wrapper.device)
    else:
        if ENV.framework_backend == BackendType.ATB:
            if (self.plugin_list and "mtp" not in self.plugin_list) or self.is_mix_model:
                result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                        attn_mask=self.plugin_data_param.mask)  # q_len spec_mask
            # old graph forward
            else:
                result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                        spec_mask=self.plugin_data_param.mask)
        else:
            result = self.generator_backend.forward(model_inputs, q_lens=self.plugin_data_param.q_len,
                                                    spec_mask=self.plugin_data_param.mask)  # q_len spec_mask
    if isinstance(result, tuple):
        logits, hidden_states = result
    else:
        logits = result
    span_end(prof, True)
    logger.info("sample")
    prof = span_start("sample", True)
    draft_filtered_logits = self.sample_preprocess_manager(logits, sampling_metadata, input_metadata)
    sampling_output = self.generator_backend.sample(draft_filtered_logits, sampling_metadata)
    span_end(prof, True)
    logger.info("postprocess")
    prof = span_start("postprocess", True)
    generation_output = self.postprocess(
        cache_ids, input_metadata, result, sampling_metadata, sampling_output)
    span_end(prof, True)
    generation_output.trace_ids = trace_ids
    return generation_output


def update_plugin_manager(module):
    logger.info(f"Patch {module}")
    origin_class = getattr(module, "PluginManager")
    origin_class.generate_token = generate_token
    origin_class.modelevalstate = True
    return module


def update_plugin_manager_class(module):
    logger.info(f"Patch {module}")
    module.generate_token = generate_token
    module.modelevalstate = True
    return module


_post_import_hooks = {
    "mindie_llm.text_generator.plugins.plugin_manager": update_plugin_manager,
    "mindie_llm.text_generator.plugins.plugin_manager.PluginManager": update_plugin_manager_class

}


class PostImportFinder:
    def __init__(self):
        self.skip = set()

    def find_module(self, fullname, path=None):
        if fullname in self.skip:
            return None
        if fullname not in _post_import_hooks:
            return None
        self.skip.add(fullname)
        return PostImportLoader(self)


class PostImportLoader:
    def __init__(self, finder):
        self.finder = finder

    def load_module(self, fullname):
        importlib.import_module(fullname)
        module = sys.modules[fullname]
        if fullname in _post_import_hooks:
            _post_import_hooks[fullname](module)
        self.finder.skip.remove(fullname)
        return module


class Patch2rc1:
    mindie_llm = "2.0rc1"
    mindie_llm_low = "2.0a9"

    @staticmethod
    def check_version(target_version):
        _t_v = version.parse(target_version)
        _c_v_up = version.parse(Patch2rc1.mindie_llm)
        _c_v_low = version.parse(Patch2rc1.mindie_llm_low)
        if _c_v_low < _t_v <= _c_v_up:
            return True
        else:
            return False

    @staticmethod
    def patch():
        sys.meta_path.append(PostImportFinder())
        logger.info("Successful patch")
