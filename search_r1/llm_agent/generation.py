# 这个文件负责RAG和强化学习的结合
# 具体来说，它从LLM的响应中提取action和content，然后根据action执行搜索或回答，最后返回observation
# 再把刚才的这些内容拼接到历史信息中，作为新的输入，继续进行下一轮的生成
import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int  # 多轮对话的最大轮数
    max_start_length: int      # 初始提示的最大长度
    max_prompt_length: int     # 整个提示的最大长度
    max_response_length: int   # 单次响应的最大长度
    max_obs_length: int        # 观察结果的最大长度
    num_gpus: int  # 这是为了确保批次大小能被GPU数量整除
    no_think_rl: bool=False  # 是否启用无思考RL，如果启用，则只保留动作，不保留思考
    search_url: str = None  # 检索系统部署的URL，例如http://127.0.0.1:8000/retrieve
    topk: int = 3  # 每个查询返回的文档数量

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,  # 演员模型的工作组，负责实际的文本生成
        config: GenerationConfig,
        is_validation: bool = False,  # 标识当前是否为验证模式
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # TensorHelper实例，用于处理张量操作，包括填充、连接、注意力掩码等
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        # 将一批文本响应（responses）批量转换为token ID张量
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False,  # 不添加特殊token（如[CLS], [SEP]等）
            return_tensors='pt',  # 返回PyTorch张量格式
            padding="longest"  # 使用最长的序列长度进行填充，确保所有序列长度一致
        )['input_ids']  # 从tokenizer返回的字典中提取input_ids字段

    # 传入token ID张量格式的responses，返回处理后的token张量和字符串列表
    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )  # 通过batch_decode将token ID张量转换为字符串列表，跳过特殊token

        # 遍历列表里每个str，在第一个 </search> 处截断，并保留完整的 </search> 标签
        # 如果包含 </answer> 在第一个 </answer> 处截断，并保留完整的 </answer> 标签
        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:  # 如果启用无思考RL，只保留动作部分
            raise ValueError('stop')#这里直接抛出异常，因为这部分还没实现
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)  # 将字符串重新转换为token ID张量
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        # 用于处理来自环境的下一批观察结果（str列表形式），将它们转换为模型可以处理的token ID张量格式
        # obs通常是RAG的检索结果，格式为<information>...</information>
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']  # 对next_obs列表里的每个str进行tokenize，返回token ID张量

        if next_obs_ids.shape[1] > self.config.max_obs_length:  # 如果长度超过最大观察长度
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]#在长度这个维度上截断

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        #  将新的响应和观察结果合并到现有的对话历史中
        # 输入：rollings是当前的对话历史，cur_responses是当前的响应（token ID张量），next_obs_ids是新的观察结果（token ID张量）
        # 输出：新的对话历史（Dict类型）
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding   
        # 使用TensorHelper的连接方法，将三个部分按顺序连接
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        # 为刚才创建的token ID张量，创建注意力掩码和位置ID
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()  # 统计非填充token的数量
        max_len = min(self.config.max_prompt_length, effective_len)# 取最小值，确保不超过最大长度

        new_rollings = DataProto.from_dict({
            # 全部保留最右边的max_len个token，确保包含最新的对话内容
            # 要保存的信息包括：token ID、位置ID、注意力掩码
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)#把原始的meta_info也更新到新的rollings中
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True#是否在左边填充
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]#把prompt和response放进tensors列表
        tensors_with_mask = [prompt_with_mask, response]#把prompt_with_mask和response放进tensors_with_mask列表
        if info is not None:
            tensors.append(info)#把info放进tensors列表
            # 创建一个与信息块相同大小的掩码，用填充token ID填充
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)#把info_mask放进tensors_with_mask列表
        
        concatenated = torch.cat(tensors, dim=1)#把tensors列表里的所有张量按列连接起来
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)#把tensors_with_mask列表里的所有张量按列连接起来
        # 如果左侧填充，则mask为非填充token的部分，否则为填充token的部分
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        # 将布尔掩码转换为整数，然后稳定排序（也就是不改变相等元素的相对位置）
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # 根据索引重新排列两个张量
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(
            1, sorted_indices)  # 信息被掩码的版本

        return padded_tensor, padded_tensor_with_info

    # 更新右侧状态字典，输入：当前的右侧状态字典，当前的响应（token ID张量），新的观察结果（token ID张量）
    # 输出：新的右侧状态字典
    def _update_right_side(self, right_side: Dict, #当前的右侧状态字典
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:  # 如果存在新的观察结果
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, #观察结果作为info
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        # right_side包含responses和responses_with_info_mask，在返回之前要在长度这个维度上截断
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    # 批次大小必须能被GPU数量整除，这个方法通过添加填充序列来满足这个要求
    # 入参：活跃的对话历史（DataProto类型）
    # 出参：处理后的对话历史（DataProto类型）
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus#计算batch_size除以num_gpus的余数
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()#把所有张量转换为long类型
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        #left_side是初始输入，字典类型，包含input_ids，right_side是初始响应，字典类型，包含responses和responses_with_info_mask
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        # 哪些序列还在活跃状态
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # 统计每个序列的轮数
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 统计每个序列的有效动作数
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 统计每个序列的有效搜索数
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 记录每轮活跃序列数量，它是active_mask的和
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch  # 滚动状态: 维护对话历史

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():  # 如果没有active_mask，说明所有序列都已结束
                break  # 退出循环
            # 确保对话历史rollings的input_ids，attention_mask，position_ids不超过有效长度
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            # 通过active_mask筛选出活跃的序列，放入rollings_active
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            # 用_postprocess_responses后处理
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 用_example_level_pad对每个序列的响应进行填充，确保所有序列的响应长度一致
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            # 执行预测，获取每个序列的下一个观察结果，以及是否结束、是否有效动作、是否有效搜索
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            # 根据done，更新active_mask
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())#记录每轮活跃序列数量
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)#处理观察结果
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        # 当所有序列都结束时，进行最终的LLM生成，基本与之前一样，只是do_search=False
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )#注意这类execute_predictions的时候，do_search=False，所以不会执行搜索

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        # 将所有统计信息添加到元信息中
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    # 输入：left_side, right_side, meta_info
    # 输出：final_output
    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )  # 只为有效token分配位置信息
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    # 执行预测，返回str列表，每个str包含一个observation
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(
            predictions)  # 从predictions中提取动作和内容
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # 如果动作是search，则提取content
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:  # 如果启用搜索
            search_results = self.batch_search(search_queries)#调用batch_search进行批量搜索，返回搜索结果
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])#确保搜索结果数量与搜索动作数量一致
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])#如果没有搜索，则返回空字符串

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):  # 遍历每个活跃的预测
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':  # 如果动作是answer
                    next_obs.append('')
                    dones.append(1)#说明已经结束
                    valid_action.append(1)#属于有效动作
                    is_search.append(0)
                elif action == 'search':
                    # 把search_result添加info标签，添加到next_obs中
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)  # 属于有效动作
                    is_search.append(1)
                else:#如果动作是其他，则返回错误信息
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    # 后处理预测，输入：predictions，输出：动作列表和内容列表
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # 匹配<search>内容</search>或<answer>内容</answer>格式
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']#从响应中提取result
        
        return [self._passages2string(result) for result in results]#把result转换为字符串

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True#在RAG的时候是否返回分数
        }
        #用post方式发送请求，返回json格式
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            # 取第一行作为标题
            title = content.split("\n")[0]
            # 把后面的内容作为text，用\n连接
            text = "\n".join(content.split("\n")[1:])
            # 把标题和text拼接起来，用Doc {idx+1}(Title: {title}) {text}\n格式
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
