
import numpy as np
import os

class RABRDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.item_titles = np.load(os.path.join(data_dir, 'item_titles.npy'), allow_pickle=True).tolist()
        self.session_items = np.load(os.path.join(data_dir, 'session_items.npy'), allow_pickle=True).tolist()
        self.test_set = np.load(os.path.join(data_dir, 'test_set.npy'), allow_pickle=True)
        self.calibrate_set = np.load(os.path.join(data_dir, 'calibrate_set.npy'), allow_pickle=True)

    def get_prompt_from_session(self, session_id):
        history_item_ids = self.session_items[session_id]
        item_names = [self.item_titles[i] for i in history_item_ids if i < len(self.item_titles)]
        return f"The user has interacted with items: ={{{', '.join(item_names)}}}. Please generate a bundle basing on this."

    def get_calibration_prompts(self):
        prompts = []
        for session_id, gt_bundle_ids in self.calibrate_set:
            prompt = self.get_prompt_from_session(session_id)
            gt_bundle = [self.item_titles[i] for i in gt_bundle_ids if i < len(self.item_titles)]
            prompts.append((session_id, prompt, gt_bundle))
        return prompts

    def get_test_prompts(self):
        prompts = []
        for session_id, gt_bundle_ids in self.test_set:
            prompt = self.get_prompt_from_session(session_id)
            gt_bundle = [self.item_titles[i] for i in gt_bundle_ids if i < len(self.item_titles)]
            prompts.append((session_id, prompt, gt_bundle))
        return prompts
