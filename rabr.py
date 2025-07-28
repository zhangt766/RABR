
import openai
import numpy as np
from collections import defaultdict

class RABR:
    def __init__(self, item_list, item_embeddings, user_embeddings=None, gamma=0.5, temperature=0.7):
        self.item_list = item_list  # list of item titles (e.g., ["iPhone", "Laptop"])
        self.item_embeddings = item_embeddings  # dict {title: np.array}
        self.user_embeddings = user_embeddings or {}
        self.gamma = gamma
        self.temperature = temperature

    def generate_bundles(self, prompt, s):
        return [self.call_gpt(prompt) for _ in range(s)]

    def call_gpt(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=128
        )
        content = response["choices"][0]["message"]["content"]
        return self.parse_items(content)

    def parse_items(self, content):
        return [item.strip() for item in content.split(',') if item.strip() in self.item_list]

    def score_items(self, user_id, completions):
        c_i = defaultdict(int)
        sim_i = defaultdict(float)
        total_freq = 0

        for comp in completions:
            for item in comp:
                if item in self.item_list:
                    c_i[item] += 1
                    total_freq += 1
                    sim_i[item] += self._similarity(item, user_id)

        score = {
            item: (c_i[item] / total_freq) * sim_i[item]
            for item in c_i
        }
        return score

    def _similarity(self, item, user_id):
        item_vec = self.item_embeddings.get(item)
        user_vec = self.user_embeddings.get(user_id)
        if item_vec is None:
            return 0.0
        if user_vec is None:
            return 1.0  # fallback to lexical only
        lexical_sim = self._cosine(item_vec, item_vec)
        user_sim = self._cosine(user_vec, item_vec)
        return self.gamma * lexical_sim + (1 - self.gamma) * user_sim

    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def construct_bundle(self, scores, lambda_thres):
        return [item for item, score in scores.items() if score >= lambda_thres]
