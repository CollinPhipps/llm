from collections import Counter, deque, defaultdict
import json
import time

class Tokenizer:
    def __init__(self):
        self.id_to_token = {}  # num to string
        self.token_to_id = {}  # string to num
        self.merges = {}  # (id1, id2) -> merged_id

    def train(self, text, vocab_size):
        processed_text = []
        for char in text:
            if char == " ":
                processed_text.append("Ġ")
            else:
                processed_text.append(char)
        processed_text = "".join(processed_text)

        unique_chars = [chr(i) for i in range(256) if i != 32]
        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.id_to_token = {i: char for i, char in enumerate(unique_chars)}
        self.token_to_id = {char: i for i, char in enumerate(unique_chars)}

        token_ids = [self.token_to_id[char] for char in processed_text]
        next = [i+1 for i in range(len(processed_text)-1)] + [-1]
        prev = [i-1 for i in range(len(processed_text))]
        pair_counts = defaultdict(int) # (id1, id2) -> count
        pair_positions = defaultdict(set) # (id1, id2) -> set of positions (each i means "this pair currently starts at index i")

        i = 0
        while True:
            if next[i] == -1:
                break

            pair = (token_ids[i], token_ids[next[i]])
            pair_counts[pair] += 1
            pair_positions[pair].add(i)

            i = next[i]

        alive = [True] * len(token_ids)

        for new_id in range(len(self.id_to_token), vocab_size):
            if not pair_counts:
                break
            
            winning_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            a, b = winning_pair
            for pos in pair_positions[winning_pair]:
                if not alive[pos] or token_ids[pos] != a or next[pos] == -1 or not alive[next[pos]] or token_ids[next[pos]] != b:
                    continue 

                b_pos = next[pos]
                l = prev[pos]
                r = next[b_pos]
                if l != -1:
                    pair_counts[(token_ids[l], a)] -= 1
                if r != -1:
                    pair_counts[(b, token_ids[r])] -= 1

                token_ids[pos] = new_id
                next[pos] = r
                if r != -1:
                    prev[r] = pos
                alive[b_pos] = False

                if l != -1:
                    pair_counts[(token_ids[l], new_id)] += 1
                    pair_positions[(token_ids[l], new_id)].add(l)
                if r != -1:
                    pair_counts[(new_id, token_ids[r])] += 1
                    pair_positions[(new_id, token_ids[r])].add(pos)

            self.merges[(a, b)] = new_id
            del pair_counts[winning_pair]
            del pair_positions[winning_pair]

        for (p0, p1), id in self.merges.items():
            merged_token = self.id_to_token[p0] + self.id_to_token[p1]
            self.id_to_token[id] = merged_token
            self.token_to_id[merged_token] = id

    def decode(self, token_ids):
        text = "".join(self.id_to_token[i] for i in token_ids)
        return text.replace("Ġ", " ")

    def encode(self, text: str):
        processed_text = []
        for i, char in enumerate(text):
            if char == " ":
                processed_text.append("Ġ")
            else:
                processed_text.append(char)
        processed_text = "".join(processed_text)

        token_ids = []
        for char in processed_text:
            if char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                raise ValueError(f"{char} is not a valid token")

        for (id1, id2), merged_id in self.merges.items():
            token_ids = self.replace_pair(token_ids, (id1, id2), merged_id)
        return token_ids

    def save(self, path: str):
        data = {
            "id_to_token": self.id_to_token,
            # JSON has no tuple keys, so store merges as an ordered list of
            # [id1, id2, merged_id] triples instead of a dict — this also
            # preserves merge order for free, which encode() depends on.
            "merges": [
                [id1, id2, merged_id] for (id1, id2), merged_id in self.merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # JSON object keys are always strings, so int ids were silently
        # stringified on save — convert them back or every id_to_token lookup
        # by int downstream will fail to match.
        self.id_to_token = {int(i): token for i, token in data["id_to_token"].items()}
        self.token_to_id = {token: i for i, token in self.id_to_token.items()}

        self.merges = {}
        for id1, id2, merged_id in data["merges"]:
            self.merges[(id1, id2)] = merged_id

    @staticmethod
    def replace_pair(token_ids, pair_id, id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(id)
                dq.popleft()
            else:
                replaced.append(current)

        return replaced


if __name__ == "__main__":
    with open("corpus.txt", "r") as f:
        text = f.read()

    vocab_size = 5_000
    tok = Tokenizer()
    start_time = time.perf_counter()
    tok.train(text, vocab_size)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    tok.save("vocab.json")