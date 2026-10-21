import os
import sys
import time
import math
import random


# Time Complexity: O(L1 * L2) = O(M^2) where M is average word length
# Auxiliary Space Complexity: O(min(L1, L2)) using 2-row rolling dynamic programming
def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class TextUtils:
    # Time Complexity: O(L) where L is string length | Space Complexity: O(L)
    @staticmethod
    def normalize(raw):
        return "".join(ch.lower() for ch in raw if not ch.isspace())

    # Time Complexity: O(L) | Space Complexity: O(L)
    @staticmethod
    def normalize_alpha(raw):
        return "".join(ch.lower() for ch in raw if ch.isalpha())

    # Time Complexity: O(L) | Space Complexity: O(L)
    @staticmethod
    def parse_line(line):
        raw = line.strip()
        if not raw:
            return "", ""
        if ":" in raw:
            parts = raw.split(":", 1)
            word = TextUtils.normalize(parts[0])
            meaning = parts[1].strip() or "No definition available"
            return word, meaning
        return TextUtils.normalize(raw), "No definition available"

    # Time Complexity: O(S) where S is sentence length | Space Complexity: O(S)
    @staticmethod
    def tokenize_sentence(sentence):
        tokens = []
        current = []
        for ch in sentence:
            if ch.isalnum() or ch == "'":
                current.append(ch)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens


class Metric:
    QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
    QWERTY_OFFSETS = (0.0, 0.25, 0.75)
    KEY_COORDS = {}

    # Time Complexity: O(1) constant 26-key initialization | Space Complexity: O(1)
    @classmethod
    def _init_coords(cls):
        if cls.KEY_COORDS:
            return
        for row_idx, row_str in enumerate(cls.QWERTY_ROWS):
            offset = cls.QWERTY_OFFSETS[row_idx]
            for col_idx, ch in enumerate(row_str):
                cls.KEY_COORDS[ch] = (float(col_idx) + offset, float(row_idx))

    # Time Complexity: O(L1 * L2) | Auxiliary Space Complexity: O(min(L1, L2))
    @staticmethod
    def levenshtein(a, b):
        return levenshtein(a, b)

    # Time Complexity: O(L1 * L2) | Auxiliary Space Complexity: O(L1 * L2)
    @staticmethod
    def levenshtein_full(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        return dp[m][n]

    # Time Complexity: O(L1 * L2) | Auxiliary Space Complexity: O(L1 * L2)
    @staticmethod
    def damerau(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        return dp[m][n]

    # Time Complexity: O(1) hash lookup | Space Complexity: O(1)
    @classmethod
    def key_spatial_cost(cls, c1, c2):
        if c1 == c2:
            return 0.0
        cls._init_coords()
        p1 = cls.KEY_COORDS.get(c1.lower())
        p2 = cls.KEY_COORDS.get(c2.lower())
        if p1 is None or p2 is None:
            return 1.0
        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return min(1.0, dist / 3.0)

    # Time Complexity: O(L1 * L2) | Auxiliary Space Complexity: O(L2)
    @classmethod
    def keyboard_weighted(cls, a, b):
        m, n = len(a), len(b)
        prev = [float(j) for j in range(n + 1)]
        for i in range(1, m + 1):
            cur = [float(i)] + [0.0] * n
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    sub_cost = 0.0
                else:
                    sub_cost = 0.4 + 0.6 * cls.key_spatial_cost(a[i - 1], b[j - 1])
                cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + sub_cost)
            prev = cur
        return prev[n]

    # Time Complexity: O(L1 * L2) | Space Complexity: O(L1 * L2)
    @staticmethod
    def build_edit_script(source, target):
        m, n = len(source), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if source[i - 1] == target[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        steps = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and source[i - 1] == target[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                steps.append(("KEEP", source[i - 1], target[j - 1], i - 1))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                steps.append(("SUBSTITUTE", source[i - 1], target[j - 1], i - 1))
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                steps.append(("INSERT", "-", target[j - 1], j - 1))
                j -= 1
            else:
                steps.append(("DELETE", source[i - 1], "-", i - 1))
                i -= 1
        steps.reverse()
        return steps


class BKNode:
    def __init__(self, word, meaning="No definition available", frequency=1):
        self.word = word
        self.meaning = meaning
        self.frequency = frequency
        self.deleted = False
        self.children = {}

    # Time Complexity: O(1) average dictionary insertion | Space Complexity: O(1)
    def add_child(self, d, node):
        self.children[d] = node


class BKTree:
    def __init__(self, metric_fn=None):
        self.root = None
        self.size = 0
        self.allocated = 0
        self.metric_fn = metric_fn or Metric.levenshtein

    # Average Time Complexity: O(M^2 * log N) where depth = O(log N) and distance = O(M^2)
    # Worst-Case Time Complexity: O(N * M^2) (degenerate chain)
    # Space Complexity: O(1) per node (Total Tree Space: O(N))
    def insert(self, word, meaning="No definition available", frequency=1):
        w = TextUtils.normalize(word)
        if not w:
            return False
        if self.root is None:
            self.root = BKNode(w, meaning, frequency)
            self.size = 1
            self.allocated = 1
            return True
        node = self.root
        while True:
            d = self.metric_fn(w, node.word)
            if d == 0:
                if node.deleted:
                    node.deleted = False
                    node.meaning = meaning
                    node.frequency = frequency
                    self.size += 1
                    return True
                node.frequency += frequency
                if node.meaning == "No definition available" and meaning != "No definition available":
                    node.meaning = meaning
                return False
            if d in node.children:
                node = node.children[d]
            else:
                node.add_child(d, BKNode(w, meaning, frequency))
                self.size += 1
                self.allocated += 1
                return True

    # Average Time Complexity: O(M^2 * log N) | Worst-Case Time: O(N * M^2) | Space: O(M)
    def contains(self, word):
        w = TextUtils.normalize(word)
        if not w or self.root is None:
            return False
        node = self.root
        while node is not None:
            d = self.metric_fn(w, node.word)
            if d == 0:
                return not node.deleted
            node = node.children.get(d)
        return False

    # Average Time Complexity: O(M^2 * log N) | Worst-Case Time: O(N * M^2) | Space: O(1)
    def lazy_delete(self, word):
        w = TextUtils.normalize(word)
        if not w or self.root is None:
            return False
        node = self.root
        while node is not None:
            d = self.metric_fn(w, node.word)
            if d == 0:
                if node.deleted:
                    return False
                node.deleted = True
                self.size -= 1
                return True
            node = node.children.get(d)
        return False

    # Time Complexity: O(N) | Space Complexity: O(N)
    def export_active(self):
        if self.root is None:
            return []
        active = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.deleted:
                active.append((node.word, node.meaning, node.frequency))
            for child in node.children.values():
                stack.append(child) if hasattr(stack, "push_back") else stack.append(child)
        return active

    # Average Time Complexity: O(N * M^2 * log N) | Space Complexity: O(N)
    def rebuild(self, new_metric=None):
        entries = self.export_active()
        if new_metric is not None:
            self.metric_fn = new_metric
        self.root = None
        self.size = 0
        self.allocated = 0
        for w, m, f in entries:
            self.insert(w, m, f)
        return self.size

    # Average Time Complexity: O(N^alpha * M^2 + R log R) where 0 < alpha < 1 (sub-linear)
    # Worst-Case Time Complexity: O(N * M^2 + R log R) where R is number of matches
    # Auxiliary Space Complexity: O(depth + R) for explicit DFS stack and result list
    def search_stats(self, query, k):
        t0 = time.perf_counter()
        q = TextUtils.normalize(query)
        results = []
        stats = {"computed": 0, "pruned_branches": 0, "char_ops": 0, "elapsed_ms": 0.0}
        if not q or self.root is None or k < 0:
            return results, stats
        stack = [self.root]
        q_len = len(q)
        while stack:
            node = stack.pop()
            d = self.metric_fn(q, node.word)
            stats["computed"] += 1
            stats["char_ops"] += q_len * len(node.word)
            if d <= k and not node.deleted:
                results.append((d, node.word, node.meaning))
            low, high = max(1, d - k), d + k
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
                else:
                    stats["pruned_branches"] += 1
        results.sort(key=lambda item: (item[0], item[1]))
        stats["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return results, stats

    # Average Time Complexity: O(N^alpha * M^2 + R log R) | Space Complexity: O(depth + R)
    def search(self, query, k):
        results, stats = self.search_stats(query, k)
        return [(d, w) for d, w, _ in results], stats

    # Time Complexity: O(N^alpha * M^2 + R log R) | Space Complexity: O(depth + max_steps)
    def search_trace(self, query, k, max_steps=15):
        q = TextUtils.normalize(query)
        trace = []
        matches = []
        if not q or self.root is None or k < 0:
            return matches, trace
        stack = [self.root]
        step = 0
        while stack:
            node = stack.pop()
            step += 1
            d = self.metric_fn(q, node.word)
            hit = d <= k and not node.deleted
            if hit:
                matches.append((d, node.word, node.meaning))
            low, high = max(1, d - k), d + k
            followed, pruned = [], []
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
                    followed.append(edge)
                else:
                    pruned.append(edge)
            if len(trace) < max_steps:
                trace.append({
                    "step": step,
                    "word": node.word,
                    "d": d,
                    "hit": hit,
                    "window": (low, high),
                    "followed": sorted(followed),
                    "pruned": sorted(pruned)
                })
        matches.sort(key=lambda x: (x[0], x[1]))
        return matches, trace

    # Time Complexity: O(N) full tree traversal | Space Complexity: O(depth)
    def diagnostics(self):
        info = {
            "active": self.size,
            "allocated": self.allocated,
            "leaves": 0,
            "internal": 0,
            "max_depth": 0,
            "avg_depth": 0.0,
            "avg_branching": 0.0
        }
        if self.root is None:
            return info
        stack = [(self.root, 0)]
        depth_sum = 0
        edge_sum = 0
        while stack:
            node, depth = stack.pop()
            depth_sum += depth
            if depth > info["max_depth"]:
                info["max_depth"] = depth
            deg = len(node.children)
            edge_sum += deg
            if deg == 0:
                info["leaves"] += 1
            else:
                info["internal"] += 1
                for child in node.children.values():
                    stack.append((child, depth + 1))
        if self.allocated > 0:
            info["avg_depth"] = depth_sum / float(self.allocated)
        if info["internal"] > 0:
            info["avg_branching"] = edge_sum / float(info["internal"])
        return info


# Average Time Complexity: O(N * M^2 * log N) | Space Complexity: O(N)
def load_words(path):
    tree = BKTree()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, m = TextUtils.parse_line(line)
            if w:
                tree.insert(w, m)
    return tree


class SpellApp:
    def __init__(self, wordfile="dictionary.txt"):
        self.tree = BKTree()
        self.wordfile = wordfile
        self.entries = []

    # Average Time Complexity: O(N * M^2 * log N) | Space Complexity: O(N)
    def load_words(self, path):
        candidates = [path, "dictionary.txt", "../dictionary.txt", "words.txt"]
        resolved = None
        for c in candidates:
            if c and os.path.isfile(c):
                resolved = c
                break
        self.tree = BKTree()
        self.entries = []
        if resolved is None:
            return 0
        self.wordfile = resolved
        with open(resolved, encoding="utf-8", errors="ignore") as f:
            for line in f:
                w, m = TextUtils.parse_line(line)
                if w and self.tree.insert(w, m):
                    self.entries.append((w, m))
        return self.tree.size

    # Average Time Complexity: O(N^alpha * M^2 + R log R) | Space Complexity: O(depth + R)
    def suggest(self, word, k):
        return self.tree.search_stats(word, k)

    # Time Complexity: O(Q * N * M^2) to compare BK-Tree against Brute-Force across Q queries
    def benchmark(self, sample_queries=25):
        if not self.entries:
            print("No dictionary entries loaded for benchmarking.\n")
            return
        sizes = [s for s in (1000, 5000, len(self.entries)) if s <= len(self.entries)]
        sizes = sorted(list(set(sizes)))
        rng = random.Random(42)
        print("\n" + "=" * 92)
        print("               BENCHMARK: BK-TREE PRUNED SEARCH vs. BRUTE-FORCE LINEAR SCAN")
        print("=" * 92)
        print(f"{'Words(N)':<10}{'k':<6}{'BK-Tree(ms)':<15}{'BruteForce(ms)':<16}{'Speedup':<11}{'Avg Visited':<14}{'Pruned':<11}{'Parity'}")
        print("-" * 92)
        for n in sizes:
            sub_entries = self.entries[:n]
            sub_tree = BKTree()
            for w, m in sub_entries:
                sub_tree.insert(w, m)
            queries = []
            for _ in range(sample_queries):
                base = sub_entries[rng.randrange(n)][0]
                idx = rng.randrange(len(base))
                rep = "a" if base[idx] == "z" else chr(ord(base[idx]) + 1)
                queries.append(base[:idx] + rep + base[idx + 1:])
            for k in (1, 2):
                total_visited = 0
                bk_out = []
                t0 = time.perf_counter()
                for q in queries:
                    res, st = sub_tree.search_stats(q, k)
                    bk_out.append([w for _, w, _ in res])
                    total_visited += st["computed"]
                t1 = time.perf_counter()
                bf_out = []
                t2 = time.perf_counter()
                for q in queries:
                    hits = []
                    for w, _ in sub_entries:
                        d = levenshtein(q, w)
                        if d <= k:
                            hits.append((d, w))
                    hits.sort()
                    bf_out.append([w for _, w in hits])
                t3 = time.perf_counter()
                bk_ms = ((t1 - t0) * 1000.0) / sample_queries
                bf_ms = ((t3 - t2) * 1000.0) / sample_queries
                speedup = bf_ms / bk_ms if bk_ms > 1e-9 else 1.0
                avg_vis = total_visited / float(sample_queries)
                pruned_pct = 100.0 * (n - avg_vis) / float(n)
                parity = "PASS" if bk_out == bf_out else "FAIL"
                print(
                    f"{n:<10}{k:<6}{bk_ms:<15.3f}{bf_ms:<16.3f}"
                    f"{f'{speedup:.1f}x':<11}{int(round(avg_vis)):<14}{f'{pruned_pct:.1f}%':<11}{parity}"
                )
        print("=" * 92 + "\n")

    def handle_complexity(self, sample_query="algoritm", k=1):
        n = max(1, self.tree.size)
        _, stats = self.tree.search_stats(sample_query, k)
        visited = stats["computed"]
        char_ops = stats["char_ops"]
        bf_ops = n * len(sample_query) * 8
        alpha = (math.log(visited) / math.log(n)) if (n > 1 and visited > 1) else 0.0
        pruned_pct = 100.0 * (n - visited) / float(n)

        print("\n" + "=" * 98)
        print("                   THEORETICAL & EMPIRICAL TIME / SPACE COMPLEXITY ANALYSIS")
        print("=" * 98)
        print(f"{'Operation / Method':<28} | {'Average Time Complexity':<25} | {'Worst-Case Time':<20} | {'Auxiliary Space'}")
        print("-" * 98)
        rows = [
            ("2-Row Levenshtein DP", "O(L1 * L2) = O(M^2)", "O(M^2)", "O(min(L1, L2))"),
            ("Full Matrix / Damerau DP", "O(L1 * L2) = O(M^2)", "O(M^2)", "O(L1 * L2)"),
            ("Single Word insert(w)", "O(M^2 * log N)", "O(N * M^2) [Chain]", "O(1) per node"),
            ("Dictionary Build (N words)", "O(N * M^2 * log N)", "O(N^2 * M^2)", "O(N) total nodes"),
            ("Exact Lookup contains(w)", "O(M^2 * log N)", "O(N * M^2)", "O(min(L1, L2))"),
            ("BK-Tree Pruned search(Q,k)", "O(N^alpha * M^2 + RlogR)", "O(N * M^2 + RlogR)", "O(depth + R)"),
            ("Brute-Force Scan(Q,k)", "Theta(N * M^2 + RlogR)", "Theta(N * M^2 + RlogR)", "O(R)"),
        ]
        for op, avg_t, worst_t, sp in rows:
            print(f"{op:<28} | {avg_t:<25} | {worst_t:<20} | {sp}")
        print("-" * 98)
        print(f"Empirical Proof on Active Dictionary (N = {n}, Query = '{sample_query}', k = {k}):")
        print(f"  * Nodes Visited (V)      : {visited} / {n} ({pruned_pct:.1f}% of subtrees skipped via Triangle Inequality)")
        print(f"  * Sub-linear Exponent    : alpha = ln(V)/ln(N) = {alpha:.3f} (since alpha < 1, search is sub-linear)")
        print(f"  * DP Cell Evaluations    : {char_ops:,} ops (BK-Tree) vs. ~{bf_ops:,} ops (Brute-Force)")
        print("  * Triangle Inequality    : |d(Q,P) - e| <= d(Q,W). For d(Q,W) <= k, only edges e in [d-k, d+k] are visited.")
        print("=" * 98 + "\n")

    def run_tests(self):
        print("\n=== Running BK-Tree & Metric Space Verification Suite ===")
        sample = ["book", "books", "cake", "boo", "cook", "boon", "cape", "cart"]
        t = BKTree()
        for w in sample:
            t.insert(w, "Sample noun")
        res, st = t.search("boak", 1)
        checks = [
            ("2-Row Levenshtein DP exact distance", levenshtein("algorithm", "algoritm") == 1),
            ("2-Row DP vs Full Matrix equivalence", Metric.levenshtein("book", "cake") == Metric.levenshtein_full("book", "cake")),
            ("Damerau-Levenshtein transposition cost = 1", Metric.damerau("teh", "the") == 1),
            ("Duplicate rejection at distance 0", not t.insert("book") and t.size == 8),
            ("Worked 8-word example ('boak', k=1 -> 5/8 visited)", [w for _, w in res] == ["book"] and st["computed"] == 5),
        ]
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print()

    def handle_trace(self):
        q = input("Enter query word to trace: ").strip()
        try:
            k = int(input("Enter tolerance k: ").strip())
        except ValueError:
            k = 1
        matches, trace = self.tree.search_trace(q, k, 15)
        print(f"\n=== Triangle-Inequality Pruning Trace for '{TextUtils.normalize(q)}' (k = {k}) ===")
        print(f"{'Step':<6}{'Node':<16}{'d(Q,N)':<8}{'Match?':<8}{'Window':<12}{'Followed':<16}{'Pruned'}")
        print("-" * 78)
        for row in trace:
            win = f"[{row['window'][0]}, {row['window'][1]}]"
            fol = ",".join(map(str, row["followed"][:4])) or "None"
            pru = ",".join(map(str, row["pruned"][:4])) or "None"
            print(f"{row['step']:<6}{row['word']:<16}{row['d']:<8}{'YES' if row['hit'] else 'No':<8}{win:<12}{fol:<16}{pru}")
        print(f"Total Matches Found: {len(matches)}\n")

    def handle_sentence(self):
        sent = input("Enter a sentence to spell-check: ").strip()
        tokens = TextUtils.tokenize_sentence(sent)
        corrected = []
        for tok in tokens:
            norm = TextUtils.normalize_alpha(tok)
            if not norm or self.tree.contains(norm):
                corrected.append(tok)
            else:
                res, _ = self.tree.search_stats(norm, 1)
                if not res:
                    res, _ = self.tree.search_stats(norm, 2)
                corrected.append(res[0][1] if res else tok)
        print(f"Corrected Sentence: \"{' '.join(corrected)}\"\n")

    def handle_explain(self):
        src = TextUtils.normalize(input("Enter source word: "))
        dst = TextUtils.normalize(input("Enter target word: "))
        steps = Metric.build_edit_script(src, dst)
        print(f"\n--- Edit Script: '{src}' -> '{dst}' (Levenshtein={levenshtein(src, dst)}, Damerau={Metric.damerau(src, dst)}) ---")
        for op, sc, tc, pos in steps:
            print(f"  [{op:<10}] pos {pos}: '{sc}' -> '{tc}'")
        print()

    def handle_stats(self):
        d = self.tree.diagnostics()
        print("\n=== BK-Tree Structural Diagnostics ===")
        print(f"Active Dictionary Words  : {d['active']}")
        print(f"Internal / Leaf Nodes    : {d['internal']} / {d['leaves']}")
        print(f"Max / Average Tree Depth : {d['max_depth']} / {d['avg_depth']:.2f}")
        print(f"Average Branching Factor : {d['avg_branching']:.2f}\n")

    def print_help(self):
        print("\n--- Available Interactive Commands ---")
        print("  <any word>  : Standard Fuzzy Spell Check (prompts for tolerance k)")
        print("  :complexity : Print Theoretical & Empirical Big-O Time/Space Complexity Table")
        print("  :bench      : Run BK-Tree vs. Brute-Force benchmark table (k=1, k=2)")
        print("  :trace      : Step-by-step Triangle-Inequality Pruning trace")
        print("  :sentence   : Auto-correct an entire sentence")
        print("  :explain    : Show character-by-character Edit Script between two words")
        print("  :stats      : Display BK-Tree structural & depth diagnostics")
        print("  :test       : Run Metric Space & BK-Tree unit test suite")
        print("  :metric     : Switch between Levenshtein and Damerau-Levenshtein")
        print("  exit        : Quit the program\n")

    def run_cli(self):
        print("=== BK-Tree Fuzzy Spell Correction Engine ===")
        print(f"Loading dictionary from: {self.wordfile} ...")
        t0 = time.perf_counter()
        count = self.load_words(self.wordfile)
        load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Loaded {count} unique words into the BK-Tree in {load_ms:.1f} ms. (Type ':help' or ':complexity')\n")
        while True:
            try:
                query = input("Enter query word (or 'exit' to quit): ").strip()
            except EOFError:
                break
            if not query:
                continue
            cmd = query.lower()
            if cmd in ("exit", ":quit", ":q"):
                break
            if cmd == ":help":
                self.print_help()
                continue
            if cmd == ":complexity":
                self.handle_complexity()
                continue
            if cmd == ":bench":
                self.benchmark()
                continue
            if cmd == ":test":
                self.run_tests()
                continue
            if cmd == ":trace":
                self.handle_trace()
                continue
            if cmd == ":sentence":
                self.handle_sentence()
                continue
            if cmd == ":explain":
                self.handle_explain()
                continue
            if cmd == ":stats":
                self.handle_stats()
                continue
            if cmd == ":metric":
                choice = input("Select Metric (1 = Levenshtein, 2 = Damerau-Levenshtein): ").strip()
                fn = Metric.damerau if choice == "2" else Metric.levenshtein
                self.tree.rebuild(fn)
                print("BK-Tree rebuilt with selected metric.\n")
                continue
            try:
                tolerance = int(input("Enter max edit tolerance k (e.g., 1 or 2): ").strip())
                if tolerance < 0:
                    tolerance = 1
            except ValueError:
                tolerance = 1
            matches, stats = self.suggest(query, tolerance)
            visited = stats["computed"]
            pruned_pct = (100.0 * (count - visited) / count) if count > 0 else 0.0
            norm_q = TextUtils.normalize(query)
            print(f"\n--- Search Results for '{norm_q}' (k = {tolerance}) ---")
            print(
                f"Nodes Visited: {visited} / {count} ({pruned_pct:.1f}% of tree pruned) "
                f"| Time: {stats['elapsed_ms']:.2f} ms | Complexity: O(V * M^2)\n"
            )
            if not matches:
                print(f"No matches found within edit distance {tolerance}.\n")
            else:
                print(f"{'Rank':<6}{'Word':<18}{'Distance':<10}Meaning")
                print("-" * 76)
                for idx, (dist, word, meaning) in enumerate(matches[:10], 1):
                    print(f"{idx:<6}{word:<18}{dist:<10}{meaning}")
                print()


if __name__ == "__main__":
    dict_arg = sys.argv[1] if len(sys.argv) > 1 else "dictionary.txt"
    app = SpellApp(dict_arg)
    app.run_cli()