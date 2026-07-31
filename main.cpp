#include "httplib.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <map>
#include <limits>
#include <cctype>
#include <unordered_set>
#include <cstdio>

static const int DIMS = 16;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// =====================================================================
//  DATA TYPES
// =====================================================================

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

int clampInt(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// =====================================================================
//  DISTANCE METRICS
// =====================================================================

float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) { float d = a[i]-b[i]; s += d*d; }
    return std::sqrt(s);
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float dot=0, na=0, nb=0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) s += std::abs(a[i]-b[i]);
    return s;
}

DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

// =====================================================================
//  BRUTE FORCE
// =====================================================================

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        for (auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if ((int)r.size() > k) r.resize(k);
        return r;
    }

    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
    }
};

// =====================================================================
//  KD-TREE
// =====================================================================

struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return;
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left  = ins(n->left,  v, d+1);
        else                              n->right = ins(n->right, v, d+1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        float dn = dist(q, n->item.emb);
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop();
        }
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d+1, dist, heap);
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { r.push_back(heap.top()); heap.pop(); }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

// =====================================================================
//  VECTOR DATABASE  (demo 16D index)
// =====================================================================

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
};

// =====================================================================
//  JSON HELPERS
// =====================================================================

std::string jS(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if      (c == '"')  o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) {
            std::ostringstream hex;
            hex << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
            o += hex.str();
        }
        else                o += (char)c;
    }
    return o + '"';
}

std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ','))
        try { v.push_back(std::stof(t)); } catch (...) {}
    return v;
}

// Extract a JSON string field value (handles basic escape sequences)
std::string extractStr(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    if (p >= body.size() || body[p] != '"') return "";
    p++;
    std::string result;
    while (p < body.size()) {
        if (body[p] == '"') break;
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            switch (body[p]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[p]; break;
            }
        } else {
            result += body[p];
        }
        p++;
    }
    return result;
}

// Extract a JSON integer field value
int extractInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    try { return std::stoi(body.substr(p)); } catch (...) { return def; }
}

bool parseBody(const std::string& b, std::string& meta,
               std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat  = extractStr(b, "category");
    auto extractArr = [&](const std::string& key) -> std::vector<float> {
        size_t p = b.find('"' + key + '"');
        if (p == std::string::npos) return {};
        p = b.find('[', p);
        if (p == std::string::npos) return {};
        size_t e = b.find(']', p);
        if (e == std::string::npos) return {};
        return parseVec(b.substr(p + 1, e - p - 1));
    };
    emb = extractArr("embedding");
    return !meta.empty() && !emb.empty();
}

void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

std::string collapseWs(const std::string& s) {
    std::string out;
    bool inWs = false;
    for (unsigned char c : s) {
        if (std::isspace(c)) {
            if (!inWs) out += ' ';
            inWs = true;
        } else {
            out += (char)c;
            inWs = false;
        }
    }
    return trim(out);
}

std::vector<std::string> wordsOf(const std::string& text) {
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);
    return words;
}

std::string joinWords(const std::vector<std::string>& words, int a, int b) {
    std::string out;
    for (int i = a; i < b; i++) {
        if (!out.empty()) out += ' ';
        out += words[i];
    }
    return out;
}

std::vector<std::string> chunkWordsWithOverlap(const std::string& text,
                                               int chunkWords = 250,
                                               int overlapWords = 40)
{
    auto words = wordsOf(text);
    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {collapseWs(text)};
    std::vector<std::string> chunks;
    int step = std::max(1, chunkWords - overlapWords);
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        chunks.push_back(joinWords(words, i, end));
        if (end == (int)words.size()) break;
    }
    return chunks;
}

std::vector<std::string> chunkText(const std::string& text,
                                   int chunkWords = 250, int overlapWords = 40)
{
    std::vector<std::string> sections;
    std::istringstream lines(text);
    std::string line, heading, block;

    auto flush = [&] {
        auto body = collapseWs(block);
        if (!body.empty()) {
            std::string section = heading.empty() ? body : (heading + "\n" + body);
            auto pieces = chunkWordsWithOverlap(section, chunkWords, overlapWords);
            sections.insert(sections.end(), pieces.begin(), pieces.end());
        }
        block.clear();
    };

    while (std::getline(lines, line)) {
        auto t = trim(line);
        bool mdHeading = !t.empty() && t[0] == '#';
        bool underline = t.size() >= 3 &&
                         (t.find_first_not_of("-=") == std::string::npos);
        if ((mdHeading || underline) && !block.empty()) flush();
        if (mdHeading) {
            heading = trim(t.substr(t.find_first_not_of('#')));
        } else if (t.empty()) {
            if ((int)wordsOf(block).size() >= chunkWords / 2) flush();
        } else {
            if (!block.empty()) block += '\n';
            block += t;
        }
    }
    flush();

    if (sections.empty()) return chunkWordsWithOverlap(text, chunkWords, overlapWords);
    return sections;
}

// =====================================================================
//  OLLAMA CLIENT  — wraps local Ollama REST API
//  Install:  https://ollama.com
//  Models:   ollama pull nomic-embed-text
//            ollama pull llama3.2
// =====================================================================

class OllamaClient {
    std::string host;
    int         port;

    // Escape a string for embedding inside a JSON string literal
    std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if      (c == '"')  o += "\\\"";
            else if (c == '\\') o += "\\\\";
            else if (c == '\n') o += "\\n";
            else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t";
            else                o += c;
        }
        return o;
    }

    // Parse {"embedding":[...]} from Ollama /api/embeddings response
    std::vector<float> parseEmbedding(const std::string& body) {
        size_t p = body.find("\"embedding\"");
        if (p == std::string::npos) return {};
        p = body.find('[', p);
        if (p == std::string::npos) return {};
        // Find matching ]  — embeddings can be large (768+ floats)
        size_t e = p + 1, depth = 1;
        while (e < body.size() && depth > 0) {
            if (body[e] == '[') depth++;
            else if (body[e] == ']') depth--;
            e++;
        }
        return parseVec(body.substr(p + 1, e - p - 2));
    }

    // Parse {"response":"..."} from Ollama /api/generate response
    std::string parseResponse(const std::string& body) {
        return extractStr(body, "response");
    }

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel   = "llama3.2:1b";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
        : host(h), port(p) {}

    bool isAvailable() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    // Returns empty vector if Ollama is not running or model not found
    std::vector<float> embed(const std::string& text) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(30, 0);
        std::string body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
        auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) return {};
        return parseEmbedding(res->body);
    }

    // Returns error string if Ollama is unavailable
    std::string generate(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);   // LLMs can be slow
        std::string body = "{\"model\":\"" + genModel + "\","
                           "\"prompt\":\"" + esc(prompt) + "\","
                           "\"stream\":false}";
        auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200)
            return "ERROR: Ollama unavailable. Run: ollama serve";
        return parseResponse(res->body);
    }
};

// =====================================================================
//  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
// =====================================================================

struct DocItem {
    int         id;
    std::string title;
    std::string text;
    std::string source;
    std::string type;
    std::vector<float> emb;
};

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    std::unordered_map<std::string, std::vector<int>> sourceIndex;
    HNSW       hnsw;
    BruteForce bf;       // brute force fallback for small sets
    std::mutex mu;
    int nextId = 1;
    int dims   = 0;      // determined from first inserted embedding
    std::string persistPath = "data/vectors.jsonl";
    bool persistenceEnabled = true;

    static std::vector<std::string> tokens(const std::string& text) {
        std::vector<std::string> out;
        std::string cur;
        for (unsigned char c : text) {
            if (std::isalnum(c)) cur += (char)std::tolower(c);
            else if (!cur.empty()) {
                if (cur.size() > 1) out.push_back(cur);
                cur.clear();
            }
        }
        if (cur.size() > 1) out.push_back(cur);
        return out;
    }

    static float keywordScore(const std::string& query, const DocItem& doc) {
        auto qs = tokens(query);
        if (qs.empty()) return 0.0f;
        auto body = tokens(doc.title + " " + doc.source + " " + doc.type + " " + doc.text);
        std::unordered_set<std::string> seen(body.begin(), body.end());
        int hit = 0;
        for (auto& q : qs) if (seen.count(q)) hit++;
        return (float)hit / (float)qs.size();
    }

    std::string embJson(const std::vector<float>& v) const {
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < v.size(); i++) {
            if (i) ss << ',';
            ss << std::setprecision(9) << v[i];
        }
        ss << ']';
        return ss.str();
    }

    void saveLocked() {
        if (!persistenceEnabled || persistPath.empty()) return;
        std::error_code ec;
        auto parent = std::filesystem::path(persistPath).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent, ec);
        std::ofstream f(persistPath, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return;
        for (auto& [id, d] : store) {
            f << "{\"id\":" << id
              << ",\"title\":" << jS(d.title)
              << ",\"text\":" << jS(d.text)
              << ",\"source\":" << jS(d.source)
              << ",\"type\":" << jS(d.type)
              << ",\"embedding\":" << embJson(d.emb)
              << "}\n";
        }
    }

public:
    DocumentDB() : hnsw(16, 200) {}

    void loadFromDisk(const std::string& path = "data/vectors.jsonl") {
        std::lock_guard<std::mutex> lk(mu);
        persistPath = path;
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            auto title = extractStr(line, "title");
            auto text = extractStr(line, "text");
            auto source = extractStr(line, "source");
            auto type = extractStr(line, "type");
            size_t p = line.find("\"embedding\"");
            if (p == std::string::npos) continue;
            p = line.find('[', p);
            size_t e = line.find(']', p);
            if (p == std::string::npos || e == std::string::npos) continue;
            auto emb = parseVec(line.substr(p + 1, e - p - 1));
            if (title.empty() || emb.empty()) continue;
            if (dims == 0) dims = (int)emb.size();
            if ((int)emb.size() != dims) continue;
            DocItem item{nextId++, title, text,
                         source.empty() ? "persisted" : source,
                         type.empty() ? "text" : type,
                         emb};
            store[item.id] = item;
            sourceIndex[item.source].push_back(item.id);
            VectorItem vi{item.id, item.title, "doc", item.emb};
            hnsw.insert(vi, cosine);
            bf.insert(vi);
        }
    }

    // Insert one chunk with its pre-computed embedding
    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb,
               const std::string& source = "manual",
               const std::string& type = "text")
    {
        std::lock_guard<std::mutex> lk(mu);
        if (dims == 0) dims = (int)emb.size();
        if ((int)emb.size() != dims) return -1;
        DocItem item{nextId++, title, text, source, type, emb};
        store[item.id] = item;
        VectorItem vi{item.id, title, "doc", emb};
        hnsw.insert(vi, cosine);
        bf.insert(vi);
        sourceIndex[source].push_back(item.id);
        saveLocked();
        return item.id;
    }

    // Semantic search — returns top-k most similar chunks
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = 0.75f,
        const std::string& queryText = "",
        const std::string& typeFilter = "",
        const std::string& sourceFilter = "")
    {
        std::lock_guard<std::mutex> lk(mu);
        if (store.empty() || (int)q.size() != dims) return {};
        int candidateLimit = clampInt(std::max(50, k * 10), k, 250);
        auto raw = (store.size() < 10)
                   ? bf.knn(q, candidateLimit, cosine)
                   : hnsw.knn(q, candidateLimit, 80, cosine);
        std::unordered_map<int, float> vectorDist;
        for (auto& [d, id] : raw) vectorDist[id] = d;

        std::vector<std::pair<float, int>> scored;
        for (auto& [id, doc] : store) {
            if (!typeFilter.empty() && doc.type != typeFilter) continue;
            if (!sourceFilter.empty() && doc.source.find(sourceFilter) == std::string::npos) continue;
            float vd = vectorDist.count(id) ? vectorDist[id] : cosine(q, doc.emb);
            if (vd > max_dist && queryText.empty()) continue;
            float semantic = std::max(0.0f, 1.0f - vd);
            float keyword = keywordScore(queryText, doc);
            float finalScore = queryText.empty()
                ? semantic
                : (0.72f * semantic + 0.28f * keyword);
            scored.push_back({1.0f - finalScore, id});
        }
        std::sort(scored.begin(), scored.end());

        std::vector<std::pair<float, DocItem>> out;
        for (auto& [d, id] : scored) {
            if ((int)out.size() >= k) break;
            out.push_back({d, store[id]});
        }
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        auto source = store[id].source;
        store.erase(id); hnsw.remove(id); bf.remove(id);
        auto it = sourceIndex.find(source);
        if (it != sourceIndex.end()) {
            auto& ids = it->second;
            ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
            if (ids.empty()) sourceIndex.erase(it);
        }
        saveLocked();
        return true;
    }

    void replaceSource(const std::string& source,
                       const std::vector<DocItem>& chunks)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto old = sourceIndex.find(source);
        if (old != sourceIndex.end()) {
            for (int id : old->second) {
                store.erase(id);
                hnsw.remove(id);
                bf.remove(id);
            }
            sourceIndex.erase(old);
        }
        for (auto chunk : chunks) {
            if (chunk.emb.empty()) continue;
            if (dims == 0) dims = (int)chunk.emb.size();
            if ((int)chunk.emb.size() != dims) continue;
            chunk.id = nextId++;
            chunk.source = source;
            store[chunk.id] = chunk;
            VectorItem vi{chunk.id, chunk.title, "doc", chunk.emb};
            hnsw.insert(vi, cosine);
            bf.insert(vi);
            sourceIndex[source].push_back(chunk.id);
        }
        saveLocked();
    }

    void removeSource(const std::string& source) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = sourceIndex.find(source);
        if (it == sourceIndex.end()) return;
        for (int id : it->second) {
            store.erase(id);
            hnsw.remove(id);
            bf.remove(id);
        }
        sourceIndex.erase(it);
        saveLocked();
    }

    std::vector<DocItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<DocItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }

    int getDims() {
        std::lock_guard<std::mutex> lk(mu);
        return dims;
    }

    std::string getPersistPath() {
        std::lock_guard<std::mutex> lk(mu);
        return persistPath;
    }
};

struct ExtractedText {
    std::string text;
    std::string type;
    std::string error;
};

std::string lowerExt(const std::filesystem::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return ext;
}

std::string psQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else out += c;
    }
    return out + "'";
}

std::string cmdQuote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    return out + "\"";
}

std::string readFileBytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return "";
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

std::string decodeEntities(const std::string& s) {
    std::string out = s;
    auto repl = [&](const std::string& a, const std::string& b) {
        size_t p = 0;
        while ((p = out.find(a, p)) != std::string::npos) {
            out.replace(p, a.size(), b);
            p += b.size();
        }
    };
    repl("&amp;", "&"); repl("&lt;", "<"); repl("&gt;", ">");
    repl("&quot;", "\""); repl("&apos;", "'"); repl("&nbsp;", " ");
    return out;
}

std::string stripMarkup(const std::string& s) {
    std::string out;
    bool tag = false;
    for (char c : s) {
        if (c == '<') { tag = true; out += ' '; }
        else if (c == '>') { tag = false; out += ' '; }
        else if (!tag) out += c;
    }
    return collapseWs(decodeEntities(out));
}

bool isTextLikeExt(const std::string& ext) {
    static const std::unordered_set<std::string> exts = {
        ".txt",".md",".markdown",".csv",".tsv",".json",".jsonl",
        ".html",".htm",".xml",".yaml",".yml",".log",
        ".cpp",".cc",".cxx",".c",".h",".hpp",".hh",
        ".py",".js",".ts",".tsx",".jsx",".java",".cs",".go",".rs",
        ".php",".rb",".swift",".kt",".sql",".sh",".ps1",".bat",
        ".toml",".ini",".cfg",".env",".dockerfile"
    };
    return exts.count(ext) > 0;
}

ExtractedText extractDocx(const std::filesystem::path& p) {
    auto tmp = std::filesystem::temp_directory_path() /
        ("vectordb_docx_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::error_code ec;
    std::filesystem::create_directories(tmp, ec);
    if (ec) return {"", "docx", "could not create temp directory"};

    std::string cmd = "powershell -NoProfile -ExecutionPolicy Bypass -Command \"Expand-Archive -LiteralPath " +
        psQuote(p.string()) + " -DestinationPath " + psQuote(tmp.string()) + " -Force\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::filesystem::remove_all(tmp, ec);
        return {"", "docx", "PowerShell Expand-Archive failed"};
    }

    auto xml = readFileBytes(tmp / "word" / "document.xml");
    std::filesystem::remove_all(tmp, ec);
    if (xml.empty()) return {"", "docx", "document.xml not found"};
    return {stripMarkup(xml), "docx", ""};
}

ExtractedText extractPdf(const std::filesystem::path& p) {
    auto tmp = std::filesystem::temp_directory_path() /
        ("vectordb_pdf_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".txt");
    std::string cmd = "pdftotext -layout " + cmdQuote(p.string()) + " " + cmdQuote(tmp.string());
    int rc = std::system(cmd.c_str());
    auto text = readFileBytes(tmp);
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    if (rc != 0 || text.empty())
        return {"", "pdf", "pdftotext unavailable or extraction failed"};
    return {collapseWs(text), "pdf", ""};
}

ExtractedText extractTextFromFile(const std::filesystem::path& p) {
    auto ext = lowerExt(p);
    if (ext == ".docx") return extractDocx(p);
    if (ext == ".pdf") return extractPdf(p);
    if (!isTextLikeExt(ext)) return {"", ext.empty() ? "file" : ext.substr(1), "unsupported file type"};

    auto raw = readFileBytes(p);
    if (raw.empty()) return {"", ext.substr(1), ""};
    if (ext == ".html" || ext == ".htm" || ext == ".xml")
        return {stripMarkup(raw), ext.substr(1), ""};
    return {collapseWs(raw), ext.empty() ? "text" : ext.substr(1), ""};
}

// =====================================================================
//  REAL-TIME FILE INGESTION
// =====================================================================

struct WatchedFile {
    std::string path;
    long long stamp = 0;
    int chunks = 0;
    std::string status = "pending";
    std::string type = "";
};

class FileWatcher {
    DocumentDB& db;
    OllamaClient& ollama;
    std::filesystem::path root;
    std::atomic<bool> running{false};
    std::thread worker;
    std::mutex scanMu;
    std::mutex mu;
    std::unordered_map<std::string, WatchedFile> files;
    std::string lastError;
    int intervalMs;

    static bool supported(const std::filesystem::path& p) {
        auto ext = lowerExt(p);
        return isTextLikeExt(ext) || ext == ".pdf" || ext == ".docx";
    }

    static long long stampFor(const std::filesystem::directory_entry& e) {
        auto t = e.last_write_time().time_since_epoch().count();
        std::error_code ec;
        auto s = e.is_regular_file(ec) ? (long long)e.file_size(ec) : 0LL;
        return (long long)t ^ (s << 1);
    }

    static std::string readText(const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open()) return "";
        return std::string(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
    }

    void setStatus(const std::string& path, long long stamp,
                   const std::string& status, int chunks = 0,
                   const std::string& type = "")
    {
        std::lock_guard<std::mutex> lk(mu);
        files[path] = {path, stamp, chunks, status, type};
    }

    void ingestFile(const std::filesystem::path& p, long long stamp) {
        auto path = p.string();
        setStatus(path, stamp, "indexing", 0);

        auto extracted = extractTextFromFile(p);
        if (!extracted.error.empty()) {
            db.removeSource(path);
            std::lock_guard<std::mutex> lk(mu);
            lastError = path + ": " + extracted.error;
            files[path] = {path, stamp, 0, "extract-error", extracted.type};
            return;
        }
        if (extracted.text.empty()) {
            db.removeSource(path);
            setStatus(path, stamp, "empty", 0, extracted.type);
            return;
        }

        auto chunks = chunkText(extracted.text, 250, 40);
        std::vector<DocItem> docs;
        docs.reserve(chunks.size());
        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = ollama.embed(chunks[i]);
            if (emb.empty()) {
                std::lock_guard<std::mutex> lk(mu);
                lastError = "Ollama embedding failed for " + path;
                files[path] = {path, stamp, 0, "ollama-error", extracted.type};
                return;
            }
            std::string title = p.filename().string();
            if (chunks.size() > 1)
                title += " [" + std::to_string(i + 1) + "/" + std::to_string(chunks.size()) + "]";
            docs.push_back({0, title, chunks[i], path, extracted.type, emb});
        }

        db.replaceSource(path, docs);
        setStatus(path, stamp, "indexed", (int)docs.size(), extracted.type);
    }

    void scanOnce() {
        std::lock_guard<std::mutex> scanLock(scanMu);
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec) {
            std::lock_guard<std::mutex> lk(mu);
            lastError = "Could not create watch directory: " + root.string();
            return;
        }

        std::set<std::string> seen;
        for (auto& e : std::filesystem::recursive_directory_iterator(root, ec)) {
            if (ec) break;
            std::error_code fileEc;
            if (!e.is_regular_file(fileEc) || !supported(e.path())) continue;
            auto path = e.path().string();
            auto stamp = stampFor(e);
            seen.insert(path);
            bool changed = false;
            {
                std::lock_guard<std::mutex> lk(mu);
                auto it = files.find(path);
                changed = it == files.end() || it->second.stamp != stamp ||
                          it->second.status == "ollama-error";
            }
            if (changed) ingestFile(e.path(), stamp);
        }

        std::vector<std::string> removed;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (auto& [path, wf] : files)
                if (!seen.count(path)) removed.push_back(path);
        }
        for (auto& path : removed) {
            db.removeSource(path);
            std::lock_guard<std::mutex> lk(mu);
            files.erase(path);
        }
    }

    void loop() {
        while (running) {
            scanOnce();
            for (int slept = 0; running && slept < intervalMs; slept += 200)
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

public:
    FileWatcher(DocumentDB& d, OllamaClient& o,
                std::filesystem::path dir = "documents",
                int interval = 3000)
        : db(d), ollama(o), root(std::move(dir)), intervalMs(interval) {}

    ~FileWatcher() { stop(); }

    void start() {
        if (running.exchange(true)) return;
        worker = std::thread([&]{ loop(); });
    }

    void stop() {
        if (!running.exchange(false)) return;
        if (worker.joinable()) worker.join();
    }

    void rescan() { scanOnce(); }

    std::string directory() const { return root.string(); }
    bool isRunning() const { return running; }

    std::vector<WatchedFile> status(std::string& error) {
        std::lock_guard<std::mutex> lk(mu);
        error = lastError;
        std::vector<WatchedFile> out;
        for (auto& [_, f] : files) out.push_back(f);
        std::sort(out.begin(), out.end(),
                  [](const WatchedFile& a, const WatchedFile& b){ return a.path < b.path; });
        return out;
    }
};

// =====================================================================
//  DEMO DATA  (16D categorical vectors)
// =====================================================================

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");
    // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math",
        {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
}

// =====================================================================
//  HTTP SERVER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    DocumentDB docDB;
    OllamaClient ollama;
    const char* watchEnv = std::getenv("VECTORDB_WATCH_DIR");
    const char* hostEnv  = std::getenv("VECTORDB_HOST");
    const char* portEnv  = std::getenv("VECTORDB_PORT");
    std::string watchDir = watchEnv ? watchEnv : "documents";
    std::string host = hostEnv ? hostEnv : "127.0.0.1";
    int port = portEnv ? clampInt(std::atoi(portEnv), 1, 65535) : 8080;
    FileWatcher watcher(docDB, ollama, watchDir);

    loadDemo(db);

    // Check Ollama at startup (non-fatal)
    bool ollamaUp = ollama.isAvailable();
    std::cout << "=== VectorDB Engine ===" << std::endl;
    std::cout << "http://" << (host == "0.0.0.0" ? "localhost" : host) << ":" << port << std::endl;
    std::cout << db.size() << " demo vectors | " << DIMS << " dims | HNSW+KD-Tree+BruteForce" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)") << std::endl;
    if (ollamaUp) std::cout << "  embed model: " << ollama.embedModel
                            << "  gen model: "   << ollama.genModel << std::endl;
    std::cout << "Watching: " << std::filesystem::absolute(watchDir).string() << std::endl;
    watcher.start();

    httplib::Server svr;

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        k = clampInt(k, 1, 100);
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        auto out = db.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id = db.insert(meta, cat, emb, getDistFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = db.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto items = db.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":"        << v.id
               << ",\"metadata\":"  << jS(v.metadata)
               << ",\"category\":"  << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        k = clampInt(k, 1, 100);
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ',';
            ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ',';
            ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert  {"title":"...","text":"..."}
    // Chunks the text, embeds each chunk via Ollama, stores in DocumentDB
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;

        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = ollama.embed(chunks[i]);
            if (emb.empty()) {
                res.set_content(
                    "{\"error\":\"Ollama unavailable. "
                    "Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2\"}",
                    "application/json");
                return;
            }
            std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            int id = docDB.insert(chunkTitle, chunks[i], emb, "manual:" + title, "manual");
            if (id > 0) ids.push_back(id);
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":"    << docDB.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // DELETE /doc/delete/123
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    // GET /doc/list
    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto docs = docDB.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            // Truncate text preview to 120 chars
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "…";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << jS(docs[i].title)
               << ",\"source\":" << jS(docs[i].source)
               << ",\"type\":" << jS(docs[i].type)
               << ",\"preview\":" << jS(preview)
               << ",\"words\":"  << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1
               << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/search {"question":"...","k":3}
    // Fast retrieval for the UI visualizer
    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = clampInt(extractInt(req.body, "k", 3), 1, 20);
        auto typeFilter = extractStr(req.body, "type");
        auto sourceFilter = extractStr(req.body, "source");
        auto typeFilter = extractStr(req.body, "type");
        auto sourceFilter = extractStr(req.body, "source");
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        auto hits = docDB.search(qEmb, k, 0.75f, question, typeFilter, sourceFilter);

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
                << ",\"title\":"    << jS(hits[i].second.title)
                << ",\"source\":"   << jS(hits[i].second.source)
               << ",\"type\":"     << jS(hits[i].second.type)
                << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/ask  {"question":"...","k":3}
    // Full RAG pipeline: embed → retrieve → generate
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = clampInt(extractInt(req.body, "k", 3), 1, 20);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        // Step 1: embed the question
        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        // Step 2: retrieve top-k relevant chunks
        auto hits = docDB.search(qEmb, k, 0.75f, question, typeFilter, sourceFilter);

        // Step 3: build prompt
        std::ostringstream ctx;
        for (int i = 0; i < (int)hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }
        std::string prompt =
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            "Just answer the question naturally.\n\n"
            "Context:\n" + ctx.str() +
            "Question: " + question + "\n\n"
            "Answer:";

        // Step 4: generate answer
        auto answer = ollama.generate(prompt);

        // Step 5: return everything
        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(ollama.genModel)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"source\":"   << jS(hits[i].second.source)
               << ",\"type\":"     << jS(hits[i].second.type)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << jS(ollama.embedModel)
           << ",\"genModel\":"         << jS(ollama.genModel)
           << ",\"docCount\":"         << docDB.size()
           << ",\"docDims\":"          << docDB.getDims()
           << ",\"persistPath\":"       << jS(std::filesystem::absolute(docDB.getPersistPath()).string())
           << ",\"watchDir\":"          << jS(std::filesystem::absolute(watcher.directory()).string())
           << ",\"watcherRunning\":"    << (watcher.isRunning() ? "true" : "false")
           << ",\"demoDims\":"         << DIMS
           << ",\"demoCount\":"        << db.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/watch/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::string error;
        auto files = watcher.status(error);
        std::ostringstream ss;
        ss << "{\"running\":" << (watcher.isRunning() ? "true" : "false")
           << ",\"directory\":" << jS(std::filesystem::absolute(watcher.directory()).string())
           << ",\"lastError\":" << jS(error)
           << ",\"files\":[";
        for (size_t i = 0; i < files.size(); i++) {
            if (i) ss << ',';
            ss << "{\"path\":" << jS(files[i].path)
               << ",\"chunks\":" << files[i].chunks
               << ",\"type\":" << jS(files[i].type)
               << ",\"status\":" << jS(files[i].status) << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/watch/rescan", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        watcher.rescan();
        res.set_content("{\"ok\":true}", "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"count\":"      << db.size()
           << ",\"dims\":"       << DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json");
    });

    // Serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html");
    });

    svr.listen(host, port);
    return 0;
}
